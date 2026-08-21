//! Oversize / thrashing refusal: why a model must not be measured on this
//! box right now.
//!
//! Ported from `eval/cli_runtime.py::oversize_refusal` + `eval/memory.py`'s
//! reclaimable-headroom arithmetic. A REFUSAL, not a warning: the warn-and-
//! continue it replaced produced a 0.1158 tok/s decode reading for a 27B model,
//! which `max_tokens / decode` turned into a ~138,000s derived timeout, which
//! then permitted a wedged server to idle 83 minutes. A timing taken while the
//! box swaps describes the swapping -- and it hardens into config exactly like
//! a real number.
//!
//! WHY PRESSURE IS ASKED FIRST. Headroom is the misleading quantity here:
//! after a sweep the page cache holds the previous model's weights as `active`
//! file-backed pages, which a naive free-memory read reports as unavailable
//! even though the kernel evicts them for free. Thrashing (swap/compressor)
//! is unambiguous by comparison -- it describes a machine ALREADY paying for
//! memory it does not have -- so it is disqualifying on its own, and headroom
//! is measured against what is RECLAIMABLE.

use crate::ztools::eval::model_resolve::model_config_path;
use crate::ztools::eval::signals::{memory_pressure, MAX_CLEAN_COMPRESSOR_GB, MAX_CLEAN_SWAP_GB};

/// Escape hatch for the deliberate case: measuring whether an oversize model
/// can run here AT ALL is a legitimate experiment; the refusal must not make
/// it impossible -- only conscious.
pub const OVERSIZE_OVERRIDE_ENV: &str = "EVAL_ALLOW_OVERSIZE";

/// Fraction of RECLAIMABLE memory a model's weights may occupy. Weights are
/// only part of the footprint -- activations and KV cache come on top.
///
/// PROVISIONAL, carried forward unchanged from the Python gate so the port
/// changes the CONSEQUENCE without silently changing the threshold.
pub const OVERSIZE_MEMORY_FRACTION: f64 = 0.8;

const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;
const PAGE_BYTES: f64 = 16384.0;

/// Total size of a model's weight files, or None if not found on disk.
///
/// On-disk bytes is what predicts fitting: qwen3.8-27b-4bit and -mxfp8 are both
/// "27b" by name and occupy 15GB and 27GB respectively. Counts weight shards
/// only; tokenizers and configs are noise at this scale.
pub fn model_disk_bytes(model: &str) -> Option<u64> {
    let config = model_config_path(model)?;
    let directory = config.parent()?;
    let mut total: u64 = 0;
    for entry in std::fs::read_dir(directory).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            if let Ok(meta) = path.metadata() {
                total += meta.len();
            }
        }
    }
    if total == 0 {
        None
    } else {
        Some(total)
    }
}

/// Memory a model needs, in GB, from its weight files where they can be found.
///
/// Rounded UP: a model needs at least its weights plus room for activations
/// and a KV cache, so the honest direction for a memory estimate is generous.
/// Falls back to the name only for models with nothing on disk to measure.
pub fn estimate_model_memory_gb(model: &str) -> u64 {
    if let Some(disk) = model_disk_bytes(model) {
        return ((disk as f64) / BYTES_PER_GB).ceil().max(1.0) as u64;
    }
    // The parameter count in the name, e.g. "ornith-1.0-35b-mxfp8" -> 35.
    let lower = model.to_lowercase();
    let start = lower.find('b').map(|b| {
        lower[..b]
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
    });
    if let Some(digits) = start {
        if let Ok(n) = digits.chars().rev().collect::<String>().parse::<u64>() {
            return n.max(1);
        }
    }
    4
}

fn vm_stat_pages(label: &str) -> Option<f64> {
    let out = std::process::Command::new("/usr/bin/vm_stat").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .find(|l| l.starts_with(label))?
        .split(':')
        .nth(1)?
        .trim()
        .trim_end_matches('.')
        .parse()
        .ok()
}

/// Memory a model can have, counting what the kernel would evict to give it.
///
/// psutil's macOS `available` covers free + inactive + speculative pages, but
/// MISSES clean file-backed pages currently `active` -- precisely what holds a
/// previously-loaded model's weights after a sweep. Those are estimated by
/// subtracting inactive+speculative from the file-backed total, which over-
/// subtracts and therefore UNDERSTATES reclaimable memory: the safe direction
/// for a gate whose failure mode is producing a wrong number.
///
/// Returns Err rather than degrading when vm_stat cannot be read: "vm_stat is
/// broken" must not become a number that looks fine and is simply wrong.
pub fn reclaimable_available_gb() -> Result<f64, String> {
    let free = vm_stat_pages("Pages free")
        .ok_or_else(|| "vm_stat: cannot read 'Pages free'".to_string())?;
    let inactive = vm_stat_pages("Pages inactive")
        .ok_or_else(|| "vm_stat: cannot read 'Pages inactive'".to_string())?;
    let speculative = vm_stat_pages("Pages speculative")
        .ok_or_else(|| "vm_stat: cannot read 'Pages speculative'".to_string())?;
    let purgeable = vm_stat_pages("Pages purgeable")
        .ok_or_else(|| "vm_stat: cannot read 'Pages purgeable'".to_string())?;
    let file_backed = vm_stat_pages("File-backed pages")
        .ok_or_else(|| "vm_stat: cannot read 'File-backed pages'".to_string())?;

    let available = (free + inactive + speculative) * PAGE_BYTES / BYTES_PER_GB;
    let active_file_backed = (file_backed - inactive - speculative).max(0.0) * PAGE_BYTES
        / BYTES_PER_GB;
    let purgeable_gb = purgeable * PAGE_BYTES / BYTES_PER_GB;
    Ok(available + active_file_backed + purgeable_gb)
}

/// Is the machine already paying for memory it does not have? None means
/// "cannot tell", which is not evidence of thrashing either way.
pub fn is_thrashing() -> Option<bool> {
    let (swap_gb, compressor_gb) = memory_pressure()?;
    Some(swap_gb > MAX_CLEAN_SWAP_GB || compressor_gb > MAX_CLEAN_COMPRESSOR_GB)
}

/// Why this model must not be measured here, or "" to proceed.
///
/// Both `available_gb` and `thrashing` are injectable so every branch is
/// testable without a 28.8GB model or a deliberately wrecked machine.
pub fn oversize_refusal(
    model_gb: f64,
    available_gb: Option<f64>,
    allow: bool,
    thrashing: Option<bool>,
) -> String {
    if allow || std::env::var_os(OVERSIZE_OVERRIDE_ENV).is_some() {
        return String::new();
    }

    let thrashing = match thrashing {
        Some(t) => t,
        None => is_thrashing().unwrap_or_default(),
    };
    if thrashing {
        let detail = match memory_pressure() {
            Some((swap, compressor)) => format!(" (swap {swap:.1}GB, compressor {compressor:.1}GB)"),
            None => String::new(),
        };
        return format!(
            "the machine is already paging{detail}. A timing taken here would \
             describe the paging, not the model. Wait for it to settle, or set \
             {OVERSIZE_OVERRIDE_ENV}=1 to measure it deliberately."
        );
    }

    let available_gb = match available_gb {
        Some(gb) => gb,
        None => match reclaimable_available_gb() {
            Ok(gb) => gb,
            Err(_) => {
                // Cannot tell how much headroom exists. Not evidence of a bad
                // fit either -- but the refusal message must say why we stopped.
                return format!(
                    "cannot read memory headroom on this machine; refusing to \
                     measure blind. Set {OVERSIZE_OVERRIDE_ENV}=1 to override."
                );
            }
        },
    };
    if model_gb <= available_gb * OVERSIZE_MEMORY_FRACTION {
        return String::new();
    }
    let limit_pct = (OVERSIZE_MEMORY_FRACTION * 100.0).round() as u64;
    format!(
        "needs ~{model_gb:.0}GB against {available_gb:.0}GB reclaimable \
         (limit {limit_pct}%). A timing taken here would \
         describe the swapping, not the model. Re-run on a quieter machine, or \
         set {OVERSIZE_OVERRIDE_ENV}=1 to measure it deliberately."
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    fn oversize_headroom_branches_are_exact() {
        // Fits comfortably under the 80% line.
        assert_eq!(oversize_refusal(10.0, Some(50.0), false, Some(false)), "");
        // Needs more than 80% of reclaimable.
        let r = oversize_refusal(28.0, Some(31.0), false, Some(false));
        assert!(r.contains("needs ~28GB against 31GB reclaimable"), "{r}");
        assert!(r.contains("limit 80%"), "{r}");
        // Thrashing disqualifies on its own, regardless of headroom.
        let r = oversize_refusal(1.0, Some(500.0), false, Some(true));
        assert!(r.contains("already paging"), "{r}");
        // Cannot-tell pressure does not refuse on its own.
        assert_eq!(oversize_refusal(1.0, Some(50.0), false, None), "");
        // The deliberate escape hatch wins over everything.
        assert_eq!(oversize_refusal(28.0, Some(31.0), true, Some(true)), "");
    }

    #[test]
    #[serial]
    fn the_env_override_matches_the_explicit_allow() {
        let prev = std::env::var_os(OVERSIZE_OVERRIDE_ENV);
        std::env::set_var(OVERSIZE_OVERRIDE_ENV, "1");
        let r = oversize_refusal(28.0, Some(31.0), false, Some(false));
        match prev {
            Some(v) => std::env::set_var(OVERSIZE_OVERRIDE_ENV, v),
            None => std::env::remove_var(OVERSIZE_OVERRIDE_ENV),
        }
        assert_eq!(r, "");
    }

    #[test]
    fn name_fallback_estimates_from_the_parameter_count_not_the_whole_name() {
        assert_eq!(estimate_model_memory_gb("totally-unknown-model"), 4);
        // "27b-4bit" and "27b-mxfp8" are BOTH 27 by name; the disk path is what
        // tells them apart, and this fallback is only for models with no disk.
        assert_eq!(estimate_model_memory_gb("qwen3.8-27b-4bit-nodisk"), 27);
        assert_eq!(estimate_model_memory_gb("4m-embedding"), 4);
    }

    #[test]
    #[serial]
    fn disk_bytes_come_from_weight_shards_only() {
        let dir = tempfile::tempdir().unwrap();
        // models_dir/hf layout via MLX_MODELS_DIR env seam.
        let models_root = dir.path().join("MLXModels/TestOrg/TestModel-2b");
        std::fs::create_dir_all(&models_root).unwrap();
        std::fs::write(models_root.join("config.json"), "{}").unwrap();
        std::fs::write(models_root.join("model-a.safetensors"), vec![0u8; 1000]).unwrap();
        std::fs::write(models_root.join("tokenizer.json"), b"noise").unwrap();

        let prev = std::env::var_os("MLX_MODELS_DIR");
        std::env::set_var("MLX_MODELS_DIR", dir.path().join("MLXModels"));
        let bytes = model_disk_bytes("testmodel-2b");
        match prev {
            Some(v) => std::env::set_var("MLX_MODELS_DIR", v),
            None => std::env::remove_var("MLX_MODELS_DIR"),
        }
        assert_eq!(bytes, Some(1000), "tokenizers are excluded");
    }
}
