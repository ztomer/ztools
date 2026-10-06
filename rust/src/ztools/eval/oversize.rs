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
//!
//! BOTH READERS, ON BOTH SUPPORTED PLATFORMS. Headroom used to come from
//! `vm_stat` alone, so on Linux -- which standing policy still supports -- it
//! returned `Err` and every measurement on that platform was refused with
//! "cannot read memory headroom". That is the safe direction, but it is still
//! the class this module keeps hitting: a supported platform reading nothing.
//! Linux's `MemAvailable` is the honest counterpart (see
//! [`meminfo_available_gb_in`]), and `signals_platform` carries the same
//! argument for the pressure half.

use std::path::Path;

use crate::units::{unsigned, whole_u64};
use crate::ztools::eval::model_resolve::model_config_path;
use crate::ztools::eval::signals::{
    PROC_MEMINFO, VM_STAT, file_text, memory_pressure, thrashing_verdict, tool_output,
};

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
#[must_use]
pub fn model_disk_bytes(model: &str) -> Option<u64> {
    let config = model_config_path(model)?;
    let directory = config.parent()?;
    let mut total: u64 = 0;
    for entry in std::fs::read_dir(directory).ok()?.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors")
            && let Ok(meta) = path.metadata()
        {
            total += meta.len();
        }
    }
    if total == 0 { None } else { Some(total) }
}

/// Memory a model needs, in GB, from its weight files where they can be found.
///
/// Rounded UP: a model needs at least its weights plus room for activations
/// and a KV cache, so the honest direction for a memory estimate is generous.
/// Falls back to the name only for models with nothing on disk to measure.
#[must_use]
pub fn estimate_model_memory_gb(model: &str) -> u64 {
    if let Some(disk) = model_disk_bytes(model) {
        return whole_u64((unsigned(disk) / BYTES_PER_GB).ceil().max(1.0));
    }
    // The parameter count in the name, e.g. "ornith-1.0-35b-mxfp8" -> 35.
    let lower = model.to_lowercase();
    let start = lower.find('b').map(|b| {
        lower[..b]
            .chars()
            .rev()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
    });
    if let Some(digits) = start
        && let Ok(n) = digits.chars().rev().collect::<String>().parse::<u64>()
    {
        return n.max(1);
    }
    4
}

/// One page counter out of one `vm_stat` reading. Pure over the text.
fn vm_stat_pages_in(text: &str, label: &str) -> Option<f64> {
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
/// Returns Err rather than degrading when no headroom reader is available:
/// "cannot read memory" must not become a number that looks fine and is simply
/// wrong.
///
/// # Errors
///
/// When neither reader on this host could be read, or the one that answered is
/// missing the fields it needs. The message names what was tried, because the
/// likely causes are different per platform and guessing at one is how a
/// refusal becomes an outage nobody can diagnose.
pub fn reclaimable_available_gb() -> Result<f64, String> {
    reclaimable_available_gb_from(Path::new(VM_STAT), Path::new(PROC_MEMINFO))
}

/// [`reclaimable_available_gb`] from injected paths, in the pressure reader's precedence.
///
/// `/proc/meminfo` when it exists -- it is a stat, not a spawn, and macOS has no
/// `/proc` at all, so the order cannot misclassify a real host -- else
/// `vm_stat`.
///
/// # Errors
///
/// When the chosen reader is unreadable, or the chosen reader's output is
/// missing a field. The message names that reader AND says which the other one
/// is, so a reader on the wrong platform is not left guessing.
pub fn reclaimable_available_gb_from(vm_stat: &Path, meminfo: &Path) -> Result<f64, String> {
    if meminfo.exists() {
        let text = file_text(meminfo).ok_or_else(|| {
            headroom_read_error(PROC_MEMINFO, &format!("{VM_STAT} (macOS; not this host)"))
        })?;
        return meminfo_available_gb_in(&text);
    }
    let text = tool_output(vm_stat, &[]).ok_or_else(|| {
        headroom_read_error(VM_STAT, &format!("{PROC_MEMINFO} (Linux; not this host)"))
    })?;
    reclaimable_available_gb_in(&text)
}

/// The refusal text for a headroom reader that could not be read.
///
/// It names the reader that answered `None` AND the one that was not tried
/// because it does not exist here. A message naming only one of them is the
/// shape that sends the next reader to the wrong platform.
fn headroom_read_error(tried: &str, not_here: &str) -> String {
    format!(
        "cannot read memory headroom from {tried} on this machine ({not_here} does not exist here)"
    )
}

/// Linux headroom: `MemAvailable`, in GiB.
///
/// The honest analogue, and the reason this is not a second refusal on a
/// supported platform: proc(5) defines `MemAvailable` as "an estimate of how much
/// memory is available for starting new applications, WITHOUT SWAPPING",
/// computed from `MemFree` plus reclaimable page cache and slab. That is
/// exactly the quantity the macOS arithmetic above APPROXIMATES by hand -- and
/// it is the kernel's own figure rather than a sum this repo guesses at, so it
/// does not inherit the over-subtraction caveat above.
///
/// Its limit, stated rather than hidden: it is an ESTIMATE, computed from a
/// recent-reading watermark and a fraction of page cache, so a box under
/// pressure can still report more headroom than it can honour. The direction of
/// that error is toward measuring a model as fitting that does not, which is
/// the same exposure the thrashing gate exists to catch -- so both are read
/// together and neither is trusted alone.
///
/// # Errors
///
/// When `text` has no readable `MemAvailable`, which is a kernel older than 3.14
/// (the line arrived there) or a truncated read. Named rather than guessed at:
/// substituting `MemFree` would silently turn "headroom including reclaimable
/// cache" into "headroom excluding it", understating by gigabytes.
pub fn meminfo_available_gb_in(text: &str) -> Result<f64, String> {
    let kib = text
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse::<f64>().ok())
        .ok_or_else(|| {
            headroom_read_error(PROC_MEMINFO, &format!("{VM_STAT} (macOS; not this host)"))
        })?;
    Ok(kib / (1024.0 * 1024.0))
}

/// [`reclaimable_available_gb`]'s arithmetic over ONE reading.
///
/// It used to spawn `vm_stat` FIVE times, once per page counter, so the five
/// numbers came from five different moments and their sum described no instant
/// at all. One reading, one sum.
///
/// # Errors
///
/// When `text` is missing any of the five page counters this sums. The message
/// names the missing line, because a changed `vm_stat` format is the likely
/// cause and guessing at it would report a plausible-looking wrong number
/// instead.
pub fn reclaimable_available_gb_in(text: &str) -> Result<f64, String> {
    let page = |label: &str| {
        vm_stat_pages_in(text, label).ok_or_else(|| format!("vm_stat: cannot read '{label}'"))
    };
    let free = page("Pages free")?;
    let inactive = page("Pages inactive")?;
    let speculative = page("Pages speculative")?;
    let purgeable = page("Pages purgeable")?;
    let file_backed = page("File-backed pages")?;

    let available = (free + inactive + speculative) * PAGE_BYTES / BYTES_PER_GB;
    let active_file_backed =
        (file_backed - inactive - speculative).max(0.0) * PAGE_BYTES / BYTES_PER_GB;
    let purgeable_gb = purgeable * PAGE_BYTES / BYTES_PER_GB;
    Ok(available + active_file_backed + purgeable_gb)
}

/// Is the machine already paying for memory it does not have? None means
/// "cannot tell", which is not evidence of thrashing either way.
#[must_use]
pub fn is_thrashing() -> Option<bool> {
    thrashing_verdict(memory_pressure())
}

/// Why this model must not be measured here, or "" to proceed.
///
/// Both `available_gb` and `thrashing` are injectable so every branch is
/// testable without a 28.8GB model or a deliberately wrecked machine.
#[must_use]
pub fn oversize_refusal(
    model_gb: f64,
    available_gb: Option<f64>,
    allow: bool,
    thrashing: Option<bool>,
) -> String {
    if allow || std::env::var_os(OVERSIZE_OVERRIDE_ENV).is_some() {
        return String::new();
    }

    // ONE reading, used for BOTH the verdict and the detail. It used to call
    // `is_thrashing()` and then `memory_pressure()` again: two spawns of
    // `sysctl` and `vm_stat` each, and a swap figure that crossed
    // MAX_CLEAN_SWAP_GB between them produced a message describing a state the
    // machine was never in. An INJECTED verdict needs no reading at all.
    let injected = thrashing;
    let pressure = if injected.is_some() {
        None
    } else {
        memory_pressure()
    };
    let thrashing = injected.unwrap_or_else(|| thrashing_verdict(pressure).unwrap_or_default());
    if thrashing {
        // The reading names its own platform, and a Linux reading says in the
        // message that swap was the only quantity that gated it. A detail that
        // read "swap 9.0GB" on both platforms would present a one-signal
        // conclusion as the two-signal one the macOS message is.
        let detail =
            pressure.map_or_else(String::new, |reading| format!(" ({})", reading.describe()));
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
    let limit_pct = whole_u64((OVERSIZE_MEMORY_FRACTION * 100.0).round());
    format!(
        "needs ~{model_gb:.0}GB against {available_gb:.0}GB reclaimable \
         (limit {limit_pct}%). A timing taken here would \
         describe the swapping, not the model. Re-run on a quieter machine, or \
         set {OVERSIZE_OVERRIDE_ENV}=1 to measure it deliberately."
    )
}

#[cfg(test)]
#[path = "oversize_tests.rs"]
mod tests;
