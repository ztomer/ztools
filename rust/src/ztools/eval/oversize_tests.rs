//! `oversize`'s tests.
//!
//! Three rules, all of them bought here:
//!
//! - A test that touches the disk seams takes the one guard (`TestEnv`), which
//!   holds a process-wide lock -- `#[serial]` is one-sided and excludes serial
//!   tests only from each other, not from the non-serial siblings cargo runs
//!   alongside them.
//! - The thrashing rule is pinned as a PURE predicate over one reading. The
//!   live reader used to be compared against a second live reading of the same
//!   machine, so the test went red whenever swap crossed 8GB between two
//!   `sysctl` spawns -- which is precisely when this repo runs.
//! - A live check degrades on a machine that has no macOS `vm_stat` instead of
//!   panicking, because Linux x86-64 and aarch64 are supported and the suite
//!   has to run there.

use super::*;
use crate::test_env::TestEnv;

/// A path that is not there, standing in for a reader this host does not have.
const ABSENT_HEADROOM: &str = "/nonexistent/ztools-no-such-reader";
use crate::ztools::eval::signals::{
    MAX_CLEAN_RECLAIM_GB, MAX_CLEAN_SWAP_GB, MemoryPressure, PressureSource,
};
use serial_test::serial;

/// A `vm_stat` reading with the five page counters this module sums, so the
/// fixture is the tool's real output SHAPE and not a hand-written number.
const RECLAIMABLE_VM_STAT: &str = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               65536.
Pages active:                            131072.
Pages inactive:                            65536.
Pages speculative:                          65536.
Pages wired down:                         65536.
File-backed pages:                       262144.
Pages purgeable:                            65536.
Pages occupied by compressor:              65536.
";

/// The same tool's output with one counter missing, which is a refusal that
/// names the line rather than a smaller number.
const COMPRESSOR_VM_STAT: &str = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               65536.
Pages occupied by compressor:              65536.
";

#[test]
#[serial]
fn oversize_headroom_branches_are_exact() {
    let env = TestEnv::new();
    // Fits comfortably under the 80% line.
    assert_eq!(oversize_refusal(10.0, Some(50.0), false, Some(false)), "");
    // Needs more than 80% of reclaimable.
    let r = oversize_refusal(28.0, Some(31.0), false, Some(false));
    assert!(r.contains("needs ~28GB against 31GB reclaimable"), "{r}");
    assert!(r.contains("limit 80%"), "{r}");
    // Thrashing disqualifies on its own, regardless of headroom.
    let r = oversize_refusal(1.0, Some(500.0), false, Some(true));
    assert!(r.contains("already paging"), "{r}");
    assert!(
        !r.contains("cannot read memory headroom"),
        "injected headroom must not be replaced by a real read"
    );
    // The deliberate escape hatch wins over everything.
    assert_eq!(oversize_refusal(28.0, Some(31.0), true, Some(true)), "");
    drop(env);
}

/// The single-sample property, observable from the outside: a refusal built
/// from an INJECTED verdict cannot carry a pressure detail, because there was
/// no reading to format one from. The detail used to come from a SECOND
/// `memory_pressure()` call after `is_thrashing()` had already spawned
/// `sysctl` and `vm_stat` once each -- and a swap figure that crossed
/// `MAX_CLEAN_SWAP_GB` between the two spawns produced a refusal describing a
/// state the machine was never in.
#[test]
#[serial]
fn an_injected_thrashing_verdict_carries_no_reading() {
    let env = TestEnv::new();
    let r = oversize_refusal(1.0, Some(500.0), false, Some(true));
    assert!(r.contains("already paging"), "{r}");
    assert!(
        !r.contains("swap") && !r.contains("compressor"),
        "an injected verdict must not have read the machine to decorate the \
         message: {r}"
    );
    assert!(
        r.contains(OVERSIZE_OVERRIDE_ENV),
        "the refusal must name the escape hatch: {r}"
    );
    // The cannot-tell direction: an unreadable reading is not thrashing.
    assert_eq!(thrashing_verdict(None), None);
    drop(env);
}

/// The five-counter arithmetic, against the tool's real output shape.
#[test]
fn reclaimable_headroom_is_one_reading_summed() {
    // 3GB available (free+inactive+speculative) + 2GB active file-backed
    // (file-backed 2GiB minus the 1GiB already counted) + 1GB purgeable.
    assert_eq!(reclaimable_available_gb_in(RECLAIMABLE_VM_STAT), Ok(6.0));
    let err = reclaimable_available_gb_in(COMPRESSOR_VM_STAT).unwrap_err();
    assert_eq!(
        err, "vm_stat: cannot read 'Pages inactive'",
        "a stubbed tool missing a counter is refused by name, not summed short"
    );
}

#[test]
#[serial]
fn the_env_override_matches_the_explicit_allow() {
    let env = TestEnv::new();
    assert_eq!(OVERSIZE_OVERRIDE_ENV, "EVAL_ALLOW_OVERSIZE");
    env.set(OVERSIZE_OVERRIDE_ENV, "1");
    let r = oversize_refusal(28.0, Some(31.0), false, Some(false));
    assert_eq!(r, "");
    drop(env);
}

#[test]
#[serial]
fn the_env_override_leaves_the_operators_environment_untouched() {
    let env = TestEnv::new();
    assert_eq!(OVERSIZE_OVERRIDE_ENV, "EVAL_ALLOW_OVERSIZE");
    assert!(
        std::env::var_os(OVERSIZE_OVERRIDE_ENV).is_none(),
        "the guard clears it, so this test exercises the never-set arm"
    );
    // With no override in the environment the refusal is the ORDINARY one.
    // That is the assertion that matters: an `EVAL_ALLOW_OVERSIZE` left over
    // from a previous test would turn this into "".
    let r = oversize_refusal(28.0, Some(31.0), false, Some(false));
    assert!(r.contains("needs ~28GB against 31GB reclaimable"), "{r}");
    drop(env);
}

#[test]
fn name_fallback_estimates_from_the_parameter_count_not_the_whole_name() {
    assert_eq!(estimate_model_memory_gb("totally-unknown-model"), 4);
    // "27b-4bit" and "27b-mxfp8" are BOTH 27 by name; the disk path is what
    // tells them apart, and this fallback is only for models with no disk.
    assert_eq!(estimate_model_memory_gb("qwen3.8-27b-4bit-nodisk"), 27);
    assert_eq!(estimate_model_memory_gb("4m-embedding"), 4);
    // Uppercase names resolve identically to lowercase ones.
    assert_eq!(estimate_model_memory_gb("ORNITH-1.0-35B-MXFP8"), 35);
}

#[test]
#[serial]
fn disk_bytes_come_from_weight_shards_only() {
    let env = TestEnv::new();
    // models_dir/hf layout via the `MLX_MODELS_DIR` env seam.
    let models_root = env.path("MLX_MODELS_DIR").join("TestOrg/TestModel-2b");
    std::fs::create_dir_all(&models_root).unwrap();
    std::fs::write(models_root.join("config.json"), "{}").unwrap();
    std::fs::write(models_root.join("model-a.safetensors"), vec![0u8; 1000]).unwrap();
    std::fs::write(models_root.join("tokenizer.json"), b"noise").unwrap();

    let bytes = model_disk_bytes("testmodel-2b");
    assert_eq!(bytes, Some(1000), "tokenizers are excluded");
    drop(env);
}

#[test]
#[serial]
fn unknown_models_and_shardless_directories_measure_nothing() {
    let env = TestEnv::new();
    assert_eq!(model_disk_bytes("no-such-model"), None);

    // A directory that exists but holds no weight shards is also nothing:
    // configs and tokenizers do not make a model loadable.
    let shardless = env.path("MLX_MODELS_DIR").join("Org/BareModel");
    std::fs::create_dir_all(&shardless).unwrap();
    std::fs::write(shardless.join("config.json"), "{}").unwrap();
    std::fs::write(shardless.join("tokenizer.json"), b"noise").unwrap();
    assert_eq!(model_disk_bytes("baremodel"), None);
    assert_eq!(model_disk_bytes("org/baremodel/nested-deeper"), None);
    drop(env);
}

#[test]
#[serial]
fn disk_estimates_round_up_and_never_report_less_than_one_gb() {
    let env = TestEnv::new();
    let model_dir = env.path("MLX_MODELS_DIR").join("Org/TinyModel");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), "{}").unwrap();
    std::fs::write(model_dir.join("w.safetensors"), vec![0u8; 1000]).unwrap();
    assert_eq!(
        estimate_model_memory_gb("tinymodel"),
        1,
        "1000 bytes rounds up to at least 1GB"
    );

    let big_dir = env.path("MLX_MODELS_DIR").join("Org/BigModel");
    std::fs::create_dir_all(&big_dir).unwrap();
    std::fs::write(
        big_dir.join("w1.safetensors"),
        vec![0u8; 2 * 1024 * 1024 * 1024 + 1],
    )
    .unwrap();
    std::fs::write(
        big_dir.join("w2.safetensors"),
        vec![0u8; 1024 * 1024 * 1024],
    )
    .unwrap();
    assert_eq!(
        estimate_model_memory_gb("bigmodel"),
        4,
        "2GiB+1 plus 1GiB sums to just over 3GB and rounds UP"
    );
    drop(env);
}

/// The `vm_stat` reader, pinned against the real output SHAPE.
#[test]
fn vm_stat_page_parsing_finds_real_labels_and_rejects_unknown_ones() {
    const VM_STAT_TEXT: &str = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               12345.
Pages active:                            987654.
Pages inactive:                            4321.
Pages speculative:                          321.
Pages throttled:                              0.
Pages wired down:                        222222.
File-backed pages:                       555555.
Pages purgeable:                            999.
Pages occupied by compressor:             12345.
";
    assert_eq!(vm_stat_pages_in(VM_STAT_TEXT, "Pages free"), Some(12345.0));
    assert_eq!(
        vm_stat_pages_in(VM_STAT_TEXT, "File-backed pages"),
        Some(555_555.0)
    );
    assert_eq!(
        vm_stat_pages_in(VM_STAT_TEXT, "Pages occupied by compressor"),
        Some(12_345.0)
    );
    assert_eq!(vm_stat_pages_in(VM_STAT_TEXT, "No Such Label Exists"), None);
}

/// The arithmetic, pinned against one reading.
#[test]
fn reclaimable_headroom_is_free_plus_inactive_plus_speculative_plus_active_file_backed_plus_purgeable()
 {
    // 1 page = 16384 B and 1 GiB = 2^30 B, so each page is 1/65536 GiB.
    let text = "\
Pages free: 65536.
Pages inactive: 65536.
Pages speculative: 65536.
Pages purgeable: 65536.
File-backed pages: 65536.
";
    // free+inactive+speculative = 3 GiB; file-backed(1) - inactive(1) -
    // speculative(1) floors at 0, so no active-file-backed term; purgeable 1.
    assert_eq!(reclaimable_available_gb_in(text), Ok(4.0));

    // More file-backed than inactive+speculative: the active file-backed term
    // appears, which is the whole reason this is not a plain free-memory read.
    let with_active = "\
Pages free: 0.
Pages inactive: 0.
Pages speculative: 0.
Pages purgeable: 0.
File-backed pages: 131072.
";
    assert_eq!(reclaimable_available_gb_in(with_active), Ok(2.0));
}

/// A missing counter is a refusal that NAMES it, not a smaller number.
#[test]
fn a_missing_page_counter_is_refused_by_name() {
    let err = reclaimable_available_gb_in("Pages free: 1.\n").unwrap_err();
    assert_eq!(err, "vm_stat: cannot read 'Pages inactive'");
}

/// The Linux headroom reader, pinned against the kernel's real output shape.
///
/// The macOS arithmetic above APPROXIMATES reclaimable memory by summing five
/// page counters; `MemAvailable` is the kernel's own figure for the same thing,
/// so the two branches are two implementations of one quantity rather than two
/// definitions of it.
const MEMINFO_HEADROOM: &str = "\
MemTotal:       65536000 kB
MemFree:         5242880 kB
Buffers:          262144 kB
Cached:         20971520 kB
SReclaimable:    4194304 kB
MemAvailable:   47185920 kB
SwapTotal:       8388608 kB
SwapFree:        8388608 kB
";

#[test]
fn meminfo_headroom_is_memavailable_in_gib() {
    // 47185920 KiB is exactly 45 GiB.
    assert_eq!(meminfo_available_gb_in(MEMINFO_HEADROOM), Ok(45.0));
}

/// A reading with no `MemAvailable` is refused BY NAME, and the refusal says the
/// macOS reader does not exist here -- so the operator is not sent to the wrong
/// platform. Substituting `MemFree` here would understate headroom by gigabytes
/// of reclaimable cache, which is the direction this gate must never err in.
#[test]
fn a_meminfo_without_memavailable_is_refused_by_name() {
    let err = meminfo_available_gb_in("MemTotal: 65536000 kB\nMemFree: 5242880 kB\n").unwrap_err();
    assert!(err.contains("meminfo"), "names the reader it read: {err}");
    assert!(
        err.contains("does not exist here"),
        "and says which reader would have been right on the other platform: {err}"
    );
}

/// The injected Linux headroom reader, over a real file.
#[test]
#[serial]
fn an_injected_meminfo_is_the_linux_headroom_reader() {
    let env = TestEnv::new();
    let meminfo = env.root().join("meminfo");
    std::fs::write(&meminfo, MEMINFO_HEADROOM).unwrap();
    assert_eq!(
        reclaimable_available_gb_from(Path::new("/nonexistent/vm_stat"), &meminfo),
        Ok(45.0),
        "a host with /proc/meminfo has headroom to measure, instead of the \
         refuse-to-measure-blind this used to return on every supported Linux box"
    );
    drop(env);
}

/// The headroom reader's refusals, both branches, both of them reachable.
///
/// The unreadable-meminfo branch used to be written as "make it mode 000", which
/// is vacuous as root -- and there is no reason to ask permission enforcement to
/// cooperate: a path that `exists()` but cannot be READ is a directory, and
/// `read_to_string` on a directory fails on every platform and for every user.
/// A refusal that has never been rendered is a refusal nobody has read, so both
/// of these messages are asserted by what they contain.
#[test]
#[serial]
fn an_unreadable_headroom_reader_names_itself_and_the_platform_it_is_not_on() {
    let env = TestEnv::new();
    let not_a_file = env.root().join("meminfo");
    std::fs::create_dir_all(&not_a_file).unwrap();
    let err = reclaimable_available_gb_from(Path::new(ABSENT_HEADROOM), &not_a_file).unwrap_err();
    assert!(err.contains("meminfo"), "names the reader it tried: {err}");
    assert!(err.contains("does not exist here"), "{err}");

    // Neither reader: the macOS refusal, because no meminfo means no meminfo.
    let err = reclaimable_available_gb_from(Path::new(ABSENT_HEADROOM), Path::new(ABSENT_HEADROOM))
        .unwrap_err();
    assert!(err.contains("vm_stat"), "names the reader it tried: {err}");
    assert!(
        err.contains("meminfo") && err.contains("does not exist here"),
        "and says which reader would have been right on the other platform: {err}"
    );
    drop(env);
}

/// The refusal path the CLI actually takes. `cli_ztools.rs` passes `thrashing:
/// None`, which is the branch that spawns the pressure readers itself -- and it
/// was uncovered, because every other test injects a verdict and therefore never
/// reaches the live read.
#[test]
#[serial]
fn an_uninjected_thrashing_verdict_reads_the_machine_and_still_answers() {
    let env = TestEnv::new();
    let r = oversize_refusal(0.001, Some(50.0), false, None);
    assert!(
        r.is_empty() || r.contains("the machine is already paging"),
        "uninjected thrashing read either permits a tiny model or names live paging: {r}"
    );
    drop(env);
}

/// The thrashing rule, over one reading, exactly. Both platform shapes, because
/// the Linux one -- swap alone, reclaim ABSENT -- is the shape that had no
/// spelling here before the Linux reader existed.
#[test]
fn the_thrashing_verdict_is_the_threshold_comparison_over_one_reading() {
    let macos = |swap: f64, reclaim: f64| {
        MemoryPressure::new(swap, Some(reclaim), PressureSource::MacOsVmStat)
    };
    let linux = |swap: f64| MemoryPressure::new(swap, None, PressureSource::LinuxMeminfo);
    // `None` cannot tell, in either direction.
    assert_eq!(thrashing_verdict(None), None);
    // At the threshold is NOT over it: the comparison is strict, so the clean
    // boundary belongs to the clean side.
    assert_eq!(
        thrashing_verdict(Some(macos(MAX_CLEAN_SWAP_GB, MAX_CLEAN_RECLAIM_GB))),
        Some(false)
    );
    assert_eq!(
        thrashing_verdict(Some(macos(MAX_CLEAN_SWAP_GB + 0.001, 0.0))),
        Some(true)
    );
    assert_eq!(
        thrashing_verdict(Some(macos(0.0, MAX_CLEAN_RECLAIM_GB + 0.001))),
        Some(true)
    );
    assert_eq!(thrashing_verdict(Some(macos(0.0, 0.0))), Some(false));
    // Linux: the swap gate alone, and the absent second half does not make the
    // reading unreadable -- an absent quantity is not an unreadable reading.
    assert_eq!(thrashing_verdict(Some(linux(0.0))), Some(false));
    assert_eq!(
        thrashing_verdict(Some(linux(MAX_CLEAN_SWAP_GB + 0.001))),
        Some(true)
    );
    // The live reader IS this composition, so it cannot drift from the rule.
    assert_eq!(is_thrashing(), thrashing_verdict(memory_pressure()));
}

/// The live reader, checked for sanity WITHOUT a second live reading to
/// compare it to, and without failing where macOS `vm_stat` does not exist.
#[test]
#[serial]
fn a_live_pressure_reading_is_finite_or_absent_and_never_wrong_sided() {
    let env = TestEnv::new();
    match memory_pressure() {
        None => {
            // The documented degrade, and the only honest thing on a host with
            // neither reader.
            assert_eq!(is_thrashing(), None);
        }
        Some(reading) => {
            assert!(reading.swap_gb().is_finite(), "swap: {}", reading.swap_gb());
            if let Some(reclaim) = reading.reclaim_gb() {
                assert!(
                    reclaim.is_finite() && reclaim >= 0.0,
                    "compressor: {reclaim}"
                );
            }
            // Whatever it says, it is the pure rule applied to that reading.
            assert_eq!(
                thrashing_verdict(Some(reading)),
                Some(reading.is_thrashing())
            );
        }
    }
    drop(env);
}

/// The live headroom, checked for sanity, degrading where there is no
/// `vm_stat` instead of panicking -- this assertion used to be an `expect`.
#[test]
#[serial]
fn live_reclaimable_headroom_is_sane_or_names_the_tool_it_could_not_read() {
    let env = TestEnv::new();
    match reclaimable_available_gb() {
        Ok(gb) => {
            assert!(gb.is_finite(), "{gb}");
            assert!(
                gb > 0.0,
                "a live machine always has some reclaimable memory"
            );
            assert!(
                gb < 1_000_000.0,
                "{gb} GB is beyond any real Mac's memory map"
            );
        }
        // The refusal must name the reader it COULD NOT read, and this host may
        // legitimately be either kind -- naming only one platform's tool is what
        // sends the next reader to the wrong one.
        Err(msg) => assert!(
            msg.contains("vm_stat") || msg.contains("meminfo"),
            "a refusal that does not name the tool it could not read: {msg}"
        ),
    }
    drop(env);
}

/// A tiny model against the live machine's reclaimable memory must fit; this
/// exercises the un-injected headroom path inside the refusal itself.
#[test]
#[serial]
fn refusal_with_uninjected_headroom_measures_the_real_machine() {
    let env = TestEnv::new();
    match reclaimable_available_gb() {
        Ok(_) => assert_eq!(
            oversize_refusal(0.001, None, false, Some(false)),
            "",
            "1MB trivially fits any real machine's headroom"
        ),
        Err(_) => assert!(
            oversize_refusal(0.001, None, false, Some(false))
                .contains("cannot read memory headroom"),
            "without a headroom reading the refusal must say why it stopped"
        ),
    }
    drop(env);
}
