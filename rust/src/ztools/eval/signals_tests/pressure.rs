//! What the machine is doing: the parsers for BOTH published shapes, the two
//! verdicts, and the lock.
//!
//! These are the tests that used to compare the live `memory_pressure()` with a
//! SECOND live `sysctl` reading and to PANIC when the reading was absent. That
//! is a test whose subject is whatever this box happened to be doing, with a
//! 0.01 GB tolerance between two reads of a value that moves -- and it was
//! unrunnable on any Linux host, because `/usr/bin/vm_stat` does not exist
//! there.
//!
//! The rule is pinned over INJECTED text and INJECTED numbers, for BOTH
//! platforms, which is the only way a test written on a Mac says anything about
//! a Linux reading. The live reader is checked for sanity only, and degrades to
//! the documented "cannot tell" where its reader is absent rather than failing.
//!
//! The `None` cases are the load-bearing half here. `None` is what every caller
//! turns into "unverified", so a `None` that should have been a reading is a
//! measurement silently lost -- which is how Linux went years with no clean
//! capability sample at all.

use serial_test::serial;

use super::super::*;
use super::{MEMINFO_QUIET, MEMINFO_SWAPPED};
use crate::test_env::TestEnv;
use crate::ztools::eval::gpu_lock::foreign_holder;

/// The real `vm_stat` compressor line, as macOS prints it.
const VM_STAT_COMPRESSOR: &str = "\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               65536.
Pages active:                            131072.
Pages occupied by compressor:             131072.
";

/// A macOS-shaped reading, for the verdicts.
fn macos_reading(swap_gb: f64, reclaim_gb: f64) -> MemoryPressure {
    MemoryPressure::new(swap_gb, Some(reclaim_gb), PressureSource::MacOsVmStat)
}

/// A Linux-shaped reading: swap only, and the second quantity ABSENT rather
/// than zero. That absence is the whole point of this helper -- a helper that
/// filled in 0.0 would let every verdict here pass on a reading shape the
/// platform does not produce.
fn linux_reading(swap_gb: f64) -> MemoryPressure {
    MemoryPressure::new(swap_gb, None, PressureSource::LinuxMeminfo)
}

/// REGRESSION: the parser used to grab the literal token "used" and then fail
/// to parse it, so `memory_pressure()` was None on EVERY machine -- every
/// capability sample UNVERIFIED, `derived_timeout` permanently 0. The happy path
/// must work against the real output SHAPE.
#[test]
fn swap_parsing_reads_the_used_figure_after_the_literal_token() {
    let parsed =
        parse_swap_used_gb("total = 4096.00M  used = 1024.00M  free = 3072.00M  (encrypted)")
            .expect("a real swapusage line parses");
    assert_exact!(parsed, 1.0); // 1024M is one GiB, not 1024

    let gigabytes = parse_swap_used_gb("total = 16.00G  used = 2.00G  free = 14.00G")
        .expect("a G-suffixed value parses");
    assert_exact!(gigabytes, 2.0); // the G suffix is NOT divided

    // The token the old code grabbed was "used" ITSELF.
    assert_eq!(
        parse_swap_used_gb("total = 4096.00M  used  free = 3583.75M"),
        None,
        "no value after `used` is no reading"
    );
    assert_eq!(parse_swap_used_gb(""), None);
    assert_eq!(
        parse_swap_used_gb("total = 4096.00M"),
        None,
        "no `used` is no reading"
    );
}

#[test]
fn compressor_parsing_reads_the_page_count_and_converts_to_gb() {
    let gb = parse_compressor_gb(VM_STAT_COMPRESSOR).expect("a real compressor line parses");
    // 131072 pages of 16384 B = 2 GiB exactly.
    assert!(
        (gb - 2.0).abs() < 1e-9,
        "131072 pages of 16KiB is 2GB, got {gb}"
    );
    assert_eq!(parse_compressor_gb("Pages free: 1.\n"), None);
    assert_eq!(
        parse_compressor_gb("Pages occupied by compressor: nope.\n"),
        None
    );
}

/// REGRESSION, and the H4 fix's own contract: Linux publishes swap in
/// `/proc/meminfo`, in KiB. Getting the divisor wrong is silent -- a 2^20 slip
/// turns 15GiB of swap into 15TiB (always "thrashing") and a 10^6 slip into
/// 15.7GB (always "clean"), so the unit is pinned against the arithmetic rather
/// than against a shape.
#[test]
fn meminfo_swap_is_parsed_in_kib_and_converts_to_gib() {
    assert_eq!(
        parse_meminfo_swap_used_gb(MEMINFO_QUIET),
        Some(0.0),
        "SwapTotal == SwapFree is a real measurement of zero, not an absent one"
    );
    // (16777216 - 1048576) KiB = 15728640 KiB = 15 GiB.
    assert_eq!(parse_meminfo_swap_used_gb(MEMINFO_SWAPPED), Some(15.0));
    // A box that swapped exactly one 4KiB page: 4 KiB / 1048576 KiB-per-GiB,
    // i.e. 1/262144 GiB. Small enough that any wrong divisor (a 10^6 slip, a
    // page slip) shows up instead of rounding away.
    assert_eq!(
        parse_meminfo_swap_used_gb("SwapTotal: 4194304 kB\nSwapFree: 4194300 kB\n"),
        Some(4.0 / 1_048_576.0)
    );
}

/// The Linux cases that must NOT produce a number, each for a different reason.
///
/// Every one of these used to be impossible to express: the reader only ever
/// parsed `sysctl` output, so the whole question of what a Linux reading looks
/// like when a field is missing had no answer at all.
#[test]
fn an_unmeasurable_linux_reading_is_none_rather_than_zero() {
    assert_eq!(
        parse_meminfo_swap_used_gb("SwapTotal: 0 kB\nSwapFree: 0 kB\n"),
        None,
        "a host with swap DISABLED has no swap figure to gate on. Reading that \
         as 0.0GB of swap would certify every measurement on it as clean on the \
         strength of a quantity that does not exist -- the same shape as a clean \
         reading invented from nothing, which is what this gate exists to \
         refuse. Every container reports SwapTotal 0 by default."
    );
    assert_eq!(
        parse_meminfo_swap_used_gb("MemTotal: 65536000 kB\nMemFree: 1 kB\n"),
        None,
        "no SwapTotal line at all is an absent figure"
    );
    assert_eq!(
        parse_meminfo_swap_used_gb("SwapTotal: 4194304 kB\n"),
        None,
        "SwapTotal without SwapFree cannot be subtracted into a used figure"
    );
    assert_eq!(
        parse_meminfo_swap_used_gb(""),
        None,
        "an empty reading is no reading"
    );
}

/// The parsers must not read each other's keys. `SwapCached` and `Cached` are
/// both real meminfo lines that a sloppy prefix match would accept, and
/// `SwapCached` in particular is a real trap: it is non-zero exactly on a box
/// under swap pressure, so matching it would make the reading rise with the very
/// pressure it is supposed to detect.
#[test]
fn meminfo_keys_are_matched_exactly() {
    let only_neighbours = "SwapCached: 1048576 kB\nCached: 2097152 kB\nSwapTotalting: 8 kB\n";
    assert_eq!(
        parse_meminfo_swap_used_gb(only_neighbours),
        None,
        "a prefix that also matches SwapCached/SwapTotalting is not an exact match"
    );
}

/// The contention rule, over one reading of each input, on BOTH platforms.
#[test]
fn an_unreadable_pressure_reading_is_never_evidence_of_contention() {
    // Documented contract: None means "cannot tell" and uncontended must then be
    // FALSE -- an unverifiable sample must not masquerade as clean. This is the
    // clause Linux used to live inside permanently.
    assert!(!uncontended_verdict(false, None));
    assert!(!uncontended_verdict(true, None));
    assert!(uncontended_verdict(
        false,
        Some(macos_reading(MAX_CLEAN_SWAP_GB, MAX_CLEAN_RECLAIM_GB))
    ));
    assert!(!uncontended_verdict(
        false,
        Some(macos_reading(MAX_CLEAN_SWAP_GB + 0.1, 0.0))
    ));
    assert!(!uncontended_verdict(
        false,
        Some(macos_reading(0.0, MAX_CLEAN_RECLAIM_GB + 0.1))
    ));
    // A foreign holder outranks clean pressure.
    assert!(!uncontended_verdict(true, Some(macos_reading(0.0, 0.0))));
}

/// The Linux arm of the same rule, and the one that has no macOS precedent: with
/// the reclaim half absent, the swap gate stands alone and still produces a
/// verdict in BOTH directions. If a Linux reading were refused whenever the
/// second quantity is missing, this whole fix would buy nothing.
#[test]
fn a_linux_reading_is_gated_on_swap_alone_and_still_verdicts_both_ways() {
    assert!(
        uncontended_verdict(false, Some(linux_reading(0.0))),
        "a quiet Linux box is uncontended -- the reading it never got is why this \
         could not be true before"
    );
    assert!(
        !uncontended_verdict(false, Some(linux_reading(MAX_CLEAN_SWAP_GB + 0.1))),
        "a swapped Linux box is contended on the swap figure alone"
    );
    assert!(
        uncontended_verdict(false, Some(linux_reading(MAX_CLEAN_SWAP_GB))),
        "at the ceiling is not over it, on either platform"
    );
    assert_eq!(linux_reading(0.0).reclaim_gb(), None);
    assert_eq!(macos_reading(0.0, 0.0).reclaim_gb(), Some(0.0));
}

/// The same rule for the thrashing half: cannot tell is neither verdict.
#[test]
fn the_thrashing_verdict_compares_one_reading_against_the_clean_ceilings() {
    assert_eq!(thrashing_verdict(None), None);
    assert_eq!(
        thrashing_verdict(Some(macos_reading(MAX_CLEAN_SWAP_GB, MAX_CLEAN_RECLAIM_GB))),
        Some(false),
        "at the ceiling is not over it"
    );
    assert_eq!(
        thrashing_verdict(Some(macos_reading(MAX_CLEAN_SWAP_GB + 0.1, 0.0))),
        Some(true)
    );
    assert_eq!(
        thrashing_verdict(Some(macos_reading(0.0, MAX_CLEAN_RECLAIM_GB + 0.1))),
        Some(true)
    );
    assert_eq!(
        thrashing_verdict(Some(macos_reading(0.0, 0.0))),
        Some(false)
    );
    assert_eq!(thrashing_verdict(Some(linux_reading(0.0))), Some(false));
    assert_eq!(
        thrashing_verdict(Some(linux_reading(MAX_CLEAN_SWAP_GB + 0.1))),
        Some(true)
    );
}

/// The detail an operator reads when a measurement is refused must not claim a
/// two-signal conclusion on a platform that only had one. A Linux refusal that
/// said "swap 9.0GB, compressor 0.0GB" would be asserting a compressor figure
/// nobody read.
#[test]
fn the_refusal_detail_says_how_many_quantities_actually_gated() {
    let mac = macos_reading(1.5, 2.5).describe();
    assert!(
        mac.contains("swap 1.5GB") && mac.contains("compressor 2.5GB"),
        "{mac}"
    );
    let linux = linux_reading(9.0).describe();
    assert!(linux.contains("swap 9.0GB"), "{linux}");
    // Exactly ONE quantity printed, because exactly one gate fired. The detail
    // may NAME the compressor to explain its absence; what it must not do is
    // print a figure for it.
    assert_eq!(
        linux.matches("GB").count(),
        1,
        "a one-gate reading must print one figure: {linux}"
    );
    assert_eq!(
        mac.matches("GB").count(),
        2,
        "and a two-gate reading must print two: {mac}"
    );
    assert!(
        linux.contains("only quantity"),
        "and it must say that swap was the only gate, not imply a second: {linux}"
    );
    assert!(macos_reading(1.0, 2.0).source().label().contains("macOS"));
    assert!(linux_reading(1.0).source().label().contains("meminfo"));
    // The third shape: macOS with the compressor figure MISSING. The reader cannot
    // produce it (a macOS reading with no compressor line is `None` overall), but
    // the constructor is public and the detail must not print a compressor figure
    // it does not have -- so the arm exists and is pinned rather than left to be
    // discovered as an uncovered branch.
    let mac_without = MemoryPressure::new(4.0, None, PressureSource::MacOsVmStat).describe();
    assert!(
        mac_without.contains("swap 4.0GB") && !mac_without.contains("compressor"),
        "{mac_without}"
    );
}

/// A live foreign lock holder makes the machine contended -- over a FIXTURE
/// lock inside the sandbox, so the answer is this machine's to give only
/// because the sandbox put the holder there.
#[test]
#[serial]
fn a_live_foreign_lock_holder_makes_the_machine_contended() {
    let env = TestEnv::new();
    // The parent process is alive, belongs to this user (so kill(pid, 0)
    // succeeds -- signalling launchd would EPERM), and is not this process: a
    // real foreign holder by the lock's own liveness rules.
    let holder_pid = std::os::unix::process::parent_id();
    let start = crate::ztools::eval::gpu_lock::start_time(holder_pid);
    let lock_dir = env.path("ZTOOLS_GPU_LOCK_DIR");
    std::fs::create_dir_all(&lock_dir).unwrap();
    std::fs::write(
        lock_dir.join("owner"),
        format!("{holder_pid}\n{start}\na concurrent eval run\n"),
    )
    .unwrap();
    assert!(
        foreign_holder().is_some(),
        "pid {holder_pid} holding the fixture lock is foreign"
    );
    assert!(
        !machine_is_uncontended(),
        "a foreign holder outranks a clean pressure reading"
    );
    drop(env);
}

/// The live reader: sane when it can read, and absent where its reader is not.
///
/// Before the Linux reader this test's "absent" arm was the ONLY thing a Linux
/// host could ever produce, and it asserted only that the verdicts degraded --
/// which they did, correctly and uselessly, forever. What is asserted now is
/// that whichever reading this host publishes is internally consistent with the
/// platform it claims to be.
#[test]
#[serial]
fn a_live_pressure_reading_is_finite_nonnegative_and_platform_coherent() {
    let env = TestEnv::new();
    match memory_pressure() {
        None => {
            // The documented degrade: a host with neither reader.
            assert_eq!(thrashing_verdict(None), None);
        }
        Some(reading) => {
            let swap = reading.swap_gb();
            assert!(swap.is_finite() && swap >= 0.0, "swap: {swap}");
            if let Some(reclaim) = reading.reclaim_gb() {
                assert!(
                    reclaim.is_finite() && reclaim >= 0.0,
                    "compressor: {reclaim}"
                );
            }
            // A reclaim figure exists only where a reader publishes one.
            let claims_macos = reading.source() == PressureSource::MacOsVmStat;
            assert_eq!(
                reading.reclaim_gb().is_some(),
                claims_macos,
                "only macOS publishes a compressor figure; a {} reading claiming \
                 one (or denying it) means the dispatch and the shape disagree",
                reading.source().label()
            );
            // Whatever it says, it is the pure rule applied to that reading.
            assert_eq!(
                thrashing_verdict(Some(reading)),
                Some(reading.is_thrashing())
            );
        }
    }
    drop(env);
}
