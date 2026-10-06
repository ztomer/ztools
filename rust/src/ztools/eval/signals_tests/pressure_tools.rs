//! The injected-seam tests: `tool_output`, `file_text` and `memory_pressure_from`
//! over injected PATHS rather than over a mock of the reading.
//!
//! `signals.rs` used to call `/usr/bin/vm_stat` by absolute path from inside
//! the reader, with no way to point it anywhere. Two consequences, both real:
//! the suite could not say anything about Linux x86-64 or aarch64 (which
//! standing policy still supports) because the tool is not at that path, and no
//! test could exercise the reader against a reading it chose.
//!
//! So every path is a DEFAULT and every path is an argument. This is where that
//! is proved, with real executables and a real `/proc/meminfo`-shaped FILE
//! rather than a mock of either.
//!
//! The Linux reader is a FILE, not a program. That is why `file_text` exists
//! beside `tool_output`: running `/proc/meminfo` as a program fails, and that
//! failure reads as "this platform publishes nothing" -- which is the exact
//! misdiagnosis this file exists to make impossible.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use serial_test::serial;

use super::super::*;
use super::{MEMINFO_QUIET, MEMINFO_SWAPPED};
use crate::test_env::TestEnv;

/// A path that is not there, standing in for a reader this host does not have.
const ABSENT: &str = "/nonexistent/ztools-no-such-reader";

/// REGRESSION, H4: with only macOS readers injected, a Linux-shaped fixture
/// could not be expressed at all. This is the test that says the dispatch reads
/// `/proc/meminfo` when that is what the host has.
#[test]
#[serial]
fn an_injected_meminfo_is_the_linux_reading() {
    let env = TestEnv::new();
    let meminfo = write_meminfo(&env.root().join("meminfo"), MEMINFO_SWAPPED);
    let reading = memory_pressure_from(Path::new(ABSENT), Path::new(ABSENT), &meminfo)
        .expect("an injected meminfo is a reading");
    assert_eq!(reading.source(), PressureSource::LinuxMeminfo);
    // `assert_exact!` and not `assert_eq!`: this value is EXACT in binary
    // floating point (15728640 KiB / 2^20), which is the whole condition the
    // shared macro states. `assert_eq!` on an f64 is also `clippy::float_cmp`
    // -- fired at -D warnings by the clippy that ships with the MSRV floor
    // (1.93.1), which is why this line can be read as more than style.
    assert_exact!(reading.swap_gb(), 15.0, "15GiB of used swap, in GiB");
    assert_eq!(
        reading.reclaim_gb(),
        None,
        "Linux publishes no compressor figure, and this reading says so rather \
         than filling in a zero it did not measure"
    );
    assert!(
        reading.is_thrashing(),
        "15GiB of swap is over the 8GiB ceiling"
    );
    drop(env);
}

/// A quiet Linux host is uncontended -- the arm that was IMPOSSIBLE before the
/// Linux reader existed, since the only reading a Linux box could produce was
/// `None`, and `None` is not clean.
#[test]
#[serial]
fn a_quiet_injected_meminfo_is_an_uncontended_reading() {
    let env = TestEnv::new();
    let meminfo = write_meminfo(&env.root().join("meminfo"), MEMINFO_QUIET);
    let reading = memory_pressure_from(Path::new(ABSENT), Path::new(ABSENT), &meminfo)
        .expect("an injected meminfo is a reading");
    assert!(!reading.is_thrashing());
    assert!(
        uncontended_verdict(false, Some(reading)),
        "the clean window can open on Linux now, which is the whole point: \
         `clean_estimate` was permanently None there, so `derived_timeout` was \
         permanently 0 and the median-of-clean recovery could never outvote \
         anything"
    );
    drop(env);
}

/// `/proc/meminfo` wins when BOTH readers are present, and the reason is a
/// property of the hosts rather than an accident: macOS has no `/proc` at all,
/// so this ordering cannot misclassify a real machine. It CAN reorder a test's
/// fixtures, which is why it is pinned.
#[test]
#[serial]
fn meminfo_takes_precedence_over_vm_stat_when_both_exist() {
    let env = TestEnv::new();
    let dir = env.root().join("tools");
    std::fs::create_dir_all(&dir).unwrap();
    let vm_stat = stub(
        &dir,
        "fake-vm_stat",
        "Pages occupied by compressor: 65536.\n",
    );
    let meminfo = write_meminfo(&env.root().join("meminfo"), MEMINFO_QUIET);
    let reading =
        memory_pressure_from(&vm_stat, Path::new(ABSENT), &meminfo).expect("meminfo answered");
    assert_eq!(
        reading.source(),
        PressureSource::LinuxMeminfo,
        "a meminfo that exists is the reader in use, because it is the cheaper \
         test and no macOS host has one"
    );
    drop(env);
}

/// An absent tool reads as "cannot tell", never as zero and never as a panic.
#[test]
fn an_unrunnable_tool_is_cannot_tell() {
    let missing = Path::new(ABSENT);
    assert_eq!(memory_pressure_from(missing, missing, missing), None);
    assert_eq!(tool_output(missing, &[]), None);
    assert_eq!(file_text(missing), None);
}

/// The macOS seam, end to end, through two real executables.
#[test]
#[serial]
fn injected_tools_produce_the_parsed_reading() {
    let env = TestEnv::new();
    let dir = env.root().join("tools");
    std::fs::create_dir_all(&dir).unwrap();
    let sysctl = stub(
        &dir,
        "fake-sysctl",
        "total = 4096.00M  used = 1024.00M  free = 3072.00M  (encrypted)",
    );
    let vm_stat = stub(
        &dir,
        "fake-vm_stat",
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages occupied by compressor: 65536.\n",
    );
    let reading =
        memory_pressure_from(&vm_stat, &sysctl, Path::new(ABSENT)).expect("both stub tools ran");
    assert_eq!(reading.source(), PressureSource::MacOsVmStat);
    assert!(
        (reading.swap_gb() - 1.0).abs() < 1e-9,
        "1024M of used swap is 1GB, got {}",
        reading.swap_gb()
    );
    assert_eq!(
        reading.reclaim_gb(),
        Some(1.0),
        "65536 pages of 16KiB is 1GB"
    );
    drop(env);
}

/// A stub that prints nothing is an UNREADABLE reading, not a clean one.
#[test]
#[serial]
fn a_tool_that_prints_nothing_reads_as_cannot_tell() {
    let env = TestEnv::new();
    let dir = env.root().join("tools");
    std::fs::create_dir_all(&dir).unwrap();
    let silent = stub(&dir, "silent-sysctl", "");
    let vm_stat = stub(&dir, "vm-stat", "Pages occupied by compressor: 1.\n");
    let absent = Path::new(ABSENT);
    assert_eq!(
        memory_pressure_from(&vm_stat, &silent, absent),
        None,
        "an empty swapusage line is not a zero-swap reading"
    );
    let sysctl = stub(&dir, "sysctl", "total = 0.00M  used = 0.00M  free = 0.00M");
    assert_eq!(
        memory_pressure_from(&silent, &sysctl, absent),
        None,
        "an empty vm_stat is not a zero-compressor reading"
    );
    drop(env);
}

/// A meminfo that EXISTS but cannot be read must not fall through to the macOS
/// reader, and must not become a reading either.
///
/// Two failure shapes live here and they are opposite, which is why both are
/// asserted. Reading it anyway (a mode-000 file is readable as root) would
/// fabricate a measurement; falling through to `vm_stat` would answer a Linux
/// question with a macOS reader and report the wrong platform. `None` is the
/// only answer that is honest in either case.
///
/// PERMISSION-DEPENDENT, and vacuous as root -- hence the probe and the loud
/// skip rather than a quiet pass.
#[test]
#[serial]
fn an_unreadable_meminfo_is_cannot_tell_and_never_falls_through_to_vm_stat() {
    let env = TestEnv::new();
    if !file_modes_enforced() {
        skip(
            "an_unreadable_meminfo_is_cannot_tell_and_never_falls_through_to_vm_stat: \
             this process can read a mode-000 file, so the kernel is not enforcing \
             unix modes here (root, or CAP_DAC_OVERRIDE). A mode-000 meminfo would \
             be parsed as ordinary text and the assertion would prove nothing.",
        );
        drop(env);
        return;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let dir = env.root().join("tools");
        std::fs::create_dir_all(&dir).unwrap();
        // BOTH macOS readers are WORKING here, deliberately. The first version of
        // this test left `sysctl` absent, so a reader that fell through to
        // vm_stat still produced None -- and the test passed while the fall
        // through it exists to catch was live. A guard can only see the wrong
        // answer when the wrong answer is reachable: the alternative reading has
        // to be a REAL one, not an absent one.
        let sysctl = stub(
            &dir,
            "fake-sysctl",
            "total = 4096.00M  used = 1024.00M  free = 3072.00M",
        );
        let vm_stat = stub(&dir, "fake-vm_stat", "Pages occupied by compressor: 1.\n");
        let meminfo = write_meminfo(&env.root().join("meminfo"), MEMINFO_SWAPPED);
        std::fs::set_permissions(&meminfo, std::fs::Permissions::from_mode(0o000)).unwrap();
        let reading = memory_pressure_from(&vm_stat, &sysctl, &meminfo);
        let _ = std::fs::set_permissions(&meminfo, std::fs::Permissions::from_mode(0o644));
        assert_eq!(
            reading, None,
            "an unreadable reader is no reading, and two WORKING macOS readers \
             standing right there must not be used to answer a Linux one"
        );
    }
    drop(env);
}

/// The documented defaults are the absolute paths production used all along,
/// plus the Linux one; changing them is a deliberate act, so they are pinned.
#[test]
fn the_documented_reader_defaults_are_the_platform_paths() {
    assert_eq!(VM_STAT, "/usr/bin/vm_stat");
    assert_eq!(SYSCTL, "sysctl");
    assert_eq!(
        PROC_MEMINFO, "/proc/meminfo",
        "absolute, not bare `meminfo`: a surprising PATH must not be able to \
         redirect the swap reading to a file this repo does not control"
    );
}

/// Is the kernel enforcing unix file modes for THIS process?
///
/// The case it guards is stated as "the OS refuses", which depends on WHO is
/// running: as root (or with `CAP_DAC_OVERRIDE`) a mode-000 file is readable, so
/// the assertion passes while proving nothing -- the file is read, parses fine,
/// and the answer comes out for an unrelated reason. So the enforcement is
/// PROBED rather than assumed, and a probe that comes back "not enforced" skips
/// LOUDLY instead of passing quietly.
///
/// The same probe, for the same reason, lives in
/// `eval/model_resolve/model_resolve_tests/disk.rs`. The two cannot share one
/// helper without editing that module, which is owned by a different change, so
/// it is duplicated rather than moved; when that file's owner next touches it,
/// both should collapse onto one `test_env` helper. Duplication of a nine-line
/// probe is the cheap half of that trade -- a WRONG probe, shared once, would be
/// silently wrong in two places.
#[cfg(unix)]
fn file_modes_enforced() -> bool {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let probe = dir.path().join("mode-000");
    std::fs::write(&probe, b"x").unwrap();
    std::fs::set_permissions(&probe, std::fs::Permissions::from_mode(0o000)).unwrap();
    // The only evidence that matters: can THIS process still read it? Restoring
    // the mode keeps the temp dir removable on the way out.
    let enforced = std::fs::read(&probe).is_err();
    let _ = std::fs::set_permissions(&probe, std::fs::Permissions::from_mode(0o644));
    enforced
}

/// Nothing below is compiled without `unix`, so there is nothing to skip.
#[cfg(not(unix))]
fn file_modes_enforced() -> bool {
    true
}

/// Announce a skip no reader can mistake for a pass.
///
/// `eprintln!` alone is NOT enough: libtest captures it, so a default
/// `cargo test` prints `ok` for a test that asserted nothing -- the exact
/// failure the probe exists to prevent, repeated inside the fix. Writing the
/// stderr HANDLE bypasses that capture, so the line is visible in a plain run.
/// Same idiom as `eval/model_resolve/model_resolve_tests/disk.rs`.
fn skip(reason: &str) {
    let mut err = std::io::stderr();
    let _ = writeln!(err, "SKIP: {reason}");
}

/// Write a `/proc/meminfo`-shaped file and hand back its path.
fn write_meminfo(path: &Path, text: &str) -> PathBuf {
    std::fs::write(path, text).unwrap();
    path.to_path_buf()
}

/// Write a runnable stub that prints `stdout`.
fn stub(dir: &Path, name: &str, stdout: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(
        &path,
        format!("#!/bin/sh\ncat <<'ZTOOLS_EOF'\n{stdout}ZTOOLS_EOF\n"),
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    path
}
