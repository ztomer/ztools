//! Which reading this host publishes, and what each field of it MEANS.
//!
//! Split out of `signals.rs` at the 500-line cap, and split out for a reason
//! beyond the cap: the platform dispatch and the parsers for the two shapes are
//! one unit of reasoning ("what can this host actually measure, and what does
//! each field of the answer stand for"), while everything left in `signals.rs`
//! is the store and the timeout arithmetic that consumes it.
//!
//! WHY A SUPPORTED PLATFORM COULD NOT BE LEFT READING NOTHING. Standing policy
//! supports macOS on Apple silicon AND Linux `x86_64`/`aarch64`, and this reader
//! used to ask macOS two questions -- `sysctl -n vm.swapusage` for swap,
//! `/usr/bin/vm_stat` for compressor pages -- so on Linux it answered `None`.
//! That is not a neutral answer. `None` reaches `machine_is_uncontended()`,
//! which returns false for it, so on Linux EVERY capability sample was tagged
//! unclean no matter how quiet the box was. Two things then go wrong at once,
//! and neither is visible in the output:
//!
//! - `clean_estimate` only ever sees clean samples, so with none it returns
//!   `None` forever and `derived_timeout` is permanently 0 -- the learned
//!   per-model timeout never exists on Linux and every request falls back to
//!   the configured floor, which is the opposite of what a measurement buys;
//! - `add_sample` -> `estimate_from` falls back to the median of ALL samples
//!   when none are clean, so the scalar written into `_capabilities` is a
//!   median of unverified readings, and the median-of-clean window that exists
//!   precisely to outvote a contaminated reading can never open.
//!
//! So the failure is not "recorded clean" -- it is "permanently unverifiable,
//! with no path back to verified". Reading the real quantity on Linux is what
//! makes that window reachable again.
//!
//! THE TWO QUANTITIES, AND WHAT EACH ONE IS ON EACH PLATFORM.
//!
//! 1. SWAP IN USE. Portable, and mapped one-to-one. macOS: the figure after the
//!    literal token `used` in `vm.swapusage`. Linux: `SwapTotal - SwapFree` from
//!    `/proc/meminfo`, whose units proc(5) documents as KiB -- 1024 bytes -- so
//!    the divisor is 2^20, not 10^6 and not 2^30. It is the same quantity in the
//!    same unit, so it is compared against the SAME `MAX_CLEAN_SWAP_GB` on both
//!    platforms. No platform-specific threshold is invented for it, because a
//!    second threshold for the same quantity is a second thing to be wrong.
//!
//! 2. MEMORY THE KERNEL HAD TO CLAW BACK. macOS publishes this as compressor
//!    pages: memory moved out of RAM into a compressed form because the machine
//!    could not afford to keep it. Linux has NO equivalent, and the two
//!    candidates that look closest are both wrong in opposite directions:
//!
//!    a. RECLAIMABLE PAGE CACHE (`Cached`, `Buffers`, `SReclaimable`). This is
//!    not pressure at all -- it is the kernel's normal way of using spare
//!    memory, and its size scales with RAM, not with how hard the machine is
//!    working. Reading it as "clawed-back memory" would report 10GB of
//!    perfectly idle cache as thrashing on a 16GB box, mark every sample
//!    unclean forever, and re-create the permanently-unverifiable failure
//!    this file exists to remove.
//!    b. PSI (`/proc/pressure/memory`). This IS pressure -- it is the kernel's
//!    own "fraction of the last 10s in which some task was stalled on
//!    memory" -- but it is a STALL FRACTION, not bytes. Comparing it with
//!    `MAX_CLEAN_RECLAIM_GB` (a byte count calibrated on macOS compressor
//!    pages) would be a number compared against a number of a different
//!    kind: the threshold stops meaning anything the moment it is used, and
//!    it could not be calibrated here anyway, because this file is written on
//!    macOS. So PSI is NOT read, and the second quantity is reported as
//!    ABSENT rather than substituted.
//!
//! WHAT THAT LEAVES, stated plainly so nobody has to reverse-engineer it: on
//! Linux this gate is swap-only. It is strictly weaker than the macOS gate, and
//! the hole is real: a Linux box under heavy anonymous-page reclaim with NO
//! swap in use is invisible to it. Closing that hole honestly needs a
//! stall-fraction threshold MEASURED on Linux, which is follow-up work on a
//! Linux box, not a constant invented here.
//!
//! THE SWAP-DISABLED LINUX BOX, which is the common case in a container: proc
//! reports `SwapTotal: 0`, so the quantity that would gate is not zero, it is
//! ABSENT. That reading is `None`, which every caller already treats as
//! "cannot tell" -- so the sample is tagged UNVERIFIED rather than clean. That
//! is the conservative direction and it is the same trade the macOS side already
//! makes when `vm_stat` cannot be run. Returning `Some(0.0)` instead would
//! turn "I measured nothing" into "I measured zero pressure", which is how a
//! supported platform ends up certifying measurements it never took.
//!
//! THE LIMITS OF THE MAPPING, for a reader who wants to audit it:
//! - Swap-used is a high-water mark in practice, not an instantaneous rate: it
//!   falls only as pages are faulted back in or the machine reboots. So the
//!   macOS gate has always had this property and the Linux gate inherits it
//!   exactly rather than introducing it.
//! - Neither platform's reading can see GPU or CPU contention. That is what
//!   `tools/gpu_lock.sh` and the whole OS are for; see its header.

use std::path::Path;

use crate::ztools::eval::gpu_lock::foreign_holder;

/// Swap above this, in GiB, disqualifies a timing. The same ceiling on every
/// platform, because it gates the same byte quantity.
pub const MAX_CLEAN_SWAP_GB: f64 = 8.0;

/// Memory-the-kernel-had-to-claw-back above this, in GiB, disqualifies a timing.
///
/// Named for what it MEANS rather than for the macOS tool that reports it, so
/// that a reader does not go looking for a compressor on Linux. Only macOS
/// publishes such a figure; see the module header for why nothing stands in for
/// it.
pub const MAX_CLEAN_RECLAIM_GB: f64 = 15.0;

const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;
/// macOS page size on `arm64`/`x86_64`.
const PAGE_BYTES: f64 = 16384.0;
/// `/proc/meminfo` reports KiB (proc(5)), which is 1024 bytes -- NOT 1000, and
/// not a page. So one GiB is 1024*1024 of them.
const KIB_PER_GB: f64 = 1024.0 * 1024.0;

/// macOS `vm_stat`, named by ABSOLUTE path.
///
/// Two reasons, both about the answer being wrong rather than missing. A
/// stripped or surprising `PATH` must not turn a memory reading into "cannot
/// tell" -- and `None` here means contended-or-unverified, so the failure mode
/// is a permanently unclean estimate, not a crash. And a hardcoded
/// `/usr/bin/...` is a hardcoded `macOS`: Linux x86-64 and aarch64 remain
/// supported. Every read goes through [`memory_pressure_from`], whose paths are
/// the seam; the constants are only the defaults.
pub const VM_STAT: &str = "/usr/bin/vm_stat";

/// The swap reading's source on macOS. `sysctl` is on `PATH` wherever it
/// exists, so it is named bare; it is a constant for the same reason as
/// [`VM_STAT`].
pub const SYSCTL: &str = "sysctl";

/// The Linux reader, absolute for the same reason as [`VM_STAT`]: it is a file,
/// not a `PATH` lookup, so a surprising `PATH` cannot silently redirect it.
pub const PROC_MEMINFO: &str = "/proc/meminfo";

/// Run `prog` with `args` and hand back its stdout.
///
/// `None` when it cannot be run or produces nothing readable: an unreadable
/// reading is "cannot tell", which is the only safe direction, because the
/// alternative -- a plausible number -- is how a contended box's readings got
/// enshrined as measurements.
#[must_use]
pub fn tool_output(prog: &Path, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(prog).args(args).output().ok()?;
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// The file-reading half of the same seam, for the Linux reader.
///
/// A separate function rather than an extension of [`tool_output`] because
/// `/proc/meminfo` is a FILE: running it as a program would spawn a process that
/// fails, and the failure would read as "this platform publishes nothing" --
/// which is exactly the misdiagnosis this file exists to remove.
#[must_use]
pub fn file_text(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// Which host published a reading.
///
/// Named, because a bare pair of numbers does not say which platform's
/// semantics it is on -- and that is what a reader needs in order to know
/// whether the second number is present at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureSource {
    /// `sysctl vm.swapusage` plus `vm_stat` compressor pages. Both quantities
    /// byte-denominated, both gated.
    MacOsVmStat,
    /// `/proc/meminfo`. Swap is byte-denominated and gated; the reclaim half has
    /// no counterpart here and is reported absent rather than substituted.
    LinuxMeminfo,
}

impl PressureSource {
    /// How to name this platform in a message meant for a person.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::MacOsVmStat => "macOS vm_stat+sysctl",
            Self::LinuxMeminfo => "Linux /proc/meminfo",
        }
    }
}

/// One memory-pressure reading, with its platform attached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryPressure {
    swap_gb: f64,
    reclaim_gb: Option<f64>,
    source: PressureSource,
}

impl MemoryPressure {
    /// Build a reading. `reclaim_gb: None` is the honest Linux shape, and is
    /// also how a macOS reading says `vm_stat` did not print the line.
    #[must_use]
    pub const fn new(swap_gb: f64, reclaim_gb: Option<f64>, source: PressureSource) -> Self {
        Self {
            swap_gb,
            reclaim_gb,
            source,
        }
    }

    /// Swap in use, GiB.
    #[must_use]
    pub const fn swap_gb(&self) -> f64 {
        self.swap_gb
    }

    /// Memory the kernel had to claw back, GiB, or `None` where the platform
    /// publishes no such figure. `None` here is NOT "zero pressure": it means
    /// this reading has one gate instead of two, and the caller that formats it
    /// says so.
    #[must_use]
    pub const fn reclaim_gb(&self) -> Option<f64> {
        self.reclaim_gb
    }

    #[must_use]
    pub const fn source(&self) -> PressureSource {
        self.source
    }

    /// Is either gated quantity past its ceiling? Pure over one reading.
    ///
    /// Strictly greater, so a reading exactly AT a ceiling is on the clean side.
    /// `reclaim_gb` absent contributes nothing -- the swap gate still stands on
    /// its own, which is why a Linux reading is `Some` rather than `None`.
    #[must_use]
    pub fn is_thrashing(self) -> bool {
        self.swap_gb > MAX_CLEAN_SWAP_GB
            || self.reclaim_gb.is_some_and(|g| g > MAX_CLEAN_RECLAIM_GB)
    }

    /// The detail an operator reads when a measurement is refused.
    ///
    /// It names the platform and, on Linux, that swap was the only quantity that
    /// gated -- so "the machine is already paging" is never presented as a
    /// two-signal conclusion it did not reach.
    #[must_use]
    pub fn describe(&self) -> String {
        match (self.source, self.reclaim_gb) {
            (PressureSource::MacOsVmStat, Some(reclaim)) => {
                format!("swap {:.1}GB, compressor {reclaim:.1}GB", self.swap_gb)
            }
            (PressureSource::MacOsVmStat, None) => format!("swap {:.1}GB", self.swap_gb),
            (PressureSource::LinuxMeminfo, _) => format!(
                "swap {:.1}GB (from {}; Linux publishes no compressor figure, so \
                 swap is the only quantity gating this reading)",
                self.swap_gb,
                self.source.label()
            ),
        }
    }
}

/// The reading for this host, or `None` when this host publishes nothing this
/// repo knows how to read -- which every caller must treat as "cannot tell",
/// never as "fine".
#[must_use]
pub fn memory_pressure() -> Option<MemoryPressure> {
    memory_pressure_from(
        Path::new(VM_STAT),
        Path::new(SYSCTL),
        Path::new(PROC_MEMINFO),
    )
}

/// [`memory_pressure`] from injected paths: the seam every caller needs.
///
/// The production reader is this with the three documented defaults, and a test
/// -- or a host whose tools live somewhere else -- can point any of the three
/// anywhere without the reading and the parsing diverging. All three are paths
/// rather than a mix of paths and argv, because the Linux reader is a file.
#[must_use]
pub fn memory_pressure_from(
    vm_stat: &Path,
    sysctl: &Path,
    meminfo: &Path,
) -> Option<MemoryPressure> {
    // Which host this is, decided by WHAT IS PRESENT rather than by
    // `cfg!(target_os)`: the point of the seam is that a test can stand a fake
    // meminfo next to a fake vm_stat and be told which reader it is exercising.
    // meminfo is checked first because it is the cheaper test (a stat, not a
    // spawn) and because macOS has no `/proc` at all, so the order cannot
    // misclassify a real host.
    if meminfo.exists() {
        let text = file_text(meminfo)?;
        return parse_meminfo_swap_used_gb(&text).map(|swap_gb| MemoryPressure {
            swap_gb,
            reclaim_gb: None,
            source: PressureSource::LinuxMeminfo,
        });
    }
    if !vm_stat.exists() {
        return None;
    }
    let swap_gb = parse_swap_used_gb(&tool_output(sysctl, &["-n", "vm.swapusage"])?)?;
    let reclaim_gb = parse_compressor_gb(&tool_output(vm_stat, &[])?)?;
    Some(MemoryPressure {
        swap_gb,
        reclaim_gb: Some(reclaim_gb),
        source: PressureSource::MacOsVmStat,
    })
}

/// `sysctl -n vm.swapusage` -> "total = 4096.00M  used = 512.25M  free = ...".
///
/// The VALUE follows the literal token "used" (and an "=" sign). Grabbing the
/// token that STARTS WITH "used" grabs "used" itself, whose
/// `.split('=').nth(1)` is `None` -- so this returned `None` on every machine,
/// every time, which tagged every capability sample UNVERIFIED, zeroed
/// `derived_timeout`, and silently disabled the median-of-clean estimator's
/// recovery path. Found by coverage work: the happy path was unreachable.
///
/// Pure, so the contract is pinned against the real output SHAPE rather than
/// against whatever this box is paging right now.
#[must_use]
pub fn parse_swap_used_gb(text: &str) -> Option<f64> {
    let mut tokens = text.split_whitespace();
    while let Some(token) = tokens.next() {
        if token != "used" {
            continue;
        }
        for field in tokens.by_ref() {
            if field == "=" {
                continue;
            }
            let value: f64 = field.trim_end_matches(['M', 'G']).parse().ok()?;
            let multiplier = if field.ends_with('G') {
                1.0
            } else {
                1.0 / 1024.0
            };
            return Some(value * multiplier);
        }
    }
    None
}

/// `vm_stat`'s compressor line, in GiB.
#[must_use]
pub fn parse_compressor_gb(text: &str) -> Option<f64> {
    let line = text
        .lines()
        .find(|l| l.starts_with("Pages occupied by compressor"))?;
    let raw = line.split(':').nth(1)?;
    let pages: f64 = raw.trim().trim_end_matches('.').parse().ok()?;
    Some(pages * PAGE_BYTES / BYTES_PER_GB)
}

/// One `Key: value kB` figure out of a `/proc/meminfo` reading, in KiB.
///
/// `None` when the key is absent, which is NOT the same as zero: proc omits
/// lines it has nothing for, and `SwapTotal: 0` is written out explicitly while
/// an absent `SwapTotal` means the kernel did not report swap at all.
#[must_use]
fn meminfo_kib(text: &str, key: &str) -> Option<f64> {
    text.lines()
        .find_map(|line| line.strip_prefix(key))
        .filter(|rest| rest.starts_with(':'))
        .map(|rest| rest.trim_start_matches(':').trim())
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

/// Swap in use from one `/proc/meminfo` reading, in GiB.
///
/// `SwapTotal - SwapFree`, in KiB, over 2^20 -- see [`KIB_PER_GB`] for why the
/// divisor is that and not 10^6.
///
/// `None` for a reading with NO swap: proc reports `SwapTotal: 0` on a host with
/// swap disabled (every container, by default), and "this host has no swap
/// figure to gate on" must not be reported as "zero swap in use". See the module
/// header for the consequence.
#[must_use]
pub fn parse_meminfo_swap_used_gb(text: &str) -> Option<f64> {
    let total = meminfo_kib(text, "SwapTotal")?;
    if total <= 0.0 {
        return None;
    }
    let free = meminfo_kib(text, "SwapFree")?;
    Some((total - free).max(0.0) / KIB_PER_GB)
}

/// Is this pressure reading past either clean threshold? Pure over one reading.
///
/// `None` in, `None` out: an unreadable pressure reading is not evidence of
/// thrashing either way. Extracted so the live [`memory_pressure`] read cannot
/// be sampled twice and disagree with itself -- it did exactly that, once per
/// eval, in a caller that asked `is_thrashing()` and then formatted the detail
/// from a SECOND reading.
#[must_use]
pub fn thrashing_verdict(pressure: Option<MemoryPressure>) -> Option<bool> {
    pressure.map(MemoryPressure::is_thrashing)
}

/// Is the machine quiet enough for a timing to mean anything? False also when
/// it cannot tell -- an unverifiable sample must not masquerade as clean.
#[must_use]
pub fn machine_is_uncontended() -> bool {
    uncontended_verdict(foreign_holder().is_some(), memory_pressure())
}

/// The contention rule over one reading of each input.
///
/// Pure, so it can be pinned without asking the box what it is doing right
/// now. `None` pressure means "cannot tell", and an unverifiable sample must not
/// masquerade as clean.
#[must_use]
pub fn uncontended_verdict(foreign_lock_held: bool, pressure: Option<MemoryPressure>) -> bool {
    if foreign_lock_held {
        return false;
    }
    pressure.is_some_and(|p| !p.is_thrashing())
}
