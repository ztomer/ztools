//! `signals`' tests, split by the domain each targets.
//!
//! Split out of one 492-line file for the 500-lines cap (no test exemption; see
//! CLAUDE.md) and to keep the three domains apart: WHERE the store lives
//! (`paths`), WHAT the machine is doing (`pressure` and `pressure_tools`), and
//! WHAT gets learned from it (`learning`).
//!
//! `identity` is the fourth, and the only one about the store's memory rather
//! than its arithmetic: what a per-task series does when the task under its
//! name is replaced. It is separate because it is the one place where a stored
//! number is deliberately DISCARDED, and the rule for that needs a file to be
//! wrong in.
//!
//! Every test that touches the environment constructs [`crate::test_env::TestEnv`]
//! and nothing else. The `EnvGuard` and `Fixture` that used to live here each
//! re-implemented the same save/restore loop over their own variable list,
//! which is how the `EVAL_OUTPUT_DIR` omission got in; there is now one list,
//! one lock and one restore, and `crate::test_env::audit` fails if a new
//! variable is read without it.
//!
//! Each submodule reaches the module under test with `use super::super::*`, so
//! the private constants the parsers share are in scope without widening the
//! module's public surface.
//!
//! THE LINUX FIXTURES LIVE HERE, not in either test file, because both need the
//! same bytes: `pressure` pins the pure parser against them and
//! `pressure_tools` pins the file-reading seam against them. A fixture copied
//! per file is a fixture that drifts, and a drift between those two would look
//! exactly like a reader bug.

mod identity;
mod learning;
mod paths;
mod pressure;
mod pressure_tools;

/// A real `/proc/meminfo`, trimmed to the fields the reader uses and to the
/// neighbours a sloppy prefix match would accept (`SwapCached`, `Cached`,
/// `Active`). The kernel's own `Key:   value kB` alignment is preserved, so a
/// parser that accidentally depended on column positions fails here.
pub(super) const MEMINFO_QUIET: &str = "\
MemTotal:       65536000 kB
MemFree:        42108416 kB
MemAvailable:   50123456 kB
Buffers:          131072 kB
Cached:         15728640 kB
SwapCached:            0 kB
Active:         10485760 kB
Inactive:        5242880 kB
SwapTotal:       8388608 kB
SwapFree:        8388608 kB
";

/// The same file on a box that has swapped 15GiB of its 16GiB.
pub(super) const MEMINFO_SWAPPED: &str = "\
MemTotal:       65536000 kB
MemFree:          524288 kB
MemAvailable:    3145728 kB
Buffers:          131072 kB
Cached:          1048576 kB
SwapCached:      1048576 kB
SwapTotal:      16777216 kB
SwapFree:        1048576 kB
";
