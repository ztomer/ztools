//! Path helpers for the ztools crate.
//!
//! Extracted from `routines/src/manifest.rs` when the port moved into its own
//! crate; the ztools modules only ever used `expand_tilde`, so that is all
//! that came over.
//!
//! [`checkout_roots`] joined them: WHERE THE SHIPPED DATA LIVES, derived rather
//! than typed. Five config defaults used to name the historical home checkout
//! literally, so the installed binary found its `conf/`, its eval inputs and
//! its task snapshots on exactly one machine at exactly one path, and the house
//! gate that says so (`GOH_NO_HOME_PATHS`) could not be turned on because
//! turning it on went red. The derivation is here because a default that reads
//! the environment belongs to the one module that already owns `~`.

use std::path::{Path, PathBuf};

/// Expand a leading `~` so paths can be written portably in config.
#[must_use]
pub fn expand_tilde(p: &str) -> PathBuf {
    dirs::home_dir().map_or_else(|| PathBuf::from(p), |home| expand_tilde_in(&home, p))
}

/// The expansion, against a caller-supplied home.
///
/// The narrow seam [`expand_tilde`] is built on, in the shape
/// `twitter/cookies.rs::find_profile_dbs_under` uses: without it every branch
/// here is decided by the machine running the tests — `$HOME` is real, so the
/// "no home directory" path never executes and a pass would say nothing about
/// it.
#[must_use]
pub fn expand_tilde_in(home: &Path, p: &str) -> PathBuf {
    // Either half missing -- not a `~/` path, or no home directory -- leaves
    // the path exactly as written rather than half-expanded.
    p.strip_prefix("~/")
        .map_or_else(|| PathBuf::from(p), |rest| home.join(rest))
}

/// Overrides the executable the checkout derivation starts from.
pub const EXE_OVERRIDE_ENV: &str = "ZTOOLS_EXE";

/// The executable every live default derives its checkout from: [`EXE_OVERRIDE_ENV`]
/// when set, else `current_exe`.
///
/// WHY A SEAM. A test binary's location is a BUILD LAYOUT, not a fact about the
/// test: under the shared build dir it has no checkout above it, and under
/// `cargo llvm-cov` it sits in `rust/target/llvm-cov-target/`, INSIDE the
/// checkout -- so every sandboxed default resolved to the operator's real
/// `conf/`, and a test that passed under `cargo test` failed under coverage.
/// `TestEnv` points this at a sandbox path with nothing above it, which makes
/// "no default names the operator's disk" true in every build layout rather
/// than in the one that happened to be used.
#[must_use]
pub fn running_exe() -> Option<PathBuf> {
    std::env::var_os(EXE_OVERRIDE_ENV)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::current_exe().ok())
}

/// The checkout roots the shipped data is looked for in, in precedence order,
/// against the real environment.
///
/// See [`checkout_roots_from`] for the derivation and for why the order is what
/// it is; this is the one-liner the config defaults call.
#[must_use]
pub fn checkout_roots() -> Vec<PathBuf> {
    let exe = running_exe();
    checkout_roots_from(exe.as_deref(), dirs::home_dir().as_deref())
}

/// The checkout roots, from an explicit executable path and home.
///
/// THE DERIVATION, and why each candidate beats the next. This used to be one
/// literal naming a checkout under the operator's home, so the binary worked on
/// the machine whose home held one at that exact path and nowhere else.
/// path-ok: the pre-derivation literal, quoted as the defect being fixed.
///
/// 1. **The root above the running executable.** `install.sh` builds from a
///    checkout and can install anywhere (`ZTOOLS_INSTALL_DIR`), so "the checkout
///    this binary was installed from" is the only location that is right by
///    construction rather than by coincidence. A root is only accepted when the
///    directory holds BOTH `conf/config.toml` and `rust/Cargo.toml`: one marker
///    can be satisfied by an unrelated `conf/` — a Python package's, a
///    Homebrew prefix's — and then every default would silently point at
///    whatever that directory happened to contain.
/// 2. **The historical home checkout.** The literal that was there before, kept
///    LAST and kept working: an installed binary under a Homebrew prefix has no
///    checkout above it, so for that install this is the whole derivation, and an
///    operator whose checkout has always been there must not be broken by
///    finding their own binary. The join is `home.join("Projects/ztools")`, not a
///    literal, which is what makes it portable between accounts at all.
///    path-ok: the relative name, quoted in prose to identify which fallback —
///    the code resolves it through the caller's home.
///
/// A root that resolves to a path already in the list is dropped rather than
/// repeated. Two identical candidates in a "first that exists wins" list are
/// noise in an error message that names every path tried, and on the machine
/// this was written on the two candidates ARE the same directory — so without
/// the dedupe every default would carry a duplicate.
#[must_use]
pub fn checkout_roots_from(exe: Option<&Path>, home: Option<&Path>) -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = exe.and_then(root_above).into_iter().collect();
    if let Some(home) = home {
        roots.push(home.join("Projects/ztools"));
    }
    roots.dedup();
    roots
}

/// The first ancestor of `exe` that is a ztools checkout, or `None`.
///
/// The executable's own directory is tried first, so an install that puts the
/// binary beside its `conf/` resolves; then its parents, to the filesystem
/// root. There is no bound on the walk beyond "stop at the root": a directory
/// either has both markers or it does not, and a checkout is at most a handful
/// of levels below wherever it was installed.
fn root_above(exe: &Path) -> Option<PathBuf> {
    exe.ancestors()
        .skip(1)
        .find(|dir| is_checkout_root(dir))
        .map(Path::to_path_buf)
}

fn is_checkout_root(dir: &Path) -> bool {
    dir.join("conf/config.toml").is_file() && dir.join("rust/Cargo.toml").is_file()
}

/// One candidate path per checkout root, rendered the way the config file
/// spells paths.
///
/// A path under the home directory is written `~/…` rather than as an absolute
/// path, and that is not cosmetic.
///
/// Every other candidate in these lists is `~`-relative and a config file is
/// meant to be portable between accounts. It is also what makes the derived root
/// and the historical home fallback the SAME STRING when they are the same
/// directory, which is the only way [`checkout_roots_from`]'s dedupe can see
/// that they are.
/// path-ok: identifying which fallback, in the `~` spelling this function emits.
fn render(root: &Path, home: Option<&Path>, rel: &str) -> String {
    // `join("")` appends a trailing separator, which would make a
    // single-directory default spell `/checkout/` rather than `/checkout`.
    let full = if rel.is_empty() {
        root.to_path_buf()
    } else {
        root.join(rel)
    };
    home.and_then(|home| full.strip_prefix(home).ok())
        .map_or_else(
            || full.to_string_lossy().into_owned(),
            |rest| {
                let mut spelled = String::from("~/");
                spelled.push_str(&rest.to_string_lossy());
                spelled
            },
        )
}

/// The candidate list for one shipped file: the caller's home-relative OVERLAY
/// first, then the file under each checkout root.
///
/// The overlay comes first because it is the operator's own answer, and a
/// checkout is a guess about where somebody's code lives.
#[must_use]
pub fn config_paths(overlay: &[&str], rel: &str) -> Vec<String> {
    let exe = running_exe();
    config_paths_from(exe.as_deref(), dirs::home_dir().as_deref(), overlay, rel)
}

/// [`config_paths`] against an explicit executable and home.
///
/// Every precedence claim in `config_tests.rs` is pinned through this, and that
/// is the reason it exists: the real `current_exe` during `cargo test` is the
/// test binary in a build directory with no checkout above it, so a test
/// written against the live default would pin the BUILD LAYOUT instead of the
/// rule.
#[must_use]
pub fn config_paths_from(
    exe: Option<&Path>,
    home: Option<&Path>,
    overlay: &[&str],
    rel: &str,
) -> Vec<String> {
    let mut out: Vec<String> = overlay.iter().map(|p| (*p).to_string()).collect();
    for root in checkout_roots_from(exe, home) {
        let candidate = render(&root, home, rel);
        if !out.contains(&candidate) {
            out.push(candidate);
        }
    }
    out
}

/// The FIRST checkout root as a path, for a default that is one directory
/// rather than a list.
///
/// `"."` when there is no root at all, which is what `store.rs` already falls
/// back to: a relative path that resolves against the working directory is a
/// wrong answer, but an EMPTY one is not a path at all and every reader would
/// have to invent its own fallback.
#[must_use]
pub fn first_checkout_root() -> String {
    first_checkout_root_from(running_exe().as_deref(), dirs::home_dir().as_deref())
}

/// [`first_checkout_root`] against an explicit executable and home.
#[must_use]
pub fn first_checkout_root_from(exe: Option<&Path>, home: Option<&Path>) -> String {
    checkout_roots_from(exe, home)
        .first()
        .map_or_else(|| ".".to_string(), |root| render(root, home, ""))
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod tests;
