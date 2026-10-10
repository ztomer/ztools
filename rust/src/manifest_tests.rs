//! `expand_tilde` against a fixture home, and the real one.
//!
//! Every config path in the crate resolves through here (`config.rs`,
//! `weekend/fetch.rs`, `weekend_cache.rs`, `twitter/*`), so the rules are
//! pinned as rows rather than left to `$HOME` on whatever machine runs them:
//! `~/` expands, anything else is returned EXACTLY as written, and a leading
//! `~` that is not followed by `/` is not ours to resolve.
//!
//! The second half is the checkout derivation, and its tests are all written
//! against [`checkout_roots_from`] / [`config_paths_from`] rather than against
//! the real `current_exe`. That is deliberate and it is the whole reason these
//! are tests: the real executable is the TEST BINARY, so a test written against
//! it would pin the build layout — `rust/target/debug/deps/` — and pass on this
//! machine while the rule it claims to check changed underneath it. A rule
//! about PRECEDENCE needs a fixture it controls on both sides.

use super::*;

/// A checkout on disk that [`is_checkout_root`] will accept: both markers, at
/// the paths the marker rule names.
fn fake_checkout(root: &Path) {
    std::fs::create_dir_all(root.join("conf")).unwrap();
    std::fs::create_dir_all(root.join("rust")).unwrap();
    std::fs::write(root.join("conf/config.toml"), "[best_models]\n").unwrap();
    std::fs::write(root.join("rust/Cargo.toml"), "[package]\n").unwrap();
}

/// A home holding a checkout at the historical path, which is what the
/// derivation has to keep working.
fn home_with_the_old_layout(home: &Path) -> PathBuf {
    let checkout = home.join("Projects/ztools");
    fake_checkout(&checkout);
    checkout
}

/// A checkout somewhere else entirely, with no `conf/` anywhere above the
/// executable — the shape a Homebrew install has.
fn checkout_elsewhere(tmp: &Path, name: &str) -> PathBuf {
    let root = tmp.join(name);
    fake_checkout(&root);
    root
}

/// A home that cannot be anyone's real one.
fn fake_home() -> &'static Path {
    Path::new("/fixture/home/ztomer")
}

/// The historical candidate, spelled ONCE so the pins below cannot disagree
/// about it.
///
/// A literal on purpose: these assertions guard the *derivation*, so deriving the
/// expected value from `checkout_roots_from` would be the code compared with
/// itself — the class `docs/TESTING.md` records as "an assertion whose expected
/// value is the fallback cannot see which branch ran". The gate's `path-ok:`
/// marker is what lets a literal live in a test at all.
/// path-ok: the pre-derivation candidate these tests pin, not a resolved location.
const HISTORICAL: &str = "~/Projects/ztools";

#[test]
fn a_leading_tilde_expands_against_the_given_home() {
    for (given, want) in [
        (
            "~/conf/config.toml",
            "/fixture/home/ztomer/conf/config.toml",
        ),
        (
            "~/.cache/twitter/debug_tweets.json",
            "/fixture/home/ztomer/.cache/twitter/debug_tweets.json",
        ),
        (
            // The `~/` INPUT this row expands; HISTORICAL is the same directory
            // without the suffix. A table of `&str` pairs cannot hold a
            // `format!`ed row, so the literal stays.
            // path-ok: the pre-derivation candidate, as the input to the expansion.
            "~/Projects/ztools/conf/weekend.toml",
            "/fixture/home/ztomer/Projects/ztools/conf/weekend.toml",
        ),
        ("~/a", "/fixture/home/ztomer/a"),
    ] {
        assert_eq!(
            expand_tilde_in(fake_home(), given),
            PathBuf::from(want),
            "{given}"
        );
    }
}

/// Everything that is not a `~/` path comes back byte-for-byte: no absolute-ising
/// against the working directory, no cleanup, no rewriting of `./`.
#[test]
fn a_path_without_a_tilde_is_returned_exactly_as_written() {
    for given in [
        "conf/config.toml",
        "/already/absolute.toml",
        "relative/without/tilde",
        "./here.toml",
        "../parent.toml",
        "",
        "not a path at all",
    ] {
        assert_eq!(
            expand_tilde_in(fake_home(), given),
            PathBuf::from(given),
            "{given:?} must not be touched"
        );
    }
}

/// The cases where a leading `~` exists but is NOT ours to expand. Half-
/// expanding them would resolve another user's home (which may not exist) or
/// silently drop the intent; `~user` is the shell's, and a bare `~` has no
/// slash for the config format to anchor on.
#[test]
fn a_tilde_that_is_not_a_home_prefix_is_left_alone() {
    for given in ["~", "~user/conf/config.toml", "~ztomer", " ~", "~-backup"] {
        assert_eq!(
            expand_tilde_in(fake_home(), given),
            PathBuf::from(given),
            "{given:?} must not be expanded"
        );
    }
}

/// Exactly ONE `~/` is consumed: `~/~/x` is a real directory named `~` under
/// the home, not a double expansion. Pinned because `Path::join` never
/// normalises, and a `canonicalize` added here later would change the string
/// every caller compares.
#[test]
fn expansion_consumes_one_prefix_and_does_not_normalise() {
    assert_eq!(
        expand_tilde_in(fake_home(), "~/~/x"),
        PathBuf::from("/fixture/home/ztomer/~/x")
    );
    assert_eq!(
        expand_tilde_in(fake_home(), "~/a/../b"),
        PathBuf::from("/fixture/home/ztomer/a/../b"),
        "`..` survives: the result is a path, not a resolved one"
    );
}

/// The wrapper reads `$HOME` through `dirs::home_dir()`, so the one thing
/// asserted here is that that branch expands: an absolute path out, never the
/// literal `~`. It reads the SANDBOX home: a test that resolves the operator's
/// real `~` is one edit away from writing there, which is the class the audit
/// gate's home-resolver hazards exist for. (The "no home directory" arm is left
/// unpinned; it fails closed — the input path.)
#[test]
fn the_real_home_expands_a_leading_tilde_to_an_absolute_path() {
    let env = crate::test_env::TestEnv::new();
    let expanded = expand_tilde("~/probe/only");
    assert!(
        expanded.is_absolute(),
        "expand_tilde did not expand: {expanded:?}"
    );
    assert_ne!(expanded, PathBuf::from("~/probe/only"));
    if let Some(home) = dirs::home_dir() {
        assert_eq!(expanded, home.join("probe/only"));
    }
    assert!(expanded.starts_with(env.root()), "{expanded:?}");
    // A path with no tilde is untouched by the real branch too.
    assert_eq!(expand_tilde("conf/x.toml"), PathBuf::from("conf/x.toml"));
    drop(env);
}

// MARK: - The checkout derivation, and the precedence behind it

/// The rule the whole thing exists for: the binary finds its data where it was
/// INSTALLED FROM, not at a path somebody's home happens to hold. A checkout
/// beside the executable wins over the historical home checkout because the
/// executable is the only input that says which checkout this build came from.
/// path-ok: naming which fallback, in the spelling [`HISTORICAL`] holds.
#[test]
fn the_checkout_above_the_executable_beats_the_one_under_the_home_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    let checkout = home_with_the_old_layout(&home);
    let installed = checkout_elsewhere(tmp.path(), "opt/ztools");
    // The documented install layout: a release build, installed beside its own
    // `conf/` and `rust/`.
    let bin = installed.join("rust/target/release");
    std::fs::create_dir_all(&bin).unwrap();
    let exe = bin.join("ztools");
    std::fs::write(&exe, b"").unwrap();

    assert_eq!(
        checkout_roots_from(Some(exe.as_path()), Some(home.as_path())),
        vec![installed, checkout],
        "the root above the executable is FIRST, and the historical home path is \
         still in the list so a Homebrew install keeps working"
    );
}

/// An installed binary with no checkout above it — which is what
/// `/opt/homebrew/bin/ztools` is — falls back to the historical layout rather
/// than to nothing. This is the case that decides whether the change is a
/// product improvement or an outage.
#[test]
fn a_binary_with_no_checkout_above_it_still_finds_the_home_layout() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let checkout = home_with_the_old_layout(&home);
    let exe = tmp.path().join("prefix/bin/ztools");
    std::fs::create_dir_all(exe.parent().unwrap()).unwrap();

    assert_eq!(
        checkout_roots_from(Some(exe.as_path()), Some(home.as_path())),
        vec![checkout],
        "one candidate, and it is the historical one: nothing above /prefix is a \
         ztools checkout, so the derivation must not invent one"
    );
}

/// A marker is TWO files, and the reason is a false positive: any directory
/// with a `conf/` would otherwise pass, and then every default would point into
/// whatever that directory happened to contain — a Homebrew prefix's `conf`, a
/// Python package's.
#[test]
fn one_marker_is_not_a_checkout() {
    let tmp = tempfile::tempdir().unwrap();
    let conf_only = tmp.path().join("opt/ztools");
    std::fs::create_dir_all(conf_only.join("conf")).unwrap();
    std::fs::write(conf_only.join("conf/config.toml"), "[best_models]\n").unwrap();
    let crate_only = tmp.path().join("elsewhere");
    std::fs::create_dir_all(crate_only.join("rust")).unwrap();
    std::fs::write(crate_only.join("rust/Cargo.toml"), "[package]\n").unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let checkout = home_with_the_old_layout(&home);

    for exe in [
        conf_only.join("bin/ztools"),
        crate_only.join("bin/ztools"),
        crate_only.join("target/debug/deps/ztools-abc"),
    ] {
        std::fs::create_dir_all(exe.parent().unwrap()).unwrap();
        assert_eq!(
            checkout_roots_from(Some(exe.as_path()), Some(home.as_path())),
            vec![checkout.clone()],
            "{} looks like a checkout on one marker alone",
            exe.display()
        );
    }
}

/// A candidate that resolves to a path already in the list is dropped. On the
/// machine this was written on the derived root IS [`HISTORICAL`], so
/// without the dedupe every default in the crate would have carried the same
/// directory twice — and the "no eval conf dir found" error names every path it
/// tried, twice.
#[test]
fn a_derived_root_that_is_the_home_fallback_collapses_into_one_candidate() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let checkout = home_with_the_old_layout(&home);
    // Built INTO the checkout, which is `cargo build`'s layout under the
    // machine's shared cargo build dir's project-local `target/`.
    let exe = checkout.join("rust/target/debug/ztools");
    std::fs::create_dir_all(exe.parent().unwrap()).unwrap();

    assert_eq!(
        checkout_roots_from(Some(exe.as_path()), Some(home.as_path())),
        vec![checkout],
        "one root, not the same root twice"
    );
    assert_eq!(
        config_paths_from(
            Some(exe.as_path()),
            Some(home.as_path()),
            &["~/.config/ztools"],
            "conf"
        ),
        vec!["~/.config/ztools".to_string(), format!("{HISTORICAL}/conf")],
        "and the shipped candidate is spelled the way every other one is, so the \
         dedupe above can see that it IS the fallback"
    );
}

/// The overlay beats every checkout, and the derived checkout beats the home
/// fallback — the full order, as one list, because "each default's precedence"
/// is only meaningful as the order they are read in.
#[test]
fn the_overlay_comes_first_then_the_installed_checkout_then_the_home_fallback() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    home_with_the_old_layout(&home);
    let installed = checkout_elsewhere(tmp.path(), "opt/ztools");
    let exe = installed.join("bin/ztools");
    std::fs::create_dir_all(exe.parent().unwrap()).unwrap();

    assert_eq!(
        config_paths_from(
            Some(exe.as_path()),
            Some(home.as_path()),
            &["~/.config/ztools/config.toml"],
            "conf/config.toml"
        ),
        vec![
            String::from("~/.config/ztools/config.toml"),
            // The derived root is outside the home, so it is spelled absolutely:
            // it is a path only THIS install can know. The historical root is
            // under the home and keeps its `~` spelling, so a config file stays
            // portable and a reader diffing two of them sees the same string.
            format!("{}/conf/config.toml", installed.display()),
            format!("{HISTORICAL}/conf/config.toml"),
        ],
        "operator's own file, then the checkout this binary came from, then the \
         historical one; swapping any two of these changes whose answer wins"
    );
}

/// A checkout that happens to live under the home directory is still spelled
/// `~/…`, so a config file stays portable.
#[test]
fn a_root_under_the_home_directory_is_spelled_with_a_tilde() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let checkout = home.join("work/ztools");
    fake_checkout(&checkout);
    let exe = checkout.join("bin/ztools");
    std::fs::create_dir_all(exe.parent().unwrap()).unwrap();

    assert_eq!(
        config_paths_from(
            Some(exe.as_path()),
            Some(home.as_path()),
            &[],
            "conf/weekend.toml"
        ),
        vec![
            String::from("~/work/ztools/conf/weekend.toml"),
            format!("{HISTORICAL}/conf/weekend.toml"),
        ]
    );
}

/// No home at all is not a crash and not an empty string: the historical
/// fallback needs a home to be relative to, so with none the list is simply the
/// overlay, and every reader that consumes it names the candidates it tried.
#[test]
fn no_home_means_the_overlay_alone_and_a_single_dir_default_says_dot() {
    let tmp = tempfile::tempdir().unwrap();
    assert_eq!(
        config_paths_from(
            Some(tmp.path().join("bin/ztools").as_path()),
            None,
            &["~/.config/ztools"],
            "conf"
        ),
        vec!["~/.config/ztools"],
        "an unresolvable `~` is the reader's problem to report, not a silent guess"
    );
    assert_eq!(
        first_checkout_root_from(None, None),
        ".",
        "a directory default cannot be empty: every caller would have to invent \
         its own fallback, and `.` is the one every other path helper here uses"
    );
}

/// The single-directory defaults read the FIRST root, so the collector points at
/// the checkout the binary came from rather than at a second copy in the home.
#[test]
fn a_single_directory_default_is_the_first_root_not_the_home_fallback() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    let checkout = home_with_the_old_layout(&home);
    let installed = checkout_elsewhere(tmp.path(), "opt/ztools");
    let exe = installed.join("bin/ztools");
    std::fs::create_dir_all(exe.parent().unwrap()).unwrap();

    assert_eq!(
        first_checkout_root_from(Some(exe.as_path()), Some(home.as_path())),
        installed.to_string_lossy().into_owned(),
        "the derived root itself, spelled absolutely because it is not under the \
         home -- and with NO trailing separator, which `Path::join(\"\")` would \
         have added"
    );
    assert_eq!(
        first_checkout_root_from(
            Some(checkout.join("rust/target/debug/ztools").as_path()),
            Some(home.as_path())
        ),
        HISTORICAL.to_string(),
        "and when the checkout IS the home fallback, the single-directory default \
         is the same string the operator had before this change"
    );
}

/// The sandbox decides the executable, so the BUILD LAYOUT cannot decide the
/// defaults.
///
/// Under `cargo llvm-cov` the test binary lives in
/// `rust/target/llvm-cov-target/`, inside the real checkout, and every sandboxed
/// default resolved to the operator's `conf/`: a config test passed under
/// `cargo test` and failed under coverage. Both halves are asserted, the seam
/// first -- an executable inside a checkout DOES find it -- so the sandbox half
/// cannot pass by the seam being ignored.
#[test]
fn the_sandbox_executable_has_no_checkout_above_it_in_any_build_layout() {
    let env = crate::test_env::TestEnv::new();
    let checkout = env.root().join("checkout");
    fake_checkout(&checkout);
    let inside = checkout.join("rust/target/llvm-cov-target/debug/deps/ztools-0");
    env.set_managed(EXE_OVERRIDE_ENV, inside.as_os_str());
    assert_eq!(running_exe().as_deref(), Some(inside.as_path()));
    assert_eq!(
        checkout_roots().first(),
        Some(&checkout),
        "the seam is read"
    );
    drop(env);

    // A fresh sandbox: the only root left is the home fallback, and it is the
    // SANDBOX's home -- nothing derived from where the test binary was built.
    let env = crate::test_env::TestEnv::new();
    let home = dirs::home_dir().unwrap();
    assert_eq!(
        checkout_roots(),
        vec![home.join("Projects/ztools")],
        "derived from {:?}",
        running_exe()
    );
    assert!(home.starts_with(env.root()), "{home:?}");
    drop(env);
}
