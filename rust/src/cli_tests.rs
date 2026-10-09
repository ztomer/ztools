//! The argv the SYMLINKED binary runs on, and the config `load_config` resolves.
//!
//! `install.sh` puts eight names in `$(brew --prefix)/bin`, each a symlink to
//! the one `ztools` binary, so for an installed user `argv[0]` is the only
//! thing separating `oeval` from `ztools weekend-plan`. Nothing in
//! `tests/cli_dispatch_ztools.rs` exercises that: it always spawns the binary
//! under its own name, so the implication branch never fires anywhere in the
//! suite. These tests ARE that branch — and the symlink list is read out of
//! `install.sh` rather than copied here, so a tenth alias is a red test
//! instead of a silent gap.

use super::*;

/// argv as the process hands it over.
fn argv(items: &[&str]) -> Vec<std::ffi::OsString> {
    items.iter().map(std::ffi::OsString::from).collect()
}

/// The repository root, so the gate reads the script that ships the aliases.
fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("rust/ has a parent")
        .to_path_buf()
}

/// The names `install.sh` symlinks, parsed out of its `for cmd in` loop.
///
/// An empty result means the parse missed the loop, which the one caller
/// asserts against: a gate that matched nothing would otherwise report
/// compliance over zero aliases.
fn installed_symlink_names() -> Vec<String> {
    let script = repo_root().join("install.sh");
    let text = std::fs::read_to_string(&script)
        .unwrap_or_else(|e| panic!("install.sh at {}: {e}", script.display()));
    let mut names = Vec::new();
    let mut in_loop = false;
    for line in text.lines() {
        let t = line.trim();
        let body = if in_loop {
            t
        } else if let Some(rest) = t.strip_prefix("for cmd in ") {
            in_loop = true;
            rest
        } else {
            continue;
        };
        if body.starts_with("done") {
            break;
        }
        // The last name carries the loop's own `; do`; the ones before it end
        // in a `\`. Either way that line is the last one to read.
        let (body, last) = body.strip_suffix("do").map_or_else(
            || {
                (
                    body.strip_suffix('\\')
                        .unwrap_or(body)
                        .trim_end()
                        .to_string(),
                    false,
                )
            },
            |before| {
                let before = before.trim_end();
                (before.strip_suffix(';').unwrap_or(before).to_string(), true)
            },
        );
        names.extend(body.split_whitespace().map(str::to_string));
        if last {
            break;
        }
    }
    names
}

fn names_of(table: &[(&str, &str)]) -> Vec<String> {
    table.iter().map(|(name, _)| (*name).to_string()).collect()
}

/// `(alias, subcommand)` — what `install.sh` installs and what each name must
/// therefore run. The table the script and the code must both agree with.
const INSTALLER_ALIASES: &[(&str, &str)] = &[
    ("twitter", "twitter-summarize"),
    ("twitter-summarize", "twitter-summarize"),
    ("weekend", "weekend-plan"),
    ("weekend-plan", "weekend-plan"),
    ("rename_images", "image-renamer"),
    ("image-renamer", "image-renamer"),
    ("oeval", "model-eval"),
    ("model-eval", "model-eval"),
];

/// The whole point of the table: every name the installer puts on `PATH`
/// resolves to a subcommand — and the list compared against is the installer's
/// own, read from the script.
#[test]
fn every_symlink_install_sh_creates_implies_its_subcommand() {
    let installed = installed_symlink_names();
    // Never a vacuous pass: an unparsable script must fail here, not compare
    // an empty list against a table and shrug.
    assert_nonempty!(installed.as_slice());
    assert_eq!(
        installed,
        names_of(INSTALLER_ALIASES),
        "install.sh's symlink list, or this table, changed"
    );
    for (alias, sub) in INSTALLER_ALIASES {
        assert_eq!(
            implied_subcommand(alias),
            Some(*sub),
            "{alias} is on PATH (install.sh) and must imply {sub}"
        );
    }
}

/// The names that are NOT installed must not acquire a subcommand they cannot
/// run: an unknown `argv[0]` is clap's problem, not ours.
#[test]
fn a_name_that_is_not_an_alias_gets_no_subcommand() {
    for name in [
        "ztools",           // the binary's own name: a subcommand is still required
        "ab_test",          // the dev harness install.sh explicitly removes
        "status",           // a subcommand name, not a program name
        "twitter_status",   // likewise
        "oeval-extra",      // a near miss must not match
        "WEEKEND",          // argv[0] names are matched exactly, not case-folded
        "weekend-plan.exe", // the installer links extensionless names
        " ztools",          // and not whitespace-trimmed either
    ] {
        assert_eq!(implied_subcommand(name), None, "{name} must imply nothing");
    }
    // The hyphenated spelling of rename_images is an accepted alias, though the
    // installer uses the underscored one.
    assert_eq!(
        implied_subcommand("rename-images"),
        Some("image-renamer"),
        "rename-images is an alias of rename_images"
    );
}

/// The name is the LAST component of `argv[0]`: the installer links
/// `/opt/homebrew/bin/weekend`, and only the leaf is in the table.
#[test]
fn the_installers_absolute_paths_take_their_last_component() {
    for (argv0, sub) in [
        ("/opt/homebrew/bin/weekend", "weekend-plan"),
        ("/usr/local/bin/weekend", "weekend-plan"), // a ZTOOLS_INSTALL_DIR override
        ("/tmp/ztools-install/bin/oeval", "model-eval"),
        ("/opt/homebrew/bin/rename_images", "image-renamer"),
        ("/opt/homebrew/bin/twitter", "twitter-summarize"),
    ] {
        assert_eq!(
            argv_with_implied_subcommand(argv(&[argv0])),
            argv(&[argv0, sub]),
            "{argv0}"
        );
    }
    // A directory in the path that IS an alias name must not decide: only the
    // last component is the program's name.
    assert_eq!(
        argv_with_implied_subcommand(argv(&["/opt/homebrew/bin/weekend/ztools"])),
        argv(&["/opt/homebrew/bin/weekend/ztools"]),
    );
}

/// The implied subcommand goes BEFORE the user's flags, so `weekend
/// --location X` parses. Inserting anywhere else would put a flag in the
/// subcommand slot.
#[test]
fn the_implied_subcommand_lands_before_the_users_flags() {
    assert_eq!(
        argv_with_implied_subcommand(argv(&["weekend", "--location", "Vaughan/Toronto"])),
        argv(&["weekend", "weekend-plan", "--location", "Vaughan/Toronto"]),
    );
    assert_eq!(
        argv_with_implied_subcommand(argv(&["/opt/homebrew/bin/twitter", "--json", "-"])),
        argv(&[
            "/opt/homebrew/bin/twitter",
            "twitter-summarize",
            "--json",
            "-"
        ]),
    );
    assert_eq!(
        argv_with_implied_subcommand(argv(&["rename_images", "/tmp/images", "--apply"])),
        argv(&["rename_images", "image-renamer", "/tmp/images", "--apply"]),
    );
    // Bare name, no flags at all (args.len() == 1).
    assert_eq!(
        argv_with_implied_subcommand(argv(&["oeval"])),
        argv(&["oeval", "model-eval"]),
    );
}

/// `weekend weekend-plan …` must stay ONE subcommand: the implication exists
/// for the spellings that omit it.
#[test]
fn an_explicit_subcommand_is_neither_overridden_nor_duplicated() {
    for (given, sub) in [
        (vec!["weekend", "weekend-plan"], "weekend-plan"),
        (vec!["oeval", "model-eval", "--suite", "full"], "model-eval"),
        (
            vec![
                "/opt/homebrew/bin/twitter",
                "twitter-summarize",
                "--json",
                "-",
            ],
            "twitter-summarize",
        ),
    ] {
        let after = argv_with_implied_subcommand(argv(&given));
        assert_eq!(after, argv(&given), "{sub} was already given");
        assert_eq!(
            after
                .iter()
                .filter(|a| a.as_os_str() == std::ffi::OsStr::new(sub))
                .count(),
            1,
            "{sub} must appear exactly once in {after:?}"
        );
    }
}

/// An explicitly named subcommand placed AFTER a global flag is duplicated
/// today. Pinned because that is what the code does, not because it is right:
/// the implication only looks at `argv[1]`, so `weekend --config p
/// weekend-plan` gets a second copy inserted in front of the flag and clap
/// then sees the real one as a stray argument. The spellings that work are
/// `weekend --config p` (implied) and `ztools --config p weekend-plan`
/// (explicit, no symlink). If the implication ever learns to skip leading
/// global options this goes red — that is the change, not a broken test.
#[test]
fn an_explicit_subcommand_after_a_global_flag_is_duplicated_today() {
    assert_eq!(
        argv_with_implied_subcommand(argv(&[
            "weekend",
            "--config",
            "/tmp/z.toml",
            "weekend-plan"
        ])),
        argv(&[
            "weekend",
            "weekend-plan",
            "--config",
            "/tmp/z.toml",
            "weekend-plan"
        ]),
    );
}

/// No alias, no rewrite, no error: an unknown `argv[0]` reaches clap
/// untouched, which then fails with "a subcommand is required". The last three
/// shapes have no usable file name at all — `file_name()` is `None` for `.`,
/// `..` and an empty path — and take the same route.
#[test]
fn an_unknown_program_name_is_handed_to_clap_untouched() {
    for argv0 in [
        "ztools",
        "/opt/homebrew/bin/ztools",
        "ab_test",
        ".",
        "..",
        "weekend/..",
        "",
    ] {
        let before = argv(&[argv0]);
        assert_eq!(
            argv_with_implied_subcommand(before.clone()),
            before,
            "argv[0] {argv0:?} must acquire nothing"
        );
    }
    // And with args: `ztools weekend-plan` is already the explicit shape.
    assert_eq!(
        argv_with_implied_subcommand(argv(&["ztools", "weekend-plan", "--location", "Vaughan"])),
        argv(&["ztools", "weekend-plan", "--location", "Vaughan"]),
    );
}

/// `args.first()` is `None` only for an empty argv, and it must come back
/// empty rather than panicking on the insert.
#[test]
fn an_empty_argv_is_left_empty() {
    assert_empty!(argv_with_implied_subcommand(Vec::new()));
}

// ── load_config ────────────────────────────────────────────────────────────────

/// A temp dir holding one config file. The dir comes back so it outlives the
/// call: `TempDir` deletes on drop.
fn config_file(name: &str, body: &str) -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join(name);
    std::fs::write(&path, body).expect("write config");
    (dir, path)
}

/// An explicit `--config` is AUTHORITATIVE: the file is exactly what runs, so
/// no `[best_models]` overlay and no shared-prompts layer may be applied on top
/// of it — that is what lets a test or a CI job point the URLs at stubs
/// without the operator's own config reaching the run. Asserted as EXACT
/// equality with the parsed file, so any layer added here shows up as a diff.
#[test]
fn an_explicit_config_is_exactly_what_runs() {
    // The serde defaults derive from the running executable and $HOME (TestEnv).
    let env = crate::test_env::TestEnv::new();
    let body = r#"osaurus_url = "http://127.0.0.1:9/stub"
twitter_model = "stub-model"
llm_timeout_secs = 7
twitter_config_paths = ["/tmp/twitter.toml"]
"#;
    let (_dir, path) = config_file("ztools.toml", body);
    let loaded = load_config(Some(path)).expect("the file parses");
    assert_eq!(
        loaded,
        toml::from_str::<ZtoolsConfig>(body).expect("the same file parses"),
        "the explicit file IS the config: nothing may be layered over it"
    );
    assert_eq!(loaded.osaurus_url, "http://127.0.0.1:9/stub");
    assert_eq!(loaded.twitter_model, "stub-model");
    assert_eq!(loaded.llm_timeout_secs, 7);
    drop(env);
}

/// A partial file must not zero the fields it omits: each carries a
/// `#[serde(default)]`, and those defaults are the production values.
#[test]
fn an_explicit_config_takes_serde_defaults_for_the_fields_it_omits() {
    // The serde defaults derive from the running executable and $HOME (TestEnv).
    let env = crate::test_env::TestEnv::new();
    let (_dir, path) = config_file("partial.toml", "osaurus_url = \"http://127.0.0.1:9\"\n");
    let loaded = load_config(Some(path)).expect("a one-field config parses");
    assert_eq!(loaded.osaurus_url, "http://127.0.0.1:9");
    assert_eq!(loaded.max_image_filename_len, 50, "omitted field default");
    assert_eq!(loaded.llm_warmup_timeout_secs, 900, "omitted field default");
    assert_eq!(loaded.llm_max_tokens, 4096, "omitted field default");
    assert_eq!(
        loaded.twitter_cache_path, "~/.cache/twitter/debug_tweets.json",
        "omitted field default"
    );
    assert_eq!(
        loaded.eval_conf_dirs,
        crate::manifest::config_paths(&["~/.config/ztools"], "conf"),
        "omitted list default: the operator's overlay first, then the \\
         checkout's conf/ -- derived through `manifest`, because the literal \\
         this replaces named one machine's checkout and the derivation's own \\
         precedence is pinned in `config_tests.rs`"
    );
    drop(env);
}

/// The two failure branches, each naming the file it could not use. A message
/// naming neither would leave an operator choosing between a typo'd path and a
/// bad key.
#[test]
fn a_config_that_cannot_be_read_or_parsed_names_the_file() {
    // The serde defaults derive from the running executable and $HOME (TestEnv).
    let env = crate::test_env::TestEnv::new();
    let (dir, _written) = config_file("gone.toml", "osaurus_url = \"http://x\"\n");
    let missing = dir.path().join("not-written.toml");
    let err = load_config(Some(missing.clone())).unwrap_err().to_string();
    assert!(err.contains("cannot read config"), "{err}");
    assert!(err.contains(&missing.display().to_string()), "{err}");

    let (_dir, bad) = config_file("bad.toml", "osaurus_url = [1, 2\n");
    let err = load_config(Some(bad.clone())).unwrap_err().to_string();
    assert!(err.contains("cannot parse config"), "{err}");
    assert!(err.contains(&bad.display().to_string()), "{err}");
    drop(env);
}

/// `--config` is the ONE config path in the crate that does not go through
/// `expand_tilde`: a quoted `~/…` reaches `read_to_string` verbatim and fails
/// as an unreadable file. Asserted because that is what the code does — the
/// interactive spelling works because the SHELL expands the tilde — and
/// because every other config path in the crate is tilde-expanded.
#[test]
fn an_explicit_config_path_is_not_tilde_expanded() {
    // The serde defaults derive from the running executable and $HOME (TestEnv).
    let env = crate::test_env::TestEnv::new();
    let (dir, _real) = config_file("real.toml", "osaurus_url = \"http://127.0.0.1:9\"\n");
    let looks_like_home = dir.path().join("~").join("ztools.toml");
    let err = load_config(Some(looks_like_home.clone()))
        .unwrap_err()
        .to_string();
    assert!(err.contains("cannot read config"), "{err}");
    assert!(
        err.contains(&looks_like_home.display().to_string()),
        "{err} — the path must be reported as given, unexpanded"
    );
    drop(env);
}

/// The no-`--config` branch cannot fail, and what it returns must be a config a
/// run can use: a scheme on every endpoint, and — the part this pins — every
/// path default rooted at `~` or already absolute, because all of them are
/// resolved through [`crate::manifest::expand_tilde`]. A relative default would
/// silently resolve against the process's working directory.
///
/// The model NAMES are deliberately not asserted here: this branch layers
/// `[best_models]` on top, so on a machine that has the file (this one) an
/// emptied `default_twitter_model` is invisible — the overlay replaces it. The
/// claim that a slot default names a model belongs to
/// `config::tests::embedded_slot_defaults_match_conf_best_models`, which is
/// where it can actually be seen to fail.
#[test]
fn the_defaults_config_is_usable_and_every_path_default_expands_absolute() {
    // The serde defaults derive from the running executable and $HOME (TestEnv).
    let env = crate::test_env::TestEnv::new();
    let cfg = load_config(None).expect("the defaults branch has no way to fail");
    for (slot, url) in [
        ("osaurus_url", &cfg.osaurus_url),
        ("duckduckgo_url", &cfg.duckduckgo_url),
        ("bing_url", &cfg.bing_url),
        ("brave_url", &cfg.brave_url),
    ] {
        assert!(
            url.starts_with("http"),
            "{slot} = {url:?} carries no scheme"
        );
    }

    let mut every_path: Vec<(&str, &String)> = vec![
        ("search_record_path", &cfg.search_record_path),
        ("twitter_cache_path", &cfg.twitter_cache_path),
        ("twitter_collector_dir", &cfg.twitter_collector_dir),
    ];
    for (slot, list) in [
        ("twitter_config_paths", &cfg.twitter_config_paths),
        ("weekend_exclusions_paths", &cfg.weekend_exclusions_paths),
        ("weekend_region_paths", &cfg.weekend_region_paths),
        ("eval_conf_dirs", &cfg.eval_conf_dirs),
        ("eval_tasks_dirs", &cfg.eval_tasks_dirs),
    ] {
        every_path.extend(list.iter().map(move |p| (slot, p)));
    }
    assert_nonempty!(every_path.as_slice());
    for (slot, path) in every_path {
        assert!(
            crate::manifest::expand_tilde(path).is_absolute(),
            "{slot} = {path:?} would resolve against the working directory"
        );
    }
    drop(env);
}
