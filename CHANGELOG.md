# Changelog

All notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); entries are added
with each committed batch.

This file starts at v2.2.0 — earlier history is in git.

## v3.0.0 — zero Python at runtime _(2026-09-13)_

The Python reference tree is gone. Every behaviour it carried is ported — with the
Python verdicts frozen as goldens the Rust tests assert — or explicitly retired in
`docs/PORT_PARITY.md`. Major because the Python entry points (`tw`, `wk`, `rn`, `ev`,
`python -m twitter`) and `TWITTER_COLLECTOR` no longer exist.

### Changed
- **Twitter collect is the native camoufox-rs driver, unconditionally.** The Python
  subprocess path, `pyenv`, the post-run cache scan and the `TWITTER_COLLECTOR` switch
  are deleted. Both collectors now assert the captured traffic came from the Following
  endpoint (markers in `conf/twitter.toml [endpoints]`) and fail loudly otherwise.
- **Collect parity is shared-record agreement**, not ID-set equality: the Following
  endpoint is served as a per-load sample (the same collector minutes apart shares
  ~7 of 50 IDs), so only the tweets both legs captured are compared, byte for byte.
- **`routines.toml` / `routines-twitter.toml` run the binary.** Status commands are
  `ztools status` and the new `ztools twitter-status`.
- **One weekend plan store.** The planner always writes the dated
  `weekend_plan_<Month>_<dd>_to_<Month>_<dd>_<yyyy>.md` into `~/Documents/weekend_plans`
  (or `WEEKEND_OUTPUT_DIR`), and the status page, the dashboard tab and the writer
  resolve that one directory. Previously the status read `~/Documents` while the tab
  read `weekend_plans/`, so a fresh plan read as stale.
- **Weekend tables use the C4 missing-value sentinel** in every cell; the fabricated
  "Family activity in GTA" filler, the constant "Outdoor/Indoor" column and a
  hardcoded "Vaughan" are gone.

### Added
- **Summarizer fallback chain with provenance** (`twitter/chain.rs`): the intended
  model resolved against the server roster, then `conf/twitter.toml [fallback]`
  models; a degraded run prints its reasons and the artifact opens with the
  DEGRADED OUTPUT banner. `TWITTER_FALLBACK_MODELS` overrides for one run.
- **`model-eval --suite full` runs the full task roster** (`eval/tasks.rs`, the 24
  Python table rows + taxes snapshots), graded 0-100 by the ported validators with the
  prompt as their source; the mixed-signal scorers (`validators/mixed_text.rs`), the
  vision task (`eval/vision.rs`, fixtures in `conf/eval_vision.toml`, images as OpenAI
  content parts) and real `parse_json` handling (including the FORMAT/PARSE failure
  classes) landed with it.
- Golden parity suites: `tests/validator_parity.rs`, `tests/weekend_parity.rs`,
  `eval/tasks_tests.rs`, `validators/mixed_text_tests.rs`, `eval/prompts` drift test
  against `conf/prompts.toml`, `tests/gpu_lock_shell_parity.rs`.
- `.gatesrc` runs `python3 -m pytest tools/tests` (the shell lock's tests, moved
  beside the tool).

### Removed
- `references/` (286 files), `pyproject.toml`, `uv.lock`, the `.venv`, the root
  `ztools` Python TUI wrapper, `routines_twitter_status.py`, `eval_tasks/__init__.py`,
  and the Python-only tools (`gen_rust_prompts.py`, `ab_eval_parity.sh`, `mutate.py`,
  the unwired `check_config_debt.py` / `check_file_size.py`). `tools/sweep_models.sh`
  and `rerun_truncated.sh` drive `ztools model-eval` and read pressure in shell.

### Deferred (stated, see `docs/ROADMAP.md`)
Weekend phase retry and timeout learning, eval token/verbosity metrics, the
`diff_from_last_run` presenter, drain-mode SIGINT.

## v2.3.0 — pedantic, audited, and one real bug _(2026-09-07)_

Part of an estate-wide Rust quality campaign. The crate went from no lint policy
at all to zero findings under `clippy::pedantic` + `nursery` at `-D warnings`,
and the sweep turned up a defect that had nothing to do with style.

### Fixed
- **`gpu_lock::is_owner_alive` could report a dead lock owner as alive forever.**
  It probed the owning process with `kill(pid as i32, 0)`. `std::process::id()`
  returns `u32` and `kill(2)` takes `i32`, and an `as` cast of a value above
  `i32::MAX` goes NEGATIVE — where a negative pid is not an invalid argument but
  a request to signal an entire process GROUP, which succeeds whenever anything
  in that group is running. The lock would never be reclaimed. The conversion is
  now checked once in `src/units.rs`, saturating to `i32::MAX` — a pid that
  simply does not exist, which fails safely.
- **An assertion that could not fail was deleted**, not weakened:
  `assert!(sys.contains(CARRY_FIELDS) || true)` in the weekend-phases tests.

### Added
- **`cargo audit` is a gate**, wired into `.gatesrc` rather than run by memory,
  and the yanked crate it found on its first run is gone.
- **`src/units.rs`** — the narrowing conversions this crate does everywhere,
  done once and tested, instead of an `as` cast per call site each making its own
  silent decision about the boundary.
- **`src/ztools/eval/scoring_math.rs`** — `ratio()` and `rounded()`, with the
  zero-denominator behaviour stated rather than implied.

### Changed
- **Every fallible or panicking public function now says so.** `# Errors` and
  `# Panics` sections throughout, which is `pedantic`'s point rather than its
  ceremony: the ones that were hard to write were the ones whose failure modes
  were not understood.
- **Fourteen `if let/else` become combinators**, and four say in an `#[expect]`
  why they should not.
- **`float_cmp`: exact equality kept, with the reason.** A tolerance would be
  worse here — these compare values that are either the same float or a
  different one by construction.
- **Five identifiers renamed whose leading underscore was a lie** — they were
  read.

## v2.2.0 — A Probed Interpreter, and a Gate _(2026-09-02)_

### Fixed
- **The Python shell-outs pick an interpreter that can actually run them.**
  Three call sites used a bare `Command::new("python3")` — the twitter browser
  scrape (`twitter/browser.rs`, twice) and the weekend planner's multi-engine
  search helper (`weekend/mod.rs`). That is correct in a login shell and wrong
  everywhere else: the `routines` menubar app's GUI environment has
  `PATH=/usr/bin:/bin:/usr/sbin:/sbin`, so `python3` resolved to
  `/usr/bin/python3`, which has none of the dependencies.

  The twitter summary died on `ModuleNotFoundError: No module named
  'requests'` and reported only "exit code 1". The weekend planner swallowed
  the identical failure into `except Exception: print([])` and reported "no
  candidates found" — which was false; nothing had searched. Measured on the
  affected machine: **0/14 → 2/67 candidates**, and the plan gained a real
  "Transient / Limited-Time Events" section instead of the fixed-activity
  fallback.

  New `ztools::pyenv` resolves explicitly and **probes**: a candidate is used
  only after it has been asked to import the exact modules that pipeline
  needs, so "the binary exists" can no longer pass for "the binary can run
  this". Candidates in order: `$ZTOOLS_PYTHON`, the project `.venv`,
  `/opt/homebrew/bin/python3`, `/usr/local/bin/python3`, then `python3`. When
  none survive, the error names every path tried and what each was missing.
- **The search helper no longer reports a failure as an empty result.**
  `collect_snippets_external` returns `Result` rather than a bare `Vec`, so a
  helper that could not start and a search that genuinely found nothing are no
  longer the same answer.

### Added
- **This repo has a gate.** `tools/gate.sh --full` (the pre-push hook) ran only
  the structural layer, with every language layer commented out and nothing
  else to pick them up — so the Rust half was entirely ungated. It had drifted
  to **223 `cargo fmt` violations** while every gate run reported all-green,
  because none of them had ever claimed to look. (Clippy was clean, which is
  precisely why an ungated repo is dangerous rather than obviously broken.)

  `--full` and `make ci` now delegate to the same runner over one declared step
  list in `.gatesrc`: the house Rust gate, the emoji and file-length checks,
  the test suite, and a coverage floor. A `Makefile` gives the same named
  targets the sibling `routines` repo has.
- **A `CHANGELOG.md`**, starting here.

### Changed
- **The whole Rust tree is `cargo fmt` clean** — a one-time mechanical pass
  over 62 files, now held by the gate.
- **`cli_ztools.rs` split** at the seam it already had: `--capabilities`
  reports what a model IS without running anything, and moved to
  `cli_ztools_capabilities.rs`, bringing the file back under the 500-line cap.

- **The `--task` filter no longer silently runs the whole suite.** Entries were
  split on `,` and trimmed but never checked for emptiness, and
  `name.ends_with("")` is true of every name — so `--task taxes,` (a trailing
  comma, easy to type) matched every task and ran the full eval instead of one.
  Found by extracting the rule into `task_matches_filter` and testing it.

### Changed
- **Four home-anchored lookups grew a seam** so their branches are provable
  without writing into the developer's own home directory: the shared-prompt
  layering (`ZtoolsConfig::with_shared_prompts_from`), the twitter cache scan
  (`tweets_from_cache`), the helper-output classifier
  (`weekend::classify_helper_output`), and the task filter above. Coverage of
  the affected files went 93.51% → 94.43% overall, but the point is that the
  failure branches — file absent, unreadable, malformed, present-but-empty —
  are now pinned rather than merely believed.

### Known
- The coverage floor is **94%**, not the house 95%, and stated rather than
  quietly set. Measured 94.43%. What remains uncovered is `cli_ztools.rs`'s
  model-eval run loop and the live twitter scrape — code that takes the
  machine-wide GPU lock and drives a headless browser against a live model
  server. Reaching 95 by unit test would mean inventing seams for the number's
  sake rather than because the logic deserves isolation. The floor is a
  ratchet: it bites today, and it rises when a live-server integration harness
  exists.
