# Changelog

All notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); entries are added
with each committed batch.

This file starts at v2.2.0 — earlier history is in git.

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
