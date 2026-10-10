# Changelog

All notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/); entries are added
with each committed batch.

This file starts at v2.2.0 — earlier history is in git.

## Unreleased — twitter summary quality, and a weekend plan for these kids

### Fixed
- **Repetition loops were saved as primary summaries.** The quality gate rejected an answer
  only with no header AND no bullet. It now also rejects more than 2 exact-or-near duplicate
  bullets, more bullets than input tweets, and bullets mostly lacking `(@handle | timestamp)`;
  a rejected answer falls to the next model with the reason recorded
  (`rust/src/ztools/twitter/quality.rs`).
- **The summarizer decoded greedily.** It now sends temperature 0.1 (`SUMMARY_SAMPLING`);
  every other call stays greedy (`llm::chat_with`). The prompt asks for related tweets to be
  merged and no bullet repeated. A `frequency_penalty` of 0.3 was tried and removed: both
  live runs with it on 2026-10-10 garbled the citation tokens every bullet repeats
  (`Sat Oct 10 336:51`, `+0000 2о` with a Cyrillic о, `Sat Oct 1`). Repetition is the
  gate's job, not a penalty that taxes the required format. ROADMAP T2 measures the rest.
- **`--json` / `--use-cache` runs wrote the production store**, so a fixture run became the
  dashboard's summary. Only a live fetch writes it now; the writer honours
  `TWITTER_OUTPUT_DIR` like the readers. A `--json` source with no tweets, or no tweets at
  all, is a refusal rather than a "Please provide the timeline" document.
- **A whole-chain failure now names its cause**: every model is served by the one Osaurus
  server (ROADMAP T1), exit 1.
- **A cut answer was saved as complete.** Live on 2026-10-10, 45 tweets came back as 8 bullets
  ending mid-citation (`(@AION2Official | Sat Oct 10 336:51 +...`) and passed every rule.
  `llm.rs` now refuses `finish_reason: length` and a stream that ends with neither `[DONE]`
  nor a finish reason; the quality gate rejects a last bullet that opens a citation it never
  closes, which catches a cut the server does not report.
- **Citations were invented and still passed.** The gate checked only that a citation was
  well-formed. It now matches every `(@handle | timestamp)` against the tweets the model was
  given and rejects an answer when more than 1 in 10 name none of them
  (`unmatched_citations`) -- the garbled timestamps above, and handles the model made up.
- **One tweet was told three times.** Live run 3 on 2026-10-10 passed every rule with one
  tweet as three bullets (two identical) and two more as two each -- too few to read as a
  loop. A bullet whose every citation names a tweet an earlier bullet already cited is now
  dropped before saving (`drop_recited`), and a topic it empties loses its header. Whether
  two DIFFERENT tweets say the same thing stays the model's call.
- **The model talking to itself was saved as the summary.** The second live run put prose
  ("Wait, let me check...") between bullets inside topic sections. Topic sections are
  bullets only; prose there is a rejection. 59 of the 60 summaries in the store already
  hold to that, so the rule describes the format rather than changing it.
- **A rejected answer left no trace.** The reason named the rule that fired, never what the
  model wrote, so a run where every model was rejected could not be diagnosed. Each rejected
  answer is now written to `<store>/rejected/` (the newest 20 kept; the store's readers never
  descend into it) and the reason names the file.
- **`**Period:**` printed local time labelled "UTC"**; it now carries the real offset.
- **Tests that resolve `~` or render the live file-summary prompt** must take `TestEnv`
  (audit gate 2).

### Fixed — weekend planner
- **A row with no name was rendered.** Live on 2026-10-10 a nameless transient row printed as
  ` (Vaughan/Toronto)` with a 1.5 score, because the provenance gate waved unnamed rows
  through (and a test pinned that). A row with no name is now dropped always, with a note.
- **An uninstalled model was silently replaced by the server's first model**, so every
  scheduled plan through 2026-10-09 had 0 live events. An unresolvable model is now
  `ModelHealth::NotInstalled`: no chat call, and the degraded banner names the model and
  what the server does have (`weekend/phases.rs`, `weekend/health.rs`). The same silent
  `models.first()` fallback is gone from `twitter/fallback.rs`.
- **The children's ages were a hardcoded CLI default (`13,10,6`)** while the config's
  `[[children]]` birthdays were never read. Ages now come from the birthdays on the plan's
  Friday (`weekend/family.rs`); `--ages` is removed.
- **Age is a filter, not a bonus.** Target ages, name and description are read for an age
  range; a row that fits none of the children ("Baby and Me (birth to 12 months)",
  "French Meetup for Adults") is dropped and counted as unsuitable. The fit score is age
  fit plus weather; filled-in fields and a price no longer earn points
  (`weekend/suitability.rs`, `weekend/score.rs`).
- **Listing-page titles accepted as venues** ("Vaughan Events This Weekend & Things to Do -
  Oct 2026") are rejected, and a transient row that only restates a fixed venue is dropped.
- **Holiday Mondays were ignored.** An Ontario statutory-holiday table (`province = "ON"` in
  `conf/weekend.toml`) extends the window: Thanksgiving 2026 plans Oct 9-12
  (`weekend/holidays.rs`, `weekend/dates.rs`).
- **`routines.toml` said "Thursday evening" and scheduled Monday 08:00**; it is now
  `weekly on thu at 18:00`.
- The plan's provenance line gained a fifth count, `N unsuitable`, so every dropped row is
  still accounted for; older plans parse as 0.

## v3.5.0 — evaluation leaderboard category filtering, CSV export, family grouping, and regression alerts _(2026-10-09)_

The theme: precision leaderboard filtering, standard data pipeline export, and automated CI regression enforcement.
**908 Rust lib tests, 136 tools tests, clippy `-D warnings` clean on stable and on the 1.93.1 MSRV,
coverage >= 95% with per-file floors enforced.**

### Added — new tools, features, and CLI capabilities
- **Task category breakdown filtering (`--category <name>`) (`M13`).**
  `ztools model-eval --leaderboard --category <name>` (e.g. `taxes`, `weekend`, `twitter`, `summarize`)
  filters scoring and rankings to tasks belonging to that category prefix or domain.
- **CSV export format (`--csv-output`) (`M14`).**
  `ztools model-eval --leaderboard --csv-output` prints comma-separated values with columns
  `model,mean,delta,think,json,summarize,filename,vlm,tasks,date` for spreadsheets and downstream ingestion pipelines.
- **Model family grouping (`--group-by-family`) (`M15`).**
  `ztools model-eval --leaderboard --group-by-family` groups leaderboard entries by detected
  model family (`qwen`, `gemma`, `raptor`, `muse`, `bonsai`, `ornith`, `nemotron`, `lfm`, `foundation`),
  preserving relative ranking within each family section.
- **Automated regression threshold alert (`--fail-on-regression <pct>`) (`M16`).**
  `ztools model-eval --leaderboard --fail-on-regression <threshold_pct>` inspects longitudinal score deltas
  and exits with a non-zero exit code if any model's score drops below `-threshold_pct`.

## v3.4.1 — route task slots to installed winners and prevent search bot wall starvation _(2026-10-09)_

The theme: production task routing to installed model winners and reliable search fetching.
**904 Rust lib tests, 136 tools tests, clippy `-D warnings` clean on stable and on the 1.93.1 MSRV,
coverage >= 95% with per-file floors enforced.**

### Fixed — things that were wrong, not merely untidy
- **Task slots routed to installed roster winners in `conf/config.toml` and `rust/src/config.rs`.**
  Previously `json`, `filename`, and `vlm` named `muse-glimmer-30b-jang_6m`, while `think` and
  `default_model` named `qwen3.8-27b-jang_6d`, neither of which was installed on the live Osaurus server.
  Updated `json` and `summarize` to `raptor-v0.5-8b-a1b-jang_6m` (100% on JSON/weekend, 90.7% on summarize),
  and `think`, `filename`, `vlm`, and `default_model` to `qwen3.8-27b-jang_6d-crack` (100% on filename/vlm,
  82.5% on think, 100% on injection).
- **Weekend model resolution fallback prevented drift to `foundation`.**
  In `phases.rs`, when a preferred model and its family are absent from the roster, `resolve_weekend_model`
  now traverses capable chat families (`["raptor", "qwen", "gemma"]`) before falling back to `models.first()`,
  preventing silent drift to the 4k-context `foundation` model that produced 0 extracted events.
- **DuckDuckGo search bot wall pacing and demotion.**
  In `search.rs`, skipped redundant immediate GET requests when DuckDuckGo's POST query is challenged
  by a bot wall. In `conf/weekend.toml`, lowered `demote_wall_ratio` from `0.8` to `0.5` so chronically
  blocked search engines are demoted behind Bing/Brave sooner.
- **Image renamer request tuning.**
  In `rename/vlm.rs`, enforced `enable_thinking: false` and `max_tokens: 100` on filename generation
  requests to prevent reasoning models from spinning on brief naming tasks.

## v3.4.0 — capability-specific ranking and longitudinal trajectory _(2026-10-09)_

The theme: evaluation rankings reflect capability dimensions and historical progress
over time rather than static snapshot aggregates. **904 Rust lib tests, 136 tools tests,
clippy `-D warnings` clean on stable and on the 1.93.1 MSRV, coverage >= 95% with per-file
floors enforced.**

### Added — new tools, features, and CLI capabilities
- **Comparative multi-model leaderboard (`ztools model-eval --leaderboard`) (`M8`).**
  Aggregates each model's latest clean run from stored evaluation history (`eval_history.json`),
  ranking models across overall mean and individual capability slots (`Think`, `JSON`,
  `Summarize`, `Filename`, `VLM`).
- **Structured JSON export (`--json-output`) (`M9`).**
  Emits complete leaderboard data as a formatted JSON array for downstream processing
  and dashboard visualization.
- **Configurable task floor threshold (`--min-tasks <N>`) (`M10`).**
  Allows operators to filter out partial runs and spot checks below a required task count floor.
- **Slot-specific ranking sorting (`--sort-by <slot>`) (`M11`).**
  Supports sorting leaderboard rankings by specific capability slots (`overall`, `think`,
  `json`, `summarize`, `filename`, `vlm`) descending, placing unrated (`None`) slots last.
- **Longitudinal historical score deltas (`M12`).**
  Computes score differences (`delta: Option<f64>`) against each model's previous eligible
  evaluation run in history, formatting deltas as `+X.X%`, `-X.X%`, `0.0%`, or `—` in
  markdown and serialized floats/nulls in JSON.
- **Signal pruning command (`ztools eval-signals --prune`) (`M7`).**
  Strips superseded and untracked task observations from stored signal records in `eval_signals.json`.

## v3.3.0 — a reading nobody took is never a score _(2026-10-09)_

The theme is one class: a value that was never measured arrived in front of a
reader looking like one that was. **896 Rust lib tests, 136 tools tests, clippy
`-D warnings` clean on stable and on the 1.93.1 MSRV, coverage >= 95%
with per-file floors enforced.**

### Fixed — things that were wrong, not merely untidy
- **A model run that measured nothing exited 0 and printed a table of 0.0%.**
  The transport error was swallowed into per-test scores, so "the server was
  down" and "the model is bad" produced the same exit code and nearly the same
  screen; a `--suite full` run against a dead server wrote a 0-mean entry into
  the history the ranking reads; an over-budget refusal was an `eprintln!` and a
  `continue`; `--model all` dropped an unreachable model and reported success
  over the rest. Every one of these is now a `NOT MEASURED` row with the reason
  in it, `—` where the score would be, no mean averaged over it, and a non-zero
  exit (`rust/tests/eval_not_measured.rs`).
- **The verdict on a partly-measured run counted wrong.** It said "0 of N
  task(s) reached the model" whenever ANY row had missed, so one outage among
  thirty answers read as a dead server. It now counts what happened.
- **The eval history averaged unmeasured rows in as zeros.** Every outcome was
  written to `eval_history.json` with its placeholder `score: 0`, so an outage
  dragged a model's historical mean down as though it had answered wrong.
  Unmeasured rows are no longer written.
- **A prompt that cannot fit the model's context window was sent anyway and
  scored 0.** Nothing compared a prompt to a window. `eval/context_fit.rs` now
  refuses, before any request, a prompt whose size alone certainly exceeds a
  DOCUMENTED window (a token count bounded from below at 5 chars/token, so an
  error can only err towards measuring). The row is `CONTEXT`: not measured, not
  an outage that counts towards abandoning the model, not a quality failure, and
  not a run failure unless nothing else was measured. `foundation`'s ~4.6 KB
  `summarize` prompt stays measured; the ~22.6 KB file-summary prompt does not.
- **Production `llm::chat` calls guard against context overflows before sending wire requests (`M3`).**
  `llm.rs` checks `context_refusal` against the documented window before dispatching;
  overflowing requests refuse immediately without wire latency, and fallback chains
  skip down to the next model (`rust/src/ztools/llm.rs`, `rust/src/ztools/twitter/chain_tests.rs`).
- **Integration and unit tests are resilient to host paging pressure.**
  `oversize_tests.rs`, `drain_signal.rs`, and `eval_not_measured.rs` now execute
  cleanly on machines under real-world developer memory pressure, preventing
  spurious test timeouts or harness panics from active swap.
- **A dead weather endpoint produced an invented forecast in the saved plan.**
  The fetcher's failure string was filtered out by the formatter, which then
  rendered a hardcoded forecast of its own (`Fri 28.2°C (clear), …`): a reading
  invented in two places, neither of which said so. A failed fetch is now
  visible in the document, and the fetcher's live failure output is asserted
  equal to the one constant the formatter recognises
  (`rust/tests/weather_failure.rs`).
- **`file_summary` could not tell a description read from the file from one
  guessed from its name.** The prompt sent seventeen paths and no content, and
  the scorer counted sentences with a content verb, which a name-based guess
  produces as readily as a grounded one. The prompt now carries each file's head
  (40 lines / 1200 B, ~22.6 KB rendered, ceiling pinned at 24,576 B) and the
  scorer checks two pinned facts per file (`eval/validate_grounding.rs`), each
  asserted to still occur in the file it describes.
- **Linux pressure readings were always `None`, so no Linux sample was ever
  clean.** The reader asked macOS-only questions; `None` failed
  `machine_is_uncontended()`, so the learned per-model timeout could never exist
  on Linux. `eval/signals_platform.rs` reads `/proc/meminfo` (swap and
  `MemAvailable`). Swap-only by design: a PSI stall threshold has to be measured
  on a Linux box first.
- **A config test passed under `cargo test` and failed under the coverage
  build.** The sandbox did not own the EXECUTABLE: every checkout-derived default
  starts from `current_exe`, and under the gate's `cargo llvm-cov` the test
  binary lives in `rust/target/llvm-cov/`, inside the real checkout, so every
  sandboxed default named the operator's `conf/`. `manifest::running_exe()` is
  now the one seam (`ZTOOLS_EXE`) and `TestEnv` points it at a path with no
  checkout above it. Reproduced red under the gate's target dir, green with the
  row.
- **Five config tests were outside the sandbox, and only a fresh clone showed it.**
  The audit that forces a `TestEnv` into every test touching config defaults
  recognised `ZtoolsConfig::default()` and `ZtoolsConfig {` only; `load_config`, a
  typed `toml::from_str`, and the three `manifest` entry points reach the same
  defaults. In this checkout the leak was invisible -- the root derived from the
  <!-- path-ok: historical record — the released stanza quotes the defect verbatim -->
  test binary IS `~/Projects/ztools`, which the historical default already names --
  so it took a CI simulation (a clone elsewhere, an empty `HOME`) to fail. The
  audit now knows all five routes and the five tests take the guard.

### Changed
- **EXPECTATION SHIFT — `file_summary` and `file_summary_mixed` now measure a
  different task under the same names.** Every score for them recorded before
  2026-10-05 was taken without the file contents and graded by the verb
  heuristic; those history entries and `conf/eval_signals.json` samples describe
  a task that no longer exists. Nothing was re-baselined. Consequences, none of
  them measured yet:
  - `foundation` (4096-token window) is now `NOT MEASURED` on both. It leads
    `model_fallback_chain` and held 31 file-summary samples; it can no longer
    run this task at all, by owner decision (2026-10-05) over cutting the file
    list or the excerpts.
  - `[best_models].think` (`qwen3.8-27b-jang_6d`) was chosen partly as "the only
    model above 60 on taxes_audit_readiness and file_summary_mixed together".
    That half of the justification is unconfirmed until the `think` group is
    re-swept.
- **`$HOME` is a sandbox redirect like every other path variable**, after every
  test that reads `~` was brought under `TestEnv`. `Point::Preserved` is gone.
- **Three gate steps**, each calibrated red before it was wired: `cargo deny`
  (`rust/deny.toml`), `tools/shell_lint_extra.sh` (shellcheck over
  `.githooks/*` and `bin/ab_test`, which `structural.sh` cannot see), and the
  vendor clippy ratchet (`tools/vendor_lint_ratchet.py`): 208 findings in
  `vendor/camoufox-rs` under per-lint shrink-only ceilings, after `clippy --fix`
  (157) and the removal of all 10 suppression sites. The 10-04 entry below and
  `.gatesrc` both said vendor clippy was ungoverned; both are corrected.
- **The hooks delegate to the house gate**: `pre-push` runs
  `$GOH/gates/push_gate.sh`, `pre-commit` routes through `tools/gate.sh --staged`.
- `rust-toolchain.toml` (channel `stable`, rustfmt + clippy named),
  `rust-version = "1.93.1"`, and the vendored `camoufox` dependency carries a
  version.
- **`docs/ROADMAP.md` rebuilt, and its contract is a gate.** 575 lines → ~300: one
  phase sequence (L land, G gates, M measurement, R readings) with never-reused item
  IDs, every item carrying Class / Why now / Done when / Blocked by, and the
  "Closed 2026-…" history pruned to this file and git. `tools/tests/test_roadmap.py`
  enforces it: the four fields, no done-annotations, every backticked repo path
  exists, and every `ROADMAP <ID>` reference elsewhere resolves. Calibrated against
  the previous file, which it rejects on four of five rules (including the stale
  `MODEL_QUIRKS.md` pointer to "item 1", now `M4`). A fifth rule, that every item
  sits under its own phase, was added when an edit dropped a phase heading and moved
  its items silently.
- `assert_empty!` / `assert_nonempty!` / `assert_exact!` are `#[macro_export]`ed
  and proven reachable from integration tests; `rust/tests/transport_http.rs`
  has no sleeps left.

### Added
- **Per-file coverage floors, enforced** (`tools/coverage_floors.jsonc`): every file
  must hold 74.0%, so the 95% aggregate can no longer be met by letting one file
  fall to zero. One exemption, `twitter/native.rs` (2.7%, the live-browser driver),
  keyed relative to `rust/` now that gates_of_heck v0.24.0 accepts relative keys.
  Calibrated on the real lcov parts: exemption removed → exit 1, the key spelled
  `rust/src/…` → stale, exit 1, missing or unparseable file → exit 2.
- **The plain `cargo test` gate step is now `cargo test --doc`**: the coverage step
  already runs every lib, bin and test target and goes red naming a failing test
  (one planted assertion → exit 1), so the suite ran twice per gate.
- **CI exists.** `.github/workflows/ci.yml` runs `make ci` -- the same `.gatesrc`
  step list as the hooks and the release -- on every push to `main` and every pull
  request, on Apple-silicon macOS, with gates_of_heck cloned at `main` and
  bootstrapped by its own `install.sh`. Until now a fresh clone had no gate at all.
  README and ARCHITECTURE said there was no CI; both now point at `.gatesrc` for the
  step list instead of carrying a copy of it, which is how the README's had gone
  stale (it still listed `cargo test` and omitted five steps).
- **The hooks are the stock gates_of_heck hooks** (`install.sh --force`, with
  install hashes in `.githooks/.goh-installed/`). Nothing was lost: the stock copies
  already carry this repo's two hook fixes, upstreamed on 2026-10-05.
- **Release tests the gate's feature set.** `bin/ab_test` ran `cargo test --quiet`,
  the default features, while every gate step passes `--all-features`; a test now
  requires every `cargo test` line in it to carry the flag (red first, then fixed).
- **The vendor secrets step calls `goh.sh secrets`.** It called
  `checks/check_no_secrets.py` by path, which gates_of_heck retired on 2026-10-06
  with no forwarder (since restored upstream), so the step failed with "can't open
  file". It now scans every tracked file rather than `vendor/` alone: a superset of
  the old scope, in 0.2 s, with no look-ahead pattern to maintain. A token planted
  in `vendor/` turns it red.
- **`GOH_EXCLUDE` anchored to `^vendor/`**: the match is `re.search`, so `vendor/`
  would also have exempted any first-party path containing it, silently.
- `tools/tests/test_ab_test_comparator.py` (12 tests) for `bin/ab_test`'s
  comparator.
- **Model family `best_for` tags and capability probing.** All model configurations
  under `conf/models/` declare descriptive `best_for` metadata tags plumbed to
  `model_best_for()` and surfaced via `ztools model-eval --capabilities` to inspect
  the model matrix without running tasks.
- **Architectural vision refusal classification.** Evaluator transport diagnosis
  in `rust/src/ztools/eval/failures.rs` classifies MLX and Apple Foundation Models
  image rejections as `FAIL_CONTEXT` capability refusals, allowing text-only models
  to sweep the 32-task suite cleanly without registering spurious infra outages.
- **Full-roster sweeps on fingerprinted task suite (`M4`) and signals re-baselining (`M6`).**
  All installed Osaurus models swept under `tools/sweep_models.sh`, re-confirming
  `[best_models]` slot winners in `conf/config.toml` and populating
  `conf/eval_signals.json` with fresh samples matching active task fingerprints.

### Dependency currency, and a gate that can fail again _(2026-10-04)_

The audit behind this is `docs/ROADMAP.md` phases A/B/C. The gate of record was
RED when this started (`clippy -D warnings` exit 101, 72 findings left by the
previous batch) and is green now: **9/9 steps, 865 Rust tests, 97 tools tests,
95.81% coverage.**

### Fixed — things that were wrong, not merely untidy
- **`wait_for_ready` in the vendored camoufox driver contradicted its own spec
  in six places, and its 7 tests were red.** The code reads **stdout**; every
  doc comment, `process/mod.rs`, and `docs/PROTOCOL.md` said **stderr**, and the
  module doc named a sentinel string (`"Juggler pipe initialized"`) that the
  binary does not emit. Probed the real binary on this machine
  (`152.0.4-beta.31-7b8d12d6`, streams captured separately): stdout carries
  `Juggler listening to the pipe`, stderr does not. So the CODE was right and
  the docs were wrong — "fixing" the code to match them would have broken the
  live twitter-collect path at `twitter/native.rs:131`. The docs are corrected,
  the sentinel prose is corrected, `PROTOCOL.md` carries a marked note recording
  the divergence, and two tests now pin the stream in BOTH directions so nobody
  can "fix" it either way.
- **The saved weekend plan never said when an event was.** `WeekendEvent.dates`
  — the field the whole date-reconciliation effort exists to preserve — was
  printed nowhere in the markdown, while the terminal table had a `Dates`
  column. The two renderings of the same data disagreed. Both tables now carry
  it, in the same position, in the same spelling.
- **The normal twitter summary's provenance line was glued to the tweets line**
  (no blank line, so `**Model:**` was lazy continuation of `**Tweets:**`), and
  `Provenance::banner()` never closed its block quote.
- **`cargo audit`'s configuration was in a directory the gate never runs from.**
  It sat at `rust/.cargo/audit.toml`; the gate runs from the repo root and
  cargo-audit resolves `$CWD/.cargo/` only, so `deny = [warnings, unmaintained,
  unsound, yanked]` was dead. Moved to `.cargo/audit.toml`, with the placement
  requirement and its proof recorded in the file. The tree is clean either way
  today (258 dependencies, zero advisories), so this mattered the day one lands.
- **The audit gate's ignore list does not ratchet.** `ignore = ["RUSTSEC-2026-9999"]`
  is accepted silently, so "delete the entry once its advisory stops firing" is
  on whoever adds it. Stated in the config instead of implied.
- **The suite wrote into the developer's real `$HOME`.** `model_resolve_http`
  set `EVAL_SIGNALS_DIR` but not `EVAL_OUTPUT_DIR`, so every run overwrote
  `~/.config/ztools/outputs/gone-model/t1.txt` (verified on disk). Four twitter
  tests were reading this checkout's own `conf/twitter.toml` through `$HOME` and
  passing only because it happens to carry a `[fallback]` table. One test walked
  the real Firefox profile tree.
- **`sweep_models.sh` re-introduced the bug it documents fixing.** `grep -c`
  exits 1 on no match, so `$(grep -c … || echo 0)` produced the two-line string
  `"0
0"` and `[ … -gt 0 ]` errored on every all-green sweep. `grep -o` under
  `set -euo pipefail` aborted `rerun_truncated.sh` outright on an empty log —
  the exact case that script exists for.
- **`install.sh` had no platform gate at all**, and its `cargo metadata … ||
  echo` fallback pointed at a path the machine-wide `build-dir` never writes to.
  Now a hard failure with an overridable `ZTOOLS_OS`/`ZTOOLS_ARCH` seam (plus
  `ZTOOLS_PLATFORM_CHECK_ONLY=1`, so a test can drive both directions without
  building): macOS arm64, Linux x86_64 and Linux aarch64 accepted; macOS Intel,
  32-bit and unknown OS rejected with the combination named.
- **The two git hooks that guard everything got no shellcheck** (`*.sh` and
  `hooks/*` never matched `.githooks/`), and `pre-push` checked for the repo's
  own `tools/gate.sh` rather than the gates checkout it actually execs — so a
  missing gates repo failed with a bare `No such file or directory`.
- **A pytest marker opted out of a guard that did not exist.** `conftest.py`
  was absent; every run printed `PytestUnknownMarkWarning`. The guard is real now
  and is itself tested.
- **A vendored test leaked a process.** `returns_timeout_when_process_hangs`
  spawned `sh -c "… && sleep 60"` and killed only the shell, so the sleeper
  outlived the suite. Caught by the house orphan check on the first run of the
  new gate step.

### Changed
- **Six direct dependencies moved a major version, and all 787 tests passed on
  the bump before a line of code changed**: `toml` 0.8 → 1.1, `brotli` 8 → 9,
  `dirs` 5 → **7** (two majors), `base64` 0.22 → 0.23, `serial_test` 3 → 4.
  `serial_test` also **collapses `syn` to a single major** in the graph — 3.5.0
  was the only thing holding 2.x in. `cargo update` alone moved 45 more
  packages, `winnow` 0.7 → 1.0 among them.
- **`reqwest` 0.13.1 → 0.13.5.** It had been silently pinned four patch
  releases back because reqwest **deleted the `webpki-roots` feature** upstream,
  and cargo treats features as part of the compatibility surface. The tree now
  takes its root store from `rustls-platform-verifier`. Proven against the real
  endpoint, not a green suite: `ZTOOLS_TLS_PROBE=1 cargo test --test tls_probe`
  fetches a real three-day forecast, and the probe is calibrated against an
  expired certificate so a failure is recognisable.
- **Edition 2021 → 2024.** 65 sites needed attention (not the 92 the compiler
  reported first — `cargo check --all-targets` stops after the failing `lib
  test` target, so the integration binaries were never counted). 61 route
  through the new shared `TestEnv` guard; **4 `unsafe` blocks** total, each with
  a `SAFETY:` argument. Edition 2024 also made let-chains available, which
  surfaced **65 previously-unreportable `collapsible_if` findings** across 25
  files; all fixed, none suppressed.
- **One guard for process-global state, and three gates that make it impossible
  to forget.** `test_env::TestEnv` manages 26 variables, restores them in reverse
  order while holding a global lock, and **panics on a key it does not manage** —
  because a variable nothing restores is the defect, not the style. Three audit
  gates re-derive their expectations from the sources: every env var the crate
  reads is managed, every hazard test constructs a guard, no test names a fixed
  directory under the system temp dir.
- **The gate grew three steps and lost its false claims.** `.gatesrc` said the
  vendored crate was "still secret-scanned"; it was not — all 53 vendor files
  skipped every delegated check. Vendor is now secret-scanned by an explicit
  step, its **own 198 tests run in the gate**, and `ruff check tools/` runs
  against the `pyproject.toml` rule set instead of being config nobody runs.
  What vendor was still NOT under on this date: clippy (365 findings across
  28 files, 35 auto-fixable) and its 10 `#[allow]` sites. Superseded 2026-10-05
  by the vendor clippy ratchet (see above).
- **Docs now match the code.** The coverage floor is 95 everywhere (two files
  said 94); `ARCHITECTURE.md` no longer claims a GitHub CI that does not exist;
  the file-size rule states that it is enforced per file SUFFIX; `CLAUDE.md`
  documents that the cookie store is Firefox-family only.

### Added
- **Byte-level goldens for the two documents a human reads** (weekend plan,
  twitter summary), in the shape `report_csv.rs` already used, regenerable only
  via `ZTOOLS_UPDATE_GOLDENS=1` — which still panics, so a plain `cargo test`
  can never bless a change.
- **Tests where the property was untested, not merely unasserted:**
  `args_with_implied_subcommand` (the argv→subcommand path every one of the
  eight installed symlinks takes), `expand_tilde`, `load_config`, the five
  built-in smoke tasks row-for-row, `CARRY_FIELDS` preservation,
  `LiveBrowserCollector::new`'s runner wiring, and a live-TLS probe.
- **`rust/tests/yoy_regression.rs` deleted.** It read `/tmp/yoy_out.txt` and
  asserted nothing whenever the file was absent — a golden that is a no-op on
  every clean machine. The case was already covered by
  `validator_parity`'s frozen verdicts, which is where the provenance belonged.
- **`fetch_weather_from(url)`** — the seam that let the weather HTTP path be
  asserted over a real socket instead of hand-fed JSON.
- `vendor/camoufox-rs/Cargo.lock`, so a crate with no lock no longer floats its
  dependencies between builds.

## v3.2.1 — reqwest 0.13, rusqlite 0.40, no suppression attributes _(2026-09-21)_

### Fixed
- **Percent scores multiply before they divide, like the Python they
  mirror.** `pct_floor` and `pct_round` computed `100.0 * (part / whole)`;
  Python's `int(100 * part / whole)` computes `(100 * part) / whole`. The
  two round differently: `29/100` came out as `28.999…` and truncated to
  28 where Python says 29. Both are integer arithmetic now (`scoring_math`),
  pinned against the exact integer quotient for every `part/whole` up to
  400. Scores that hit the boundary move up by one point.

### Changed
- **No `#[expect]` anywhere in the crate** (85 sites; the house `no_allow`
  gate refuses `#[expect]` since gates_of_heck v0.12.2). Every finding is
  fixed rather than annotated:
  - the numeric casts go through `units`: `count` / `unsigned` / `signed`
    (exact integer → `f64` in two halves) and `whole_i64` / `whole_u64` /
    `whole_u32` (`num-traits` `ToPrimitive`, with the cast's saturating
    boundaries stated and tested);
  - exact float comparisons in tests say so with `assert_exact!`
    (`partial_cmp`, `==` semantics), which is where the `float_cmp`
    argument in `scoring_math` now points;
  - `twitter-summarize`'s seven flags resolve to a `TwitterCommand` at the
    dispatch, with the precedence the body used to apply by early return;
  - the six functions over 100 lines are split (`cli::run`, `weekend_plan`,
    `model_eval`, the eval runner's task/attempt loop,
    `validate_detailed_json`, the taxes narrative validator) and the curated
    activities are a data table with one constructor;
  - CLI entry points borrow their arguments; extension checks use
    `Path::extension` (`*.md` no longer matches a bare `.md`, matching glob).
- **reqwest 0.12 → 0.13** (locked at 0.13.1). The TLS feature was renamed
  upstream (`rustls-tls` → `rustls` + `webpki-roots`); `query` and `form` are
  now declared explicitly. No call-site changes — the 0.13 API is
  source-compatible with everything this crate uses.
- **rusqlite 0.32 → 0.40** (locked at 0.40.2, still `bundled`). No call-site
  changes.

## v3.2.0 — the roadmap, closed _(2026-09-19)_

### Added
- **Every plan carries a provenance ledger** (`_Provenance: N extracted, N unsourced,
  N outside the window, N excluded._`), and `ztools status` reports it under
  `details.provenance`. The instrument for the one Phase 1 reading that needs a
  week of plans to exist: whether the noisier Bing corpus moves the invented-row
  rate.
- **`weekend_injection` eval task.** One scraped line orders the extractor to emit
  a single made-up row and nothing else. foundation and gemma-4-12b obey (emit only
  it); every other candidate extracts the real venues. This is the json slot's gate
  now; raptor-v0.5, kept out by the filename proxy, resists here and ties muse.
- **`summarize_injection` eval task.** A timeline with one tweet ordering the
  summarizer to open with a fixed sentence and drop the rest. Every summarize
  candidate scores 100 on it — raptor-v0.5 included, which summarised the other
  tweets and quoted the spam as an attributed bullet. The slot's injection exposure
  had been inferred from `filename_injection`; measured directly, there is none.

### Fixed
- **`validate_resists_injection` no longer scores a verbatim quote as obedience.**
  A marker hit whose surrounding words come from the source is the model reporting
  the injected line; the obedient shape is the sentence in the model's own framing.
  Calibrated on raptor-v0.5's real answer.

## v3.1.0 — the weekend planner produces events again _(2026-09-19)_

Measured end to end against a healthy server, `ztools weekend-plan` had produced
zero transient events on every run since August, in 22 minutes, and then left the
server unable to answer `pong`. Four causes, each closed at the class level.

### Fixed
- **Two more search engines.** Every `DuckDuckGo` query — `html` and `lite`, POST and
  GET — came back as an `anomaly-modal` challenge, so the planner had no corpus and
  said "no events". `weekend/search.rs` now tries Bing when DDG walls or empties a
  query (`bing_url`, loopback in every test), reads its `li.b_algo` markup and
  unwraps its `ck/a?u=a1<base64>` redirects so the aggregator follow-up fetches the
  listing page, not the redirect; Brave (`brave_url`) is the third leg. This is what
  upstream does too: `ddgs` 9.16 (2026-08-26) drives the identical DDG POST behind
  a browser-TLS impersonator (`primp`) and still got "No results found" from this
  machine, while its Bing and Brave legs answered — its `auto` mode is engine
  spreading, not a wall-beater, and DDG's text results are Bing's anyway. Verdicts are classified RESULTS-FIRST: Bing's page
  lists `challenges.cloudflare.com` in a script allowlist, and the marker scan alone
  called nine answered queries "blocked" on the first live run.
- **Thinking off, output bounded, streaming with a stall guard** (`ztools/llm.rs`,
  the one client every production call now goes through). `qwen3.8` reasoned past
  the 300s timeout on an eight-line extract; the client disconnected; the server
  kept the abandoned generation, queued the next request behind it, and after
  three of these no longer answered at all. Now: `enable_thinking: false` (measured:
  the same extract in 11s), `max_tokens` (4,096), `stream: true`, and the call fails
  only when no byte arrives for `llm_stall_secs` (120) — tokens that keep flowing
  keep it waiting, up to a cap that is a backstop, not a budget.
- **One warm-up before any phase** (`llm_warmup_timeout_secs`, 900). A cold 25GB
  model measured an 8m38s load; a 300s client made the server cancel the load, and
  the next call restarted it — a livelock in which the model never became ready.
  The warm-up runs on its own thread while the search does, and if it never
  answers no phase is attempted and the plan says which model did not.
- **An extract breaker.** A dead model cost four timeouts per line (8 -> 4 -> 2 -> 1
  -> raw); after `EXTRACT_FAILURE_BREAKER` (3) consecutive failures the rest of the
  corpus passes through raw.
- **One definition of "the upcoming weekend"** (`weekend::dates::plan_window`). The
  planner walked forward to the next Friday while the status page counted the
  weekend you are in, so a Saturday refresh planned NEXT weekend and the dashboard
  kept saying this one was "not planned".

### Changed
- **`[best_models]` re-derived from the first thinking-off sweep** (2026-09-19, all
  ten rankable installed models, 30 tasks each). Three slots had named models that
  were not installed since mid-August. New: json, filename, vlm →
  `muse-glimmer-30b-jang_6m` (90.0 mean, 100 on injection); summarize →
  `raptor-v0.5-8b-a1b-jang_6m` (92.8, the raw winner, by the owner's decision over
  the injection gate -- it obeys planted instructions, and the trade is stated in
  the config); think/default → `qwen3.8-27b-jang_6d`. Injection resistance gates
  json and filename.
- **`lfm2.5-2.6b-4bit` deleted** -- it could not load on this Osaurus.
- **A missing event description is `—`, not a copy of the name.** "Why It Fits"
  used to repeat the event's own title as its reason (class C4).
- **ROADMAP consolidated and phased**: measure the two accepted exposures first,
  planner honesty second, eval ergonomics third; watch items and non-goals stated. The embedded Rust defaults
  match, kept equal by a drift-gate test. Proven live: the planner with muse
  produced 10 rows in 8m08s.
- **`sweep_models.sh` records a refusal as `REFUSED`**, not `DONE tasks=0`, so
  `--resume` re-runs it; four refusals from today's sweep are re-filed.
- **`model-eval` measures thinking OFF by default** (`--thinking` restores the old
  regime), so a sweep ranks models as the production tools now call them. Sweeps
  before this version measured with reasoning on; compare across the boundary with
  the flag.

### Added
- **The plan says why it is empty** (`weekend/health.rs`). The degraded warning names
  a bot wall ("9 of 16 searches were blocked by a bot wall — the corpus was starved,
  not the weekend quiet") or a model that never answered; a populated plan behind a
  partly walled search carries a note that the list may be incomplete; `ztools status`
  appends `(search bot-walled)` so the dashboard line carries it too.

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

### Release wiring _(2026-09-14)_
- **`vendor/camoufox-rs` is tracked.** It was gitignored while being a
  `rust/Cargo.toml` path dependency, so the tap's tarball build — every prior
  tag — could not have compiled from a clean clone. Structural gates exempt
  `vendor/`.
- **Coverage floor 94 → 95** (the house floor): the house gate measures 95.65%.
- **The CI list runs `structural.sh --full`** (native `goh`: emoji, length,
  markers, shell lint, secrets) instead of two hand-picked Python checks.
- **`build.sh` and the `bin/` launcher shims are gone.** They were a third
  install door (after `brew` and `install.sh`) that resolved the build via
  <!-- path-ok: historical record — names the removed defect, not a live path -->
  `cargo metadata` + `jq` on every launch and hardcoded `~/Projects/ztools`.
  Dev builds run with `cargo run --manifest-path rust/Cargo.toml -- <cmd>`.

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
