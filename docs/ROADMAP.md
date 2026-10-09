# Roadmap — `ztools`

Forward-looking only. What shipped is in `CHANGELOG.md`; what was planned and is done
is in git history (house rule #13). This file is the open work, in the order it has to
happen, and nothing else.

## The contract this file keeps

Enforced by `tools/tests/test_roadmap.py` (gate step `python3 -m pytest tools/tests`),
so it holds whether or not anybody remembers it:

1. **Every item has an ID that is never reused** (`L1`, `G3`, …) and four fields:
   **Class** (the failure mode it is an instance of, because the class is what
   recurs), **Done when** (a command, test or gate that goes from red to green, never
   a judgement), **Blocked by** (an item ID, an owner decision, a cross-repo ask, or
   `nothing`), and **Why now** (what it hides while it stays open).
2. **A done item is deleted, not annotated.** "Closed 2026-…" paragraphs are how the
   previous version of this file grew to 575 lines of history around 20 lines of
   open work. The gate rejects them.
3. **Every repo path in backticks exists**, unless the item marks it `(new)`.
   Sixteen prompt rows once named a deleted tree for a month; a roadmap that names
   files is the same instrument and gets the same check.
4. **Every inbound reference resolves.** Docs that point here say `docs/ROADMAP.md <ID>`,
   and the gate fails on an ID this file does not define.

**How work is ranked.** The tools exist to run unattended and be trusted, so rank by
what a failure HIDES: a plan that says "quiet weekend" over a blocked search, a
summary steered by a tweet, a slot winner measured on a task that has since changed.
The same rule orders the machinery, and it puts the machinery first: a gate that
cannot fail hides more than a plan that reads oddly, because it is the thing that
would have noticed. Phases are SEQUENTIAL where a later phase's evidence depends on an
earlier one, and each phase says what it depends on.

## Where things stand — 2026-10-09

A dated summary of what the runs below already print. Nothing in this section is
checked by a gate, so re-derive it before editing it:
`git log --oneline origin/main..main`, `gh run list --limit 3`,
`tools/gate.sh --full`, and `"$GOH_DIR/gates/goh.sh" home-paths --exclude ^vendor/`.

- Release v3.3.0 cut, tagged, and published on GitHub.
- Remote CI run 37998212140 passed in 2m47s on macOS arm64 (`make ci` clean across all 12 steps).
- Items `L2`, `M4`, `M6`, `M7`, `M8`, `M9`, and `M10` are closed and deleted.
- Landed today, each proven by a test made to fail first:
  **`L2`** (green CI gate of record on GitHub Actions),
  **`M1`** (repo-relative file_summary checkout rows), **`M2`** (task identity
  fingerprinting and set-aside history aggregation), **`M3`** (production context-fit
  guard in `rust/src/ztools/llm.rs`, fallback skip down `model_fallback_chain`, and
  live Osaurus 8k window overflow probe documented in `docs/MODEL_QUIRKS.md`),
  **`M4`** (full-roster sweep across all installed models on 32-task fingerprinted roster,
  architectural vision refusal diagnosis, `[best_models]` confirmed and cited in `conf/config.toml`,
  and embedded slot defaults drift test passing),
  **`M5`** (smoke prompts fit all documented windows),
  **`M6`** (eval signals re-baselined with fresh task fingerprints in `conf/eval_signals.json`),
  **`M7`** (`ztools eval-signals --prune` strips unrecorded and superseded task observations),
  **`M8`** (`ztools model-eval --leaderboard` formats latest clean runs into comparative rankings),
  **`M9`** (`ztools model-eval --leaderboard --json-output` exports structured ranking JSON array),
  **`M10`** (`ztools model-eval --leaderboard --min-tasks <N>` filters models by run task count floor),
  **`G2`** (307 files clean under `GOH_NO_HOME_PATHS=1`), **`G3`** (toolchain header cleaned),
  **`G4`** (CI gate tool manifest dynamic install + pinned tsv), **`R1`** (provenance ledger reading).
- Local verification on `main`:
  `tools/gate.sh --full` passes all 12 CI steps (coverage floor 95%, clippy, cargo audit,
  cargo deny, structural, pytest, ruff, camoufox tests, and secrets scan),
  `python3 -m pytest tools/tests -q` green (136 passed),
  `cargo test --manifest-path rust/Cargo.toml --all-features --lib` passes 905 lib tests.
- All temporary agent sessions (OpenCode and Claude worktrees/caches) are purged.

## Phase M — what the eval records must mean what it says

Depends on nothing. The class for the whole phase: **a stored number that no longer
describes the thing its name says it describes.** `M1` through `M10` landed on
2026-10-08/09 (task fingerprinting, context guards, full-roster sweeps, signal
re-baselining, signal pruning, comparative leaderboards, JSON serialization, and
configurable task thresholds); what remains is capability-specific sorting and historical
delta tracking.

### M11 — Slot-specific ranking sort for comparative model-eval leaderboard
- **Class:** fixed one-dimensional ranking over multi-dimensional capability profiles.
- **Why now:** `ztools model-eval --leaderboard` ranks models exclusively by overall mean;
  operators routing specialized workloads (e.g. reasoning, JSON extraction, vision) need
  to rank models by specific capability slots.
- **Done when:** `ztools model-eval --leaderboard --sort-by <slot>` (supporting `overall`,
  `think`, `json`, `summarize`, `filename`, `vlm`) sorts the leaderboard by the chosen slot
  score descending, covered by a test in `rust/src/ztools/model_eval_tests.rs`.
- **Blocked by:** nothing.

### M12 — Historical score delta comparison in comparative leaderboard
- **Class:** static snapshot ranking lacking longitudinal performance trajectory.
- **Why now:** `ztools model-eval --leaderboard` shows the latest score for each model but
  does not display whether a model improved or regressed relative to its prior run in
  stored evaluation history.
- **Done when:** `ztools model-eval --leaderboard` computes and displays score deltas
  against each model's previous run, covered by a test in `rust/src/ztools/model_eval_tests.rs`.
- **Blocked by:** nothing.

## Phase G — gates that still cannot fail

Depends on nothing. The class for the whole phase: **a check that reports green over
something it never inspected.** Both items left here are about the CHECKER rather
than the code: which checker ran (`G6`) and who owns the lint path (`G5`).

### G5 — Retire `tools/shell_lint_extra.sh` once the house shellcheck reaches it
- **Class:** a local fork of shared machinery. The hooks themselves converged on the
  stock ones on 2026-10-06 (`install.sh --force`, recorded in
  `.githooks/.goh-installed/`); this is what is left.
- **Why now:** the repo-local step exists only because the house shellcheck glob misses
  `.githooks/*` and `bin/*` (X5). Two lint paths for one class of file drift.
- **Done when:** X5 has shipped, the house structural gate goes red on a syntax error
  planted in `.githooks/pre-push` and in `bin/ab_test`, and the local step and script
  are deleted.
- **Blocked by:** `X5`.

### G6 — CI runs a NAMED release of `gates_of_heck`, not its `main`
- **Class:** a gate whose checker can change under it, so a verdict move cannot be
  attributed to the repo and `git bisect` of a gate change is impossible.
- **Why now:** `.gatesrc`'s hooks delegate to whatever `gates_of_heck` currently is,
  and CI clones its `main` at depth 1, so a push is checked by a different checker
  than the last push was. The standing house rule is that `main` is never a pin; a
  local checkout already sits 52 commits past v0.24.0 (`G2`'s counts, `L2`'s history
  and `X1`-`X3` all shipped inside that window unpinned).
- **Done when:** the workflow's clone in `.github/workflows/ci.yml` names a released
  tag, CI passes on it, and the local `GOH_DIR` reports the same tag.
- **Blocked by:** the owner's pick of the first pinned tag.

## Phase R — readings, not builds

Depends on nothing in this file; each is a measurement someone has to take. `R1`
was taken and recorded in `docs/MODEL_QUIRKS.md`; `R2` is the one that needs a
machine this desk does not have.

### R2 — A Linux pressure threshold from a Linux box
- **Class:** a reader that is right about what it reads and silent about what it
  cannot.
- **Why now:** `rust/src/ztools/eval/signals_platform.rs` reads swap and
  `MemAvailable` on Linux and deliberately not PSI, because a stall threshold that was
  not measured would be a guess.
- **Done when:** PSI `some`/`full` is measured under a quiet and a loaded sweep on a
  Linux host, and a threshold lands with that measurement cited.
- **Blocked by:** access to a Linux host.
