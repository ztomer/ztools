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

## Where things stand — 2026-10-06

- The 2026-10-04/05/06 work is ONE commit on `main`, ahead of `origin/main` (`40148ae`)
  and not pushed (`L2`). The two unreleased stanzas in `CHANGELOG.md` describe it.
- Verified on that tree: `tools/gate.sh --full` 12/12 green (every step's log a real
  run, not a cache answer), 967 Rust tests, 123 tools tests, coverage 96.85% with
  per-file floors, clippy `-D warnings` on stable 1.99 and the 1.93.1 MSRV.
- CI (`.github/workflows/ci.yml`) exists and has never run (`L2`).
- `gates_of_heck` owns every hook and gate step this repo delegates to; its v0.24.0
  shipped X1-X3, and the hooks here are its stock ones. X4 and X5 are open.

## Phase L — land what is built

Depends on nothing. Everything after it depends on it: a measurement or gate built on an
uncommitted tree describes a state nobody can check out.

### L2 — Push, and CI's first run is green
- **Class:** a gate that has not been run is indistinguishable from one that passes.
- **Why now:** `.githooks/pre-push` delegates to `$GOH/gates/push_gate.sh`, so a push
  runs whatever `gates_of_heck` currently is, and that repo is mid-change.
- **Done when:** `git push` succeeds through the pre-push hook without `--no-verify`, and
  the first `.github/workflows/ci.yml` run on that commit is green. CI is new and has
  never run off this machine: what the first run finds is this item's work.
- **Blocked by:** the owner's yes to push.

## Phase G — gates that still cannot fail

Depends on `L`. The class for the whole phase: **a check that reports green over
something it never inspected.**

### G2 — `GOH_NO_HOME_PATHS=1` is on
- **Class:** a binary or test that works only on this machine at this path.
- **Why now:** 84 findings measured 2026-10-06 (86 on 2026-10-05) — 51 in
  `rust/src/ztools/eval/prompts/file_summary.rs` (`M1`), the other 35 spread over 22
  files (`routines.toml` 4, `docs/EVALUATION_WORKFLOW.md` 3, the twitter test files 7,
  `rust-toolchain.toml` 1, …). Enabling the key red is not an option.
- **Done when:** `"$GOH_DIR/gates/goh.sh" home-paths --exclude ^vendor/`
  exits 0, the key is in `.gatesrc`, and adding one literal home path goes red.
- **Blocked by:** `M1` (51 of the 86).

### G3 — `rust-toolchain.toml` says what it does
- **Class:** stated vs implemented.
- **Why now:** its header says the file stops a background `rustup update` from
  changing gate verdicts. `channel = "stable"` does not: the next stable still arrives
  unannounced. Point-pinning would contradict the house rule (always the newest
  stable), so the claim is what is wrong, not the channel.
- **Done when:** the header claims only what the file guarantees (the channel and the
  named components), and the gate prints `rustc -V` so a verdict change can be
  attributed to a toolchain move after the fact.
- **Blocked by:** nothing.

### G5 — Retire `tools/shell_lint_extra.sh` once the house shellcheck reaches it
- **Class:** a local fork of shared machinery. The hooks themselves converged on the
  stock ones on 2026-10-06 (`install.sh --force`, recorded in
  `.githooks/.goh-installed/`); this is what is left.
- **Why now:** the repo-local step exists only because the house shellcheck glob misses
  `.githooks/*` and `bin/*` (X5). Two lint paths for one class of file drift.
- **Done when:** X5 has shipped, the house structural gate goes red on a syntax error
  planted in `.githooks/pre-push` and in `bin/ab_test`, and the local step and script
  are deleted.
- **Blocked by:** X5.
## Phase M — what the eval records must mean what it says

Depends on `L`. Ordered: `M1` and `M2` change what a task IS, so they land before
`M4`, or the sweep measures a task that is about to change and has to be re-run.
The class for the whole phase: **a stored number that no longer describes the thing
its name says it describes.**

### M1 — file_summary reads the checkout it runs from
- **Class:** an eval input bound to one machine.
- **Why now:** the 17 rows in `rust/src/ztools/eval/prompts/file_summary.rs` are
  absolute paths under one home, and
  `rust/src/ztools/eval/prompts/file_summary_content.rs` reads the excerpts from them —
  so an eval run from any other checkout, worktree included, grades the model against
  the HOME checkout's files.
- **Done when:** the rows are repo-relative, the excerpts are read through the
  `manifest` checkout seam, and a test renders the prompt from a fixture checkout.
  This is a reference-behaviour change: `validate_mixed_file_summary` finds signal
  rows by "line starts with `/`", and the prompt text the model sees changes, so
  `rust/src/ztools/eval/validators/mixed_text_tests.rs` is re-pinned deliberately, with
  the diff saying so.
- **Blocked by:** nothing.

### M2 — A task's identity includes what it asks
- **Class:** a replaced task under the same name. The file_summary change on
  2026-10-05 is the instance; the model-side sibling ("a replaced build under the same
  name") is in `CLAUDE.md` and is cleared by hand today.
- **Why now:** history entries and signal samples are keyed by task NAME. Every
  file_summary score before 2026-10-05 was taken without the file contents and graded
  by a different scorer, and the trend tables and medians will average them with the
  new ones.
- **Done when:** history entries and samples carry a fingerprint of the task's
  messages and checks, aggregates read only the current fingerprint, the trend table
  says how many were set aside, and a test that changes a task's prompt sees its old
  history excluded.
- **Blocked by:** nothing.

### M3 — Production never sends a prompt the model cannot hold
- **Class:** the production sibling of `context_fit`. The eval now refuses a prompt
  that certainly overflows a documented window; `rust/src/ztools/llm.rs` does not, and
  `foundation` (4096 tokens) leads `model_fallback_chain`.
- **Why now:** unknown whether any production prompt reaches 20 KB, and unknown what
  osaurus does with one that does — a refusal is a fallback, a silent truncation is a
  wrong answer.
- **Done when:** a probe records both answers (`probe-first`). If either is reachable,
  the chain skips the model with a stated reason, and a test pins that.
- **Blocked by:** nothing.

### M4 — Re-sweep, and re-derive `[best_models]`
- **Class:** a slot winner measured on a task that has since changed.
- **Why now:** `[best_models].think` (`qwen3.8-27b-jang_6d`) was chosen partly as "the
  only model above 60 on taxes_audit_readiness and file_summary_mixed together"; that
  half is unconfirmed. `foundation` is now NOT MEASURED on both file-summary tasks.
  `docs/MODEL_QUIRKS.md` points here for re-deriving its tables.
- **Done when:** a full-roster sweep under the final tasks is in the history with
  current fingerprints, `conf/config.toml`'s slot table and comments cite it, and the
  embedded-defaults drift test passes against the new slots.
- **Blocked by:** `M1`, `M2`. Run serially on an idle GPU through
  `tools/osaurus_one.sh` (see `CLAUDE.md`).

### M5 — The smoke roster provably fits the smallest window
- **Class:** an assumption standing in for a check. The smoke path
  (`rust/src/ztools/model_eval.rs`) does not consult `context_fit`, on the grounds
  that its prompts are small.
- **Why now:** cheap, and it converts "small" from a belief into a gate before
  someone adds a large smoke fixture.
- **Done when:** a test asserts every smoke prompt passes `context_refusal` for every
  documented window in `conf/models/`.
- **Blocked by:** nothing.

## Phase R — readings, not builds

Depends on nothing in this file; each is a measurement someone has to take.

### R1 — Read a week of the provenance ledger
- **Class:** an instrument nobody reads.
- **Why now:** every plan ends with `_Provenance: N extracted, N unsourced, N outside
  the window, N excluded._` since 2026-09-19; first reading was 5 extracted, 0 dropped.
- **Done when:** a week of lines is compared and the result recorded in
  `docs/MODEL_QUIRKS.md`; if `unsourced` climbs, the region list or a per-engine result
  cap is the lever.
  `for f in ~/Documents/weekend_plans/weekend_plan_*.md; do grep -H _Provenance "$f"; done`
- **Blocked by:** a week of plans.

### R2 — A Linux pressure threshold from a Linux box
- **Class:** a reader that is right about what it reads and silent about what it
  cannot.
- **Why now:** `rust/src/ztools/eval/signals_platform.rs` reads swap and
  `MemAvailable` on Linux and deliberately not PSI, because a stall threshold that was
  not measured would be a guess.
- **Done when:** PSI `some`/`full` is measured under a quiet and a loaded sweep on a
  Linux host, and a threshold lands with that measurement cited.
- **Blocked by:** access to a Linux host.

## Cross-repo asks (`~/Projects/gates_of_heck`)

A dedicated session owns that repo; nothing here edits it. X2 (unreadable floors file
refused, exit 2) and X3 (project-relative exempt keys; per-target floors refused) shipped
in `22c8d49`, and X1 (`--floors-json` absolute before any `cd`) and a consumer-pytest
regression in `eaed4e0`, all on its `main` and wired or verified here. Open:

| ask | what | unblocks |
|---|---|---|
| X4 | span forgiveness: a line counts as covered when its FUNCTION ran, and a span ends at the next function, so a large `match` whose first arm ran reads as covered | what the floor means |
| X5 | the shellcheck glob misses `.githooks/*` and `bin/*` | `G5` |

## Owner decisions

The push (`L2`) · the region-filter divergence (open question below).

## Watch items (no action until they move)

- **26GB in an interactive slot.** filename and vlm are `muse-glimmer-30b`; `rn` is
  interactive, and a cold load measured up to 8m38s on a contended box. Fine while
  `rn` runs after something has warmed muse; if the first rename of the day is the
  complaint, `rn` warms on launch or the slot takes the 6.3GB raptor-v0.5 (100
  quality, 0 injection — the same trade as summarize).
- **Osaurus wedges on abandoned requests.** The production client never abandons one,
  and `tools/osaurus_one.sh --restart` clears it. A wedge with no abandoned call in the log
  is a server bug to report, not ours.

## Open questions

- **Region filter divergence** (pinned, consensus fixtures only): Rust's foreign-city
  blocklist beats a local token; Python's whitelist-only did not. Remaining surface:
  London-Ontario / Chicago-with-Toronto / ON-postal-only. Both lists live in
  `conf/weekend.toml [region]`. Someone picks.
- **Collect parity is shared-record agreement, not ID-set equality**, because the
  Following endpoint is served as a per-load sample. If x.com serves it
  reverse-chronologically again, the stronger criterion can come back; `bin/ab_test`
  keeps the comparator.

## Non-goals

- No per-(model, phase) timeout learning for the planner. Every production call is
  streamed and fails only on a stall, and output is bounded by `max_tokens`; a learned
  cap would learn a number nothing consults.
- No DuckDuckGo wall-beater. The wall is per IP; engine spreading is the answer and it
  is built.
- No keychain AES port (the persistent-profile design killed the Chrome-cookie path),
  no bs4 port (aggregator text extraction is bounded raw-HTML stripping), no per-model
  prompt variants (`conf/prompts.toml` is canonical).
- `tools/*.py` gates stay Python permanently.
- No branch-coverage gate: a second number with its own forgiveness. Per-file floors
  (`tools/coverage_floors.jsonc`) carry the weight, and X4 is where line coverage gets honest.
- No clippy campaign on `vendor/camoufox-rs`: 208 findings sit under shrink-only
  per-lint ceilings (`tools/vendor_lint_baseline.jsonc`), which is the honest
  instrument for code this repo does not own.
- Shrinking file_summary to fit `foundation`: decided 2026-10-05. The task keeps its
  17 rows and 40-line excerpts, and `foundation` is NOT MEASURED on it.
