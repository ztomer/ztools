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

## Where things stand — 2026-10-10

A dated summary of what the runs below already print. Nothing in this section is
checked by a gate, so re-derive it before editing it:
`git log --oneline origin/main..main`, `gh run list --limit 3`,
`tools/gate.sh --full`, and `"$GOH_DIR/gates/goh.sh" home-paths --exclude ^vendor/`.

- Releases v3.7.0 (twitter repair, weekend graders that read fields, retention in days) and v3.8.0 (the on-device fallback, T1) cut 2026-10-10. Nothing is open: every phase below has landed.
- Items `L2`, `M1` through `M16` landed and deleted from open work; items `G5`, `G6`, and `R2` dropped.
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
  **`M11`** (`ztools model-eval --leaderboard --sort-by <slot>` slot-specific capability sorting),
  **`M12`** (`ztools model-eval --leaderboard` longitudinal score delta tracking against previous run),
  **`M13`** (`ztools model-eval --leaderboard --category <name>` category-filtered evaluation ranking),
  **`M14`** (`ztools model-eval --leaderboard --csv-output` spreadsheet/pipeline CSV export format),
  **`M15`** (`ztools model-eval --leaderboard --group-by-family` model family grouping),
  **`M16`** (`ztools model-eval --leaderboard --fail-on-regression <pct>` automated regression threshold check),
  **`G2`** (307 files clean under `GOH_NO_HOME_PATHS=1`), **`G3`** (toolchain header cleaned),
  **`G4`** (CI gate tool manifest dynamic install + pinned tsv), **`R1`** (provenance ledger reading).
- Local verification on `main`:
  `tools/gate.sh --full` passes all 12 CI steps (coverage floor 95%, clippy, cargo audit,
  cargo deny, structural, pytest, ruff, camoufox tests, and secrets scan),
  `python3 -m pytest tools/tests -q` green (136 passed).
- All temporary agent sessions (OpenCode and Claude worktrees/caches) are purged.

### Sources consolidated — 2026-10-10

Every other agent session that touched ztools, and where its content went. Nothing below is a
plan anymore; this file is.

- **Claude Code / desktop:** no ztools transcripts or desktop sessions remain. The 10-08 backlog
  chip "Bring up ztools CI: machete and 17 eval tests" is superseded: fixed in 64d67f3, CI green.
- **OpenCode** (`ses_ee23c056…`, `ses_ef8dce2a…`, and nine Python-era sessions 07-16 → 08-24): every
  item landed (M1–M3, M5, G2–G4, R1, edition 2024, weather endpoint, grounding) or was superseded
  by the Rust cutover (ddb7d53). Project row and snapshot removed (Trash).
- **Antigravity** (`f43ba45d…`, `46b086a1…`, `608907e3…`, `2901c3f7…`, plus backups): Python-era,
  superseded by the Rust port; removed (Trash). The cross-repo conversation `460d804a…` (Python
  3.15 move) is not this project's and was left alone; its one ztools finding, CI pinned to 3.14,
  is fixed (CI reads `.python-version`).
- **Gemini CLI** (`~/.gemini/tmp/ztools/`, April–June chats): Python-era; removed (Trash).
- **Found by the consolidation itself:** tags v3.4.0, v3.4.1 and v3.5.0 had no GitHub release
  (tagged by hand); the release kit now backfills a tag that lacks one (gates_of_heck 2573b86).

## Phase W — the weekend plan must read right to a parent

All Phase W items (`W1` through `W6`) landed on 2026-10-10. No open work remains in this phase.

## Phase M — what the eval records must mean what it says

`M1` through `M22` landed; M17-M22 were found while landing W5 (2026-10-10), when the
weekend eval rows started sending the planner's own prompts. No open work remains in this
phase.

## Phase T — the twitter summary must be trustworthy when nobody is watching

All Phase T items landed on 2026-10-10: T2 (the loop-rate measurement) and T3 (the repair it
motivated) are in `docs/MODEL_QUIRKS.md`; T1 gave the chain a model outside the Osaurus
server -- `foundation` answered by macOS 27's `fm` (`conf/twitter.toml [fallback] on_device`).
No open work remains in this phase.

