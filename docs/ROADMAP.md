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

- Release v3.5.0 cut, tagged, and published on GitHub (following v3.4.1).
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

## Phase M — what the eval records must mean what it says

All Phase M items (`M1` through `M16`) have landed. No open work remains in this phase.



## Phase T — the twitter summary must be trustworthy when nobody is watching

Depends on nothing. Ranked by what a failure hides: T1 hides nothing (the outcome is
a non-zero exit naming the cause) but loses every run while Osaurus is down; T2 hides
whether the decoding change helped at all.

### T1 — the fallback chain has no model outside the one Osaurus server

- **Class:** a fallback that shares the primary's failure domain is not a fallback.
  Every entry in `conf/twitter.toml` `[fallback] models`, `foundation` included, is
  served by the same server at `osaurus_url`, so when it is down the whole chain fails
  at once (seen 2026-10-03). Nothing in `rust/src/ztools/llm.rs` or `conf/` reaches a
  model by any other route; the Python-era direct-MLX tier was retired with no Rust
  counterpart. Today the outcome is explicit — exit 1 and "every model in the fallback
  chain (...) is served by the same Osaurus server at ..., so the chain has no fallback
  outside that server" (`rust/src/ztools/twitter/mod.rs`, pinned by
  `rust/tests/cli_refusals.rs`) — but no summary is produced.
- **Why now:** the summarizer runs three times a day unattended; every Osaurus outage
  is a lost run that only the status page reports.
- **Done when:** a `[fallback]` entry routed to a second backend that does not depend on
  the Osaurus process (a separate server, or an in-process runtime) answers in a
  `rust/tests/cli_refusals.rs` case where the Osaurus URL is a closed port, and that
  test exits 0 with a DEGRADED banner naming the second backend.
- **Blocked by:** owner decision on which second backend to run (none is installed).

### T2 — the summary decoding change is unmeasured on the live server

- **Class:** a decoding knob believed rather than measured. The summarizer now sends
  `temperature: 0.1` (`SUMMARY_SAMPLING`, `rust/src/ztools/twitter/mod.rs`) because
  temperature 0 looped. A `frequency_penalty` of 0.3 was dropped after two live runs
  (2026-10-10) garbled the citation tokens it penalised; how often temperature 0.1
  alone still loops on this model was not measured.
- **Why now:** the quality gate now REJECTS loops, so a decoding setting that still
  loops costs a fallback (a DEGRADED summary) rather than a silently bad one.
- **Done when:** repeated `ztools twitter-summarize --use-cache` runs on one cached
  timeline (they write a scratch directory, never the store), under the GPU lock, at
  `Sampling::GREEDY` and at `SUMMARY_SAMPLING`, are recorded in `docs/MODEL_QUIRKS.md`
  with how many answers the quality gate rejected for each. (`model-eval` cannot
  answer this: it pins temperature 0 by design.)
- **Blocked by:** a free slot on the GPU lock (`tools/gpu_lock.sh`).
