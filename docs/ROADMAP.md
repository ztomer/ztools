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

- Release v3.6.0 cut (twitter summary quality, 7-day retention toggles, the test-home sandbox, weekend supply and plan fixes). The kit backfilled the GitHub releases v2.2.0, v3.0.0, v3.4.0, v3.4.1 and v3.5.0 lacked; 57 older tags have no CHANGELOG stanza and stay without one.
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

Found while fixing the weather labels (2026-10-10). Ranked by how much of the plan
each one makes wrong.

### W2 — the other fields fill placeholders the way the label did

- **Class:** C4 (a field the model fills because empty reads as failure). The weather
  label stopped defaulting once "unknown" was a named answer; price and duration have no
  such word, and on 2026-10-10 Pumpkinfest came back price "Unlimited" and duration
  "Unlimited", lifted from "Unlimited midway rides".
- **Why now:** a placeholder price reads as a fact to a parent choosing between two events.
- **Done when:** a replay of a saved structure batch shows price and duration either
  copied from text that states them or empty, measured on the same rows before and after.
- **Blocked by:** a free slot on the GPU lock for the replay (`tools/gpu_lock.sh`).

### W3 — the draft phase ignores its count

- **Class:** an instruction the model does not follow, with nothing checking it. "Suggest
  10" returned about 80 rows on a 2026-10-10 replay, nav labels included; the listing gate
  and structure batching absorb it, at the cost of prompt size and time.
- **Why now:** every extra row costs a structure batch on a shared GPU, three runs a weekend.
  The other direction too: on a later 2026-10-10 run the draft prompt reached 29,490 chars
  and got no answer on either attempt, so the plan came from the monolithic fallback.
- **Done when:** the cause is measured (followed pages kept in order, or the prompt) and
  the draft's row count on a saved corpus is recorded before and after the change.
- **Blocked by:** a free slot on the GPU lock for the measurement.

### W4 — adult nightlife reaches a family plan

- **Class:** a suitability judgement nobody is asked. The 2026-10-10 17:17 run (monolithic
  fallback) planned "BAESHARAM BOLLYWOOD AFTER DARK" (10:30 PM, a bar), "BARBIE NIGHT @
  FICTION" (a club) and an R&B brunch party for kids 6, 11 and 14. `suitability.rs` reads
  stated age ranges; these state none, so they pass.
- **Why now:** it is the one defect a parent reading the plan notices first.
- **Done when:** the structure schema asks the model whether the activity is for families
  (its judgement, with the reason), the plan drops a "no", and a replay of the 17:17 corpus
  drops those three while keeping Kidchella and the Woodbridge Fall Fair.
- **Blocked by:** a free slot on the GPU lock for the replay.

### W5 — model-eval scores a weekend prompt production no longer sends

- **Class:** a parallel copy of a prompt (engineering rule 9). `conf/models/*.toml`
  `weekend_transient` / `weekend_fixed` carry their own event schema with the old weather
  rule; production now asks for `place_kind` and labels from it (`weekend/setting.rs`).
  Rankings for the weekend slots measure the old prompt.
- **Why now:** the slot defaults in `[best_models]` are chosen by those rankings.
- **Done when:** the eval tasks render the production structure prompt
  (`phases::structure_system`) instead of a copy, and a test fails if they differ.
- **Blocked by:** nothing.

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
