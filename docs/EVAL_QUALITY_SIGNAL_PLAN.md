# Eval Quality-Signal Improvement Plan

Date: 2026-07-09
Goal: make `model_eval` produce *actionable* quality signals across ALL tasks, not just
a single pass/fail number, and make signal-from-noise filtering *measurable* everywhere.

## Status (2026-07-09) — DONE except Step 2b

- Step 1 (mixed-signal validators for ALL mixed tasks): DONE.
  - `validate_mixed_signal` (weekend), `validate_mixed_summary`,
    `validate_mixed_file_summary`, `validate_mixed_filename` (rename) added.
  - `eval/run.py` now passes `source_text` to every validator (incl. text ones).
  - Fixed inverted signal/noise parsing in `TWITTER_PROMPT_MIXED` (NOISE now appended
    after the real timeline, not inside it).
  - Fixed underscore tokenization in name matching (json + text validators).
  - Fixed markdown/JSON parsing for `file_summary_mixed` (models return either).
  - Redesigned `RENAME_PROMPT_MIXED` to be self-contained (SNIPPETS + NOISE) and
    list-based so filtering is measurable.
  - Removed duplicate `summarize_mixed` / `file_summary_mixed` task defs.
  - Source-grounding cap in `validate_detailed_json` now only applies when a source is
    provided (no longer crushes un-source tasks to 15).
- Step 2a (per-dimension reporting): PARTIAL. Added a "Signal/Noise Filtering" block to
  the eval report (`run.py`) that surfaces `included N/M noise` per mixed task.
- Step 2b (wire weekend quality scorers as a dimension breakdown in eval output):
  DEFERRED. The scorers exist in `lib/quality_weekend_scorers.py` but are not yet
  surfaced per-model in `model_eval` (they need reference metadata per task). The
  precision/recall filtering signal is the higher-value, now-shipping piece.
- Step 3 (cleanup): DONE (no dup task defs).
- Step 4 (full sweep): DONE — see `docs/MODEL_QUIRKS.md` "Signal/Noise Filtering".
- Step 5 (docs + release): DONE — v0.6 cut.

## Key learning
A single 0-100 pass/fail cannot distinguish "filters noise" from "dumps everything":
noise IS in the source, so source-grounding always matched. Precision/recall against
explicit signal vs noise sets is the only way to measure filtering. The mixed sweep
exposed real leakage (ornith 4/8 tweets; qwen35b + ornith include ALL 6 noise files)
that the old eval scored as 100%.

## Current state (problems)

1. **Mixed-task filtering is unmeasurable for non-weekend tasks.**
   `weekend_transient_mixed` / `weekend_fixed_mixed` now use `validate_mixed_signal`
   (precision = excluded noise, recall = kept signal). The other mixed tasks still run
   their plain validators on the full output and never verify the noise was dropped:
   - `summarize_mixed` — noise tweets should be excluded from the summary.
   - `file_summary_mixed` — only real files should be summarized.
   - `rename_mixed` / `filename_mixed` / `image_rename_mixed` — garbage inputs must map
     to empty, not to invented filenames.

2. **Single 0-100 score, no dimension breakdown.** The rich quality scorers in
   `lib/quality_weekend_scorers.py` (Completeness/Weather/Age/Source/Exclusions) and the
   weighted validators (`text_validator.py`) compute dimensions but eval only reports one
   integer. You cannot see *why* a model lost points.

3. **`Quality Check Summary` is weekend-only** and only reports source-matching ratio.
   No per-model filtering comparison, no precision/recall table.

4. **Duplicate task definitions** in `eval/tasks_core.py` (`summarize_mixed` and
   `file_summary_mixed` are each defined twice; later def overrides earlier).

## Plan

### Step 1 — Mixed-signal validators for ALL mixed tasks
Add to `lib/validators/`:
- `validate_mixed_summary(data, source_text)` — parse signal vs NOISE tweets from the
  mixed timeline; score = 0.5*recall(signal tweets covered) + 0.5*precision(noise tweets
  excluded). Noise tweet markers: senders/phrases in the `NOISE` block.
- `validate_mixed_file_summary(data, source_text)` — parse real file paths vs noise
  entries; precision/recall on which files were summarized.
- `validate_mixed_filename(data, source_text)` — for image_rename_mixed style
  test_cases: each garbage input must yield empty/refusal; each real input must yield a
  filename. Score = correct rejections / total noise cases + correct accepts / total
  signal cases.

Wire into `TASKS`: point `summarize_mixed`, `file_summary_mixed`, `rename_mixed`,
`filename_mixed`, `image_rename_mixed` at their `validator` and pass `source` for
signal/noise parsing. Reuse the `NOISE` marker convention already in the prompts.

### Step 2 — Per-dimension reporting in eval output
- In `eval/run.py`, after validation, also run the dimension scorers for weekend tasks
  (`_score_weekend_*`) and capture per-dimension scores onto the result dict.
- Extend `eval/report.py` Rich tables: add a "Dimensions" table (model × dimension) for
  weekend tasks and a "Filtering" table (model × precision/recall) for mixed tasks.
- Keep the existing `Quality Check Summary` but add precision/recall lines for mixed tasks.

### Step 3 — Clean up `tasks_core.py`
Remove the duplicate `summarize_mixed` / `file_summary_mixed` blocks. Consolidate the
mixed task list so each appears once with the correct `validator` + `source`.

### Step 4 — Full mixed-eval sweep across all models
Run every `*_mixed` task on all available models (`osaurus list`):
foundation, qwen3.6-27b-mxfp8-mtp, qwen3.6-35b-a3b-mxfp8-mtp, gemma-4-12b-it-mxfp8,
gemma-4-e4b-it-8bit, diffusiongemma-26b-a4b-it-mxfp8, ornith-1.0-35b-mxfp8,
qwen-agentworld-35b-a3b-mxfp8, potion-base-4m.
Capture precision/recall per model per task.

### Step 5 — Update docs + release
- Write filtering comparison table into `docs/MODEL_QUIRKS.md` (new "Signal Filtering"
  section) and note any model that dumps noise.
- Commit + push; cut **v0.6**.

## Acceptance
- Every `*_mixed` task reports precision & recall, not just a 0-100 pass.
- A model that includes noise items scores <100 on the mixed task (proven by feeding a
  known-noisy model OR by a hand-crafted bad output unit test).
- Dimension breakdown visible in eval report for weekend tasks.
- No duplicate task definitions; full mixed sweep runs on all models without errors.
