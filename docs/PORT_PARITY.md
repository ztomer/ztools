# Python ↔ Rust parity ledger

Two implementations of the same four tools exist, and they have already drifted.

- **Python** — `references/**` in this repo, entry points `tw` / `wk` / `rn` / `ev`
  (only inside `.venv/bin`, so they need the venv active).
- **Rust** — `rust/src/ztools/*.rs` in this repo, subcommands `twitter-summarize`,
  `weekend-plan`, `image-renamer`, `model-eval` behind the `ztools` binary.

**The Rust port exists to escape the venv/uv dependency: one static binary, no Python
startup.** That is a real goal and it is why the port stays. This file is the price of
keeping it — the list of things that must be mirrored, so "we'll sync it later" is a
tracked debt instead of a memory.

## Which one actually runs (verified 2026-08-12)

| invocation | implementation | source |
|---|---|---|
| `twitter`, `weekend`, `rename_images`, `oeval` | **Rust** | `~/.zshrc:154-158` — shell *functions*, which beat PATH |
| the scheduled jobs | **Python** | `routines.toml:15` `uv run --frozen wk`; `routines-twitter.toml:8` `uv run --frozen tw` |
| `tw` / `wk` / `rn` / `ev` | Python | `.venv/bin/` only; not on PATH |

Interactive runs and scheduled runs therefore execute **different code**. Note also that
`$HOME/Projects/ztools/bin` is on PATH (`~/.zshrc:233`) carrying Python scripts named
`twitter`, `weekend`, `oeval`, `rename_images` — all shadowed by the shell functions and
dead since 16 Jul.

## Known divergences

### 1. The summarizer prompt is duplicated, not shared — twitter prompt now shared

The twitter summarize **instruction block** now has one canonical home:
`conf/prompts.toml` `[twitter.summarize]`. The Rust binary embeds a fallback copy
(`rust/src/config.rs`) so the static binary works with no checkout; a drift-gate
test (`test_twitter_prompt_matches_shared_conf`) fails if that fallback diverges
from the file. The Python eval harness composes `TWITTER_PROMPT` from the same
file (`references/eval/tasks_prompts.py`), keeping the eval fixture timeline as
data. Editing the prompt once edits both sides; the gate enforces it.

**Still open:** the weekend schemas and the rename task restatement remain
parallel copies (see Phase 1 in `RUST_PORT_PLAN.md`).

### 2. Model selection has already diverged, and the Rust default is not the measured best

| | twitter/summarize | source |
|---|---|---|
| Rust | `gemma-4-e2b-it-8bit` (hardcoded default) | `rust/src/config.rs:76-84` |
| Python | `summarize = gemma-4-12b-it-mxfp8`, `json = foundation` | `conf/config.toml [best_models]` |

The 2026-08-12 sweep ranks `gemma-4-e4b-it-8bit` at 82.1 and `muse-glimmer-30b-jang_6m`
at 88.2 over 22 common tasks. So the interactive path runs a model the leaderboard does
not endorse, and editing `conf/config.toml` — the obvious place — does not change it.

### 3. The eval and its scorers are Python-only

`references/eval/**` plus `references/lib/validators/**` are what rank models and gate
quality. Rust has its own 194-line `model_eval.rs`. Any scorer fix here does not reach
the Rust binary's notion of which model is best.

## Resolved & Ported into Rust (The Parity Roadmap)

- [x] **1. Broken Model & Packaging Defect Detection**
  - Ported to `rust/src/ztools/model_health.rs`.
  - Offline inspection: detects unsupported MTP speculative drafting shards (`*mtp*.safetensors`) when `runtime_available = false`, missing safetensors shards referenced in `model.safetensors.index.json`, and incomplete download artifacts (`*.incomplete`).
  - Viability guard: checks decode rate against `THRASHING_DECODE_TOKENS_PER_SEC` (1.0 tok/s) and refuses broken models before running tasks.
- [x] **2. Best Model Matrix & Dynamic Configuration**
  - Synchronized `rust/src/config.rs` with the 30-task benchmark winners:
    - `json` / Weekend: `qwen3.8-27b-8bit` (100% across all 7 weekend/json tasks).
    - `filename` / Renaming: `gemma-4-e2b-it-8bit` (100% quality + 100% prompt injection resistance).
    - `summarize` / Twitter: `gemma-4-e2b-it-8bit` (top contradiction & factual accuracy resistance).
    - `think` / Structured Fallback: `qwen3.8-27b-8bit` (100% on Taxes QA & File Summary Mixed).
    - `vlm` / Vision: `qwen3.8-27b-8bit` (100% on `image_real` & `image_rename`).
  - Added `with_ztools_best_models()` dynamic loader from `~/.config/ztools/config.toml` or `conf/config.toml`.
- [x] **3. Image Renamer Security & Untrusted Framing**
  - Wrapped extracted image text inside `<<<BEGIN_UNTRUSTED_DOCUMENT` delimiters to prevent OCR prompt injection attacks (`filename_injection` defense).
  - Stripped markdown code blocks, conversational prefixes (`"Here is the filename:"`), and file extensions during sanitization.
- [x] **4. Twitter Summarizer Prompt & Timestamp Parity (C2a fix)**
  - Synchronized prompt instructions with `TWITTER_PROMPT`.
  - Formatted tweet timestamps as `%b %d %H:%M` in the prompt payload to prevent date-dropping at the LLM boundary.
- [x] **5. Weekend Planner Schema & Exclusion Filtering (C2b, C8 fixes)**
  - Aligned JSON schema to include `start_date`, `end_date`, `price`, `day`, `weather`.
  - Matched candidate events against exclusion patterns from `conf/weekend.toml`.
  - **2026-08-19 upgrade:** the Rust matcher was a weaker port (whitespace-only
    tokenisation) and silently missed interpolated/reordered venue names — "Sky
    Zone Toronto" escaped while the Python side dropped it. Replaced with the
    faithful port (`rust/src/ztools/weekend/enforce.rs`): typographic-punctuation
    folding, connector-word and possessive handling, parenthetical stripping,
    token-subset + containment matching, and the C8 seasonal-event exception,
    wired into `weekend_cache` and the `weekend_plan` dispatch. Tests ported from
    `test_report_class_fixes.py` were proven to fail against the old matcher
    before the new one landed.
- [x] **5b. Weekend constraint suite (C5 weather, C4 constant columns, C3 window,
  C7 provenance)** — 2026-08-19: the full `enforce.py` constraint suite is now
  ported to `rust/src/ztools/weekend/` and wired into the `weekend_plan`
  dispatch in canonical order: provenance (`drop_unsourced_rows`, C7) → exclusion
  (`drop_excluded_places`, C8) → window (`drop_events_outside_window` + day
  reconcile, C3) → weather labels (C5) → constant columns (C4). The C5 weather
  correction and C4 suspect-conjunction tests proved to fail when neutered;
  window tests proved to fail when `window_overlap` was neutered; provenance
  tests proved to fail when `row_is_sourced` was forced false.
  `fetch_duckduckgo_events` returns the fetched corpus so the provenance gate has
  ground truth. `enforce.rs` was split out past the 500-line cap: constants into
  `weekend/constants.rs`, dates into `weekend/dates.rs`.
- [x] **5c. Weekend 4-phase pipeline (extract → draft → refine → structure) +
  supply prioritisation** — 2026-08-19: `weekend/prompts.rs` ports the phase
  templates verbatim with a C1-checking renderer, `weekend/phases.rs` ports
  `extract_sources` (adaptive batching, raw pass-through on failure),
  `draft_activities`, `refine_draft`, `structure_to_json` and `condense_weather`
  behind a `PlanContext`, and `weekend/supply.rs` ports `prioritise_in_window`
  / `in_window_count` using the SAME date scanner as the enforcer. Weather now
  precedes the pipeline (matching Python); the monolithic prompt remains as the
  fallback when a phase stalls. Phase tests prove the degrade-not-starve
  fallbacks (each proved-fail-first).
- [x] **6. Greedy decoding across all LLM callers** (temperature 0.0) — for deterministic reproducible leaderboard outputs.
- [x] **7. Derived request timeouts** from measured cold-start / prefill / decode rates.

## How this gets resolved & verified: Deep A/B Testing

By automated **behavioral A/B testing** using `bin/ab_test --functional`:
1. **Defect Probe Parity**: Run test model fixture bundles (clean, broken MTP, missing index shards, incomplete downloads) through both Rust and Python, asserting identical diagnostic verdicts.
2. **Security & Prompt Injection Parity**: Run adversarial OCR inputs through both Python `rn` and Rust `image-renamer`, asserting that neither is compromised and both emit identical sanitized filenames.
3. **Prompt & Date Parity**: Verify tweet payload timestamps and weekend event date schemas match between Python and Rust.
4. **Rust Quality Gates**: `cargo clippy --all-targets -D warnings` and the 500-line cap per file in `~/Projects/ztools/rust`.

## The standing hazard

This is the "parallel reimplementation" failure mode: two pipelines that must agree, with
nothing forcing them to. The structural fix is:
1. Move the shared surface — prompts and model choice — into shared config that both sides read.
2. Maintain the automated A/B test harness (`bin/ab_test`) in CI to catch divergence the day it happens.

