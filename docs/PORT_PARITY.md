# Python ↔ Rust parity ledger

Two implementations of the same four tools exist, and they have already drifted.

- **Python** — `references/**` in this repo, entry points `tw` / `wk` / `rn` / `ev`
  (only inside `.venv/bin`, so they need the venv active).
- **Rust** — `~/Projects/routines/src/ztools/*.rs`, subcommands `twitter-summarize`,
  `weekend-plan`, `image-renamer`, `model-eval`.

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

### 1. The summarizer prompt is duplicated, not shared

`routines/src/ztools/twitter.rs:105` carries its own copy of the instruction text
("Use connecting phrases and narrative verbs to show how events relate"), mirroring
`references/eval/tasks_prompts.py: TWITTER_PROMPT`. Nothing derives one from the other.
Every prompt improvement therefore has to be made twice or it silently applies to half
the runs.

The Rust side reads exactly one thing from this repo — `~/Projects/ztools/conf/weekend.toml`
(`config_ztools.rs:73`). Prompts are not in shared config.

### 2. Model selection has already diverged, and the Rust default is not the measured best

| | twitter/summarize | source |
|---|---|---|
| Rust | `gemma-4-e4b-it-8bit` (hardcoded default) | `config_ztools.rs:76-81` |
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
  - Ported to `routines/src/ztools/model_health.rs`.
  - Offline inspection: detects unsupported MTP speculative drafting shards (`*mtp*.safetensors`) when `runtime_available = false`, missing safetensors shards referenced in `model.safetensors.index.json`, and incomplete download artifacts (`*.incomplete`).
  - Viability guard: checks decode rate against `THRASHING_DECODE_TOKENS_PER_SEC` (1.0 tok/s) and refuses broken models before running tasks.
- [x] **2. Best Model Matrix & Dynamic Configuration**
  - Synchronized `config_ztools.rs` with the 30-task benchmark winners:
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
- [x] **6. Greedy decoding across all LLM callers** (temperature 0.0) — for deterministic reproducible leaderboard outputs.
- [x] **7. Derived request timeouts** from measured cold-start / prefill / decode rates.

## How this gets resolved & verified: Deep A/B Testing

By automated **behavioral A/B testing** using `bin/ab_test --functional`:
1. **Defect Probe Parity**: Run test model fixture bundles (clean, broken MTP, missing index shards, incomplete downloads) through both Rust and Python, asserting identical diagnostic verdicts.
2. **Security & Prompt Injection Parity**: Run adversarial OCR inputs through both Python `rn` and Rust `image-renamer`, asserting that neither is compromised and both emit identical sanitized filenames.
3. **Prompt & Date Parity**: Verify tweet payload timestamps and weekend event date schemas match between Python and Rust.
4. **Rust Quality Gates**: `cargo llvm-cov --all-targets --fail-under-lines 95` and 400-line cap per file in `~/Projects/routines`.

## The standing hazard

This is the "parallel reimplementation" failure mode: two pipelines that must agree, with
nothing forcing them to. The structural fix is:
1. Move the shared surface — prompts and model choice — into shared config that both sides read.
2. Maintain the automated A/B test harness (`bin/ab_test`) in CI to catch divergence the day it happens.

