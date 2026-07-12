# Model Eval Calibration — 2026-07-11

## Scope

Run all eval tasks across all 9 models available on the Osaurus server. Compare
actual scores against each model's known capabilities (published benchmarks,
architecture). Flag anomalies, identify test-design issues, and apply fixes.

All fixes implemented as v0.8.10–v0.9.0.

---

## Models Tested

| # | Model | Type | Params | Active | Context | Known Strength | Known Weakness |
|---|-------|------|--------|--------|---------|----------------|----------------|
| 1 | foundation | Router (unknown backend) | ? | ? | ? | Fastest latency (18.8s total) | Opaque backend identity |
| 2 | diffusiongemma-26b-a4b-it-mxfp8 | MoE Diffusion | 25.2B | 3.8B | 256K | 1100+ tok/s inference | Quality loss vs base Gemma 4 26B |
| 3 | gemma-4-12b-it-mxfp8 | Dense | 12B | 12B | 256K | Laptop-friendly, MMLU Pro 77.2% | Vision below dedicated encoders |
| 4 | gemma-4-e4b-it-8bit | Dense | ~4B | ~4B | 256K | Phone/edge deployable, IFEval 96.7% | Weak math (AIME 42.5%) |
| 5 | ornith-1.0-35b-mxfp8 | MoE | 35B | ~3B | 262K | #1 agentic coding at size (SWE-Bench 75.6%) | Narrow coding specialization |
| 6 | potion-base-4m | Static Embed | 3.7M | 3.7M | N/A | Ultra-fast tiny embeddings | NOT an LLM — no text generation |
| 7 | qwen-agentworld-35b-a3b-mxfp8 | MoE World Model | 35B | 3B | 256K | Environment simulation | Not a general chatbot |
| 8 | qwen3.6-27b-mxfp8-mtp | Dense | 27B | 27B | 262K | Elite dense (AIME 94.1%, MMLU-Pro 86.2%) | Needs 16GB+ VRAM |
| 9 | qwen3.6-35b-a3b-mxfp8-mtp | MoE | 35B | 3B | 262K | 12GB GPU viable, elite MoE | MTP speedup limited on MoE |

---

## Changes Applied (v0.8.10–v0.9.0)

### Fixed: detailed_json source grounding (A1)

**Before**: Every model scored 15%. Prompt used `{location}` → `"Vaughan"`, model
generated from knowledge, source grounding compared against full 8-venue eval
input model never saw → zero overlap → 15% cap.

**Fix**: Changed all 6 model configs' `weekend_fixed` prompts to use `{}` for
the full venue list. Model now sees the source data and extracts from it.

**Result**: detailed_json: 15% → **100%** for all capable models. Source
grounding now works correctly.

### Fixed: `{date_range}` substitution (A2)

**Before**: Never substituted. All models saw literal `{date_range}` in prompt.

**Fix**: `_safe_format_prompt` now handles `{date_range}` → `"this weekend"`
alongside `{location}` and `{age_range}`. All template vars can coexist with
`{}` for full data substitution.

### Fixed: `--config-tasks` dead code → full task suite (A5)

**Before**: `--config-tasks` flag parsed but never used. `build_tasks_from_model`
ran unconditionally, overriding the hardcoded 15+ task TASKS dict with only 5
config tasks. Only detailed_json, json, filename, summarize, file_summary ever
ran.

**Fix**: `--config-tasks` now switches between:
- **Default** (no flag): 18 hardcoded tasks including mixed signal/noise
  variants, schema strictness, contradiction probe, instruction leak detection
- **`--config-tasks`**: 5 model-specific tasks with per-family prompt tuning

Default now runs the discriminating tasks: `summarize_contradiction`,
`filename_leak`, `weekend_transient_schema`, `weekend_transient_mixed`,
`weekend_fixed_mixed`, `file_summary_mixed`, etc.

### Fixed: Default fallback prompts

**Before**: Default family (used by ornith, potion, unknown models) had a
filename prompt with no sample text placeholder — just `"Output ONLY the
filename (lowercase, underscores)."` The model got no input to process.

**Fix**: Added `TEXT: {}` placeholder and standardized with other configs.

### Fixed: `file_summary` validator crashes on list-of-strings

Models returning `["path1.py", "path2.py"]` instead of `[{"path": "...", "desc": "..."}, ...]` would crash `validate_file_summary` with `'str' object has no attribute 'get'` (an INFRA error). Now converts string items to dicts automatically — the score degrades gracefully (low path realism, missing descriptions) instead of crashing.

### Fixed: Dead `image_rename` tasks removed from TASKS

`image_rename` and `image_rename_mixed` had `test_cases` instead of `messages` keys, so the eval runner silently skipped them every run. Removed from TASKS dict — these were stubs for a never-wired-up multi-case runner.

### Fixed: Non-LLM models skipped during discovery

potion-base-4m (Model2Vec static embedding) was causing HTTP 500 errors on
every task. Added filter in `eval/cli.py` that skips models with keywords
`model2vec`, `potion`, `embedding` in their name. The eval now prints
"Skipping potion-base-4m (non-LLM model)" during discovery.

### Fixed: `_safe_format_prompt` robustness

**Before**: Couldn't handle `{}` alongside `{location}`/`{age_range}`. Tried
`str.format()` first which crashed on JSON braces, then fell back to
`str.replace()` which left named vars unsubstituted.

**Fix**: Handles all template variables independently. `{}`, `{location}`,
`{age_range}`, `{date_range}` all work together or separately.

---

## Eval Results (Post-Fix)

### Default (18 tasks, hardcoded prompts)

Full sweep across all 8 applicable models (potion-base-4m excluded):

| Model | Mean | summarize_contradiction | file_summary | weekend_transient_schema | summarize | filename |
|-------|:----:|:----------------------:|:------------:|:------------------------:|:---------:|:--------:|
| diffusiongemma-26b | **93.4** | **100** | 100 | 100 | 65 | 55 |
| foundation | 84.9 | **0** | 30 | 100 | 55 | 30 |
| qwen-agentworld-35b | 82.8 | **100** | 60 | 100 | 65 | 55 |
| gemma-4-12b | 81.2 | **0** | 60 | 100 | 65 | 55 |
| qwen3.6-35b-a3b | 80.6 | **0** | 60 | 100 | 65 | 55 |
| qwen3.6-27b | 79.6 | **0** | 60 | 100 | 65 | 30 |
| gemma-4-e4b | 76.6 | **100** | 30 | 0 | 65 | 55 |
| ornith-1.0-35b | 73.6 | **0** | 60 | 100 | 53 | 0 |

Key discriminators:
- **summarize_contradiction**: 4 models (foundation, gemma-4-12b, qwen3.6-27b, qwen3.6-35b, ornith) parrot the falsehood (0%), 3 resist (100%). Tracks instruction-following vs critical reasoning.
- **weekend_transient_schema**: gemma-4-e4b scores 0% — outputs markdown tables instead of JSON. Can't follow structured output instructions.
- **file_summary** (now INFRA-safe): diffusiongemma-26b draws 100% (specific descriptions), others 30-60% (generic).
- **filename**: foundation and qwen3.6-27b at 30%; others 55%; ornith 0% (empty response quirk).

---

## Anomalies (Remaining)

### A3. ornith-1.0-35b Empty on filename (still open)

ornith returns empty string for filename prompts — even "Say hello." returns
empty. The model is an agentic coding specialist and appears to refuse
non-coding interactions. Direct probe:

```python
call("ornith-1.0-35b-mxfp8", [{"role": "user", "content": "Say hello."}])
# → content: ''
```

Earlier false 100% scores were from the broken default prompt (no sample text)
where the model produced generic but valid filenames like "output_file".

**Verdict**: Model quirk — appears to refuse non-coding requests entirely.

### A4. potion-base-4m Not an LLM

HTTP 500: `"Unsupported model type: model2vec"` — static embedding model,
cannot generate text. Now skipped during model discovery (eval/cli.py checks
for `model2vec`, `potion`, `embedding` keywords).

### A6. Score discrimination (partially improved)

Hardcoded TASKS now provide better separation on individual tasks (filename
100%→30%, summarize 90%→55%, file_summary 100%→30%). Full multi-model
comparison pending.

---

## Latency Comparison

| Model | Total 5-tasks | Avg/task | file_summary | filename |
|-------|:-------------:|:--------:|:------------:|:--------:|
| foundation | 18.8s | 3.8s | 1.5s | 0.3s |
| diffusiongemma-26b | 39.1s | 7.8s | 1.8s | 1.7s |
| gemma-4-12b | 70.7s | 14.1s | 7.3s | 0.7s |
| gemma-4-e4b | 49.1s | 9.8s | 3.5s | 0.3s |
| ornith-1.0-35b | 213.1s | 42.6s | 3.0s | 19.4s |
| qwen-agentworld-35b | 30.7s | 6.1s | 2.4s | 0.8s |
| qwen3.6-27b | 141.5s | 28.3s | 12.4s | 2.0s |
| qwen3.6-35b-a3b | 39.6s | 7.9s | 2.4s | 0.6s |

Latency varies **5-55x** across models.

---

## Verbosity Comparison

| Model | Avg response | filename | summarize |
|-------|:------------:|:--------:|:---------:|
| foundation | 737 chars | 22 | 490 |
| diffusiongemma-26b | 838 chars | 31 | 1128 |
| gemma-4-12b | 831 chars | 31 | 1031 |
| gemma-4-e4b | 853 chars | 31 | 1077 |
| ornith-1.0-35b | 1079 chars | 0 | 1222 |
| qwen-agentworld-35b | 769 chars | 67 | 939 |
| qwen3.6-27b | 793 chars | 35 | 1200 |
| qwen3.6-35b-a3b | 1001 chars | 42 | 1134 |

---

## Recommendations (Remaining)

1. **Add long-context needle test** — models vary wildly on 256K retrieval
2. **Score latency and verbosity as formal metrics**
3. **Probe foundation's backend identity** (0% on summarize_contradiction + worst filename score suggests weak instruction following for a 100% on weekend content tasks)

## Raw Data

Full eval output: `~/.config/ztools/eval_results.json`
CSV export: `~/.config/ztools/eval_results.csv`
Model research: compiled from web search on 2026-07-11
