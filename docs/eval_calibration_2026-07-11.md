# Model Eval Calibration — 2026-07-11

## Scope

Run all 5 eval tasks (detailed_json, json, filename, summarize, file_summary) across
all 9 models available on the Osaurus server. Compare actual scores against each
model's known capabilities (published benchmarks, architecture). Flag anomalies,
identify test-design issues, and recommend calibrations.

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

**Expected ranking** (by general capability): qwen3.6-27b >= qwen3.6-35b-a3b >=
gemma-4-12b > diffusiongemma > qwen-agentworld > gemma-4-e4b > ornith >
foundation (unknown) > potion (not an LLM — expected 0%)

---

## Eval Results

### Scores by Model and Task

| Model | detailed_json | json | filename | summarize | file_summary | Mean |
|-------|:------------:|:----:|:--------:|:---------:|:------------:|:----:|
| foundation | 15 | 90 | 100 | 90 | 100 | **79** |
| diffusiongemma-26b | 15 | 85 | 100 | 90 | 100 | **78** |
| gemma-4-12b | 15 | 85 | 100 | 90 | 100 | **78** |
| gemma-4-e4b | 15 | 85 | 100 | 90 | 100 | **78** |
| ornith-1.0-35b | 15 | 85 | 0 | 100 | 100 | **60** |
| potion-base-4m | 0 | 0 | 0 | 0 | 0 | **0** |
| qwen-agentworld-35b | 15 | 85 | 100 | 90 | 100 | **78** |
| qwen3.6-27b | 15 | 85 | 100 | 90 | 100 | **78** |
| qwen3.6-35b-a3b | 15 | 85 | 100 | 90 | 100 | **78** |

### Score Distribution

Mean scores cluster at three values: 79 (foundation), 78 (6 models), 60 (ornith), 0 (potion).

**7 of 9 models are indistinguishable** (78-79% mean). The eval suite does not
discriminate between models of very different capability levels (compare qwen3.6-27b
AIME 94.1% vs gemma-4-e4b AIME 42.5% — both score 78).

---

## Anomalies

### A1. detailed_json 15% — All Models

Every model including the best Qwen 3.6 variants scores exactly 15% on
detailed_json with the diagnosis `"not from input (hallucinated)"`.

**Root cause**: The prompt substitutes `{location}` → `"Vaughan"`:
```
Extract 8-10 popular Vaughan venues for families with kids ages 3-7.
```
The model generates Vaughan venues from its **knowledge** (Canada's Wonderland,
Vaughan Sports Arena, etc.). But the validator's source grounding compares against
the **full eval input** (8 venues: Vaughan Sports Arena, Central Park Zoo, Bronx
Zoo, NYC Aquarium, etc.) which the model never sees. Zero overlap → 15% cap.

**Verdict**: Test design bug, not model quality. The source data is used as a
grounding reference but never passed to the model. Fix: either pass the venue list
in the prompt (making it a true extraction task), or remove source grounding for
this task (making it a pure generation task).

### A2. `{date_range}` Never Substituted

The weekend_transient prompt template contains `{date_range}`:
```
Find 5-10 events for {date_range} in Central Park. Kids ages 3-7.
```

`_safe_format_prompt` only handles `{location}` and `{age_range}` — it does not
substitute `{date_range}`. The eval input JSON also lacks a date_range field.
Every model sees the literal string `{date_range}` in their prompt.

**Impact**: Models likely ignore the nonsense token. But the prompt is formally
broken. Fix: add a date_range field to eval_inputs.toml and handle it in
`_safe_format_prompt`.

### A3. ornith-1.0-35b-mxfp8 Empty on filename (0%)

ornith scores 100% on summarize and file_summary but 0% on filename (empty
content returned). Direct probe confirms empty string:

```python
call("ornith-1.0-35b-mxfp8", [{"role": "user", "content": "Output ONLY..."
}])  # → content: ''
```

**Root cause**: ornith is an agentic coding specialist (SWE-Bench 75.6%).
It may refuse tasks that don't match its coding/structured-output training
distribution. The filename task asks for a short bare string — no code blocks,
no JSON. The model may be confused by the "no JSON" instruction since it's
trained to always produce structured output.

**Verdict**: Model quirk. Either (a) wrap filename in a JSON schema for ornith,
or (b) accept this as a known limitation of coding-specialist models.

### A4. potion-base-4m 0% on Everything

HTTP 500: `"Unsupported model type: model2vec"`

**Root cause**: potion-base-4m is a **static embedding model** (3.7M params, 128-dim
vectors). It is NOT an LLM and cannot generate text. It should be excluded from
generation eval runs.

**Verdict**: Remove from test list or skip gracefully.

### A5. `--config-tasks` Flag Is Dead Code

The `--config-tasks` CLI flag is parsed but **never referenced** in the eval
main function. `build_tasks_from_model()` runs unconditionally and overrides the
hardcoded TASKS dict. Only 5 tasks (detailed_json, json, filename, summarize,
file_summary) ever run in the default path.

The full tasks_core.py defines 15+ tasks including:
- `weekend_transient_schema` — strict JSON-only output
- `filename_leak` — instruction leak detection
- `summarize_contradiction` — faithfulness probe
- `weekend_transient_mixed` / `weekend_fixed_mixed` — noise injection
- `rename_mixed`, `summarize_mixed`, `file_summary_mixed` — signal/noise scoring

These never run.

**Verdict**: Either wire up the flag to switch between config tasks and
hardcoded tasks, or run the full suite by default (config tasks + hardcoded
augmentation).

### A6. No Score Discrimination Between Tiers

Expected ranking vs actual:
```
Expected:  27B ≈ 35B > 12B > diffgemma > agentworld > e4b > ornith > foundation > potion
Actual:    foundation (79) > all_others (78) > ornith (60) > potion (0)
```

The 5 eval tasks only test easy capabilities:
- **filename**: all models 100% (too easy — binary pass/fail)
- **file_summary**: all models 100% (too easy)
- **summarize**: 90-100% cluster (needs harder scoring)
- **json**: 85-90% cluster (mostly item count differences)
- **detailed_json**: all 15% (broken, not measuring quality)

**Verdict**: The eval suite needs harder or more targeted tasks to differentiate
models. Suggestions:
- Add a **needle-in-haystack** task testing long-context retrieval
- Use the **mixed signal/noise** tasks (already defined in tasks_core.py)
- Score **verbosity conciseness** — ornith averages 1079 chars/response vs foundation 737
- Add a **format-following** task with strict schema requirements
- Test **instruction following** with many simultaneous constraints

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

Latency varies **5-55x** across models. ornith (213s total) is 11× slower than
foundation (19s). The MoE models (qwen3.6-35b-a3b 39.6s, diffusiongemma 39.1s)
are 3-4× faster than dense models of equivalent quality (qwen3.6-27b 141.5s).

**Latency is a good discriminator** even when scores are identical.

---

## Verbosity Comparison

| Model | Avg response | filename | summarize | Pattern |
|-------|:------------:|:--------:|:---------:|:--------|
| foundation | 737 chars | 22 | 490 | Most concise |
| diffusiongemma-26b | 838 chars | 31 | 1128 | |
| gemma-4-12b | 831 chars | 31 | 1031 | |
| gemma-4-e4b | 853 chars | 31 | 1077 | |
| ornith-1.0-35b | 1079 chars | 0 | 1222 | Most verbose |
| qwen-agentworld-35b | 769 chars | 67 | 939 | |
| qwen3.6-27b | 793 chars | 35 | 1200 | |
| qwen3.6-35b-a3b | 1001 chars | 42 | 1134 | |

ornith produces the longest summaries (1222 chars) and scores 100% — suggesting
longer, more detailed responses correlate with higher summary quality. foundation
produces the shortest (490 chars) and scores 90%.

---

## Signal File Update Required

`conf/eval_signals.json` currently has signal data for many models that are no
longer present (laguna, nemotron, minimax, lfm2, m1, etc.) and is missing entries
for new models (ornith, qwen3.6-27b-mtp, qwen3.6-35b-a3b-mtp, etc.). The eval
run appends new entries automatically, but stale entries from retired models
should be pruned.

---

## Recommendations

### Immediate Fixes (test suite)

1. **Fix `{date_range}` substitution** — add field to eval_inputs and handle in
   `_safe_format_prompt`
2. **Fix detailed_json source grounding** — either pass the venue list to the
   model or remove the source check
3. **Exclude potion-base-4m** from generation evals (embedding model, not LLM)
4. **Wire up `--config-tasks` flag or remove it** — dead code is confusing

### Scoring Improvements

5. **Run the full 15+ task suite by default** — mixed signal/noise tasks will
   differentiate models better
6. **Add a long-context needle test** — models vary wildly on 256K context
   retrieval (gemma-4-e4b MRCR 25.4% vs qwen3.6-27b unknown but likely better)
7. **Score latency and verbosity as metrics** — these are real differentiators

### Model Handling

8. **Probe foundation's backend identity** — it's the fastest model and sometimes
   the best-scoring; knowing what it is matters for capacity planning
9. **Investigate ornith filename failure** — determine if it's a prompt-format
   refusal or a genuine capability gap; potentially wrap filename in JSON schema
   for coding models

---

## Raw Data

Full eval output: `~/.config/ztools/eval_results.json`
CSV export: `~/.config/ztools/eval_results.csv`
Model research: compiled from web search on 2026-07-11
