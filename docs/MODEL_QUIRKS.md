# Model Quirks & Best Practices

**Updated: June 2026** — canonical reference for model selection and prompts.

---

## TL;DR Cheat Sheet

Start server: `osaurus serve &>/dev/null & sleep 10`

---

## Best Models by Task

| Task | Model | Speed | Command |
|------|-------|-------|---------|
| eval weekend | qwen3.6-27b-mxfp8-mtp | ~60s | `python3 -m eval --task weekend` |
| rename | laguna-xs.2-mxfp4 | 2.8s | `python3 -m rename` |
| summarize | qwen3.6-27b-mxfp8-mtp | 14.8s | `python3 -m twitter` |

---

## Osaurus Server Rules

1. **Single instance only** - Multiple cause timeouts
2. **Check before run**: `osaurus status`
3. **Response parsing** - Must read ALL chunks until `done=true`

---

## Working Prompts

### Image Filenames
```
Give a short 2-4 word summary of: {text}
```
Max 35 chars, extract first 4-6 words.

### Weekend Tasks
```yaml
weekend_fixed: |
  Output JSON now. Schema: {"fixed_activities": [...]}
  {prompt}
  CRITICAL: Only use: target_ages, price, weather
  Output ONLY JSON.
```

---

## Known Issues

| Model | Issue | Fix |
|-------|-------|-----|
| gemma weather | Outputs weather data | Avoid for weekend |
| gemma-4-31b-jang | Cold start 30s then 1s | Warmup call first |
| qwen | Thinking tokens | Can't disable |
| jang models (MLX) | Wrong shape | Use server instead |
| gemma-4-e4b | Input looping | Avoid |

---

## Config Location

- `conf/config.yaml` — models, timeouts, prompts
- `conf/models/*.yaml` — per-model config
- `lib/config_core.py` — load functions (shim at `lib/config.py`)

---

## The Working Prompt Pattern (April 2026)

**CRITICAL**: For weekend tasks, prompts must use RUNTIME PLACEHOLDERS, not {}. The model generates data with specified values.

```yaml
weekend_fixed: |
    Output JSON now. Schema: {"fixed_activities": [{"name": "str", "location": "str", "target_ages": "str", "price": "str", "weather": "str"}]}

    Extract 8-10 popular {location} venues for families with kids ages {age_range}.

    CRITICAL: Each item MUST have:
    - target_ages: "{age_range}"
    - price: "$18-35 per child" or "$25-35 per family"
    - weather: "indoor" or "outdoor"

    Output ONLY JSON.

  weekend_transient: |
    Output JSON now. Schema: {"transient_events": [...]}
    
    Find 5-10 events for {date_range} in {location}. Kids ages {age_range}.
    
    Use ONLY these values:
    - day: Friday, Saturday, or Sunday
    - target_ages: "{age_range}"
    - weather: "indoor" or "outdoor"
```

Key: `{location}`, `{age_range}`, `{date_range}` are INJECTED at runtime (`weekend/cli.py`), NOT {} placeholders.

---

## Field Normalization (Critical)

Different models output different field names. **All normalization must be in `normalize_llm_items()` in `weekend/cli.py`** - do not scatter it across the code.

Known aliases:
- **name**: `name`, `activity`, `activity_name`, `title`, `event`, `event_name`, `description`
- **location**: `location`, `address`, `venue`, `place`
- **target_ages**: `target_ages`, `age_group`, `ages`, `age_range`
- **price**: `price`, `cost`, `pricing`, `fee`
- **weather**: `weather`, `setting`, `type`, `indoor_outdoor`
- **day**: `day`, `date`, `dates`, `event_date`
- **duration**: `duration`, `end_date`, `time`

---

## Critical Config

| Constant | Value | Notes |
|----------|-------|-------|
| **Osaurs port** | **1337** | Check: `osaurus status` |

---

## Pre-Generated Baseline Data (2026)

**Approach**: Task is "extract from provided JSON context" not "generate events".

- Test data in `_EVAL_TEST_INPUTS` (config.py)
- Pre-generated JSON with proper structure
- Models score on accurate extraction, not generation
- Consistent baseline across runs
- Avoids "refuses to generate fictional events" problem

### Test Data Locations:
- `config.py` lines 316-355: `_EVAL_TEST_INPUTS` dict

---

## Known Issues (Additional)

| Model | Issue | Status |
|-------|-------|--------|
| lfm2-24b | Crashes server (OOM), 30m timeout | AVOID |
| gemma weekend | Refuses fictional event data; outputs weather or questions | WONTFIX |
| qwen filename | Empty response with complex prompts | FIXED (simpler prompt) |

---

## Strict Validation Rules (Updated 2026)

### Extraction Validation
- **>80% from source**: Required for passing
- **No hallucinated items**: Items must match input data
- **Completeness**: All input items should be in output

### file_summary Validator
- **No filename inference**: "a python script" = FAIL
- **Must have content verbs**: parse, validate, extract, load, read, write, etc.
- **Filename appearing in summary** = FAIL

---

## Model-Specific Prompts

All prompts in `conf/models/{model}.yaml` must include:

```yaml
# Required for JSON output
weekend_fixed: |
  Output JSON now. CRITICAL: Use EXACT schema: {schema}
  
  REQUIRED fields for EACH item:
  - name: str
  - location: str
  ...

  Output ONLY JSON. No extra text.

filename: |
  Output JSON now. Schema: {"filename": "str"}
  Output ONLY JSON.

summarize: |
  Output the summary in bullet points. Use ## headers.
```

---

## Model Quirks

### Foundation ✅ WORKS RELIABLY
- **Fast**: 8-15s for tasks
- **Clean JSON**: No markdown, no thinking
- **Source matching**: 100% (risky - may copy directly from input)
- **Synthesis weakness**: Scores only 52-58% on synthesis (no connecting narrative, no TL;DR). Despite unified summarization prompt, it still lists events without relationship language
- **Filename**: 97% quality, 0.6s avg — best speed-to-quality ratio

### Laguna-xs.2-mxfp4 ✅ BEST BALANCE
- **Emerges as top pick** from full quality eval (May 2026)
- **Filename**: 98% quality, 3.1s avg
- **Summarize**: 92% quality — best Synthesis of non-qwopus models
- **No failures**: 0 crashes across all 8 test cases
- **Note**: Relatively unknown model but beats qwen and nemotron on consistency

### Nemotron-3-nano-omni ⚠️ INSTRUCTION LEAK
- **Instruction leak**: Often outputs `"Here is the filename: ..."` instead of the filename alone. Score drops to 50-74% on affected cases
- **Filename**: 84% avg (dragged down by leak), 3.3s avg
- **Summarize**: 89.5% — similar Synthesis weakness to foundation
- **Potential fix**: Prompt may need stricter instruction ("Output ONLY the filename, no explanation")

### Qwen Family
- **Requires**: "Output JSON now" trigger (for weekend tasks)
- **Thinking**: Plaintext blocks - handled by stripping
- **Key quirks**: Uses `category` → `target_ages`
- **qwen3.6-27b-mxfp8-mtp** ✅ best qwen: 99% filename, 100% summarize, 0 failures, 14.8s avg
- **qwen3.6-27b-mxfp4**: 93.8% filename, 100% summarize, 12.3s avg
- **qwen3.6-35b-a3b-mxfp4**: 93.8% filename, 94% summarize, 10.1s avg — good but not better than 27b variants
- **qwen3.6-35b-a3b-mxfp8-mtp** ❌ BROKEN: Consistently crashes on summarize and file_summary (returns empty). Only filename works (93.8%). MoE mixture-of-experts issue?

### Qwopus ⚠️ HIGH QUALITY BUT UNRELIABLE
- **Best quality when it works**: 98.2% filename, 98.5% summarize
- **Only model with good synthesis (94%)**: Adds rich connecting narrative
- **BUT 40% failure rate on cold start**: Produces empty output randomly
- **Very slow**: 40-220s per call
- **Inconsistent**: Same model, same prompt, same case scored 96.2% in one run, 0% in another
- **Recommendation**: Only use for quality-critical batch work where failures are acceptable

### Gemma ❌ NOT SUITABLE FOR WEEKEND
- Returns weather data instead of events
- 0 items with details in tests
- Flat dicts instead of nested structure
- **gemma-4-e4b-it-4bit/8bit**: All tasks return empty — MLX backend may not support these model formats

### Minimax-m2.7-small-jangtq ❌ UNUSABLE
- **Extremely slow**: 400s+ per single filename call
- **Generic outputs**: 3/5 filename cases return "filename.txt" or similar
- **Complex tasks**: 100% failure on summarize and file_summary
- **Conclusion**: Not suitable for any ztools use case - remove from model list

---

## Signal/Noise Filtering (Mixed Eval)

Each eval task has a `*_mixed` variant that injects clearly-labeled NOISE into the
input and measures whether the model extracts ONLY the signal. Scores are
precision (noise excluded) + recall (signal kept). A model that dumps noise scores
lower, and the leakage is named explicitly in the failure reason
(`included N/M noise items` / `included N/M noise files`).

Sweep (2026-07, all models via `osaurus list`):

| Model | weekend_transient | weekend_fixed | summarize | file_summary | rename | noise leaked? |
|-------|-------------------|---------------|-----------|--------------|--------|---------------|
| foundation | 95 | 91 | 88 (1) | 100 | 100 | minor (1 tweet) |
| qwen3.6-27b-mxfp8-mtp | 100 | 91 | 75 (2) | 100 | 100 | moderate (2 tweets) |
| qwen3.6-35b-a3b-mxfp8-mtp | 100 | 91 | 88 (1) | 91 (6/6 files) | 100 | **dumps all noise files** |
| gemma-4-12b-it-mxfp8 | 100 | 91 | 88 (1) | 100 | 100 | minor (1 tweet) |
| gemma-4-e4b-it-8bit | 100 | 91 | 75 (2) | 100 | 100 | moderate (2 tweets) |
| diffusiongemma-26b-a4b-it-mxfp8 | 100 | 0 (failed) | 88 (1) | 100 | 100 | minor (1 tweet) |
| ornith-1.0-35b-mxfp8 | 100 | 91 | 50 (4) | 91 (6/6 files) | 100 | **worst**: 4 tweets + all noise files |
| qwen-agentworld-35b-a3b-mxfp8 | 100 | 91 | 88 (1) | 25 (missed all) | 100 | barely summarized |

Notes:
- **Weekend "missed 2/12" is a recall artifact, not noise leakage.** Every weekend
  task is noise-clean (precision 100%); the model rephrases ~2 item names so the
  fuzzy name-matcher misses them. Noise exclusion itself works (no model scored
  noise on weekend).
- **summarize_mixed** is the most discriminating: models leak 1-4 of 8 noise tweets.
  ornith (4/8 → 50%) and qwen27b/gemma-e4b (2/8 → 75%) are the worst; foundation /
  gemma-12b / qwen35b / diffusiongemma / qwen-agentworld leak only 1 (88%).
- **file_summary_mixed**: qwen35b and ornith INCLUDE all 6 noise files in output
  (flagged "included 6/6 noise files"); qwen-agentworld barely summarized (25%).
  All others filter clean (100%).
- **rename_mixed**: every model filters clean (100%) — the JSON-array format with
  explicit SNIPPETS/NOISE sections is unambiguous.
- The single 0-100 pass/fail number is no longer the signal — read the
  `included N/M noise` failure reason to see actual filtering quality.

---

## Eval Commands

```bash
# Quick single model eval
python3 -m eval --model foundation --quick --task filename

# Full eval
python3 -m eval

# Quick alias via shim
python3 model_eval.py --model foundation --task filename --quick

# Quality benchmark
python3 -m eval.benchmark_quality
```

---

## Key Files

- `eval/cli.py` - Eval runner CLI
- `eval/tasks_core.py` - Task definitions
- `eval/run.py` - Eval loop
- `lib/quality_models.py` / `lib/quality_scorers.py` - Quality evaluation
- `lib/validators/` - Validator implementations
- `conf/models/*.yaml` - Model prompts

---

## Filename / Rename Task

Config-driven via `conf/config.yaml`:
```yaml
filename_models: [laguna-xs.2-mxfp4, foundation]
prompts:
  filename: "Output ONLY the filename string (no JSON, no code blocks).
  Use lowercase, underscores for spaces, no special characters.
  Keep it under 50 characters. TEXT: {text}"
```

MLX backend: OsaurusAI custom quant (MXFP4/JANGTQ) only loadable via `osaurus serve`. Standard mlx-lm supports qwen3_5, gemma4 architectures. Model discovery (`find_any_working_mlx_model`) scans all dirs and filters incompatible architectures.