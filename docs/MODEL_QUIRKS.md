# Model Quirks & Best Practices

**Updated: May 2025** - Canonical reference for ztools model selection and prompts.

---

## TL;DR Cheat Sheet

```bash
# Kill stale servers
pkill -f "osaurus" && sleep 2

# Start fresh
osaurus serve &>/dev/null & sleep 10
```

---

## Best Models by Task

| Task | Model | Speed | Command |
|------|-------|-------|---------|
| weekend (fixed/transient) | qwen3.6-35b-a3b-mxfp4 | ~100s | model_eval.py --task weekend |
| **image filename** | **nemotron-3-nano-omni-30b** | **7-10s** | image_renamer.py |
| summarize | qwen3.6-35b-a3b-mxfp4 | ~30s | - |

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

- `conf/config.yaml` - Best models, timeouts, prompts
- `conf/models/*.yaml` - Per-model settings
- `lib/config.py` - Load functions

---

## Legacy Docs

Archived in `docs/*.ARCHIVE.md`:
- PROJECT_MEMORY.ARCHIVE.md - Weekend planner history
- CODE_REVIEW*.ARCHIVE.md - Old refactoring notes  
- debug-playbook.md - Legacy debug tips

| Model | Speed | Quality | Notes |
|-------|-------|---------|-------|
| **nemotron-3-nano-omni-30b-a3b-mxfp4** | 7-10s | **Best** | Fast + accurate summaries |
| gemma-4-31b-it-jang_4m | 20-35s | Good | Slower, sometimes verbose |
| foundation | 0.3-0.7s | Generic | Fast but meaningless |

### Example Outputs (weight loss tips):
- nemotron: `common_weight_loss_errors`
- gemma: `avoid_weight_loss_mistakes`  
- foundation: `weightlossmistakes`

### Prompt Evolution:
1. `Filename: {text} (one word)` → Too short
2. `Give a short 2-4 word summary of: {text}` → **Winner**
3. Max 4-6 words, limit to ~35 chars

---

## Response Parsing Bug Found

**CRITICAL**: Osaurus server streams tokens. Previous code only read first chunk!

```python
# BROKEN - only gets first chunk
content = j.get("message", {}).get("content", "")

# FIXED - accumulate until done=true
for line in resp.text.split("\n"):
    j = json.loads(line)
    content += j.get("message", {}).get("content", "")
    if j.get("done", False):
        break
```

---

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

Key: `{location}`, `{age_range}`, `{date_range}` are INJECTED at runtime (weekend_planner.py line ~305), NOT {} placeholders.

---

## Field Normalization (Critical)

Different models output different field names. **All normalization must be in `normalize_llm_items()` in weekend_planner.py** - do not scatter it across the code.

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

## Best Models by Task (2026)

| Task | Best Model | Score | Notes |
|------|-----------|-------|-------|
| **weekend_transient** | foundation, qwen | 100% | extraction from pre-generated data |
| **weekend_fixed** | foundation | 100% | extraction from pre-generated data |
| **summarize** | foundation | 100% | clean ## headers |
| **filename** | foundation, qwen | 100% | simpler prompt |
| **file_summary** | foundation, qwen | 75% | prose format works |

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

## Known Issues (2026)

### lfm2-24b (AVOID - Crashes Server)
- Times out after 1800s (30 min)
- Crashes osaurus server (OOM)
- All tasks fail with INFRA after crash
- **DO NOT USE** - Remove from eval

### qwen filename (FIXED)
- Empty response with complex prompt
- Fix: Use simpler prompt without "Output JSON now"

### gemma weekend tasks (WONT FIX)
- Gemma refuses to generate fictional event data
- Outputs clarification questions instead
- Works on other tasks (filename, summarize)
- Use foundation/qwen for weekend tasks

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

## Known Issues

### Gemma Weather Bug
All gemma models output weather data instead of events for transient tasks:
- gemma-4-26b-a4b-it-mxfp4: Returns `{"date": "April 25", "temperature": "12°C"}`
- **Root cause**: Model doesn't follow schema - generates conversational text

### Qwen Filename Empty
qwen3.6 models return empty for filename task:
- **Fix**: Added "Output JSON now" trigger to prompt

---

## Model Quirks

### Foundation ✅ WORKS RELIABLY
- **Fast**: 8-15s for tasks
- **Clean JSON**: No markdown, no thinking
- **Source matching**: 100% (risky - may copy directly from input)

### Qwen Family
- **Requires**: "Output JSON now" trigger
- **Thinking**: Plaintext blocks - handled by stripping
- **Key quirks**: Uses `category` → `target_ages`

### Gemma ❌ NOT SUITABLE FOR WEEKEND
- Returns weather data instead of events
- 0 items with details in tests
- Flat dicts instead of nested structure

---

## Eval Commands

```bash
# Quick single model
python3 -m eval_tasks --model foundation --quick

# Full eval
python3 model_eval.py

# Test imports
python3 -c "from model_eval import run_eval, TASKS; print('OK')"
```

---

## Key Files

- `model_eval.py` - Main eval runner
- `eval_tasks/` - Task definitions
- `lib/validators_lib.py` - Core validators  
- `conf/models/*.yaml` - Model prompts

---

## IMAGE RENAMER DOCS (May 2025)

### Config-Driven Approach

In `conf/config.yaml`:
```yaml
filename_models:
  - nemotron-3-nano-omni-30b-a3b-mxfp4  # Fast + Accurate
  - gemma-4-31b-it-jang_4m  # Slower but good
  - foundation  # Fallback

prompts:
  filename: "Give a short 2-4 word summary of: {text}"
```

Load via lib/config.py:
```python
from lib.config import get_filename_models, get_filename_prompt

FILENAME_MODELS = get_filename_models()
PROMPT_TEXT_TO_FILENAME = get_filename_prompt()
```

### MLX Direct - Known Issues

- **JANG models** (gemma-4-31b-it-jang_4m): Need `jang` package
- **gemma-4-e4b-it-4bit**: Repeats input in loop
- **Qwen**: Can't disable thinking tokens

### Refactoring Summary (812 → 676 lines)

Removed:
- Duplicate `query_mlx_for_filename()` function
- Dead/commented code blocks (~20 lines)  
- Redundant wrapper functions (~50 lines)
- Hardcoded prompts → moved to config

### Test Results (nemotron)

```
10_massive_weight_loss_mistakes_to_avoid... -> common_weight_loss_mistakes
10_powerful_sentences_by_scott_adams... -> resilience_systems_growth
scott_adams -> witty_corporate_satire
business lessons -> entrepreneurial_wisdom
```