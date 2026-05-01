# Model Quirks & Best Practices

Reference for ZTools prompt engineering and model selection.

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