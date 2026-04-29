# Model Quirks & Best Practices

Reference for ZTools prompt engineering and model selection.

---

## Critical Config

| Constant | Value | Notes |
|----------|-------|-------|
| **Osaurs port** | **1337** | Check: `osaurus status` |

---

## Best Models by Task (2025)

| Task | Best Model | Score | Notes |
|------|-----------|-------|-------|
| **weekend_transient** | foundation | 100% | 8s, clean JSON |
| **weekend_fixed** | foundation | 100% | reliable |
| **summarize** | foundation | 100% | clean ## headers |
| **filename** | foundation | 100% | follows schema |
| **file_summary** | gemma | 70% | produces headers but generic content |

---

## Strict Validation Rules (Updated 2025)

### file_summary Validator
- **No filename inference**: "a python script" = FAIL
- **Must have content verbs**: parse, validate, extract, load, read, write, etc.
- **Filename appearing in summary** = FAIL

### validate_detailed_json Validator
- **8+ items required** (was 3)
- **No duplicates**: duplicate names penalized
- **All items must have details**
- **Source matching critical**: >=80% from input = bonus, <50% = FAIL

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