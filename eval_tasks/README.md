# eval_tasks - Model Evaluation Framework

## Structure

```
eval_tasks/
├── __init__.py     # Task definitions & prompts
├── validators.py   # Validator wrappers (imports from lib.validators_lib)
├── analyze.py     # Analysis & reporting functions
├── run.py        # CLI runner
└── README.md     # This file
```

## Adding a New Task

1. Define prompts in `__init__.py`:
```python
NEW_SYS = "You are a helpful assistant for {topic}."
NEW_USR = "Your question about {topic}."

TASKS["new_task"] = {
    "messages": [
        {"role": "system", "content": NEW_SYS},
        {"role": "user", "content": NEW_USR},
    ],
    "validator": validate_detailed_json,
    "parse_json": True,
    "source": NEW_USR,  # For source matching validation
}
```

2. Validator choices:
- `validate_detailed_json` - For structured JSON with details + source matching
- `validate_filename` - For filename generation
- `validate_file_summary` - For file summarization (strict: no filename inference)
- `validate_summary` - For general text summaries

## Model Prompts

Prompts come from `conf/models/{model}.yaml`. Key tasks:
- `weekend_fixed` / `weekend_transient` - Must output JSON array with 8-10 items
- `filename` - Must output JSON: {"filename": "str"}
- `summarize` - Bullet points with ## headers

## Strict Validation Rules

1. **file_summary**: No filename inference, must describe actual file content
2. **validate_detailed_json**: 8+ items required, no duplicates, source matching checked
3. **Prompts**: Must include "Output JSON now" + exact schema

## Quick Test Commands

```bash
# Test all imports
cd /Users/ztomer/Projects/ztools
python3 -c "from model_eval import run_eval, TASKS; print('model_eval OK')"
python3 -c "from eval_tasks import TASKS, load_tasks_from_config; print('eval_tasks OK')"

# Test a model (requires osaurus running)
python3 -c "
from model_eval import run_eval, TASKS
results = run_eval('foundation', TASKS)
for r in results:
    print(r['task'], r['quality_score'])
"

# Run via module
python3 -m eval_tasks.run --model qwen --task weekend

# Add new task example: see __init__.py TASKS dict
```

## Adding a New Task

1. Define prompts in `__init__.py`:
```python
NEW_SYS = "You are a helpful assistant for {topic}."
NEW_USR = "Your question about {topic}."

TASKS["new_task"] = {
    "messages": [
        {"role": "system", "content": NEW_SYS},
        {"role": "user", "content": NEW_USR},
    ],
    "validator": validate_detailed_json,  # or validate_summary, validate_filename
    "parse_json": True,  # True for JSON output, False for text
    "source": NEW_USR,  # For source matching validation (optional)
}
```

2. Validator choices:
- `validate_detailed_json` - For structured JSON output (weekend tasks)
- `validate_filename` - For filename generation
- `validate_file_summary` - For file summarization
- `validate_summary` - For general text summaries

3. Set validators in `load_tasks_from_config()` if needed

## Testing Changes

```bash
# Quick test of model_eval imports
python3 -c "from model_eval import run_eval, TASKS; print('OK')"

# Quick test eval_tasks imports  
python3 -c "from eval_tasks import TASKS, load_tasks_from_config; print('OK')"

# Run eval on a model
python3 -c "
from model_eval import run_eval, TASKS
results = run_eval('qwen', TASKS)
for r in results:
    print(r['task'], r['quality_score'])
"
```

## Key Files

- `model_eval.py` - Main eval runner (1695 lines)
- `lib/validators_lib.py` - Core validators
- `conf/models/*.yaml` - Model prompts

## Common Issues

- ImportError: Check validators are in lib.validators_lib
- "no items found": Model didn't return JSON or parse failed
- Duplicated output: Check for duplicate print in run_eval and main