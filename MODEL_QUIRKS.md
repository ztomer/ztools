# Model Quirks & Best Practices

Reference for ZTools prompt engineering and model selection.

---

## Best Models by Task

| Task | Best Model | Notes |
|------|-----------|-------|
| **json** | qwen3.6-35b-a3b-mxfp4 | 100% on eval |
| **detailed_json** | qwen3.6-35b-a3b-mxfp4 | 100% on eval |
| **summarize** | foundation | Clean ## headers, fast |
| **filename** | foundation | Fast, follows schema |
| **vlm** | gemma-4-26b-a4b-it-4bit | Vision tasks |
| **weekend_planner** | qwen3.6-35b-a3b-mxfp4 | Reliable (9-10 fixed, 5-6 transient) |
| **gemma4-31b** | Avoid | Wrong JSON structure, returns forecast instead of events |
| **MLX backend** | Disabled | Not working reliably |

---

## Model Configuration

Each model has its own config in `conf/models/{family}.yaml`:

```yaml
# qwen.yaml example
name: qwen
timeout: 120
prompts:
  json: "Output JSON now. Use EXACT schema."
  summarize: "Output the summary. Use ## headers..."
  weekend_fixed: "Extract JSON with..."
key_mappings:
  event: name
  age_group: target_ages
quirks:
  - type: prefix
    pattern: "Output JSON now."
```

### Loading
```python
from lib.config import get_model_config, get_model_prompt

config = get_model_config("qwen3.6-35b-a3b-mxfp4")
prompt = get_model_prompt("qwen3.6-35b-a3b-mxfp4", "json")
```

---

## Prompt Engineering

### Data-Driven Approach
All prompts are stored in model YAML files, not hardcoded in scripts.

### JSON Tasks
Use `"Output JSON now."` prefix to prevent thinking blocks.

### Summarize Tasks
Model-specific prompts handle ## headers + thinking differently.

---

## Post-Processing

### Qwen Plaintext Thinking Removal

Qwen 3.6 doesn't use `<think>` XML tags - it outputs thinking in plaintext. The content_processing.py handles:

1. **Markers** (start of thinking): `"Here's a thinking process:"`, `"Thinking Process:"`, `"Let me analyze"`
2. **Output markers** (end of thinking): `"Draft:"`, `"Output Generation:"`, `"I'll now generate:"`, `"Let's draft"`
3. **Self-correction blocks**: `*(Self-Correction during draft)*`, patterns like `(Self-[Cc]orrection...)`
4. **Stats tokens**: `"stats:2114;97.2952"` or similar trailing tokens

### Functions (osaurus_lib.py)

```python
# Model-aware normalization (uses config)
data = extract_json(response, model)
data = normalize_keys(data, model)

# Thinking extraction for XML tags
thinking, content = extract_thinking(response)

# Strip plaintext thinking (qwen-specific)
cleaned = strip_thinking(content)
merged = merge_thinking_with_summary(thinking, cleaned)
```

---

## Evaluation

### Quick Test
```bash
python3 model_eval.py --model <model> --task <task> --quick
```

### Tasks
`json`, `detailed_json`, `summarize`, `filename`

---

## Adding New Models

1. Create `conf/models/{family}.yaml`
2. Add prompts, key_mappings, quirks
3. Done - no code changes needed

---

## Runtime Constants

| Constant | Value | Notes |
|----------|-------|-------|
| **Osaurs port** | **1337** | Check: `osaurus status` |

---

## Model Quirks Detected

### Qwen 3.6 (qwen3.6-35b-a3b-mxfp4)
- **Thinking**: Plaintext thinking with "Here's a thinking process:" markers
- **Stats tokens**: Trailing "stats:2114;97.2952" in output
- **Required prefix**: "Output JSON now." to prevent thinking blocks
- **Key quirks**: Uses `category` → `target_ages`, `context_highlight` → `price`
- **Extraction keys**: fixed_activities, venues, activities, items

### Gemma 4-31b (gemma-4-31b-it-jang_4m)
- **Key name**: Returns `activity` instead of `name`
- **Location**: Returns `venue` instead of `location`
- **Output structure**: Generates weather forecast `{"weekend_forecast": {...}}` instead of events list
- **Items count**: Only generates 1-3 items vs expected 5-10
- **Requires transformation**: weekend_forecast nested structure extraction
- **Not suitable**: Model doesn't follow prompt to generate enough items

---

All scripts must define these required constants at module level:

| Constant | Description | Source |
|----------|-------------|--------|
| `AGE_RANGE` | Child age range "min-max" | Computed from `conf/weekend.yaml` children |
| `DATES_STR` | Date range string | Computed at runtime in main() |
| `CITY`, `REGION` | Location | From `conf/weekend.yaml` |
| `CHILDREN` | Child list | From `conf/weekend.yaml` |

Example (weekend_planner.py):
```python
AGE_RANGE = f"{min(c['age'] for c in CHILDREN)}-{max(c['age'] for c in CHILDREN)}" if CHILDREN else "4-12"
DATES_STR = "April 24 to April 26"  # Placeholder - actual in main()
```

---

## Library Architecture

```
conf/models/
├── qwen.yaml
├── gemma.yaml
├── foundation.yaml

lib/
├── osaurus_lib.py      # API calls, normalize_keys(model=)
├── config.py            # get_model_config(), get_model_prompt()
├── validators_lib.py
├── content_processing.py
└── mlx_lib.py
```
