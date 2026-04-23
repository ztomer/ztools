# Model Quirks & Best Practices

Reference for ZTools prompt engineering and model selection.

---

## Critical Config

| Constant | Value | Notes |
|----------|-------|-------|
| **Osaurs port** | **1337** | Check: `osaurus status` |

---

## Best Models by Task

| Task | Best Model | Notes |
|------|-----------|-------|
| **json** | qwen3.6-35b-a3b-mxfp4 | 100% on eval |
| **detailed_json** | qwen3.6-35b-a3b-mxfp4 | 100% on eval |
| **summarize** | foundation | Clean ## headers, fast |
| **filename** | foundation | Fast, follows schema |
| **vlm** | gemma-4-26b-a4b-it-4bit | Vision tasks |
| **weekend_fixed** | qwen3.6-35b-a3b-mxfp4 | Works (10 items) |
| **weekend_transient** | qwen3.6-35b-a3b-mxfp4 | WORKS (6-8 items) |
| **MLX backend** | Disabled | Not working reliably |

---

## Model Quirks Detected

### Qwen 3.6 (qwen3.6-35b-a3b-mxfp4 and qwen3.6-27b-mxfp4) ✅ WORKING
- **Thinking**: Plaintext thinking with "Here's a thinking process:" markers
- **Stats tokens**: Trailing "stats:2114;97.2952" in output
- **Required prefix**: "Output JSON now." to prevent thinking blocks
- **Key quirks**: Uses `category` → `target_ages`, `context_highlight` → `price`
- **Extraction keys**: fixed_activities, venues, activities, items
- **SUCCESS**: Both fixed and transient work reliably

### Gemma 4 Family ❌ NOT SUITABLE FOR WEEKEND TASKS
- **Key name**: Returns `activity` instead of `name`
- **Location**: Returns `venue` instead of `location`  
- **Output structure**: Generates weather forecast instead of events
- **Not suitable**: Generates weather data instead of events
- **Fixed activities work**: Uses `activity` key via field_mapping

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
