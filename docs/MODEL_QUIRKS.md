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
| **weekend_transient** | foundation | Fastest (8s), 100%, clean JSON |
| **weekend_fixed** | foundation | 100%, reliable |
| **summarize** | foundation | Fast, clean ## headers |
| **filename** | foundation | Fast, follows schema |
| **file_summary** | foundation | 44% - correctly detects filename inference via code-pattern check |
| **vlm** | gemma-4-26b-a4b-it-mxfp4 | Vision tasks only |
| **qwen3.6-35b** | qwen3.6-35b-a3b-mxfp4 | Good alternative to foundation |
| **qwen3.6-27b** | qwen3.6-27b-mxfp4 | Same quality, slower |
| **qwen3.6-27b-mxfp8** | ❌ DO NOT USE | Causes server hang/timeout |

---

## Known Issues

### Gemma Weather Bug
All gemma models output weather data instead of events for transient tasks:
- gemma-4-26b-a4b-it-mxfp4: Returns `{"date": "April 25", "temperature": "12°C"}` instead of events
- gemma-4-31b-it-jang_4m: Same issue
- gemma-4-e2b-it-8bit: Same issue
- **Not fixable via prompts** - model behavior issue

### Performance Notes
- Oosaurus server should run on port 1337 (not 8080)
- gemma models return different JSON structures per version
- Use qwen for weekend tasks only

---

## Model Quirks Detected

### Qwen 3.6 (qwen3.6-35b-a3b-mxfp4 and qwen3.6-27b-mxfp4) ✅ WORKING
- **Thinking**: Plaintext thinking with "Here's a thinking process:" markers
- **Stats tokens**: Trailing "stats:2114;97.2952" in output
- **Required prefix**: "Output JSON now." to prevent thinking blocks
- **Key quirks**: Uses `category` → `target_ages`, `context_highlight` → `price`
- **Extraction keys**: fixed_activities, venues, activities, items
- **SUCCESS**: Both fixed and transient work reliably
- **Markdown**: Adds `**` to names (e.g., `**Friday, April 20**`) - cosmetic, handled by libs
- **27b**: Slower (200s+) but same quality - use if you have time
- **Schema helps**: `schema_strict` prompts improve detail extraction

### Gemma 4 Family ❌ NOT SUITABLE FOR WEEKEND TASKS
- **Key name**: Returns `activity` instead of `name`
- **Location**: Returns `venue` instead of `location`  
- **Output structure**: Generates weather forecast instead of events
- **Not suitable**: Generates weather data instead of events
- **Fixed activities work**: Uses `activity` key via field_mapping
- **Flat dicts**: Outputs `[{"Location": "Park"}, {"Ages": "All"}]` instead of proper nested structure
- **Many items**: Returns 19-35 garbage items but 0 with real details
- **Explorer test**: `no_preamble` → 19 items/0 details, `schema_strict` → 35 items/0 details
- **Root cause**: Model doesn't follow JSON schema instructions - returns conversational text as items
- **ALL variants broken**: 26b, 31b, 8bit all fail same way
- **File summary**: 70% - produces ## headers but fails content line validation

### Foundation ✅ WORKS RELIABLY
- **Fast**: 8-15s for tasks
- **Clean JSON**: No markdown, no thinking blocks  
- **Source matching**: 100% match ratio
- **Perfect**: all items with details in tests
- **File summary**: 44% - uses filename inference, fails code-pattern validation

---

## File Summary Validation

The `validate_file_summary()` function detects filename inference vs actual file reading:

| Check | Points | Detection |
|-------|--------|-----------|
| `## Headers` | 20 | Structure compliance |
| `Length >= 500` | 20 | Effort indicator |
| `Python code patterns` | 20 | `.py` files: `def `, `class `, `import ` |
| `Markdown patterns` | 12 | `.md` files: headers, lists, links |
| `YAML patterns` | 3 | `.yaml` files: key-value syntax |
| `Line variance` | 8 | Variety in summary lengths |

**File-type specific**: Validation weights differ by file type. Python files get 20pts for code patterns, markdown gets 12pts for doc patterns, YAML gets 3pts.

**Score breakdown** (foundation 44%):
- 20 (headers) + 20 (length) + 8 (low variance) = 48 → capped at 44
- No code patterns matched ("A Python script for..." vs "def plan_weekend()")
- No markdown details for README/CLAUDE.md

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
