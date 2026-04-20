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

### Functions (osaurus_lib.py)

```python
# Model-aware normalization (uses config)
data = extract_json(response, model)
data = normalize_keys(data, model)

# Thinking preservation
thinking, content = extract_thinking(response)
merged = merge_thinking_with_summary(thinking, content)
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
