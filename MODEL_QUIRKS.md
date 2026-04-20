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

## Prompt Engineering

### JSON Tasks
Prefix prompts with `"Output JSON now."` to prevent thinking blocks:

```python
WEEKEND_SYS_FIXED = """
Output JSON now. Use EXACT schema: {"fixed_activities": [...]}

Extract 10 venues in Toronto/Vaughan.
Default values:
- target_ages: "6-13 years"
- price: $18-35 per child
- weather: "indoor"
- Never leave fields empty
"""
```

### Summarize Tasks
Use model-specific prompts in `conf/config.yaml`:

```yaml
summarize_prompts:
  qwen3.6: "Output the summary. Use ## headers for topics.\n\n<timeline>\n{}\n</timeline>\n\nSummarize the timeline. Include your analysis."
  foundation: "Output the summary. Use ## headers for topics.\n\n<timeline>\n{}\n</timeline>\n\nSummarize the timeline."
```

---

## Post-Processing

### Key Functions (osaurus_lib.py)

```python
# Extract thinking block from response
thinking, content = extract_thinking(response)

# Merge thinking as ## Analysis section
merged = merge_thinking_with_summary(thinking, content)

# Normalize alternate key names
normalized = normalize_keys(data)
```

### Thinking Blocks
**Preserve thinking** - it contains valuable signal for synthesis and grouping. Use `merge_thinking_with_summary()` to add as ## Analysis section.

---

## Model-Specific Quirks

### Gemma 4 Series
Uses different key names in output:
- `event` → `name`
- `age_group` → `target_ages`
- `date` → `day`
- `setting` → `weather`
- `year_round_activities` → `fixed_activities`

Handled by `normalize_keys()` in osaurus_lib.py.

### Qwen3.6
Needs `"Output JSON now."` prefix for JSON tasks.
Produces thinking blocks in summarize - preserve with post-processing.

### Foundation
Fast, clean output, no thinking blocks.
Good for simple structured tasks.

---

## Evaluation

### Quick Test
```bash
python3 model_eval.py --model <model> --task <task> --quick
```

### Tasks
`json`, `detailed_json`, `summarize`, `filename`

### Scoring
Measures:
- Valid JSON structure
- Field completeness
- Source extraction (uses input vs hallucinating)
- Content quality

---

## Library Architecture

```
lib/
├── osaurus_lib.py      # API calls, post-processing
├── config.py           # Model prompts, task config
├── validators_lib.py    # Evaluation scoring
├── content_processing.py  # Clean output
└── mlx_lib.py         # Local MLX models
```

### Flow
1. Script builds raw prompt
2. Library applies model quirks if needed
3. API returns response
4. Post-processing cleans output
5. Validators score result
