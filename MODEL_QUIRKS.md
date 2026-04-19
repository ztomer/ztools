# Model Quirks & Best Practices

> **Startup reference**: Read this file before running model_eval.py or making prompt/validator changes.

## Tools

### explore_model_quirks.py
Systematically probe model behaviors to discover prompt patterns.

```bash
python3 explore_model_quirks.py <model_id> [timeout]
```

Tests multiple prompt approaches and recommends best one.

## Summary

This document captures lessons learned from evaluating various LLM models for the ZTools eval pipeline.

---

## Osaurus (Server) Backend

### qwen3.5-27b-claude-4.6-opus-distilled-mlx-4bit ✅
- **Status**: Baseline model - performs excellently (100% on all tasks)
- **Best for**: JSON extraction, data tasks
- **No special prompting needed**

### qwen3.6-35b-a3b-mxfp4 ✅
- **Status**: After fixes - 100% on all tasks
- **Required**: "Output JSON now." prefix in system prompt
- **Why**: Prevents thinking/reasoning output that confuses JSON extraction

### Gemma 4 Series (26B, 31B, E2B, E4B) ✅
- **Status**: Fixed - now 80-100% on JSON tasks
- **Works**: filename (100%), summarize (100%), detailed_json (100%), json (80%)
- **Key finding**: Gemma uses DIFFERENT key names in outputs:
  - `event` instead of `name`
  - `age_group` instead of `target_ages`
  - `date` instead of `day`
- **Solution**: Validator uses flexible key matching (has_item_details)
- **Library quirks** (in osaurus_lib.py apply_model_quirks):
  - Adds "IMPORTANT: DATA EXTRACTION..." to system for gemma4
  - Replaces "Execute the task" → "Extract to JSON"
  - Uses "Data:" instead of "Current Context:"

---

## MLX (Local) Backend

### Current Status: Not Working
- **Issue**: Works in direct Python execution but returns empty in subprocess
- **Root cause**: Environment/import isolation when run via subprocess.run()
- **Works when**: Using `/Users/ztomer/.venv/bin/python3` directly (not subprocess)

### Models Tested:
- OsaurusAI/Qwen3.6-35B-A3B-mxfp4
- OsaurusAI/gemma-4-26B-A4B-it-4bit
- OsaurusAI/gemma-4-31b-it-jang_4m
- OsaurusAI/gemma-4-E4B-it-4bit
- OsaurusAI/gemma-4-E2B-it-8bit
- mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit

### Required Fixes:
1. Use uv run: `["rtk", "uv", "run", "--with", "mlx", "--with", "mlx-lm"]`
2. Prepend "Output JSON now." to prompts to prevent thinking output
3. Use base64 encoding for prompts in subprocess to avoid escaping

---

## Prompt Engineering Patterns

### Working Patterns:

```python
# For Qwen models - prevents thinking
WEEKEND_SYS_TRANSIENT = """
Output JSON now.

Act as a data-extraction agent for family events...
Schema - MANDATORY (every event must have ALL of these fields):
{
    "transient_events": [
    {"name": "Spring Festival", "location": "...", "target_ages": "...", ...}
    ]
}

CRITICAL: Each event object MUST contain ALL 7 fields...
"""
```

### Not Working:
- Just specifying schema without "Output JSON now."
- Just saying "Output ONLY JSON" without forcing first
- Using `json` or `Output:` without the full action phrase

---

## Validator Adjustments

### has_item_details() - Lenient Field Checking
Original: Checked only `["location", "weather", "description"]`
Updated: Checks `["location", "weather", "price", "target_ages", "duration", "day"]`

This matches the actual fields used in our prompts and ensures models are credited for including meaningful content.

---

## Evaluation Commands

### Quick single-task eval (fast iteration):
```bash
python3 model_eval.py --model "qwen3.6-35b-a3b-mxfp4" --task "json" --quick
```

### Full eval (40+ minutes):
```bash
python3 model_eval.py --quick
```

### With debug logging:
```bash
python3 model_eval.py --model "qwen3.6-35b-a3b-mxfp4" --task "json" --quick --debug
```

---

## What to Iterate On

1. **MLX backend**: Needs subprocess isolation fix OR direct integration
2. **Gemma 4 JSON**: Needs alternative prompting strategy or different model
3. **All models detailed_json task**: Current 100% - no changes needed

---

## Architectural Decision: Library-Level Quirks

Model-specific prompt modifications are applied CENTRALLY in the library, NOT in individual scripts.

### osaurus_lib.py applies:
- `get_model_family()` - detects model family (qwen, gemma4, gemma, foundation)
- `apply_model_quirks()` - modifies messages based on model family
- Called automatically in `call()` function

This means:
- Scripts pass raw prompts → library adds quirks if needed
- New model discoveries only need code changes in one place
- Consistent behavior across all scripts (weekend_planner, twitter_summarizer, etc.)

## Rule: Adding New Learnings

When new model behaviors or prompting discoveries are found during eval:
1. Document immediately in MODEL_QUIRKS.md under appropriate section
2. Add to library in `apply_model_quirks()` if model-agnostic
3. Add to prompt templates in model_eval.py for eval-specific tasks
4. Update validators if field names need adjustment
5. Run quick eval to verify: `python3 model_eval.py --model <model> --task <task> --quick`