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
  - `setting` instead of `weather`
  - `year_round_activities` instead of `fixed_activities`
  - `events` / `limited_time_events` instead of `transient_events`
- **Production**: weekend_planner.py normalizes all these variations
- **Solution**: Validator uses flexible key matching (has_item_details)

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

### Quality Scoring (2026-04-19 Update)
The eval now measures **source extraction quality** - whether models extract from provided input vs hallucinating:

- **JSON_SOURCE_WEIGHT = 25** - Points for extracting from input
- **DETAILED_SOURCE_WEIGHT = 30** - Same for detailed JSON tasks

This catches models that make up generic names vs using real venue names from web search results.

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

---

## Weekend Planner Model Quality

### Test Method
Same venue input to ALL models, then judge:
1. Does it output valid JSON schema?
2. Does it use the INPUT data (not hallucinate)?
3. Does it apply GOOD judgement (indoor for rainy Saturday)?

### Venue Input (for all models):
```
- Fun Zone (Toronto): all ages, $20, indoor
- Science Centre: ages 5-12, $25, science museum
- Legoland (Vaughan): ages 3-12, $30, indoor lego
- Riverdale Farm: all ages, $10, farm
- Ripley's Aquarium: all ages, $40, aquarium
- Ontario Science Centre: kids 5+, $25, hands-on science
```

Context: Saturday = Rain (needs indoor)

### Results

| Model | Outputs | Schema | Uses Input | Judgement |
|-------|---------|--------|------------|-----------|
| **foundation** | 10 | ✅ | ✅ (Science Centre, Legoland) | ✅ All indoor |
| **gemma-4-26b** | 0 | ❌ | ❌ | ❌ |
| **qwen3.6** | via MLX | ? | ❌ (hallucinates) | ❌ |

### Conclusion
**foundation** is the ONLY model that works for weekend_planner:
- Returns correct JSON schema
- Uses provided venues (no hallucinations)
- Applies good judgement (indoor activities for rainy day)

This is NOT about eval benchmark scores - it's about PRODUCTION quality for weekend_planner's specific task.

### Fresh Run Comparison (Same Web Data)

| Model | Time | Fixed Quality | Notes |
|------|------|--------------|-------|
| **qwen3.6** | 2.5m | ✅ **BEST** | Real venue names + prices |
| gemma-4-26b | 12m | ✅ Good | Real venue names (slower) |
| foundation | 1m | ⚠️ Generic | Makes up generic names |
| gemma-4-31b | 1.5m | ❌ Broken | Empty locations () |

### Quality Score Formula
- Priority: Quality 95%, Speed 5%
- Speed only a tie-breaker between equal-quality models

### Correction (2026-04-19)
- Initial test showed gemma extracting "real names" - this was comparison error
- **qwen3.6 is actually BEST**: Same extraction quality as gemma, 5x faster
- Both extract real venues from web search results (Ontario Science Centre, Hockey Hall of Fame, etc.)
- foundation is worse (generic names like "Guide to the Best Indoor Play")

---

## Production Script Recommendations

### weekend_planner (family activity planning)
- **Best model**: foundation
- **Why**: Only model that follows schema + uses input + applies judgement

### twitter_summarizer 
- **Best model**: gemma-4-26b-a4b-it-4bit or gemma-4-31b-it-jang_4m
- **Why**: Handles long context, good summarization

### image_renamer
- **Best model**: foundation
- **Why**: Fast, follows filename schema