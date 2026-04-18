# ZTools - Final Review - COMPLETED

## Code Quality Issues (Uncle Bob) - FIXED

| Issue | Fix |
|-------|-----|
| Duplicated get_best_model() | Using library |
| Hardcoded BEST_MODELS | Config.yaml loading |
| Bare except: | Specific exceptions |
| Config not loaded | _load_config() added |

## Code Correctness Issues (Linus) - FIXED

| Issue | Fix |
|-------|-----|
| Missing import requests | Added to generate_weekend |
| Wrong import call_model | Fixed to call_with_prompt |
| Empty responses causing crashes | Null checks added |
| No markdown cleaning before parse | Fixed order |

## JSON Extraction - IMPROVED

| Fix | Impact |
|-----|--------|
| Clean markdown BEFORE finding braces | Works with \`\`\`json wrappers |
| Extract from plain text | Handles "1. Item" format |
| Wrapper key detection | Handles {fixed_activities: [...]} |

## Model Behavior vs Code

**DISCOVERED**: foundation outputs JSON when asked. Gemma/Qwen ignore JSON instructions and output conversational text.

```
foundation -> ✓ Valid: {"activities": [...]}
gemma -> ✗ Plain text: "Here are 5 options..."
qwen -> ✗ Thinking: "Let me think..."
```

This is NOT a code bug - it's model behavior. foundation is specifically trained for JSON output.

## Eval Results (Final)

| Model | json_simple | json_medium | Reason |
|-------|------------|-------------|--------|
| foundation | 99% | 100% | Follows JSON instructions |
| gemma-4-26b | 99% | 0% | Ignores JSON instructions |
| gemma-4-31b | 99% | 0% | Ignores JSON instructions |
| gemma-4-e4b | 99% | 0% | Ignores JSON instructions |
| qwen3.5 | 100% | 33% | Slower but sometimes works |

## Recommendation

For JSON tasks: Use foundation (or train/tune models differently)
For thinking tasks: Use gemma/qwen (they're more capable but need different handling)

## Files

```
ztools/
├── osaurus_lib.py      # Generic LLM utilities
├── config.yaml         # OVERRIDES hardcoded defaults  
├── model_eval.py       # Evaluator
├── weekend.yaml        # Tool config
├── twitter.yaml        # Tool config
├── rename.yaml         # Tool config
├── REVIEW.md           # This file
```