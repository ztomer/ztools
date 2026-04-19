@MODEL_QUIRKS.md

# Project-Specific Instructions

## Mandatory Startup Read
- **Read MODEL_QUIRKS.md before any model eval work**

## Key Rules

### Model Evals
- Use quick mode for iteration: `--quick --task <task>`
- Add discovered learnings to MODEL_QUIRKS.md immediately when found
- Run: `python3 model_eval.py --model <model> --task <task> --quick`

### Prompt Engineering
- Always prepend "Output JSON now." for qwen3.6 to prevent thinking
- Test changes with quick single-task eval before full run

### MLX Backend
- Currently not working - subprocess returns empty
- Document in MODEL_QUIRKS.md when debugging