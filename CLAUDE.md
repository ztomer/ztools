@docs/MODEL_QUIRKS.md
@docs/TESTING.md

# Project-Specific Instructions

## Key Rules

### File Size Limit
- No file may exceed 500 lines of code. Enforced by split modules and shim files. Pre-existing shims:
  - `lib/config.py` → `lib/config_core.py`, `lib/config_getters.py`, `lib/config_tasks.py`
  - `lib/quality.py` → `lib/quality_models.py`, `lib/quality_scorers.py`, `lib/quality_runner.py`, `lib/quality_report.py`, `lib/quality_entry.py`
  - `lib/osaurus_lib.py` → `lib/llm/` submodules + shim
  - `image_renamer.py` → `img_helpers.py`, `img_llm.py`, `img_renamer.py`
  - `twitter_summarizer.py` → `twit_cookies.py`, `twit_browser.py`, `twit_summarize.py`, `twit_output.py`, `twit_main.py`
  - `weekend_planner.py` → `weekend_config.py`, `weekend_data.py`, `weekend_prompts.py`, `weekend_llm.py`, `weekend_output.py`, `weekend_main.py`
   - `model_eval.py` → `eval_tasks.py`, `eval_run.py`, `eval_report.py`, `eval_failures.py`, `eval_validate.py`, `eval_main.py`
   - `eval/tasks_core.py` → `eval/tasks_prompts.py` + shim

### Model Evals
- Use quick mode for iteration: `--quick --task <task>`
- Add discovered learnings to docs/MODEL_QUIRKS.md immediately when found
- Run: `python3 model_eval.py --model <model> --task <task> --quick`

### Prompt Engineering
- Always prepend "Output JSON now." for qwen3.6 to prevent thinking
- Test changes with quick single-task eval before full run

### MLX Backend
- Currently not working - subprocess returns empty
- Document in docs/MODEL_QUIRKS.md when debugging

### Testing
- See `docs/TESTING.md` for patterns, mock infrastructure, and rules
- Every test must have a non-tautological assertion
- Use the real scorer/validator when possible; mock only the LLM layer
- Test the real `__main__` block via `exec` of the real source — never re-implement it in the test
- Add discovered test patterns or bugs to `docs/TESTING.md` immediately