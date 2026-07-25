# Project-Specific Instructions

Reference docs — read these when the task touches their subject, not by default:
- `docs/MODEL_QUIRKS.md` — model selection, per-model prompt quirks, eval results
- `docs/TESTING.md` — test patterns, mock infrastructure, coverage rules

## Key Rules

### File Size Limit
No file may exceed 500 lines. Split into a package (`twitter/`, `weekend/`, `rename/`,
`eval/`) or extract a module; `lib/config.py`, `lib/quality.py`, and `lib/osaurus_lib.py`
are shims re-exporting their split submodules. Check with `wc -l` before adding to a file
that is already close.

### Testing
- Every test must have a non-tautological assertion
- Use the real scorer/validator when possible; mock only the LLM layer
- Test the real `__main__` block via `exec` of the real source — never re-implement it in the test
- No test may launch a real browser or read real browser cookies; `tests/conftest.py`
  enforces this. Opt out with `@pytest.mark.real_cookie_discovery` only to test discovery itself.
- Prove a new test can fail before trusting it green
- Run: `./.venv/bin/python -m pytest tests/ -q`
  (OCR tests need `--ignore=tests/test_img_helpers.py --ignore=tests/test_image_renamer.py`
  under `--cov` — numpy's C extension crashes with coverage tracing)
- Add discovered test patterns or bugs to `docs/TESTING.md` immediately

### Model Evals
- Quick mode for iteration: `python3 -m eval --model <model> --task <task> --quick`
- Add discovered learnings to `docs/MODEL_QUIRKS.md` immediately when found

### Prompt Engineering
- Always prepend "Output JSON now." for qwen3.6 to prevent thinking
- Test changes with a quick single-task eval before a full run
