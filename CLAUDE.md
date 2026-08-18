# Project-Specific Instructions

Reference docs — read these when the task touches their subject, not by default:
- `docs/MODEL_QUIRKS.md` — model selection, per-model prompt quirks, eval results
- `docs/TESTING.md` — test patterns, mock infrastructure, coverage rules

## Key Rules

### File Size Limit
No production file may exceed 500 lines. Split into a package (`twitter/`, `weekend/`,
`rename/`, `eval/`) or extract a module; `lib/config.py`, `lib/quality.py`,
`lib/osaurus_lib.py`, `lib/validators/text_validator.py`, `lib/quality_scorers.py`,
`eval/report.py` and `eval/cli.py` are shims re-exporting their split submodules.
Check with `wc -l` before adding to a file that is already close.

Test modules under `references/tests/` are **exempt** — they are lists of independent
cases padded with long fixtures, so splitting them buys no cohesion, and a gate that
blocks every routine test edit gets bypassed instead of obeyed. The pre-commit hook
enforces exactly this split (production gated, tests skipped) so the rule and the gate
cannot drift.

### Testing
- Every test must have a non-tautological assertion
- Use the real scorer/validator when possible; mock only the LLM layer
- Test the real `__main__` block via `exec` of the real source — never re-implement it in the test
- No test may launch a real browser or read real browser cookies; `references/tests/conftest.py`
  enforces this. Opt out with `@pytest.mark.real_cookie_discovery` only to test discovery itself.
- Prove a new test can fail before trusting it green
- Run (quick): `./.venv/bin/python -m pytest references/tests/ -q`
- Run (what the gate runs — use this before pushing; it adds the coverage floor
  and an unreachable LLM server, and a quick run passing proves neither):
  `OLLAMA_BASE_URL=http://127.0.0.1:1 MLX_MODELS_DIR=/tmp/nonexistent .venv/bin/pytest --cov --cov-fail-under=95 .`
  Run it exactly as written. It previously carried
  `--ignore=references/tests/test_img_helpers.py --ignore=references/tests/test_image_renamer.py`
  for a numpy C-extension crash under coverage tracing; that crash no longer reproduces
  (numpy 2.x / coverage 7.15, python 3.11.15), and those two files carry ~95 statements of
  `rename/helpers.py` and `rename/cli.py`. Excluding them dropped the total to **94.09%**
  against a 95 floor — the documented gate could not pass, at HEAD, for anyone.
  With them included it is **95.11%**. If the crash returns, raise it rather than
  re-adding the ignores: excluding a test file silently subtracts its module's coverage
  from a floor that is still measuring that module.
- Add discovered test patterns or bugs to `docs/TESTING.md` immediately

### Model Evals
- **Start the server with `./tools/osaurus_one.sh`, never by hand.** Models are
  4-35GB resident and a second server loads its own copy rather than queueing, so
  two servers means eviction, swapping, and requests the server cancels itself with
  `HTTP 499 request_cancelled` — which from the client is indistinguishable from a
  slow model. That is not hypothetical: it recorded qwen3.8-27b at 0.1 tok/s decode
  and a 423s cold start. The script is idempotent (a no-op when one is already
  answering) and `--check` exits 1 when there is not exactly one.
- **Never measure with anything else running against the GPU.** One command at a
  time, serially — including your own background jobs.
- **A contaminated measurement is outvoted, not permanent — but the guard is
  blind to the GPU.** This used to say a bad reading could never be displaced and
  that you had to delete the model's `_capabilities` entry by hand. That stopped
  being true when `eval/samples.py` landed: samples are a LIST, and the estimate is
  the MEDIAN OF THE LAST 5 CLEAN SAMPLES (`SAMPLE_WINDOW`), so recovery is "take
  another clean sample". Two things that ARE still true and matter more:
  - `machine_is_uncontended()` gates on SWAP (<=8GB) and COMPRESSOR (<=15GB) only.
    It cannot see GPU utilisation, so a competing Metal/GPU workload is recorded as
    CLEAN and enters the median as though the box were quiet.
  - The median only protects a model that HAS history. A new or thinly-sampled
    model's estimate is essentially its one sample, so first measurements are the
    exposed case. Still measure serially with nothing else on the GPU.
- Quick mode for iteration: `python3 -m eval --model <model> --task <task> --quick`
- Add discovered learnings to `docs/MODEL_QUIRKS.md` immediately when found

### Prompt Engineering
- Always prepend "Output JSON now." for qwen3.6 to prevent thinking
- Test changes with a quick single-task eval before a full run
