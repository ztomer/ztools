# Project-Specific Instructions

Reference docs — read these when the task touches their subject, not by default:
- `docs/MODEL_QUIRKS.md` — model selection, per-model prompt quirks, eval results
- `docs/TESTING.md` — test patterns, mock infrastructure, coverage rules

## Key Rules

### File Size Limit
No file may exceed 500 lines — **no exemptions, for tests or for any directory.**
Split into a package (`twitter/`, `weekend/`, `rename/`, `eval/`) or extract a module;
`lib/config.py`, `lib/quality.py`, `lib/osaurus_lib.py`, `lib/validators/text_validator.py`,
`lib/quality_scorers.py`, `eval/report.py`, `eval/cli.py` and `eval/run.py` are shims
re-exporting their split submodules. Check with `wc -l` before adding to a file that is
already close.

A shim re-exports NAMES, not patch targets: rebinding an attribute on the shim rebinds
a copy nobody reads. Tests must patch the module that OWNS the function, which each
submodule's docstring states. `eval/run.py`'s docstring is the worked example.

The rule covers **Rust too**, and it is enforced by one implementation,
`tools/check_file_size.py`, called by `.githooks/pre-commit` and pinned by
`references/tests/test_file_size_gate.py`. Rust splits use the same shim pattern:
`json_validator/` and `taxes_grounded/` are directories whose `mod.rs` re-exports.

**There used to be a test exemption** (Python files under `references/tests/`; Rust
`*_tests.rs`/`tests.rs` siblings and inline `#[cfg(test)] mod tests` blocks subtracted
from the count). It is gone as of 2026-08-24. Split an oversized test file the same way
as production: independent test classes/cases move into sibling `test_*.py` files
(pytest discovers every file matching that glob — no shim needed), or Rust
`#[cfg(test)]` modules move into their own `#[path = "..."] mod` file. `conftest.py`
fixtures that don't fit split into a `fixtures/` package that conftest imports from
(the shim rule still applies: import by module, patch the module that owns the fixture).

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
  (numpy 2.x / coverage 7.15, python 3.14), and those two files carry ~95 statements of
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
  answering) and `--check` exits 1 when there is not exactly one — or when another
  session holds the GPU lock.
- **Never measure with anything else running against the GPU.** One command at a
  time, serially — including your own background jobs.
- **The GPU and the osaurus server are held under a machine-wide lock**
  (`/tmp/mac-osaurus-gpu.lock`; `tools/gpu_lock.sh` + `lib/gpu_lock.py`), because
  several agent sessions run on this Mac concurrently and ONE healthy server is not
  enough on its own: restarting the server a peer is mid-measurement against
  corrupts that run exactly as badly as a second server does. The eval entry point
  holds it for the whole run; `osaurus_one.sh` holds it while it mutates the server;
  both `quit app "osaurus"` call sites REFUSE, with a stated reason, when another
  session holds it. WHY A LOCK RATHER THAN TRUSTING THE SAMPLE MEDIAN: `eval/samples.py`
  outvotes a bad reading only when it knows the reading is bad, and
  `machine_is_uncontended()` gates on swap and compressor — it cannot see the GPU, so a
  peer's eval is recorded as a CLEAN sample. The median also only protects a model that
  HAS history; a first measurement is its own estimate. Nothing to clean up by hand — a
  dead owner's lock is reclaimed (PID plus process start time, so a recycled PID cannot
  impersonate it), and the
  wedge ceiling measures PROGRESS via a per-task heartbeat rather than wall clock,
  so an honest multi-hour run never loses its lock. Blocked? `--check` names the
  holder. Deliberately NOT the desktop lock at `/tmp/mac-desktop-ui.lock`.
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
  - **A REPLACED BUILD under the same name is NOT fixed by another clean sample.**
    Contention is what the median outvotes. If the artifact behind a name changes
    (a rebuild, a requant, an MTP variant shipped as the old tag), the old samples
    describe a model that no longer exists and will outvote the new one for five
    readings. Clear that model's `_capabilities` by hand in that case, and only
    that case. Print what you cleared so nothing vanishes silently.
  - **The `clean` flag is only as useful as its consumers.** It was recorded for
    months while `_derived_timeout` read the raw scalar and never asked, so a
    thrashing box measured decode at 0.1158 tok/s, `max_tokens / decode` came to
    ~138,000s, and the resulting 2-hour per-task ceiling let a wedged server idle
    83 minutes. A contended machine makes measurements slow, slow measurements
    inflate the derived timeout, and the inflated timeout permits a longer stall.
    `eval/watchdog.py` is the backstop that depends on no measurement at all.
- Quick mode for iteration: `python3 -m eval --model <model> --task <task> --quick`
- Add discovered learnings to `docs/MODEL_QUIRKS.md` immediately when found

### Prompt Engineering
- Always prepend "Output JSON now." for qwen3.6 to prevent thinking
- Test changes with a quick single-task eval before a full run
