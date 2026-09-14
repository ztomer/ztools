# Project-Specific Instructions

Reference docs — read these when the task touches their subject, not by default:
- `docs/MODEL_QUIRKS.md` — model selection, per-model prompt quirks, eval results
- `docs/TESTING.md` — test patterns, mock infrastructure, coverage rules

## Key Rules

### File Size Limit
No file may exceed 500 lines — **no exemptions, for tests or for any directory.**
Enforced by the house gate (`gates_of_heck/checks/check_file_length.py` via
`.githooks/pre-commit` and `tools/gate.sh --full`; `.gatesrc` sets `GOH_MAX_LINES=500`).
Split into a module directory whose `mod.rs` re-exports (`json_validator/`,
`taxes_grounded/`), or move a `#[cfg(test)]` module into its own
`#[path = "..._tests.rs"] mod tests;` sibling. Check with `wc -l` before adding to a
file that is already close. A re-export forwards NAMES, not patch targets: test the
module that owns the function.

### Testing
- Every test must have a non-tautological assertion, and prove a new test can fail
  before trusting it green.
- Use the real scorer/validator; mock only the LLM layer (per-test stub HTTP servers,
  see `rust/tests/cli_dispatch_ztools.rs`). No test may launch a real browser or read
  real browser cookies.
- Golden parity tests (`rust/tests/validator_parity.rs`, `weekend_parity.rs`,
  `eval/validators/mixed_text_tests.rs`, `eval/tasks_tests.rs`) pin Rust behaviour to
  the verdicts the retired Python implementation produced on its last run. A change
  that moves one of those numbers is a change to reference behaviour: regenerate the
  golden deliberately and say so in the diff.
- Run (quick): `cargo test --manifest-path rust/Cargo.toml --all-features`
- Run (what the gate runs — use this before pushing): `tools/gate.sh --full`
  (= `make ci`): fmt, clippy `-D warnings`, no `#[allow]`, audit, emoji, file length,
  tests, coverage floor 94.
- `tools/tests/` holds the pytest for the shell tooling (`tools/gpu_lock.sh`):
  `python3 -m pytest tools/tests -q`. `tools/*.py` are dev gates, never the product.
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
  (`/tmp/mac-osaurus-gpu.lock`; `tools/gpu_lock.sh` + `rust/src/ztools/eval/gpu_lock.rs`,
  cross-checked by `rust/tests/gpu_lock_shell_parity.rs`), because
  several agent sessions run on this Mac concurrently and ONE healthy server is not
  enough on its own: restarting the server a peer is mid-measurement against
  corrupts that run exactly as badly as a second server does. The eval entry point
  holds it for the whole run; `osaurus_one.sh` holds it while it mutates the server;
  both `quit app "osaurus"` call sites REFUSE, with a stated reason, when another
  session holds it. WHY A LOCK RATHER THAN TRUSTING THE SAMPLE MEDIAN: `eval/samples.rs`
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
  being true when `eval/samples.rs` landed: samples are a LIST, and the estimate is
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
    `eval/watchdog.rs` is the backstop that depends on no measurement at all.
- Quick mode for iteration: `ztools model-eval --model <model> --suite full --task <task>`
  (`--suite smoke` for the five offline fixture tasks; `--json-output` for rows)
- Add discovered learnings to `docs/MODEL_QUIRKS.md` immediately when found

### Prompt Engineering
- Always prepend "Output JSON now." for qwen3.6 to prevent thinking
- Test changes with a quick single-task eval before a full run
