# Architecture & Implementation Roadmap — `ztools`

_Forward-looking backlog only; completed plans are pruned to git history (house rule #13).
Python `references/` is preserved for A/B parity verification only; production runs execute
Rust code._

---

## State of the port (2026-08-21)

The Rust binary `ztools` (~2.7k production LOC) is the primary implementation for all four
tools (`twitter-summarize`, `weekend-plan`, `image-renamer`, `model-eval`). The eval system
is fully ported and compiled:

- **Transport pipeline** (`eval/transport.rs`) mirrors `osaurus_lib.call` end-to-end: quirks
  applied inside the call (so a substitute re-derives them), stream-guard happy path,
  blocking fallback, and missing-model substitution (`eval/model_resolve.rs`) retrying once
  against a servable stand-in and surfacing `substituted_from`/`substitution_reason`.
- **Runner** (`eval/runner.rs`) wires learned per-task timeouts, p95 signal recording,
  prefill/cold-start/decode measurement, and the stall watchdog behind
  `run_eval_with_signals` (the CLI `--suite full` path); unit/integration tests stay on the
  hermetic `run_eval`.
- **Signals/prefill/samples/discrimination/watchdog/gpu_lock** all ported with mock-server
  integration tests (`tests/model_resolve_http.rs` proves substitution, quirk re-derivation,
  opt-out, unmarked-404 refusal, and learning-path recording over the wire; both key tests
  were proven to fail by mutation before being trusted green).
- **Model health**: `model_health.rs` + `model_health_tests.rs` cover MTP shards,
  index.json weight_map validation, `.incomplete` artifacts, thrashing viability.
- **Gates**: `#[allow]` ban clean (the `clippy::too_many_arguments` suppression was removed
  by grouping `twitter-summarize` flags into a struct), emoji gate clean, prompt parity gate
  green, full pytest coverage gate green at 95.25%.

---

## Remaining work

### 1. Live A/B parity — first run PASSED (2026-08-21); widen the model matrix

All six taxes tasks on `gemma-4-e2b-it-8bit` scored IDENTICALLY across Rust and Python
(74 / 50 / 100 / 100 / 74 / 100), and the three rubric validators produce byte-identical
verdicts on identical input (scores + reason strings) via `eval/validators/taxes_rubric.rs`
(ported from `lib/validators/taxes_validator.py` after the live run caught the generic-check
substitutes scoring structurally differently). Output budgets now resolve from config
(`eval/budgets.rs`), matching Python's `get_max_tokens_for_task` chain.

What remains of this item: repeat the comparison on a SECOND and THIRD model
(e.g. `ornith-1.0-9b-mxfp8`, then a 27B+ reasoner under the GPU lock) and on the smoke-suite
tasks once those have Python counterparts, then flip any remaining defaults so Rust is the
unambiguous production runner.

### 2. Automated CI comparison (Rust vs Python)

Automate an output-parity check (same captured model answer through both validator stacks)
so drift between the two implementations fails CI rather than waiting for the next live sweep.

### 3. Known, documented divergences (fix or accept explicitly)

Each is annotated in source where it lives:

- **Per-task configured tables**: output budgets ARE ported (`eval/budgets.rs` reads
  `conf/config.toml [max_tokens]` and narrows via `conf/models/<family>.toml`, matching
  `get_max_tokens_for_task`); the per-task TIMEOUT table is not (`signals.rs::effective_timeout`),
  so the learned value competes only with the documented floor. Family resolution in
  `budgets.rs` is name matching; the architecture-from-signals refinement is not ported.
- **Task data provenance**: the canonical snapshots are `eval_tasks/data/taxes/` (tracked).
  A stale untracked copy under `references/eval/tasks_data/` once scored 0 where Python
  scored 100; it was deleted on sight. Point `--tasks-dir` at `eval_tasks/data` only.
- **Foundation on-device fallback**: not ported; a dead server surfaces as an error result
  (`transport.rs` module doc). Decide whether the eval path ever wants it.
- **Failure-category machinery**: the Python loop's PARSE/TIMEOUT/CONTENT categories are not
  ported; `record_signal(..., is_parse_failure=false)` is honestly zero rather than guessed
  (`runner.rs`), and retry-token escalation for reasoning overruns does not exist yet.
- **`is_generative_model` gate**: the prefill probe skips embedding models in Python via a
  disk config probe; the Rust learning path probes every model name it is given.

---

## Pruned to git history

Items 1–4 (allow-gate, best-models matrix, image-renamer security port, Twitter C2a prompt/
timestamp parity) and eval conversion Phases 1–4 are complete and verified; their records
and the items 5–10 A/B matrices live in git log.
