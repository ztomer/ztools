# Architecture & Implementation Roadmap — `ztools`

_Forward-looking backlog only; completed plans are pruned to git history (house rule #13).
Python `references/` is preserved for A/B parity verification only; production runs execute
Rust code._

---

## State of the port (2026-08-21)

The Rust binary `ztools` is the primary implementation for all four tools (`twitter-summarize`,
`weekend-plan`, `image-renamer`, `model-eval`). The eval system is fully ported and verified:

- **Transport pipeline** (`eval/transport.rs`) mirrors `osaurus_lib.call` end-to-end: quirks
  applied inside the call (so a substitute re-derives them), stream-guard happy path,
  blocking fallback, and missing-model substitution (`eval/model_resolve.rs`) retrying once
  against a servable stand-in and surfacing `substituted_from`/`substitution_reason`.
- **Runner** (`eval/runner.rs`) wires learned per-task timeouts, p95 signal recording,
  prefill/cold-start/decode measurement (gated by the ported `is_generative_model` config
  probe), and the stall watchdog behind `run_eval_with_signals`; unit/integration tests stay
  on the hermetic `run_eval`.
- **Oversize/thrashing refusal** (`eval/oversize.rs`): a model whose weights exceed 80% of
  reclaimable memory — or a machine already paging — is REFUSED with the same message shape
  as Python (`EVAL_ALLOW_OVERSIZE=1` for the deliberate case). Ported after the widened
  parity run caught Rust measuring `qwen3.8-27b-8bit` where Python refused it.
- **Budgets & timeouts from config**: output budgets (`eval/budgets.rs`) and per-task
  timeouts (`signals.rs::effective_timeout`) both read their `conf/config.toml` tables,
  narrowed/maxed exactly like `get_max_tokens_for_task` / `_effective_timeout`.
- **CI validator-parity gate**: `rust/tests/validator_parity.rs` prints each fixture
  answer's verdict from the RUST validators; `references/tests/test_rust_validator_parity.py`
  computes the same verdicts with the PYTHON validators and asserts byte-for-byte agreement.
  Proven red by mutating one stack's grounding ladder before being trusted green.

### Live A/B parity matrix (all six taxes tasks, identical snapshots)

| model | result |
|---|---|
| gemma-4-e2b-it-8bit | **6/6 exact** |
| ornith-1.0-9b-mxfp8 | **5/6 exact**; the sixth (`audit_readiness`) shown to be model sampling variance — captured answers fed through BOTH stacks offline produce byte-identical verdicts (34 = 34) |
| qwen3.8-27b-8bit | **6/6 exact** (Python run under its deliberate `EVAL_ALLOW_OVERSIZE=1`) |

---

## Remaining work

### 1. Make Rust the unambiguous production runner

Flip any remaining defaults/docs that still route production through Python; keep
`references/` purely as the parity layer powering `bin/ab_test --functional` and this gate.

### 2. Known, documented divergences (fix or accept explicitly)

- **Foundation on-device fallback**: not ported; a dead server surfaces as an error result.
  Likely answer: the eval path does not want it — record the decision here when made.
- **Failure-category machinery**: the Python loop's PARSE/TIMEOUT/CONTENT categories are not
  ported (`record_signal(..., is_parse_failure=false)` is honestly zero), nor is the
  reasoning-overrun retry-token escalation (`failures.py::reasoning_retry_budget`). The
  ornith sweep's slowness is the visible symptom; port when reasoning models become regular
  parity targets.
- **Family resolution via recorded architecture**: `budgets.rs` resolves conf families by
  name matching; Python prefers the architecture recorded in eval_signals. Only matters for
  models whose ids do not contain their family name.

---

## Pruned to git history

Items 1–4 (allow-gate, best-models matrix, image-renamer security port, Twitter C2a prompt/
timestamp parity), eval conversion Phases 1–5, and the first three live A/B parity runs are
complete and verified; their records live in git log.
