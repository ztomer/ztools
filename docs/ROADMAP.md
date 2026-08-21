# Architecture & Implementation Roadmap — `ztools`

_Forward-looking backlog only; completed plans are pruned to git history (house rule #13).
Python `references/` is preserved for A/B parity verification only; production runs execute
Rust code._

---

## State of the port (2026-08-21) — COMPLETE

The Rust binary `ztools` is the sole production implementation for all four tools
(`twitter-summarize`, `weekend-plan`, `image-renamer`, `model-eval`). Every `bin/` shim
execs the native binary; `python3 -m eval` survives only as the parity reference. The eval
system is fully ported and verified:

- **Transport pipeline** (`eval/transport.rs`) mirrors `osaurus_lib.call` end-to-end: quirks
  applied inside the call (so a substitute re-derives them), stream-guard happy path,
  blocking fallback, and missing-model substitution (`eval/model_resolve.rs`) retrying once
  against a servable stand-in and surfacing `substituted_from`/`substitution_reason`.
- **Runner** (`eval/runner.rs`): learned per-task timeouts, p95 signal recording, prefill
  measurement gated by the ported `is_generative_model` config probe, stall watchdog, AND
  the ported failure-category machinery (`eval/failures.rs`): INFRA/TIMEOUT/PARSE/FORMAT/
  CONTENT/REASONING drive infra abandonment, honest parse-failure counting in the signal
  store, and reasoning-overrun retry escalation (`reasoning_retry_budget`: retry with MORE
  room, bounded at 64k).
- **Oversize/thrashing refusal** (`eval/oversize.rs`): a model whose weights exceed 80% of
  reclaimable memory — or a machine already paging — is refused exactly like Python
  (`EVAL_ALLOW_OVERSIZE=1` for the deliberate case).
- **Budgets & timeouts from config** (`eval/budgets.rs`): `[max_tokens]` and `[timeouts]`
  tables read from `conf/config.toml`; per-model caps narrow via `conf/models/<family>.toml`;
  family resolved from the RECORDED ARCHITECTURE in eval_signals first (trimmed
  qwen3_5_moe -> qwen), name matching as fallback.
- **CI validator-parity gate**: fixture answers scored by BOTH validator stacks every test
  run; byte-for-byte agreement asserted (`rust/tests/validator_parity.rs` +
  `references/tests/test_rust_validator_parity.py`). Proven red by mutation.

### Live A/B parity matrix (all six taxes tasks, identical snapshots)

| model | result |
|---|---|
| gemma-4-e2b-it-8bit | **6/6 exact** |
| ornith-1.0-9b-mxfp8 | **5/6 exact**; sixth shown to be sampling variance — captured answers through both stacks give byte-identical verdicts |
| qwen3.8-27b-8bit | **6/6 exact** |

---

## Accepted divergences (decided, not open)

- **Foundation on-device fallback: NOT wanted on the eval path.** A sweep against a dead or
  refusing server must say "cannot run here" — silently answering from a different engine
  would enshrine another model's quality under the configured name. (Substitution is
  different: it fires on EVIDENCE the configured tag is gone, retries once, and says so.)
- **parse_json tasks**: the Rust task set carries none yet; when one arrives, port the
  FORMAT/PARSE/prose-before-JSON branches annotated in `failures.rs::classify_failure`.

---

## Open items

None. New work should start a fresh section rather than resurrect this list.
