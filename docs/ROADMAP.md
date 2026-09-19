# Roadmap — `ztools`

_Forward-looking backlog only; completed plans are pruned to git history (house rule #13).
The Python-retirement plan (Phases 0-4: spikes, foundation, port behind seams, parity
proof, cutover) completed 2026-09-13 — its record, including the live-proof notes and the
collect-parity instrument recalibration, is in the history of this file before that
date and in `docs/PORT_PARITY.md`._

## State (2026-09-13)

Zero Python at runtime. The static Rust binary is the only implementation: twitter
collect + summarize (native camoufox-rs driver, fallback chain with provenance),
weekend planner (native DDG search, one plan store), image renamer, model-eval (the
full 24-row task roster + taxes snapshots, graded validators, vision task), and the
two routines status commands. `routines*.toml` run the binary. `references/`,
`pyproject.toml`, `uv.lock`, the `.venv` and the Python-only tools are gone;
`tools/*.py` are dev gates; `tools/tests/` holds the pytest for the shell lock.

## Open items (B4 deferrals — behaviour the Python had, the Rust does not, stated
rather than silently dropped)

1. **Weekend phase retry.** Python retried each LLM phase up to `LLM_MAX_RETRIES=5`
   on transport/parse failure; the Rust phases are single-shot with the configured
   `llm_extended_timeout_secs`. A transient server error fails the phase and the run
   says so. Port when a scheduled run is lost to one: wrap `weekend/phases.rs` calls
   in the same `run_chain`-style loop the summarizer now has, with the count in
   `conf/weekend.toml`, not code.
2. **Weekend phase timeout learning.** Python widened per-(model, phase) timeouts from
   observed latency (`conf/phase_signals.json`, headroom 1.5, never below the
   default). Rust uses the configured value. `eval/signals.rs` already learns
   per-model timeouts for the eval; the weekend phases could read the same store.
3. **Eval token / verbosity metrics.** `compute_token_estimates` / `compute_verbosity`
   need per-outcome content capture in `TaskOutcome` (a serialization decision — the
   record would grow by the model's full answer). Everything else in the report is
   ported (`eval/report_metrics.rs`, `report.rs`, `report_csv.rs`).
4. **Eval `diff_from_last_run` presenter.** History is saved and the trend table
   renders per-model deltas; the Python "what changed since the last run" table is
   not re-rendered as such.
5. **Drain-mode SIGINT.** The Python eval caught SIGINT, finished the in-flight task
   and released the lock in order. The Rust eval dies on SIGINT; the lock's
   dead-owner reclaim (pid + start time) restores the safety property, and the
   in-flight task's result is lost. Port with `ctrlc` if a sweep is ever interrupted
   by hand often enough to matter.

6. **`lfm2.5-2.6b-4bit` cannot load** on this Osaurus ("Unhandled keys
   [language_model] in LFM2Model") — delete it or report upstream. Every other
   installed model is ranked (sweep of 2026-09-19, thinking off).
7. **A summarize-specific injection task.** `filename_injection` is the only proxy
   and it now gates three slots; the raptors lose `summarize` on it despite 92.8 and
   88.8. A planted-instruction tweet in the summarize corpus would measure the real
   exposure and let a raptor win the slot honestly if it resists there.

## Open questions

- **Region filter divergence** (pinned, consensus fixtures only): Rust's
  foreign-city blocklist beats a local token; Python's whitelist-only did not. The
  remaining divergence surface is London-Ontario / Chicago-with-Toronto /
  ON-postal-only. Both lists live in `conf/weekend.toml [region]`; the rule is in
  `weekend_cache.rs` (`a_foreign_city_beats_any_local_token`). Someone picks.
- **Collect parity is shared-record agreement, not ID-set equality** — because the
  Following endpoint is served as a per-load sample (same collector, minutes apart:
  7 shared of ~50). If x.com ever serves it reverse-chronologically again, the
  stronger criterion can come back; `bin/ab_test` keeps the comparator.

## Explicit non-goals

- No keychain AES port: the persistent-profile design killed the Chrome-cookie path.
- No bs4 port: aggregator text extraction is bounded raw-HTML stripping.
- No per-model prompt variants (`conf/models/*.toml [prompts]`): prompts are
  `conf/prompts.toml`-canonical; the per-model tables are read only for token budgets.
- `tools/*.py` gates stay Python permanently.
