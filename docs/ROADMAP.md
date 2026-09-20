# Roadmap — `ztools`

_Forward-looking backlog only; completed plans are pruned to git history (house rule
#13). Consolidated and phased 2026-09-19. The Python-retirement plan completed
2026-09-13 and lives in this file's history before that date and in
`docs/PORT_PARITY.md`._

## State (2026-09-19)

Zero Python at runtime. The static Rust binary is the only implementation: twitter
collect + summarize (native camoufox-rs driver, fallback chain with provenance),
weekend planner (three-engine search, one plan store, health-stamped plans), image
renamer, model-eval (30-task roster, graded validators, vision task, thinking-off
regime), and the two routines status commands.

Every production LLM call goes through one client (`ztools/llm.rs`): thinking off,
`max_tokens` bounded, streamed, failed only on a stall. `[best_models]` is derived
from a complete sweep of every installed model under that regime; the embedded
defaults are drift-gated to it. Every installed model except `lfm2.5-350m` (28.7,
unrankable by score) holds or could hold a slot.

## How to rank work

The tools exist to run unattended and be trusted. Rank by what a failure HIDES:
a plan that says "quiet weekend" over a blocked search, a summary steered by a
tweet, a slot winner measured under a regime the tool does not run. Work that
only makes a visible thing nicer comes after work that closes a silence.

## Phase 1 — measure what the decisions rest on

The slot derivation carries one inferred exposure (json) and one unmeasured
noise effect (Bing). Both should be measured, not inferred.

1. **A snippet-shaped injection task for the json slot.** `summarize_injection`
   (built 2026-09-19) cleared every candidate and retired the filename proxy for
   that slot; json still leans on `filename_injection`, which keeps raptor-v0.5
   (100 on the wk tasks) out. A search-snippet corpus with one planted
   instruction, scored the same way (obeyed sentence vs. quoted line), would
   settle json honestly. Ship criterion: the task is in the roster, the slot
   comment cites it.
2. **`weekend_fabrication` for the engine chain.** Bing returns noisier results
   than DDG did ("Dropped 50 out-of-region" per run against ~4 before). The
   provenance and region gates hold, but nothing measures whether the extra
   noise moves invented-row rates in the live planner. One week of plans, count
   rows the provenance gate dropped; if it climbs, the region list or the
   per-engine result cap is the lever.

## Phase 2 — planner honesty in the plan itself

3. **Engine order learned from health.** `SearchHealth` records per-engine
   walls per run; nothing reads them back. If DDG is walled for N consecutive
   runs, try it last (or skip it) rather than pay 16 challenged POSTs per plan.
   Data, not code: the order and the threshold in `conf/weekend.toml`. Only
   worth it if the wall persists — check the health lines in a month first.
4. **Weekend phase retry** (B4 deferral). Python retried each phase up to 5× on
   transport/parse failure; Rust phases are single-shot. The stall-guarded
   client removed the main cause (abandoned calls). Port only when a scheduled
   run is actually lost to a transient error: the `run_chain`-style loop the
   summarizer has, count in `conf/weekend.toml`.
5. **Weekend phase timeout learning** (B4 deferral). Python widened per-(model,
   phase) timeouts from observed latency. The stall guard makes the cap a
   backstop, so this only matters if a call legitimately streams past 900s.
   `eval/signals.rs` has the store if it ever does.

## Phase 3 — eval ergonomics

6. **Eval "what changed since the last run" presenter** (B4). History is saved
   and the trend table renders per-model deltas; the Python diff table is not
   re-rendered. Cheap once someone wants it.
7. **Eval token / verbosity metrics** (B4). Need per-outcome content capture in
   `TaskOutcome` — a serialization decision, the record grows by the model's
   full answer. Decide before porting.
8. **Drain-mode SIGINT** (B4). The Rust eval dies on Ctrl-C; the lock's
   dead-owner reclaim restores safety and the in-flight task is lost. Port with
   `ctrlc` if sweeps get interrupted by hand often enough to matter.

## Watch items (no action until they move)

- **26GB in an interactive slot.** filename and vlm are `muse-glimmer-30b`; `rn`
  is interactive, and a cold load measured up to 8m38s on a contended box. Fine
  while `rn` runs after something else has warmed muse; if the first rename of
  the day is the complaint, `rn` warms on launch or the slot takes the 6.3GB
  raptor-v0.5 (100 quality, 0 injection — same trade as summarize).
- **Osaurus wedges on abandoned requests.** The production client never
  abandons one, and `osaurus_one.sh --restart` clears it. If a wedge recurs
  with no abandoned call in the log, it is a server bug to report, not ours.

## Open questions

- **Region filter divergence** (pinned, consensus fixtures only): Rust's
  foreign-city blocklist beats a local token; Python's whitelist-only did not.
  Remaining surface: London-Ontario / Chicago-with-Toronto / ON-postal-only.
  Both lists live in `conf/weekend.toml [region]`. Someone picks.
- **Collect parity is shared-record agreement, not ID-set equality** — because
  the Following endpoint is served as a per-load sample. If x.com serves it
  reverse-chronologically again, the stronger criterion can come back;
  `bin/ab_test` keeps the comparator.

## Explicit non-goals

- No DuckDuckGo wall-beater. `ddgs` 9.16 (2026-08) hits the same wall through
  browser-TLS impersonation; the wall is per IP. Engine spreading is the answer
  and it is built. A headless browser per query is not worth it while two
  engines answer.
- No keychain AES port: the persistent-profile design killed the Chrome-cookie path.
- No bs4 port: aggregator text extraction is bounded raw-HTML stripping.
- No per-model prompt variants (`conf/models/*.toml [prompts]`): prompts are
  `conf/prompts.toml`-canonical; the per-model tables are read only for budgets.
- `tools/*.py` gates stay Python permanently.
