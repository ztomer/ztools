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

Every slot's injection exposure is measured in its own shape
(`filename_injection`, `summarize_injection`, `weekend_injection`). What is left
is a reading that takes a week to exist.

1. **Read the provenance ledger after a week** (instrument built 2026-09-19).
   Every plan now ends with `_Provenance: N extracted, N unsourced, N outside the
   window, N excluded._` and `ztools status` carries it under
   `details.provenance`. First reading, on the Bing-heavy corpus: 5 extracted,
   0 dropped. Compare a week of plans; if `unsourced` climbs, the region list or
   a per-engine result cap is the lever. Nothing to build — a reading to take:
   `for f in ~/Documents/weekend_plans/weekend_plan_*.md; do grep -H _Provenance "$f"; done`

## Phase 2 — planner honesty in the plan itself

(The Python-parity deferrals that lived here are closed: phase retry shipped as
`[llm] phase_retries`; phase timeout learning is a non-goal, below.)

## Phase 3 — eval ergonomics

Closed 2026-09-19: the diff-since-last-run table, verbosity metrics and
Ctrl-C drain mode all shipped. Nothing open.

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

- No per-(model, phase) timeout learning for the planner (the Python planner
  had it). Every production call is streamed and fails only on a stall, and
  its output is bounded server-side by `max_tokens`; the per-call cap is a
  backstop that a legitimate answer cannot reach. A learned cap would learn a
  number nothing consults.

- No DuckDuckGo wall-beater. `ddgs` 9.16 (2026-08) hits the same wall through
  browser-TLS impersonation; the wall is per IP. Engine spreading is the answer
  and it is built. A headless browser per query is not worth it while two
  engines answer.
- No keychain AES port: the persistent-profile design killed the Chrome-cookie path.
- No bs4 port: aggregator text extraction is bounded raw-HTML stripping.
- No per-model prompt variants (`conf/models/*.toml [prompts]`): prompts are
  `conf/prompts.toml`-canonical; the per-model tables are read only for budgets.
- `tools/*.py` gates stay Python permanently.
