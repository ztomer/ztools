# Backlog

One forward-looking file. Completed work is pruned to git history rather than
accumulating here as a graveyard of finished plans.

---

## Current state, 2026-08-16

**Done and committed**

| | item | commits |
|---|---|---|
| 0 | Roster drift: config named deleted models, `wk` was dead outright | `c75adcc` |
| — | Single-server invariant, enforced not remembered | `400cfba`, `9346a9c` |
| 2 | Prefill re-measured for every model, n>=2 | `1165364`, `aba52b0` |
| — | qwen3.8 switched to the 4-bit build (326x faster here) | `374d3b1` |
| — | Capabilities DERIVED rather than concluded by hand | `22b67c2` |
| — | Those probes actually WIRED into the four name-guessing consumers | `b7c55a0` |
| — | Sweep harness shipped, truncation visible | `1d9836c` |
| — | Mutation harness, and the bytecode bug that made its first numbers wrong | `ea0733a`, `137b9fc` |
| — | Reasoning overrun diagnosed instead of mislabelled | `d22b563` |

**In flight**

1. **The sweep** (item 1 below). Restarted from scratch each time the harness changed,
   because a sweep whose scoring changes mid-run produces incomparable results. This
   run is the first on a harness that classifies a reasoning overrun correctly.

**Open, in rough priority order**

2. ~~Wire the streaming overrun guard into the eval.~~ **Done.** `eval/run.py` passes
   `stream_guard=True`, and the guard is wired INSIDE `osaurus_lib.call` rather than
   replacing the transport, so quirks, JSON extraction, model substitution and the
   Foundation fallback all still apply. A stream error falls through to the blocking
   request, because that path is the one that knows how to substitute a deleted model
   and how to fall back on-device. Verified live end to end: bonsai reasons for ~1460
   chars and answers, uncut.

3. **Mutation survivors.** `json_validator.py` is down from 54 survivors to 28
   (detection 53% -> 73%), and fixing them found a live scorer defect (see the
   too-few-items cap, `df474e5`). Remaining, in order of value:

   | function | survivors |
   |---|---|
   | `validate_detailed_json` | 6 |
   | `check_source_extraction` | 5 |
   | `get_source_matching_details` | 5 |
   | `validate_json` | 4 |
   | `_names_match` | 4 |
   | `validate_mixed_signal` | 4 |

   `text_validator.py` is at 26 (from 28); `validate_summary` 8 -> 6, of which TWO
   are EQUIVALENT MUTANTS rather than gaps: its specificity tiers (`ratio >= 0.8`,
   `>= 0.5`) are absorbed by the misattribution cap, because any ratio below 1.0
   trips that cap and flattens the score to 45. The half-credit tier cannot affect
   any returned score. Pinned and explained in
   test_summary_scorer_boundaries.py so nobody re-derives it.

   The scorers (22 survivors) are untouched.
   `python3 tools/mutate.py --preset validators`.

   One lesson worth carrying: testing a function's OUTCOME is not testing its
   BOUNDARIES. `has_item_details` has two `len(item) >= 2` returns on different
   branches, and a two-field test reached neither -- `location` is a detail field, so
   it returned True from the scan above and exited. Killing those needed an item whose
   second key is deliberately NOT a detail field.

4. **Make "does thinking help?" a measured question.** Right now nothing records
   whether a model answered better with reasoning than without. The knob exists --
   `"default_chat_template_kwargs": {"enable_thinking": false}` in a model's own
   `generation_config.json`, which osaurus reads and no installed model sets -- so the
   experiment is available and has never been run. Probe `emits_reasoning` into
   `_capabilities`, then run one task both ways per model and record which wins.

5. **`_estimate_timeout` uses one of the three quantities `ev` measures.** Prefill is
   read back per model; decode and cold start use flat constants (8 tok/s, 120s)
   while the measured values sit in `conf/eval_signals.json`.

6. **A contaminated measurement is permanent.** The recorders keep the SLOWEST
   observation, so a reading taken under memory pressure can never be displaced by a
   correct one. Nothing enforces "delete `_capabilities` before re-measuring".

7. **The TUI has no config audit.** `ztools` can start clean while the config names an
   unservable model; `ev` checks this, the TUI does not.

8. **Unexplained: why a 27GB model runs at 0.09 tok/s on a 64GB machine.** Ruled out:
   configured caps, KV pre-allocation, a second server, other apps holding RAM,
   unsupported architecture, unsupported quantization, corrupt download. See
   docs/MODEL_QUIRKS.md. Remaining untested candidate: the inline MTP shard the MXFP8
   build carries and the 4-bit does not.

**Standing lessons this session kept re-teaching**

- Calibrate the instrument before believing it. The mutation harness measured its own
  bytecode cache; the viability check condemned a known-good model; the 6-bit monitor
  never fired because it matched on a name.
- A name is not evidence of a capability. Family, vision, size, generativeness and
  memory footprint were all guessed from model names, and all five guesses were wrong
  on the current roster.
- Fix the harness before the bug, and stop the run to do it. A sweep left running
  through a known harness defect spends hours producing numbers you will throw away.

---

## 0. The installed roster drifted out from under the config (fixed 2026-08-15)

**What happened.** `qwen3.6-35b-a3b-mxfp8-mtp` and `qwen-agentworld-35b-a3b-mxfp8` were
deleted from disk; `nemotron-3.5-lightning-30b-a3b-mxfp8` and `qwen3.8-27b-mxfp8`
appeared. `conf/config.toml` still named the dead ones for `default_model`,
`best_models.think`, `best_models.vlm` and 2 of the 3 `filename_models` — four of seven
tasks routed to a model the server answers with

    HTTP 404 {"error":{"message":"Model '...' is not installed or registered ..."}}

`wk` was dead outright: both of its tasks pointed there. Nothing reported the real
problem; the tools just surfaced a status code.

**The class.** A model tag is a session-scoped identity (rule #7), and nothing probed
it before depending on it (rule #10). The connection-failed path already degraded with
a stated reason via `_try_foundation`; the model-missing path — the same failure shape —
had no equivalent and died instead.

**Fixed structurally, not per-instance.**
- `lib/model_resolve.py` — `is_missing_model_error` distinguishes "that tag is gone"
  from a 404 for a mistyped URL; `substitute_model` picks the largest installed model of
  the same family (falling back through `foundation, qwen, gemma, ornith, nemotron`)
  and returns a reason sentence; `audit_configured_models` reports which config SLOT is
  stale (`best_models.think = ...`), not merely that something is.
- `lib/osaurus_lib.call` retries once against the substitute and records
  `substituted_from` / `substitution_reason`. It deliberately does NOT fire on a
  non-404, on a 404 from a wrong path, or on an unreachable roster — an empty roster is
  no evidence, not evidence of nothing, and substituting there would trade a loud error
  for a quiet wrong answer.
- `conf/config.toml` now names installed models, annotated PROVISIONAL: `qwen3.8-27b-mxfp8`
  is a like-for-like stand-in (same `qwen3_5` architecture, largest installed qwen,
  carries a `vision_config` so it can still serve `vlm`), **not** a measured winner.
  Item 1 is what replaces the guesswork.

- `ev` runs the audit at startup and prints any stale slot before evaluating anything,
  so the config is checked without waiting for a tool to trip over it. It audits
  against the roster `ev` had already fetched rather than issuing its own request —
  the first attempt did fetch its own, which broke 26 tests, because
  `references/tests/conftest.py` forbids a live server connection on exactly that path.
  The gate was right and the second request was waste regardless.

**Still open.** The TUI has no equivalent — `ztools` can still start clean while the
config names an unservable model. Wire `audit_configured_models` into its startup panel,
passing the roster the TUI already holds. Note that pytest cannot be the gate for any of
this (the suite blocks real server connections on purpose); the check has to run at
runtime against a real roster.

## 1. Sweep zeval across every installed model, and derive the quirks from it

**Why now.** The eval harness only recently started measuring anything real:
the `filename` task was scoring 100 for naming an unfilled `{text}` placeholder,
and `validate_summary` scored template leakage 90/100 with an empty failure
reason. Every `best_models` entry in `conf/config.toml` predates those fixes, so
the current model assignments were chosen by a harness that could not tell good
output from bad. They need to be re-derived, not adjusted.

**Scope.** 11 models are installed, re-listed 2026-08-15 (the previous list was
stale — see item 0). `family` is `details.family` from `/api/tags`, i.e. the real
architecture rather than a guess from the name; `vision` is whether the model's own
`config.json` carries a `vision_config`:

| model | family | size | vision |
|---|---|---|---|
| foundation | foundation | — | n/a (on-device) |
| bonsai-27b-ternary-jang | qwen3_5 | 27B | yes |
| gemma-4-12b-it-mxfp8 | gemma4_unified | 12B | yes |
| gemma-4-e2b-it-8bit | gemma4 | 2B | yes |
| gemma-4-e4b-it-8bit | gemma4 | 4B | yes |
| muse-glimmer-30b-jang_6m | muse_glimmer | 30B | yes |
| nemotron-3.5-lightning-30b-a3b-mxfp8 | nemotron_h | 30B | **no** |
| ornith-1.0-35b-jang_4m | qwen3_5_moe | 35B | yes |
| ornith-1.0-9b-mxfp8 | qwen3_5 | 9B | yes |
| potion-base-4m | unknown | 4M | no |
| qwen3.8-27b-mxfp8 | qwen3_5 | 27B | yes |

Two things fall out of that table, both of which change what this sweep has to test:

**The name-prefix family matcher is wrong.** `lib/config_getters.py::get_model_family`
keys on the model NAME, so `bonsai-*` and `ornith-*` resolve to `"default"` and get the
built-in fallback prompts — but both are `qwen3_5`, the family `conf/models/qwen.toml`
was written for. They may need no new config at all, just correct routing. Only
`muse_glimmer` and `unknown` (potion) are genuinely unserved families. Test the
architecture-keyed routing against the name-keyed one as part of the sweep rather than
assuming either.

**The VLM keyword heuristic is wrong in the other direction.**
`lib/osaurus_models.py::DEFAULT_VLM_KEYWORDS` is `vl,vision,qwen,llamavl`, so it finds
the qwens and misses gemma, ornith, bonsai and muse-glimmer — every one of which has a
vision tower. Meanwhile nemotron, the ONLY text-only server model, is not excluded by
anything. Replace the keyword match with the `config.json` probe (item 2 already reads
that file for `max_position_embeddings`; `vision_config` is in the same dict).

**Do this.**
1. Run the full task set per model, one at a time — the GPU is a single shared
   resource and concurrent runs thrash it.
2. Read raw output per model, not just scores, and record per-model quirks in
   `docs/MODEL_QUIRKS.md`: thinking-block behaviour, JSON compliance, whether
   "Output JSON now." is needed, instruction-leak patterns, format drift.
3. Create `conf/models/<family>.toml` for any family showing a systematic quirk.
4. Re-derive `best_models` from the results, and record WHEN it was derived so
   the next harness change invalidates it visibly.

**Watch for.** A leaderboard that compresses (everything 90-100) measures
nothing; so does one that ranks by latency because a slow model timed out and
scored 0. Check `conf/eval_signals.json` for learned timeouts before trusting
any ranking.

## 2. Extend the capability probe beyond the context window

**Done.** Context length is no longer assumed. `lib/model_caps.py` reads
`max_position_embeddings` from each model's own `config.json` (walking
`text_config`), because nothing in the Osaurus API reports it — `/v1/models` and
`/api/tags` carry family, parameter size and quantization only. The measured
values make the earlier guesses look timid:

| model | real context | previous guess |
|---|---|---|
| Qwen3.6-35B/27B, Ornith 35B/9B, Bonsai-27B | 262144 | 32768 |
| gemma-4-12B/E2B/E4B, Muse-Glimmer-30B | 131072 | 32768 |
| foundation, potion-base-4m | not on disk -> None | n/a |

An unknown model probes to `None`, never a fabricated number, and the caller
falls back to its own documented default. A `context_window` entry in
`conf/models/*.toml` still overrides both.

**Correction (2026-08-11): the context throttle is gone, and it should never
have existed.** For a while `usable_context_window` capped context at
`MAX_PREFILL_SECONDS` (120) x a measured prefill rate, floored at 800 chars/sec
for unmeasured models. Both constants were invented in `lib/model_caps.py`, not
derived from anything, and they sat beside a genuine measurement which made them
read as though they had been measured too.

The premise was wrong regardless of the numbers. `tw` runs every six hours and
`wk` once a day, so ingestion time is free -- there is nothing to trade it
against. The cap handed back ~46,000 of a 262,144-token window to buy seconds
nobody was short of, and paid for them in output quality, which is the only
thing that actually matters here. `usable_context_window` now returns the
override, then the probed window, then the caller default, and nothing throttles
it. `test_no_time_based_cap_survives_anywhere` fails if the API returns.

If a SHORTER prompt ever turns out to score better -- long-context attention
genuinely does degrade -- that is a finding for the eval to make, recorded as a
`context_window` entry in `conf/models/*.toml` with the evidence beside it. It
is not a number to assume.

**Prefill measurement survives, for timeouts only.** `ev` probes each model so
`twitter/budget.py::_estimate_timeout` can wait long enough for a large prompt
instead of killing it. The direction of caution inverts for this use: an
unmeasured model is assumed SLOW (200 chars/sec) so the timeout is generous,
where a throttle would have assumed slow in order to send less. `MAX_TIMEOUT` is
5400s -- a ceiling that stops a wedged server hanging forever, not a budget to
fit inside.

Measuring it honestly took four attempts, and three produced confident wrong
answers. Every published figure before the last row is contaminated:

| method | gemma-4-12b | what it actually measured |
|---|---|---|
| assumed constant | 40 | nothing |
| whole-call timing | 85 | decode, which dominates a generation-heavy task |
| `max_tokens=1`, identical filler | 1,322 -> 3,789 -> 140,281 | the server's prefix cache |
| `max_tokens=1`, nonce, no warmup | 1,045-1,237 | prefill + model load |
| **`max_tokens=1`, nonce, warmed** | **to be re-measured** | prefill |

The warmup bug is the clearest: the probe is an eval's FIRST request, so it
timed weights moving into memory. bonsai-27b (27GB, ternary) read 59 chars/sec
against ornith-9b's 1,803 -- a difference almost entirely attributable to load
time, which was then about to be recorded as that model's throughput forever.

**Correction (2026-08-15): the recorded rates do NOT predate the warmup fix.** This
paragraph used to end by claiming they did. Checked rather than assumed: at commit
`1bb9ee0` (the warmup fix itself) every model's `prefill_chars_per_sec` was absent, so
each rate now in `conf/eval_signals.json` was measured by the fixed probe. The probe
code needs nothing. What the DATA needs is different and narrower:

- **Done: every installed model now has a rate**, measured twice, one server, idle
  machine. Five had none at all, including the then-`default_model`. The one
  remaining exception is `potion-base-4m`, and that is now a definitive answer rather
  than a gap: the server rejects it with `HTTP 500 {"message":"Unsupported model type:
  model2vec"}`. It is an embedding model. `ev` currently skips it by NAME
  (`NON_LLM_KEYWORDS` contains "potion"); the server's own error is the better signal
  and would generalise to the next embedding model someone installs.

- **The plausibility bound was discarding a real measurement.**
  `gemma-4-e2b-it-8bit` reported "unmeasured" through the whole sweep. Every step of
  its probe actually succeeded — it genuinely ingests 21,946-23,063 chars/sec, and
  `MAX_PLAUSIBLE_PREFILL_RATE` was 20,000, so the guard threw the reading away as if
  it were a cache hit. Since the guard only ever discards, the model then fell back to
  the pessimistic 200 chars/sec default: a 100x error, produced by a safety check.

  The bound has now been set from measured data on both sides (genuine readings top
  out at 23,063; prefix-cache hits ran 65,000-140,281) rather than from a guess, and
  two tests pin it inside that gap. Worth noting WHY the guess kept failing: it was
  twice set from an assumption about which model would be fastest (~3,500 "the 35B
  MoE", then 20,000). Prefill is compute-bound and parallel, so the fastest ingester
  is the SMALLEST model — a 2B model shreds a 20K-char prompt in under a second.
- **Every rate was n=1.** `prefill_samples` was 1 across the board, so nothing in the
  file could tell a reproducible rate from a one-off — which mattered because
  `gemma-4-e4b-it-8bit` read 6,908.9 chars/sec against the probe's own docstring
  claim that the fastest genuine reading on this host is ~3,500. Measured twice on
  2026-08-15: 7,183.4 and 6,552.6, so the reading was right and the DOCSTRING was
  wrong. Small dense models ingest faster than big MoE ones; the ceiling tracks size,
  not the architecture that note assumed. Corrected in `eval/prefill.py`.

- **Memory pressure silently corrupts a measurement, and the corruption is permanent.**
  The largest models are 27-35GB resident. When the machine cannot hold one
  comfortably it swaps, and the server starts raising `HTTP 499 request_cancelled`
  against itself — which from the client is indistinguishable from a slow model.
  `qwen3.8-27b-mxfp8` recorded 0.1 tok/s decode, a 423s cold start and 95.7 chars/sec
  prefill under those conditions, against 2,740 for the LARGER ornith-35b. Two
  distinct causes were in play and both must be excluded before believing a number:
  other applications holding RAM, and a second osaurus process (which does not queue
  behind the first — it loads its own copy of whatever model it is asked for).

  The permanence is the sharp edge: `record_prefill_rate` and `_record_decode_rate`
  keep the SLOWEST observation, deliberately, so that a timeout is sized for a bad run
  rather than a lucky one. The same policy means a reading taken under memory pressure
  can never be displaced by a correct one. Re-measuring is not enough — the model's
  `_capabilities` entry has to be deleted first.

  `tools/osaurus_one.sh` now enforces the single-server invariant and is the documented
  way to start the server. **Still open:** nothing enforces "delete before re-measure".
  Consider a `--remeasure` flag on `ev` that clears a model's capabilities first, or
  recording every sample with a timestamp so a contaminated era can be dropped instead
  of poisoning the minimum forever.

- **`_estimate_timeout` uses one of the three quantities `ev` measures.**
  `twitter/budget.py` budgets cold start + prefill + decode, and `ev` measures and
  stores all three per model in `_capabilities`. But only prefill is read back:
  `_prefill_rate_for_model` consults the measurement, while decode uses a flat
  `DECODE_TOKENS_PER_SEC = 8` and cold start a flat `COLD_START_BASE = 120`, both
  ignoring the recorded values. For a model whose real decode rate is well under 8
  tok/s the budget is short by that ratio and the request is killed at `MAX_TIMEOUT`
  (5400s) having produced nothing — the failure mode the measured prefill rate exists
  to prevent, left open on the other two terms. Wire both through, the same way
  prefill already is.
- **`conf/eval_signals.json` carries dead entries**, including twelve uninstalled
  models and two keyed `m` and `mock-model`, which are fixture names. These are
  RESIDUE, not an active leak: `conftest.py::_signals_files_stay_clean` is autouse and
  session-scoped and repoints `EVAL_SIGNALS_PATH` at tmp, so no current test can reach
  the real file — confirmed by full gate runs leaving it untouched. They are safe to
  delete and will not come back.
  (Note for whoever greps: two comments — `eval/signals.py:20` and `weekend/llm.py:50`
  — say conftest redirects `EVAL_SIGNALS_DIR`. It redirects `EVAL_SIGNALS_PATH`, the
  resolved path. The protection is real; only the comments name the wrong symbol, and
  they made the gate look absent when it is not.)

**Now derived rather than concluded by hand (`ev --capabilities`).** `eval/capabilities.py`
probes family, vision, weight-file size, context window, generativeness and viability,
and prints the roster table that this document used to carry as hand-typed prose. The
conclusions in the sections above are now things the tool re-derives on demand; if the
roster changes, re-run the command instead of re-reading the markdown.

**Still to wire: the probes report, but production still guesses.** This is the gap
that matters most now — having the right answer available is not the same as using it.

- `lib/config_getters.get_model_family` still matches on the model NAME. Point it at
  the probed `details.family`, keeping the name match as the fallback for when the
  server is unreachable. Until then bonsai and ornith still get fallback prompts.
- `lib/osaurus_models.DEFAULT_VLM_KEYWORDS` still selects VLMs by keyword. Point
  `select_best_vlm_model` at `probe_vision`.
- `eval/cli.py::NON_LLM_KEYWORDS` still skips embedding models by name. Point it at
  `is_generative_model`, which already reads the config and is already correct.
- `eval/cli_runtime.estimate_model_memory` guesses from the model name: it warned
  "Model needs ~27GB" for the 15GB 4-bit build, because the name says 27b. Point it
  at `model_disk_bytes`.
- `MAX_PLAUSIBLE_PREFILL_RATE` is now evidence-based but still a constant. It could be
  derived from the recorded sample population instead, which would keep it correct as
  faster models arrive.

**Still to do.**
- **Vision: probed, not yet wired** (2026-08-15). Every installed model's `config.json`
  was read; the `vision_config` column in item 1's table is the answer, and it
  contradicts the name-keyed heuristic in both directions. Replace
  `DEFAULT_VLM_KEYWORDS` with the disk probe. `potion-base-4m` is correctly skipped but
  for the wrong reason — by name, not by capability.
- **`details.family` confirmed as the right key** (2026-08-15). `/api/tags` reports
  `gemma4_unified`, `gemma4`, `qwen3_5`, `qwen3_5_moe`, `muse_glimmer`, `nemotron_h`,
  `unknown` — and it correctly identifies bonsai and ornith as qwen3_5, which the name
  matcher cannot. Re-key `get_model_family` on it, keeping the name match as the
  fallback for when the server is unreachable.
- Still unprobed: tool/function-calling support, and whether a model emits thinking
  blocks. Both are still encoded as quirks rather than measured.
- **Re-measure the prefill rates.** `osaurus bench [--model X] [--prompt-tokens
  1024,8192] [--runs 3] [--json <path>]` ships with the server and reports TTFT /
  prefill / decode separately, tagged with hardware info. That is a purpose-built
  instrument for the exact quantity the homegrown probe got wrong three times running;
  evaluate it against a known-warm model before adopting it, and if it is sound, delete
  the homegrown probe rather than maintaining two.
- foundation stays special-cased: no local config to probe, a much smaller
  window than any server model (sizing a prompt to the server budget and
  re-sending it returns HTTP 500), fast and strong on short tasks. Treat "what
  works for foundation" and "what works for the server models" as two separate
  questions.

## 3. Rewrite the tests that pass by construction (re-scoped 2026-08-15)

**The raw patch count is not a defect count, and this item was framed as though it
were.** Measured across the 89 test modules and 2,161 test functions:

| category | count | verdict on inspection |
|---|---|---|
| patch targets, total | 1,487 | — |
| ... at a real boundary (requests, subprocess, sys.argv, PIL, playwright) | 431 | legitimate, keep |
| ... on the code's own functions | 1,056 | dominated by `main()` composition-root patches, which this file already calls a design call |
| tests with NO assertion and no `pytest.raises` | **1** | legitimate: `client.connect(...)` "must not raise" IS the assertion |
| tests whose every assertion is "a mock was called" | **8** | sampled 3, all real: exact forwarded signature, `--task` filter contents, bounded fetch count |
| tests asserting a literal they injected into a patch | 71 | over-counted; sampling shows the literal is usually a vehicle through real logic (e.g. inject a 4096 window, assert the probe SHRANK below it) |

So the crude tautology classes are essentially absent, and a rewrite of ~890 patch
sites would be motion rather than progress.

**What the static analysis cannot see, and what actually found bugs.** Over one
session of changes, mutation testing — break the code, confirm the test goes red —
was run about ten times and caught **two** tests that passed for the wrong reason:

- a size-vs-alphabetical ordering test whose fixture happened to sort the same way
  under both rules, so it measured nothing;
- a recursion-guard test whose scenario converged on its own, so the guard was never
  exercised.

Both read perfectly well. Neither is reachable by grepping for patch counts or
missing asserts. That is a ~20% blind rate on freshly written, carefully reviewed
tests, and there is no reason the existing suite's rate is lower.

**So the real item is not a rewrite.** It is: run mutation testing and fix what
survives. `tools/mutate.py` does this — `--preset scorers`, `--preset validators`.

### First run, 2026-08-15

| target | mutations | killed | survived | detection |
|---|---|---|---|---|
| scorers | 94 | 72 | **22** | **77%** |
| validators | 194 | 103 | **91** | **53%** |

Survivors are concentrated rather than spread: `json_validator.py` 54,
`text_validator.py` 28, `attribution.py` 9, `scorers_filename.py` 10,
`scorers_summarize.py` 8. The single largest kind is the boundary mutation — `>=`
silently becoming `>`, 34 of the validator survivors. Thresholds are exercised
somewhere in their range but never AT their edges, which is the one place a threshold
can be wrong. That is where to start.

**The first run of this reported 100% and 56%, and BOTH numbers were wrong.**
Worth recording, because the failure is not obvious and would recur:

CPython validates a `.pyc` against the source's **(mtime, size)**. Most mutations
here are length-preserving by construction — `>=` becomes `> `, `==` becomes `!=` —
and the write-test-restore cycle takes milliseconds. When a restore lands in the same
mtime tick as the mutation, with an identical size, the interpreter accepts the
MUTATED bytecode as valid for the RESTORED source.

The consequences ran in both directions and neither was visible:

- mutated bytecode survived past the run and failed unrelated tests afterwards,
  which is how a scorer appeared to return `(100, "no items with details")` — a
  perfect score beside its own failure list. **That bug does not exist.**
  `validate_detailed_json` returns 35 for detail-free input and 100 for good input,
  both correct. The reading came entirely from stale bytecode.
- an "order-dependent test" was diagnosed on the same evidence and is likewise not
  real.
- kills were inflated, because a leftover mutated `.pyc` broke tests for a reason
  unrelated to the mutation under test and was scored as a detection. That is how
  the scorers read 100%.

`tools/mutate.py` now purges `__pycache__` before and after every run and executes
pytest with `-B` and `PYTHONDONTWRITEBYTECODE=1`. The numbers above are from the
fixed harness, with the full gate green (2217 passed, 95.17%) immediately after.

The general lesson is the one this repo keeps relearning: **calibrate the
instrument before believing it.** A mutation harness that cannot prove its mutation
actually loaded is measuring its own caching behaviour.

The seam hazard still applies — splitting a module silently breaks
`patch.object(module, name)`, because the moved function resolves its globals in the
new module and the patch then applies to a name nobody reads.

### Original note (kept for the reasoning, not the framing)

Eight classes were rewritten and 115 patches removed or converted, but roughly
890 patch sites remain that mock the code's own functions. The boundary ones
(tesseract, `requests`, `DDGS`, playwright, `subprocess`, `sys.argv`) are
legitimate. The composition-root ones on `main()` are a design call, not a
cleanup — see the reasoning in git history before touching them.

The seam hazard is real and recurred three times in one session: splitting a
module breaks `patch.object(module, name)` silently, because the moved function
resolves its globals in the new module. The patch still applies — to a name
nobody reads.
