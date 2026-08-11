# Backlog

One forward-looking file. Completed work is pruned to git history rather than
accumulating here as a graveyard of finished plans.

---

## 1. Sweep zeval across every installed model, and derive the quirks from it

**Why now.** The eval harness only recently started measuring anything real:
the `filename` task was scoring 100 for naming an unfilled `{text}` placeholder,
and `validate_summary` scored template leakage 90/100 with an empty failure
reason. Every `best_models` entry in `conf/config.toml` predates those fixes, so
the current model assignments were chosen by a harness that could not tell good
output from bad. They need to be re-derived, not adjusted.

**Scope.** 11 models are installed:

```
bonsai-27b-ternary-jang        muse-glimmer-30b-jang_6m     qwen3.6-27b-mxfp8-mtp
foundation                     ornith-1.0-35b-jang_4m       qwen3.6-35b-a3b-mxfp8-mtp
gemma-4-12b-it-mxfp8           ornith-1.0-9b-mxfp8
gemma-4-e2b-it-8bit            potion-base-4m
gemma-4-e4b-it-8bit
```

Only three families have a `conf/models/*.toml`: gemma, qwen, qwopus (plus
laguna and nemotron for models no longer installed, and foundation). So
`bonsai`, `muse-glimmer`, `ornith` and `potion-base` currently fall through to
the built-in fallback prompts — the ones whose weekend_transient schema was
malformed until recently. Whether they need their own quirks file is unknown
because nothing has measured them.

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
Every rate in `conf/eval_signals.json` predates the warmup fix and needs
re-measuring.

**Still to do.**
- Probe the other capabilities the same way instead of encoding them as quirks:
  tool/function-calling support, vision capability (`potion-base-4m` is
  currently skipped as "non-LLM" by name), and whether a model emits thinking
  blocks. `details.family` from `/api/tags` gives the real family
  (`gemma4_unified`, `qwen3_5`, `muse_glimmer`) and is a better key for
  per-family config than matching on the model name prefix.
- foundation stays special-cased: no local config to probe, a much smaller
  window than any server model (sizing a prompt to the server budget and
  re-sending it returns HTTP 500), fast and strong on short tasks. Treat "what
  works for foundation" and "what works for the server models" as two separate
  questions.

## 3. Rewrite the tests that pass by construction (partially done)

Eight classes were rewritten and 115 patches removed or converted, but roughly
890 patch sites remain that mock the code's own functions. The boundary ones
(tesseract, `requests`, `DDGS`, playwright, `subprocess`, `sys.argv`) are
legitimate. The composition-root ones on `main()` are a design call, not a
cleanup — see the reasoning in git history before touching them.

The seam hazard is real and recurred three times in one session: splitting a
module breaks `patch.object(module, name)` silently, because the moved function
resolves its globals in the new module. The patch still applies — to a name
nobody reads.
