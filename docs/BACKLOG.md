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

**Also done: what is actually sent is now measured, not chosen.** The window is
what a model CAN take; the real constraint is how long ingesting a prompt takes.
So `practical_context_cap(model)` is expressed as time — `MAX_PREFILL_SECONDS`
(120s, the only number a human picks) multiplied by that model's own measured
prefill rate. `ev` probes each model with `max_tokens=1` before running its
tasks and records the slowest observation under `_capabilities` in
`conf/eval_signals.json`; `lib/model_caps.py` reads it back.

Three wrong numbers were replaced here, and they were wrong in both directions:

| source | gemma-4-12b rate | implied cap |
|---|---|---|
| assumed `PREFILL_CHARS_PER_SEC = 40` | 40 | 1,600 |
| derived from whole-call time | 85 | 3,420 |
| `max_tokens=1`, identical filler each run | 1,322 → 3,789 → 140,281 | absurd |
| **`max_tokens=1` + per-run nonce** | **1,045-1,237** | **~46,600** |

Two distinct traps, and both produce a confident number:

- **Timing an ordinary task call measures decode, not ingestion** — ~17x too low
  on a generation-heavy task. It reads like a conservative floor and is simply
  the wrong quantity.
- **Repeating identical probe text measures the server's prefix cache.** The
  same model on the same host climbed 1,322 → 3,789 → 140,281 chars/sec across
  successive identical probes. A rate that improves every time you measure it is
  not a rate. The probe now leads with a nonce, so nothing after it can be
  reused, and it holds at 1,045-1,237 across four runs;
  `MAX_PLAUSIBLE_PREFILL_RATE` (20,000) discards any reading that lands in
  cache-hit territory anyway.

A model `ev` has never evaluated falls back to an 800 chars/sec floor, chosen to
sit below the slowest honest measurement. `twitter/budget.py::_estimate_timeout`
reads the same per-model rate, so the timeout and the prompt size come from one
measurement rather than from two independent constants.

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
