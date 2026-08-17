# Next phase — plan (2026-08-17)

Ordered by effect on whether `wk`/`tw`/`rn` can be trusted day to day, not by effect on
the eval machinery. Produced after the session that fixed five measurement defects
(token budget, attribution, config aliasing, stream deadline, and a 31GB host leak).

> **Housekeeping:** the standing docs rule is *one forward-looking backlog file; prune
> completed plans to git history*. This file is deliberately temporary — fold it into
> `docs/BACKLOG.md` once the ordering is agreed, and delete it. A per-phase plan file
> left lying around is exactly the graveyard that rule exists to prevent.

---

## P0. `rn` is outside the protection perimeter (NOT previously on the backlog)

**The invariant.** Every production LLM call goes through `lib/osaurus_lib.call` — which
knows about model substitution on a 404, the stream wall-clock deadline, quirks, and the
Foundation fallback — and every config slot naming a model tag is audited against the
roster. `wk` and `tw` honour this. `rn` violates it three times, and the violations are
live:

- `references/rename/llm.py` makes raw `requests.post` calls to `/api/chat` with its own
  parsing in `is_relevant_with_llm`, `query_llm_for_filename`, `query_vlm_for_filename`.
  No substitution, no deadline, no quirks, no fallback.
- `references/rename/llm.py:70` — the relevance-check default is
  `"qwen3.6-27b-mxfp4,gemma-4-26b-a4b-it-mxfp4"`. **Neither is installed.** The relevance
  check silently returns `None` for every image today.
- `conf/rename.toml` `vlm_preferred` names two models that are not installed.
  `audit_configured_models` only audits `conf/config.toml`, so this drift is invisible to
  both `ev` and the TUI audit.
- `references/rename/llm.py:41-44` — `FILENAME_MODELS` and the prompt are bound at
  **import time**: the same class as the `config_getters._cfg` defect fixed this session,
  surviving in a sibling.

**Why first.** `rn` is one of three user-facing tools and is quietly degraded right now.
Everything else on this list refines instruments; this fixes a tool.

**Approach.** Route the three call sites through `osaurus_lib.call` (adding image
passthrough, which P2 needs anyway); extend `audit_configured_models` to cover every
`conf/*.toml` key naming a model tag; replace import-time bindings with call-time getters.

**Verification.** Two class-level gates: (a) a structural test grepping production
packages for direct `requests` calls to LLM chat endpoints outside `lib/`, proven to fail
by running it against HEAD; (b) an audit test with a fixture `rename.toml` naming an
absent model. Wire (a) into pre-commit beside the file-size hook.

**Effort.** ~1 day. P2 depends on it.

---

## P1. `weekend_fixed_mixed` is unwinnable by construction

**The defect, found by reading rather than by symptom.** `WEEKEND_USR_FIXED`
(`references/lib/eval_data.py:63`) asks the model to **find 10** activities from **12**
signal venues. `validate_mixed_signal` (`references/lib/validators/json_validator.py:318`)
computes `recall = tp / total_signal` over all 12. A model that obeys the prompt perfectly
scores `100 * (0.5 * 10/12 + 0.5 * 1.0) = 91` — exactly the shared 91 that all eleven
models hit. They were not missing two items; they were **obeying the instruction**, and
the validator punishes compliance.

**Invariant violated:** the validator's contract must equal the prompt's contract.

**Approach.** Cap the recall denominator at the requested count (thread `expected_count`
from the task definition into the validator) rather than rewriting the prompt — "pick N of
M" mirrors what `wk` actually asks in production. Then the class fix: a structural test
that, for **every** task in `TASKS`, constructs the ideal answer implied by its own prompt
and asserts the validator returns 100. A "task is winnable" gate.

**Verification.** The winnability test is its own proof-of-failure: run it before the fix
and `weekend_fixed_mixed` must go red at 91.

**Effort.** Half a day. Independent — can go first.

---

## P2. A real image task through `rn`'s real path (backlog 9)

**Beyond the known gap.** `best_models.vlm` is UNMEASURED because no eval task sends an
image. The deeper problem: nothing proves `query_vlm_for_filename`'s transport works at
all — it posts Ollama-style `"images": [b64]` to `/api/chat` and no test, eval or probe has
confirmed osaurus accepts that shape. An image task added through a *different* path would
prove nothing about `rn`, which is precisely the class backlog 9 describes.

**Approach.** Probe the live server first to establish what image shape it accepts; add
3–5 checked-in synthetic fixture images (PIL-drawn, no OCR-able text so the task cannot be
passed blind); add task `image_real` through the **same transport `rn` uses**; validator
checks expected keywords per image.

**Calibration is mandatory:** score image A's output against image B's key and require
failure, and run a text-only model (nemotron) and require a bad score. Keep the
cross-image control as a permanent test so the validator cannot decay into shape-checking.

**Effort.** ~1 day. Depends on P0.

---

## P3. Self-correcting estimator (backlog 6) + all three quantities in the timeout (backlog 5)

One change closes both. `record_prefill_rate` / `_record_decode_rate` /
`_record_cold_start` (`references/eval/prefill.py:169-213`) keep the **slowest** observation
forever, so a contaminated reading is permanent — it feeds `_derived_timeout` and already
cost a week and a deleted model. Separately `twitter/budget.py::_estimate_timeout` reads
back only prefill, using flat constants for decode and cold start while the measured values
sit in the same dict.

**Approach.** Store samples as `{value, ts, uncontended}` with contention checked at
measure time (`vm_stat` pages-free/compressor per the MODEL_QUIRKS recipe, plus
`osaurus_one.sh --check`); estimate by median of the most recent N uncontended samples;
migrate existing scalars as one legacy-tagged sample. Then point `_estimate_timeout`'s
decode and cold-start terms at `recorded_capability`, keeping the pessimistic fallbacks.

**Verification.** Record one contaminated-slow sample then N good ones and assert the
estimator converges — run against the current slowest-wins code first, where it stays stuck.

**Effort.** ~1 day. Independent.

---

## P4. Corroborate roster claims with disk

osaurus advertises deleted models from an in-memory cache until restarted, so `/api/tags`
is a claim, not proof. Every consumer of `get_models` — substitution, audits, the TUI — can
pick a stand-in that 404s. Intersect the roster with `model_config_path` existence; flag
in-roster-but-not-on-disk as "stale roster, restart osaurus" and exclude it from
substitution candidates; treat an unfindable probe (foundation) as unknown-keep, never as
missing.

**Effort.** Half a day. Independent.

---

## P5. Two or three short, trap-dense adversarial tasks — after P1

Follow the `summarize_misattribution` recipe (short input, dense traps, ratio-graded):

- a **weekend grounding trap** for the `json` slot: a venue list where famous plausible
  venues are deliberately absent, scoring fabrication;
- a **filename prompt-injection trap** for `rn`: OCR text containing embedded
  instructions, scoring whether the model names the content or obeys the injection. This
  is `rn`'s real threat model — it feeds untrusted screenshot text into a prompt.

Only after P1's winnability gate exists, so new tasks are born provably winnable.

**Effort.** 1–2 days.

---

## Explicitly NOT doing

- **"Does thinking help?" (backlog 4)** — no consumer for the answer; no knob or slot
  choice hinges on it. Measurement without a decision attached.
- **The mutation-survivor campaign (backlog 3) as a scheduled item** — survivors are
  concentrated in files P1 touches anyway. Kill them in passing, don't grind.
- **New ranking statistics** — "the eval cannot rank" is mitigated by per-slot scoring and
  actually solved by P1 (removes a fake ceiling), P2 (makes `vlm` measurable) and P5 (adds
  separation). A composite score would be polishing a statistic instead of fixing tasks.
- **Adopting `osaurus bench`** — swapping instruments immediately after calibrating one
  repeats the class of error this repo keeps paying for. Evaluate later against a known-warm
  model; if adopted, delete the homegrown probe rather than running both.
- **De-saturating the saturated tasks** — they are regression tests now, correctly weighted
  at zero for ranking.

---

## Order

P1 (½d) → P0 (1d) → P2 (1d, needs P0) → P3 (1d) → P4 (½d) → P5 (1–2d).

P1/P3/P4 are independent and can interleave. Only P2's probe and the final `vlm`
re-derivation need the GPU, so eval-machine contention stays confined to one step.

## Files

| file | items |
|---|---|
| `references/rename/llm.py` | raw transport, dead model defaults, import-time bindings (P0) |
| `references/lib/osaurus_lib.py` | `call` gains image support; the one client everything routes through (P0, P2) |
| `references/lib/validators/json_validator.py` | recall denominator; winnability class fix (P1) |
| `references/eval/prefill.py` | slowest-wins recorders → self-correcting estimator (P3) |
| `references/lib/model_resolve.py` | audit all `conf/*.toml` slots; disk-corroborated roster (P0, P4) |
