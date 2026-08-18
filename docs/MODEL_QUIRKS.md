# Model Quirks & Best Practices

**Updated: June 2026** — canonical reference for model selection and prompts.

---

## ⚠ Half this document names models that are no longer installed (2026-08-15)

`qwen3.6-35b-a3b-mxfp8-mtp`, `qwen3.6-27b-mxfp8-mtp`, `qwen-agentworld-35b-a3b-mxfp8`
and the whole `laguna` family have been removed from disk. Every table below that names
one is describing a model this machine cannot run — including the cheat sheet
immediately following. `nemotron-3.5-lightning-30b-a3b-mxfp8` and `qwen3.8-27b-mxfp8`
arrived and appear nowhere. The numbers are not merely stale, they are unrunnable, and
they are what `conf/config.toml` was copied from.

Do not adjust these rows. `docs/BACKLOG.md` item 1 re-derives them from a sweep; until
that runs, the only measured claims in this file are the roster table below and the
2026-08-12 sweep section, and only for the models that still exist.

### ~~qwen3.8-27b-mxfp8 does not fit this machine's page cache~~ RETRACTED 2026-08-17

**There is no size threshold on this host. The machine was being drained by a leak.**

The original finding — 0.09 tok/s decode, 143s TTFT, "326x slower than the 4-bit",
"the threshold is between 18GB (fine) and 27GB (dead)" — was measured while
claude-mem's `bun` worker daemon held **31GB of this machine's 64**. Available memory
was 14.6GB, so a 27GB model genuinely could not stay resident. Not because 27GB is too
big for a 64GB box, but because half the box was gone.

Killing that process returned available memory to 42.6GB. Re-measured immediately
after — same models, same server, same prompt, machine otherwise idle:

| build | size | cold TTFT | warm TTFT | decode |
|---|---|---|---|---|
| qwen3.8-27b-4bit | 16.1GB | 38.7s | 1.3s | 19.2 tok/s |
| qwen3.8-27b-jang_6d | 25.8GB | 46.1s | 1.6s | 15.2 tok/s |
| qwen3.8-27b-mxfp8 | 28.7GB | 41.2s | 1.6s | **14.7 tok/s** |

14.69 against the 0.08 recorded before: **184x off**. The real cost of 1.8x the bytes
is 1.3x on decode — which is simply what reading more bytes per token costs. All three
builds are usable, and no model should be excluded here on size grounds.

**Why the investigation reached the wrong answer, which is the part worth keeping.**
The table of ruled-out hypotheses above contained the correct answer, and it was
dismissed with a broken instrument:

> | other apps holding RAM | reproduced with the machine 62% free, top consumer 0.4GB |

Both halves of that are measurement errors:

- **"top consumer 0.4GB"** was read from RSS. The leaking daemon's RSS *was* 0.55GB —
  because its 31GB had been squeezed into the compressor and swapped out. RSS shows
  what is resident, and a leak that is never touched again is never resident. The
  section correctly warned that "RSS is the WRONG instrument" for the model, then
  used RSS to clear the other suspects.
- **"62% free"** came from a metric that ignores the compressor. `memory_pressure`
  was reporting "42% free" at a moment when `Pages free` was 59MB and 49GB of data
  was compressed into 29GB.

**Instruments that would have caught it**, and that any future memory claim here has
to use:

```bash
vm_stat | grep -E "Pages free|occupied by compressor"   # free was 59MB, not 42%
top -l 1 -o mem -n 12 -stats pid,command,mem,cmprs      # CMPRS column: bun 31G
python -c "import psutil; print(psutil.virtual_memory().available/1024**3)"
```

`top`'s `CMPRS` column is the one that found it. A process holding 31GB compressed
appears as 31G there and as 0.55GB in RSS.

The general rule, now a standing one: **a measurement taken on a contended machine
describes the contention, not the thing measured.** This one hardened into a design
rule that reached `conf/config.toml`, this document, and a `default_model` choice, and
would have permanently excluded every model above 18GB from a machine that handles
them fine. See backlog item 6 — `record_*` keeping the SLOWEST observation is what
made the bad number impossible to displace.

**Not a configured limit, and not KV pre-allocation** (checked 2026-08-15, because
"64GB should be enough" is the obvious objection and it deserved an answer). The
server publishes its own budget at `GET /health` under `ram_feasibility`:

    physical_memory              68.7 GB
    recommended_max_working_set  55.7 GB   (Metal)
    gpu_budget                   51.5 GB
    soft_limit / hard_limit      55.0 / 61.8 GB
    kv_headroom                   5.2 GB   <- ~20% of weights
    projected (weights + KV)     31.0 GB   for the 24GB build
    exceeds_gpu_budget           False

Two things follow. KV headroom is a fraction of WEIGHTS, not the declared context, so
the 16GB a 262144-token cache would cost is never reserved — the arithmetic that
suggested otherwise was modelling a system this is not. And the 27GB build projects
to roughly 33GB against a 51.5GB budget, so no configured cap rejects it: the server
believes it feasible and admits it.

The knobs exist (`memorySafety.customPhysicalMemoryFraction`,
`customDefaultMaxKVSize`, `customAllocatorCacheBytes`) and NONE are set, so the
defaults are in force and they are generous rather than restrictive. Raising them
would not help.

**ANSWERED 2026-08-17, and the answer was not in this list.** A leaked plugin daemon
held 31GB of the machine's 64, leaving 14.6GB available. The MTP shard was never the
cause and needed no investigation. See the retraction at the top of this file.

There is no threshold on this host. With the leak cleared (42.6GB available) the MXFP8
build decodes at 14.7 tok/s against the 4-bit's 19.2 — a 1.3x cost for 1.8x the bytes,
which is what reading more bytes per token costs.

**Resolved.** Pulled `mlx-community/Qwen3.8-27B-4bit` (15GB on disk, vision retained)
and measured it back to back against the MXFP8 build on the same machine:

| variant | on disk | TTFT (1024 tok) | decode | prefill | cold start |
|---|---|---|---|---|---|
| qwen3.8-27b-mxfp8 | 27 GB | 143,022 ms | 0.08-0.2 tok/s | 116.7 c/s | 182 s |
| **qwen3.8-27b-4bit** | **15 GB** | **30,744 ms** | **26-35 tok/s** | **449-643 c/s** | **0.79 s** |

The "326x" read as proof that requantizing fixed a cache problem. It was not: the
4-bit was merely small enough to survive in the 14.6GB the leak left behind. On a
clean machine the same pair measures 19.2 vs 14.7 tok/s. Requantizing moving the
number is consistent with a memory shortage of ANY cause, so it never distinguished
"this model is too big for this host" from "something else is eating the host".

The MXFP8 build was deleted on the strength of that wrong conclusion and re-pulled
2026-08-17 (`mlx-community/Qwen3.8-27B-mxfp8`, 28.68GB). It is usable and is included
in the corrected sweep. Do not delete a model on a performance number measured while
something else held a third of the machine.

`osaurus pull` writes to `~/.osaurus/models`, which the server does NOT scan; models
have to be moved to `~/MLXModels/<org>/<Name>` before `osaurus list` sees them. A pull
that reports "Done. Model saved to: ..." can therefore leave you with a model the
server will 404.

`mlx-community/Qwen3.8-27B-MTP-4bit` (260MB drafter, for speculative decoding) is also
installed but NOT yet wired in — osaurus lists it as a separate model rather than
pairing it automatically. Unmeasured.

### Installed roster, measured 2026-08-15

`family` is `details.family` from `/api/tags` — the real architecture, not the name.
`vision` is whether the model's own `config.json` carries a `vision_config`.

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
| qwen3.8-27b-mxfp8 | qwen3_5 | 27B | yes (14.7 tok/s once the leak was cleared) |
| qwen3.8-27b-4bit | qwen3_5 | 27B | yes (15GB, 26-35 tok/s -- use this one) |
| qwen3.8-27b-mtp-4bit | qwen3_5 | drafter | speculative-decoding weights, unwired |

Two routing bugs fall straight out of it:

- **`bonsai` and `ornith` are qwen3_5**, but `get_model_family` keys on the model NAME,
  so both resolve to `"default"` and get built-in fallback prompts instead of
  `conf/models/qwen.toml`. Only `muse_glimmer` and `unknown` are genuinely unserved.
- **Nearly everything has a vision tower.** `DEFAULT_VLM_KEYWORDS`
  (`vl,vision,qwen,llamavl`) finds the qwens and misses gemma, ornith, bonsai and
  muse-glimmer, while nemotron — the only text-only server model — is excluded by
  nothing. Read `vision_config` off disk instead of matching names.

---

## `rn` was renaming images from hallucinated descriptions (fixed 2026-08-18)

osaurus exposes an OpenAI-compatible endpoint and **silently drops** the Ollama-style
`{"images": [b64]}` key inside a message. It does not error; it answers as though no
image were attached. `rn` used that key, so every image it renamed was described from
nothing.

Measured against the live server with a picture of a red circle:

| payload | model's reply |
|---|---|
| `{"images": [b64]}` | "Please provide the image you are referring to..." |
| no image at all | "Please provide the image you are referring to..." |
| content parts + `image_url` | **"Red semi-circle."** |

The first two being identical is the proof: the key was ignored.

Three unmistakable, mutually unrelated fixtures through the old path:

| image | before (blind) | after |
|---|---|---|
| red circle | "large white building blue sky" | "red curved shape" |
| white dots on navy | "large brown dog" | "small white circles" |
| green + yellow bars | "large brown bear forest" | "...tan rectangle, ...green rectangle" |

**The lesson, which is why P2 in the plan said "probe first".** The plan assumed this
transport worked and only needed an eval task built on top. Had that been done in the
stated order, the new task would have measured hallucination and produced a confident
`best_models.vlm` ranking from it. Nothing in the codebase had ever verified that an
image reached a model -- `image_rename` sends its prompt as TEXT, so ten models scoring
100 on it proved only that they can emit a filename-shaped string.

Also fixed alongside: `apply_model_quirks` assumed `content` is a string and raised
`AttributeError: 'list' object has no attribute 'lower'` on multimodal messages. Quirks
reword prompts; a list of content parts has no prose to reword, so it passes through.

## Why our leaderboard disagreed with published benchmarks

Asked directly, and worth keeping because the answer was not "quantization" or "these
are unusual community builds". Two causes, both ours.

### 1. The token budget decided the score, not the model

`get_max_tokens_for_task` reads `[max_tokens]` keyed by TASK NAME and falls back to
`DEFAULT_MAX_TOKENS` for anything absent. Exactly **1 of the 24 eval tasks** was named
in that table. The other 23 inherited a different number, so whether a task got the
configured budget or the fallback came down to whether someone had happened to list
it.

The proof needs no statistics. `filename` and `filename_leak` send a **byte-identical
185-character prompt**:

| task | in `[max_tokens]`? | budget | nemotron scored |
|---|---|---|---|
| `filename` | yes | 1000 | **0%** |
| `filename_leak` | no | 16000 (fallback) | **100%** |

Re-run directly to confirm: at 1000 it aborts mid-reasoning; at 16000 it returns
`screenshot_login_error_invalid_credentials`.

**Scale of the distortion.** 19 of 37 zeros across the sweep were reasoning overruns —
nemotron 10 of 10, ornith-9b 9 of 10 — and those are exactly the two models at the
bottom. Excluding only the tasks the harness killed:

| model | as measured | excluding harness kills |
|---|---|---|
| nemotron-3.5-lightning-30b | 51.3 (last) | **90.8 (first)** |
| ornith-1.0-9b-mxfp8 | 49.5 | 81.4 |

A 30B model was never plausibly worst. The measurement was wrong.

**Why the budget was too small.** Reasoning models emit their chain of thought before
any content, and it scales with the TASK, not with the budget handed to them. Measured
on nemotron:

    trivial ("what is 2+2")             ~400 chars reasoning, answers
    summarize_misattribution (8 lines) 19,009 chars reasoning, answers at 8192
    summarize (40 tweets)              77,208 chars reasoning, answers at 32000

It returns EMPTY at every budget below what it needs — guard on or off — and ignores
`detailed thinking off` and `/no_think`. Both models answer trivial prompts normally
(~400 chars, `finish=stop`), so the integration was never broken.

Budgets are now a uniform 32000 with a structural test pinning every task to the same
number, and the overrun retry escalates instead of shrinking. See the commit for the
full list.

### 2. The suite does not measure scale-dependent capability at all

**Pearson r(parameters, score) = +0.064** across 10 models — and **+0.092** even after
excluding the two the harness killed. The remaining 8 span just **10.2 points** from 2B
to 35B.

That is the structural answer. Public benchmarks measure knowledge and reasoning, which
scale with parameters. This suite measures whether a model honours an output contract:
emit this JSON shape, end every bullet with `(@handle | time)`, do not invent venues. A
2B model can honour a contract as well as a 35B one — which is why gemma-4-e2b beat
gemma-4-12b at summarisation, and why that result is not anomalous.

Consequences: do not read this leaderboard as a capability ranking, and do not expect
it to track public benchmarks. It answers "which installed model will reliably produce
what `wk`/`tw`/`rn` need", which is a different and more useful question here.

**Ruled out:** prompt or quirk asymmetry between families. All models receive
byte-identical messages for a given eval task; quirks are task-gated. Verified, not
assumed.

## Scores before 2026-08-16 are not comparable with scores after it

Four scorer defects were fixed on 2026-08-16, all of the same class: a scorer that
could not fail a bad answer. The last two are below, under "Attribution was never
actually checked"; the `summarize` scores in any earlier sweep are affected.

**`validate_file_summary` scored boilerplate as real work.** `BOILERPLATE_RE`
("not specified|n/a|unknown|not provided") has always existed and `validate_summary`
has always checked it, but the file-summary specificity test used a separate local
list of generic WORDS ("personal", "document", ...) that none of those phrases
contain. A summary whose every description read "not specified" counted every one as
SPECIFIC and scored a full 100 -- identical to one describing real files. "unknown"
scored lower only by accident: it is shorter than MIN_SPECIFIC_DESC_LEN, so the
length test caught what the content test missed.

    all "not specified"   before 100   after 60 ("generic descriptions only")
    all "not provided"    before 100   after 60
    real descriptions     before 100   after 100

Boilerplate is now excluded from BOTH the specificity count and the set of
descriptions, so it reaches the "generic descriptions only" failure rather than
collecting consolation credit for having said something.

**The `file_summary` and `file_summary_mixed` results in any sweep started before
this are wrong** for models that emitted boilerplate, and both need re-running per
model. They are 2 of 23 tasks; the other 21 are unaffected.

### Attribution was never actually checked (two defects, one property)

Misattribution is the failure `tw` exists to avoid -- telling the user the wrong
person said a thing, which the reader has no way to spot. Two independent defects
meant the check never ran.

**1. The `summarize` task passed no `source`.** `validate_summary` gates its
strongest rule on having one:

```python
if source_text and total_bullets and faithful < total_bullets:
    score = min(score, MISATTRIBUTION_MAX_SCORE)
```

With no source the cap cannot fire and the attribution ratio contributes no
specificity credit either, so a summary crediting every quote to the wrong person
scored the same as a correct one. Same class as `filename` scoring 100 for
summarising an unfilled `{text}` placeholder: **the input did not carry the property
under test, so the validator skipped the test and graded shape.**

Fixed by setting `source` on the task. Worth stating plainly, though: **fixing it did
not de-saturate `summarize`.** gemma-4-e2b, gemma-4-12b and foundation still score
100. The models genuinely attribute correctly on that timeline -- every claim there
has exactly one plausible author, so getting it right takes no care. The saturation
was in the INPUT's difficulty, not only in the scorer.

**2. `_BULLET_TAG_RE` was measuring punctuation, not attribution.** It anchored on
`\)\s*$`, so a bullet ending with a full stop or wrapped in a stray bracket parsed as
UNTAGGED:

    - claim (@Reuters | 07:10)      recognised
    - claim (@Reuters | 07:10).     NOT recognised   <- gemma-4-12b, gemma-4-e2b
    - claim ((@mchen | 07:10))      NOT recognised   <- foundation

Downstream, "no tags" is indistinguishable from "no attributions to check": the cap is
gated on `total_bullets`, so **any model that punctuates its bullets was never
attribution-checked at all** -- in the eval or in `tw`. Two of the three most-used
models punctuate.

This one is worth remembering as a method, not just a bug. It surfaced because a new
task reported `0%, no attributed bullets` for gemma-4-12b, whose raw output tagged
every bullet correctly AND got the hardest trap right. The same 0 came back for
foundation, which genuinely failed that trap. **An instrument returning the same
reading for a right answer and a wrong one is not measuring the thing it names.**
Always read the raw output behind a surprising score before recording it.

### `summarize_misattribution` (new, 2026-08-16)

Ranks models on attribution, which `summarize` cannot. A compact 8-line timeline
built entirely from the three ways attribution fails in practice:

| trap | shape |
|---|---|
| quoted speaker | `[@Reuters]: Analyst @mchen said X` — @mchen also posts, about something else |
| adjacent contradiction | two outlets, one line apart, saying opposite things about one company |
| same handle, two timestamps | `@Bloomberg` posts twice; the claim must pair with the right time |

Graded as a RATIO (`validate_attribution`), not through validate_summary's
all-or-nothing cap. That cap is correct for `tw` -- one wrong attribution is
disqualifying for something a user acts on -- but it is a poor instrument, because
every model with a single slip lands on the same number and the task separates
nobody. First results:

| model | score | slip |
|---|---|---|
| ornith-1.0-35b-jang_4m | 100 | — |
| muse-glimmer-30b-jang_6m | 100 | — |
| gemma-4-12b-it-mxfp8 | 88 | reformatted a timestamp to `0732` |
| gemma-4-e2b-it-8bit | 88 | swapped the adjacent-contradiction authors |
| foundation | 86 | credited the quoted speaker (@mchen) with Reuters' claim |

Note the timeline is deliberately SHORT. An earlier version injected the same traps
into the full 40-tweet timeline; every model scored 96-99 because one or two trap
errors were diluted across forty easy bullets. Trap density is what makes a task rank.

### The too-few-items cap

`validate_detailed_json` gained a too-few-items CAP. Before it, the count credit was
additive and the weights summed to 120 against a ceiling of 100, so the 20-point
overhang absorbed the penalty whole: a 4-item weekend report scored exactly the same
100 as a 12-item one, on tasks whose entire purpose is producing a list.

    items      before    after
    2-4          100       85      (forgoes the whole count weight, 15)
    5-9          100       95      (forgoes the GOOD/OK difference, 5)
    10+          100      100

The caps are derived from the credit not earned rather than picked, so they stay
correct if the weights are ever retuned. Found by mutation testing: nine `>=`
mutations on this function's thresholds survived the entire suite, because the credit
they gated was invisible under saturation.

An existing integration test asserted the old behaviour with the comment "only fails
on <10 items check, not penalized" -- the defect written down and accepted.

---

## TL;DR Cheat Sheet

Start server: `./tools/osaurus_one.sh` -- NEVER by hand. A second osaurus does not
queue, it loads its own copy of the model, and two of them thrash a machine sized
for one. The script is idempotent and `--check` exits 1 unless exactly one is up
AND no other session holds the GPU lock.

---

## Best Models by Task

Derived 2026-08-16 from the full sweep (11 installed models x 23 tasks). Each slot is
scored only over the eval tasks its own consumer exercises — a mean across all 23 is
meaningless when most of them are saturated. The reasoning and the tiebreakers are
written out in `conf/config.toml`; this table is the summary.

| Slot | Model | Group score | Consumer |
|------|-------|-------------|----------|
| `json` | gemma-4-12b-it-mxfp8 | 98.7 | `wk` |
| `summarize` | gemma-4-e2b-it-8bit | 89.8 | `tw` |
| `filename` | foundation | 100.0 | `rn` |
| `think` / `default_model` | ornith-1.0-35b-jang_4m | 88.4 | fallback for every unslotted task |
| `vlm` | gemma-4-12b-it-mxfp8 | **unmeasured** | `rn` image path |

Overall means across all 23 tasks, for context only — do NOT pick a slot with this
column, it is what the per-slot scoring exists to avoid:

| Model | Mean | Zeros | Size |
|-------|------|-------|------|
| gemma-4-12b-it-mxfp8 | 86.9 | 1 | 13.4GB |
| muse-glimmer-30b-jang_6m | 86.5 | 2 | 27.6GB |
| qwen3.8-27b-jang_6d (6-bit) | 85.9 | 2 | 25.8GB |
| gemma-4-e4b-it-8bit | 85.0 | 2 | 9.0GB |
| ornith-1.0-35b-jang_4m | 84.0 | 2 | 19.8GB |
| qwen3.8-27b-4bit | 83.3 | 3 | 16.1GB |
| foundation | 82.6 | 0 | on-device |
| gemma-4-e2b-it-8bit | 79.8 | 2 | 5.9GB |
| bonsai-27b-ternary-jang | 76.9 | 3 | 8.0GB |
| nemotron-3.5-lightning-30b | 51.3 | 10 | 34.0GB |
| ornith-1.0-9b-mxfp8 | 49.5 | 10 | 10.1GB |

### A 5.9GB model beats the 13.4GB one at summarising, reproducibly

`gemma-4-e2b-it-8bit` took the `summarize` slot away from `gemma-4-12b-it-mxfp8`,
which is surprising enough that both were re-run three times on the two adversarial
tasks. The result is deterministic, not sampling noise:

|  | `summarize_contradiction` | `summarize_factual_accuracy` |
|---|---|---|
| gemma-4-e2b-it-8bit | 100, 100, 100 | 67, 67, 67 |
| gemma-4-12b-it-mxfp8 | 0, 0, 0 | 34, 34 |

The source in the first contains a self-contradiction; the second contains three
planted falsehoods. The 12B parrots the contradiction verbatim every time and repeats
all three falsehoods. The e2b is the only installed model that clears both gates.

The lesson generalises past these two models: **parameter count predicts fluency, not
faithfulness.** Fluency is what the saturated tasks measure, which is why nine to
eleven models tie at 100 on them and why they cannot rank anything. Only the
adversarial tasks separate the roster, so they should carry the weight in any future
slot decision. e2b is genuinely weak elsewhere (file_summary 0, detailed_json 45,
weekend_fixed 45) and must not be promoted out of `summarize` on the strength of this.

### The vision slot is not measured by anything

`image_rename` and `image_rename_mixed` send `IMAGE_RENAME_PROMPT` as **text**. No eval
task in the suite feeds an actual image. Ten of the eleven installed models score 100
on both, and those 100s say nothing at all about vision — they measure whether a model
can emit a filename-shaped string. `best_models.vlm` is therefore a static-probe pick
(`vision_config` present in config.json, via `probe_vision` in `lib/model_caps.py`),
not a measurement. The probe barely narrows the field either: ten of eleven claim
vision. Backlog item 9 covers building a real one.

---

## Osaurus Server Rules

1. **Single instance only** - enforce it with `./tools/osaurus_one.sh`, do not rely on
   remembering. Two servers do not merely cause timeouts, they silently corrupt
   MEASUREMENTS: contention shows up as `HTTP 499 request_cancelled`, which from the
   client is indistinguishable from a slow model.
2. **Check before run**: `./tools/osaurus_one.sh --check` (exits 1 unless exactly one
   process is up, it holds the port, AND no other session holds the GPU lock).
   `osaurus status` is weaker -- `osaurus stop` leaves the process resident, so status
   can read "stopped" while the memory is still occupied.
3. **One session at a time, enforced by a machine-wide lock.** Several agent sessions
   now run on this Mac concurrently, and ONE healthy server is not enough on its own:
   restarting the server a peer is measuring against corrupts that run exactly as
   badly as a second server would. `/tmp/mac-osaurus-gpu.lock` (see `tools/gpu_lock.sh`
   and `lib/gpu_lock.py`) is held for the whole of an eval and for any server mutation.
   - Deliberately NOT the desktop lock at `/tmp/mac-desktop-ui.lock`. Different
     resources; sharing a name would make every screenshot run a false "GPU busy".
   - The two `quit app "osaurus"` call sites REFUSE, with a stated reason, when
     another session holds it. They do not queue: a tweet summary should not block
     for the hours an eval runs.
   - Its wedge ceiling measures PROGRESS, not duration -- the eval heartbeats after
     every task -- because an honest run holds the GPU for hours and a wall-clock
     ceiling would reclaim the lock out from under a healthy measurement.
   - Held by a dead session? It reclaims itself (PID plus process start time, so a
     recycled PID cannot impersonate the owner). Nothing to clean up by hand.
   - Blocked? `./tools/osaurus_one.sh --check` names the holder.
   - Why a lock rather than trusting the sample median: `eval/samples.py` estimates
     from the median of the last 5 CLEAN samples, but `machine_is_uncontended()`
     gates on SWAP and COMPRESSOR and cannot see the GPU, so a peer's eval is
     recorded as CLEAN and enters the median as though the box were quiet. And the
     median only protects a model that HAS history — a first measurement is its own
     estimate. The lock covers exactly what that guard is blind to.
4. **Response parsing** - Must read ALL chunks until `done=true`
5. **A contaminated measurement is permanent.** The recorders keep the SLOWEST
   observation, so delete the model's `_capabilities` from `conf/eval_signals.json`
   before re-measuring, or the bad number wins forever.

---

## Working Prompts

### Image Filenames
```
Give a short 2-4 word summary of: {text}
```
Max 35 chars, extract first 4-6 words.

### Weekend Tasks
```yaml
weekend_fixed: |
  Output JSON now. Schema: {"fixed_activities": [...]}
  {prompt}
  CRITICAL: Only use: target_ages, price, weather
  Output ONLY JSON.
```

---

## Known Issues

| Model | Issue | Fix |
|-------|-------|-----|
| gemma weather | Outputs weather data | Avoid for weekend |
| gemma-4-31b-jang | Cold start 30s then 1s | Warmup call first |
| qwen | Thinking tokens | Can't disable |
| jang models (MLX) | Wrong shape under stock `mlx_lm` | Use `mlx-vlm` (git main) or the Osaurus server |
| gemma-4-e4b | Input looping | Avoid |
| foundation (0%), gemma-4-e4b (0%) | Parrots ALL planted falsehoods (3/3) | Worst fact-checkers; use qwen-agentworld or ornith for factual summarization |
| gemma-4-12b (67%) | Resists most falsehoods (1/3 parroted) | Adequate for most use cases |
| diffusiongemma-26b (34%), qwen3.6-27b (34%), qwen3.6-35b (34%) | Resists few falsehoods (2/3 parroted) | Not reliable for truth-sensitive tasks |
| ornith-1.0-35b | Summarize_contradiction single-falsehood test produces false positive: model correctly flags the falsehood as "FAKE/SATIRE" but token-matching validator counts it as parroting. Multi-falsehood factual_accuracy test confirms 100%. | Use factual_accuracy test instead of summarize_contradiction |
| qwen-agentworld-35b (100%) | Resists ALL planted falsehoods reliably | Best fact-checker in the suite |

---

## Config Location

- `conf/config.toml` — models, timeouts, prompts
- `conf/models/*.toml` — per-model config
- `lib/config_core.py` — load functions (shim at `lib/config.py`)

## Universal Model Steering Directives (August 2026)

All 7 model configs (`foundation.toml`, `gemma.toml`, `gemma_versions.toml`, `laguna.toml`, `nemotron.toml`, `qwen.toml`, `qwopus.toml`) now incorporate 4 universal prompt steering rules:

1. **Context Bounding (`file_summary`)**:
   > *"Rely ONLY on provided content context. DO NOT infer functionality from file names, words, or puns (e.g. 'osaurus' is an LLM client server wrapper, not dinosaur data)."*
   - Prevents small local models from hallucinating domain stories based on filename tokens.

2. **Location Precision (`weekend_fixed` / `weekend_transient`)**:
   > *"location: Copy street address or city name. NEVER output generic 'Indoor venue' or 'Outdoor venue'."*

3. **Weather Enforcement (`weekend_fixed` / `weekend_transient`)**:
   > *"weather: 'indoor', 'outdoor' or 'both'. Venues with 'park', 'nature', 'garden', 'trail', or 'walk' in their name MUST be labeled 'outdoor'."*
   - Backed up by `OUTDOOR_MARKERS` in `weekend/enforce.py` to auto-correct inverted labels.

4. **Executive Narrative & Bracket Attributions (`summarize`)**:
   > *"Start with a brief ## Executive Summary paragraph... Use narrative verbs... Conclude EVERY bullet point with `(@username | Mon DD HH:MM)`."*

---

## The Working Prompt Pattern (April 2026)

**CRITICAL**: For weekend tasks, prompts must use RUNTIME PLACEHOLDERS, not {}. The model generates data with specified values.

```yaml
weekend_fixed: |
    Output JSON now. Schema: {"fixed_activities": [{"name": "str", "location": "str", "target_ages": "str", "price": "str", "weather": "str"}]}

    Extract 8-10 popular {location} venues for families with kids ages {age_range}.

    CRITICAL: Each item MUST have:
    - target_ages: "{age_range}"
    - price: "$18-35 per child" or "$25-35 per family"
    - weather: "indoor" or "outdoor"

    Output ONLY JSON.

  weekend_transient: |
    Output JSON now. Schema: {"transient_events": [...]}
    
    Find 5-10 events for {date_range} in {location}. Kids ages {age_range}.
    
    Use ONLY these values:
    - day: Friday, Saturday, or Sunday
    - target_ages: "{age_range}"
    - weather: "indoor" or "outdoor"
```

Key: `{location}`, `{age_range}`, `{date_range}` are INJECTED at runtime (`weekend/cli.py`), NOT {} placeholders.

---

## Field Normalization (Critical)

Different models output different field names. **All normalization must be in `normalize_llm_items()` in `weekend/cli.py`** - do not scatter it across the code.

Known aliases:
- **name**: `name`, `activity`, `activity_name`, `title`, `event`, `event_name`, `description`
- **location**: `location`, `address`, `venue`, `place`
- **target_ages**: `target_ages`, `age_group`, `ages`, `age_range`
- **price**: `price`, `cost`, `pricing`, `fee`
- **weather**: `weather`, `setting`, `type`, `indoor_outdoor`
- **day**: `day`, `date`, `dates`, `event_date`
- **duration**: `duration`, `end_date`, `time`

---

## Critical Config

| Constant | Value | Notes |
|----------|-------|-------|
| **Osaurs port** | **1337** | Check: `osaurus status` |

---

## Pre-Generated Baseline Data (2026)

**Approach**: Task is "extract from provided JSON context" not "generate events".

- Test data in `conf/eval_inputs.toml` (`[test_inputs]`)
- Pre-generated JSON with proper structure
- Models score on accurate extraction, not generation
- Consistent baseline across runs
- Avoids "refuses to generate fictional events" problem

### Test Data Locations:
- `conf/eval_inputs.toml`: `[test_inputs]` dict

---

## Known Issues (Additional)

| Model | Issue | Status |
|-------|-------|--------|
| lfm2-24b | Crashes server (OOM), 30m timeout | AVOID |
| gemma weekend | Refuses fictional event data; outputs weather or questions | WONTFIX |
| qwen filename | Empty response with complex prompts | FIXED (simpler prompt) |

---

## Contradiction / Faithfulness Test (July 2026)

Planted a falsehood in the input ("quantum giraffes of Manitoba won the Stanley Cup") and checked if models parroted it in the summary.

| Pass (100%) | Fail (0%) |
|-------------|-----------|
| qwen3.6-35b-a3b-mxfp8-mtp | foundation |
| gemma-4-12b-it-mxfp8 | qwen3.6-27b-mxfp8-mtp |
| | gemma-4-e4b-it-8bit |
| | diffusiongemma-26b-a4b-it-mxfp8 |
| | ornith-1.0-35b-mxfp8 |
| | qwen-agentworld-35b-a3b-mxfp8 |

**Implication**: Only the two largest/most capable models resist instruction-based falsehoods. All smaller or less-capable models parrot the planted fact. For quality-critical summarize tasks, use a passing model.

---

## Sweep results, 2026-08-12 (greedy decoding, derived timeouts)

Run by `benchmarks/sweep_models.sh`, one model at a time with a server restart between
each. Means over the 22 tasks common to all six completed models:

| mean | model | all-task mean |
|---|---|---|
| **88.2** | **muse-glimmer-30b-jang_6m** | 88.7 (23 tasks) |
| 82.5 | gemma-4-12b-it-mxfp8 | 83.3 |
| 82.1 | gemma-4-e4b-it-8bit | 82.9 |
| 81.2 | qwen3.6-27b-mxfp8-mtp | 81.2 |
| 80.0 | foundation | 80.9 |
| 73.3 | gemma-4-e2b-it-8bit | 74.4 |

Still pending when this was written: bonsai-27b, ornith-9b, ornith-35b, qwen3.6-35b.

**muse-glimmer is the only model that resists planted falsehoods.** It scored 100 on
`summarize_factual_accuracy` where the field scored 34, 34, 67, 0, 0 — it repeated none
of the three hoax tweets. That matches the July finding above that only the most capable
models filter them, and it is the single largest quality difference in the table. Its one
hard failure is `file_summary` (0%: "only 2 items, need 3+"), where gemma-4-e4b scores 100.

**Two models were recovered by deriving timeouts from measured rates.** muse-glimmer and
bonsai-27b previously produced nothing at all — every task hit a flat 900s ceiling, and
the abandoned requests leaked server inference slots until the server returned HTTP 503.
muse's `weekend_transient` alone takes 737s legitimately.

**Half the eval no longer discriminates.** 10 of the 22 common tasks score 100 for every
model: filename, filename_leak, filename_mixed, image_rename, image_rename_mixed, json,
rename_mixed, summarize, weekend_fixed_mixed, weekend_transient_schema. They still work as
regression floors, but they cost GPU time and separate nothing. The tasks that do
discriminate are file_summary, detailed_json, weekend_fixed, weekend_transient_mixed,
summarize_contradiction, summarize_factual_accuracy, and the three taxes tasks.

**Do not read the `summarize_factual_coverage` column across the fix boundary** — see below.

## `summarize_factual_coverage` scores before 2026-08-12 are not comparable

Every model that produced real results failed this task — 16, 11, 16, 33, 27 out of 100.
Identical failure across independent models looked like a prompt weakness. It was the
scorer: `validate_factual_coverage` matched each key fact as an exact case-insensitive
substring, while the prompt orders the model to reword. It was measuring verbatim copying.

Two controls settled it in minutes, neither needing a model:

| control | old scorer | fixed scorer |
|---|---|---|
| the source timeline (contains every fact by construction) | 94 | 100 |
| all 18 topics covered, in the model's own words | 5 | 77 |
| 4 of 18 topics | 5 | 16 |
| no facts at all | 0 | 0 |

The 94 was its own defect: `'Amazon launches drone delivery in Toronto'` is not a substring
of its source line, which says `'...in select Toronto neighborhoods'`. No output could ever
have matched that fact.

**Consequence for the leaderboard**: any `summarize_factual_coverage` figure recorded before
this fix was produced by a different instrument and must not be compared with one recorded
after. Re-run that single task per model rather than reasoning across the boundary.
`references/tests/test_factual_coverage.py` keeps both controls as gates.

---

## Strict Validation Rules (Updated 2026)

### Extraction Validation
- **>80% from source**: Required for passing
- **No hallucinated items**: Items must match input data
- **Completeness**: All input items should be in output

### file_summary Validator
- **No filename inference**: "a python script" = FAIL
- **Must have content verbs**: parse, validate, extract, load, read, write, etc.
- **Filename appearing in summary** = FAIL

---

## Model-Specific Prompts

All prompts in `conf/models/{model}.toml` must include:

```toml
# Required for JSON output
weekend_fixed: |
  Output JSON now. CRITICAL: Use EXACT schema: {schema}
  
  REQUIRED fields for EACH item:
  - name: str
  - location: str
  ...

  Output ONLY JSON. No extra text.

filename: |
  Output JSON now. Schema: {"filename": "str"}
  Output ONLY JSON.

summarize: |
  Output the summary in bullet points. Use ## headers.
```

---

## Model Quirks

### Foundation ✓ WORKS RELIABLY
- **Fast**: 8-15s for tasks
- **Clean JSON**: No markdown, no thinking
- **Source matching**: 100% (risky - may copy directly from input)
- **Synthesis weakness**: Scores only 52-58% on synthesis (no connecting narrative, no TL;DR). Despite unified summarization prompt, it still lists events without relationship language
- **Filename**: 97% quality, 0.6s avg — best speed-to-quality ratio

### Laguna-xs.2-mxfp4 ✓ BEST BALANCE
- **Emerges as top pick** from full quality eval (May 2026)
- **Filename**: 98% quality, 3.1s avg
- **Summarize**: 92% quality — best Synthesis of non-qwopus models
- **No failures**: 0 crashes across all 8 test cases
- **Note**: Relatively unknown model but beats qwen and nemotron on consistency

### Nemotron-3-nano-omni ⚠ INSTRUCTION LEAK
- **Instruction leak**: Often outputs `"Here is the filename: ..."` instead of the filename alone. Score drops to 50-74% on affected cases
- **Filename**: 84% avg (dragged down by leak), 3.3s avg
- **Summarize**: 89.5% — similar Synthesis weakness to foundation
- **Potential fix**: Prompt may need stricter instruction ("Output ONLY the filename, no explanation")

### Qwen Family
- **Requires**: "Output JSON now" trigger (for weekend tasks)
- **Thinking**: Plaintext blocks - handled by stripping
- **Key quirks**: Uses `category` → `target_ages`
- **qwen3.6-27b-mxfp8-mtp** ✓ best qwen: 99% filename, 100% summarize, 0 failures, 14.8s avg
- **qwen3.6-27b-mxfp4**: 93.8% filename, 100% summarize, 12.3s avg
- **qwen3.6-35b-a3b-mxfp4**: 93.8% filename, 94% summarize, 10.1s avg — good but not better than 27b variants
- **qwen3.6-35b-a3b-mxfp8-mtp** ✓ NOW WORKS: Previously consistently crashed on summarize/file_summary (returned empty) — may have been a server issue. July 2026 sweep: passes weekend_transient_schema (100%), filename_leak (100%). **Best all-rounder** (92% mean). NOTE: summarize_contradiction is **stochastic** (~33% pass rate) — sometimes resists falsehood, sometimes parrots. Not deterministic for truthfulness.

### Qwopus ⚠ HIGH QUALITY BUT UNRELIABLE
- **Best quality when it works**: 98.2% filename, 98.5% summarize
- **Only model with good synthesis (94%)**: Adds rich connecting narrative
- **BUT 40% failure rate on cold start**: Produces empty output randomly
- **Very slow**: 40-220s per call
- **Inconsistent**: Same model, same prompt, same case scored 96.2% in one run, 0% in another
- **Recommendation**: Only use for quality-critical batch work where failures are acceptable

### Gemma ✗ NOT SUITABLE FOR WEEKEND
- Returns weather data instead of events
- 0 items with details in tests
- Flat dicts instead of nested structure
- **gemma-4-e4b-it-8bit**: loads + generates via `mlx-vlm` (git main) at ~71 tok/s; stock `mlx_lm` returns empty (cannot build the multimodal arch).

### Minimax-m2.7-small-jangtq ✗ UNUSABLE
- **Extremely slow**: 400s+ per single filename call
- **Generic outputs**: 3/5 filename cases return "filename.txt" or similar
- **Complex tasks**: 100% failure on summarize and file_summary
- **Conclusion**: Not suitable for any ztools use case - remove from model list

---

## Signal/Noise Filtering (Mixed Eval)

Each eval task has a `*_mixed` variant that injects clearly-labeled NOISE into the
input and measures whether the model extracts ONLY the signal. Scores are
precision (noise excluded) + recall (signal kept). A model that dumps noise scores
lower, and the leakage is named explicitly in the failure reason
(`included N/M noise items` / `included N/M noise files`).

Sweep (2026-07, all models via `osaurus list`):

| Model | weekend_transient | weekend_fixed | summarize | file_summary | rename | noise leaked? |
|-------|-------------------|---------------|-----------|--------------|--------|---------------|
| foundation | 95 | 91 | 88 (1) | 100 | 100 | minor (1 tweet) |
| qwen3.6-27b-mxfp8-mtp | 100 | 91 | 75 (2) | 100 | 100 | moderate (2 tweets) |
| qwen3.6-35b-a3b-mxfp8-mtp | 100 | 91 | 88 (1) | 91 (6/6 files) | 100 | **dumps all noise files** |
| gemma-4-12b-it-mxfp8 | 100 | 91 | 88 (1) | 100 | 100 | minor (1 tweet) |
| gemma-4-e4b-it-8bit | 100 | 91 | 75 (2) | 100 | 100 | moderate (2 tweets) |
| diffusiongemma-26b-a4b-it-mxfp8 | 100 | 0 (failed) | 88 (1) | 100 | 100 | minor (1 tweet) |
| ornith-1.0-35b-mxfp8 | 100 | 91 | 50 (4) | 91 (6/6 files) | 100 | **worst**: 4 tweets + all noise files |
| qwen-agentworld-35b-a3b-mxfp8 | 100 | 91 | 88 (1) | 25 (missed all) | 100 | barely summarized |

Notes:
- **Weekend "missed 2/12" is a recall artifact, not noise leakage.** Every weekend
  task is noise-clean (precision 100%); the model rephrases ~2 item names so the
  fuzzy name-matcher misses them. Noise exclusion itself works (no model scored
  noise on weekend).
- **summarize_mixed** is the most discriminating: models leak 1-4 of 8 noise tweets.
  ornith (4/8 → 50%) and qwen27b/gemma-e4b (2/8 → 75%) are the worst; foundation /
  gemma-12b / qwen35b / diffusiongemma / qwen-agentworld leak only 1 (88%).
- **file_summary_mixed**: qwen35b and ornith INCLUDE all 6 noise files in output
  (flagged "included 6/6 noise files"); qwen-agentworld barely summarized (25%).
  All others filter clean (100%).
- **rename_mixed**: every model filters clean (100%) — the JSON-array format with
  explicit SNIPPETS/NOISE sections is unambiguous.
- The single 0-100 pass/fail number is no longer the signal — read the
  `included N/M noise` failure reason to see actual filtering quality.

---

## Eval Commands

```bash
# Quick single model eval
python3 -m eval --model foundation --quick --task filename

# Full eval
python3 -m eval

# Quick alias via shim
python3 model_eval.py --model foundation --task filename --quick

# Quality benchmark
python3 -m eval.benchmark_quality
```

---

## Key Files

- `eval/cli.py` - Eval runner CLI
- `eval/tasks_core.py` - Task definitions
- `eval/run.py` - Eval loop
- `lib/quality_models.py` / `lib/quality_scorers.py` - Quality evaluation
- `lib/validators/` - Validator implementations
- `conf/models/*.toml` - Model prompts

---

## Filename / Rename Task

Config-driven via `conf/config.toml` (the real values — this list must stay
above the first `[table]` header or TOML nests it inside that table and
`get_filename_models()` never sees it):
```toml
filename_models = [
  "foundation",
  "qwen-agentworld-35b-a3b-mxfp8",
  "qwen3.6-35b-a3b-mxfp8-mtp",
]

[prompts]
filename = """
Output ONLY the filename string (no JSON, no code blocks).
Use lowercase, underscores for spaces, no special characters.
Keep it under 50 characters.

TEXT: {text}
"""
```

Per-model templates in `conf/models/*.toml` may use either the positional `{}`
slot or `{text}`; `rename.llm` renders them through `lib.prompt_render`, never
`str.format()`.

MLX backend: OsaurusAI MXFP8/4 quants (Gemma4 `gemma4`/`gemma4_unified`, Qwen `qwen3_5_moe`) load via `mlx-vlm` (git main) — proven for `gemma-4-E4B-it-8bit` and `Qwen3.6-35B-A3B-MXFP8-MTP`. Stock `mlx-lm` supports plain-text qwen3_5/gemma4 only and rejects the multimodal checkpoints. Model discovery (`find_any_working_mlx_vlm_model`) scans dirs and load-probes via `mlx-vlm`.

## ornith-1.0 (9B, 35B) — unbounded reasoning, answer never emitted (2026-08-11)

Ornith returns its chain of thought in a separate `reasoning_content` field and
leaves `content` empty until the reasoning ends. On simple prompts it finishes
in ~230 completion tokens and answers correctly. On the harder eval tasks it
does not stop: given `max_tokens=16000` it spends all 16,000 on reasoning and
returns `finish_reason: length` with `content: ''`, which is the 523-second,
0-scoring `image_rename` and `image_rename_mixed` results in the sweep.

Confirmed reproducible, not sampling noise -- identical at temperature 0 and
0.1:

    max_tokens=120    finish=length   content=''         reasoning truncated
    max_tokens=512    finish=stop     content=valid JSON  234 completion tokens
    max_tokens=16000  finish=length   content=''          complex prompt, 523s

Two things this rules out. It is NOT a token-budget shortage: raising the budget
makes it worse, because the reasoning expands to fill whatever it is given. And
"Output JSON now." does NOT suppress it -- that string was in the prompt for
every run above. Whatever works for the qwen family does not transfer here.

Also note `content` is empty rather than absent, so a caller checking only for a
missing key sees a successful response with nothing in it.

Ornith has no `conf/models/*.toml`, so it currently runs on the built-in
fallback prompts. Any fix belongs there, and needs to make the model STOP
reasoning rather than give it more room.


## Decoding policy for the eval: greedy, decided 2026-08-12

`ev` pins temperature to 0 (`EVAL_TEMPERATURE`). Production runs at 0.1, so this
is deliberately NOT production-identical, and the reason is comparability: with
sampling on, ornith scored 100% and then 0% on an unchanged `image_rename`
across two runs, and a leaderboard built on that ranks the sampler.

The cost of the choice, recorded so it is not rediscovered as a surprise:
greedy scored ornith WORSE on 4 of 11 shared tasks than sampling did
(`image_rename` and `weekend_transient_mixed` both 100 -> 0). Greedy decoding is
prone to repetition loops, which is plausibly what feeds ornith's unbounded
reasoning. So a greedy-sensitive model can rank below where it would ship.

What this means in practice:
- Cross-model rankings and before/after prompt comparisons are valid, because
  every model is measured the same way.
- ABSOLUTE scores are not a prediction of production quality.
- Re-validate the winning model at temperature 0.1 before writing it into
  `best_models`, since that is the setting it will actually run under.
