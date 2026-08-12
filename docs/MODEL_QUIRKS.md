# Model Quirks & Best Practices

**Updated: June 2026** — canonical reference for model selection and prompts.

---

## TL;DR Cheat Sheet

Start server: `osaurus serve &>/dev/null & sleep 10`

---

## Best Models by Task

| Task | Model | Speed | Command |
|------|-------|-------|---------|
| eval weekend | qwen3.6-35b-a3b-mxfp8-mtp | ~45s | `python3 -m eval --task weekend` |
| rename | laguna-xs.2-mxfp4 | 2.8s | `python3 -m rename` |
| summarize | gemma-4-12b-it-mxfp8 | ~30s | `python3 -m twitter` |
| think/analysis | qwen3.6-35b-a3b-mxfp8-mtp | ~45s | `python3 -m eval --task weekend` |
| json/schema | qwen3.6-35b-a3b-mxfp8-mtp | ~45s | `python3 -m eval --task weekend_transient_schema` |
| filename | foundation | 0.6s | `python3 -m rename` |
| filename (fallback) | laguna-xs.2-mxfp4 | 3.1s | `python3 -m rename` |

---

## Osaurus Server Rules

1. **Single instance only** - Multiple cause timeouts
2. **Check before run**: `osaurus status`
3. **Response parsing** - Must read ALL chunks until `done=true`

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
