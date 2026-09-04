# Bug report: Twitter summarizer LLM + MLX fallback chain

Symptom (observed run):

```
· Trying qwen-agentworld-35b-a3b-mxfp8 (29 tweets)...
⚠ qwen-agentworld-35b-a3b-mxfp8 error: HTTPConnectionPool(host='localhost', port=1337): Read timed out. (read timeout=495)
· Trying foundation (29 tweets)...
⚠ foundation error: 500 Server Error: Internal Server Error for url: http://localhost:1337/v1/chat/completions
· Sending 0/36 tweets to MLX model Qwen-AgentWorld-35B-A3B-MXFP8 ...
MLX model error: [MLX GENERATE ERROR] NameError: name 'MLX_MAX_TOKENS' is not defined
⚠ All server models failed
⚠ [LLM error: both local MLX and server failed]
```

Osaurus **is** running and reachable (`/v1/models` and `/v1/chat/completions` both 200 for
small requests). The failure is not a connection failure — it is a cascade of three separate
defects that together make the whole fallback chain collapse.

---

## Bug 1 — `foundation` fallback overflows its context window (server HTTP 500)

**File:** `twitter/summarize.py`

**Root cause.** The prompt is sized once against a single fixed budget
(`OSAURUS_CONTEXT_WINDOW`, default 8192 tokens) in `summarize_with_llm`, and
`_summarize_with_model` / `call_fn` rebuild the *same large prompt* for **every** server
model — regardless of that model's actual context window.

`call_with_fallback` then hands the identical oversized prompt to `foundation`, whose context
window is far smaller than 8192 tokens. Osaurus rejects it:

```
{"error":{"message":"Exceeded model context window size","type":"internal_error"}}  # HTTP 500
```

Reproduced directly: a ~49 KB prompt to `model=foundation` returns HTTP 500 "Exceeded model
context window size"; a short prompt returns 200. So the 500 is deterministic context overflow,
not a transient server error.

**Class of bug:** prompt sized from a *global* budget, not the *per-model* context window
(cf. CLAUDE.md rule 6 / rule 12 — one instance of "sized from the wrong signature").

**Secondary defect (same file):** the default fallback list is stale.
`TWITTER_FALLBACK_MODELS` defaults to `qwen3.6-35b-a3b-mxfp4,foundation`, but the server
serves `qwen3.6-35b-a3b-mxfp8-mtp` (mxfp8, not mxfp4). The `mxfp4` entry can never match a
served model, so it is a dead fallback slot.

**Tertiary defect (same file):** `get_available_models()` in `summarize_with_llm` is called
with **no host argument**, so it always probes `localhost:1337` and ignores `--host` /
`$OLLAMA_BASE_URL`. On a non-default host this yields an empty model list and a spurious
`ensure_server()` restart of the wrong (local) server. Not the cause of *this* run (host was
default) but the same connect-path defect and worth fixing together.

### Fix plan (Bug 1)

1. Size the prompt **per model**. When the model being tried is `foundation` (or any model
   with a known small window), rebuild the prompt against that model's real context window
   instead of the global `OSAURUS_CONTEXT_WINDOW`. Add a per-model context lookup (env-driven
   override `TWITTER_FOUNDATION_CONTEXT_WINDOW`, default conservative e.g. 4096 tokens) and
   pass a `ctx_chars` computed from it into `_build_prompt`.
2. Make `_summarize_with_model` accept a per-model `ctx_chars` rather than the single shared
   value computed once in `summarize_with_llm`.
3. Pass the caller's `base_url` into `get_available_models(base_url)` so the model probe and the
   restart decision honor `--host` / `$OLLAMA_BASE_URL`.
4. Fix the stale default fallback model id (`mxfp4` → `mxfp8-mtp`) OR make the fallback list
   filter against the live `/v1/models` list so a stale id degrades gracefully instead of
   wasting an attempt.

### Test plan (Bug 1)

- Unit: `_build_prompt` with a small `max_chars` produces a prompt whose estimated token count
  is under the foundation window (non-tautological: assert the char budget is actually smaller
  for foundation than for the primary model).
- Integration (mocked LLM layer): `summarize_with_llm` where the primary model returns
  a timeout and `foundation` is the fallback — assert the prompt handed to `foundation` is
  sized to the foundation window, and that a stale/unavailable fallback id is skipped, not
  attempted against the server.

---

## Bug 2 — MLX direct fallback is dead: `NameError: MLX_MAX_TOKENS`

**File:** `lib/mlx_lib.py`, function `call_mlx` (the generated subprocess script,
lines ~185–216).

**Root cause.** `call_mlx` writes an MLX generation script to a temp file using a Python
**f-string**. Inside that f-string, real interpolations use single braces
(`{model_path_escaped}`) and literal braces are escaped as `{{ }}`. The generation call is:

```python
for r in stream_generate(model, tokenizer, prompt, max_tokens=MLX_MAX_TOKENS):
```

`MLX_MAX_TOKENS` is a **bare name** — it is neither interpolated (single braces) nor a defined
name **inside** the emitted script. It only exists in the parent process. The emitted script
therefore references an undefined global, and every MLX generation dies with:

```
[MLX GENERATE ERROR] NameError: name 'MLX_MAX_TOKENS' is not defined
```

Because `call_mlx` is the shared entry point for the direct-MLX fallback, **the entire MLX
fallback path has been inert** — it always errors out before generating a single token.

**Class of bug:** value that lives in the parent scope referenced by name inside a generated
child script instead of being interpolated as a literal (a code-generation seam defect).

### Fix plan (Bug 2)

1. Interpolate the value into the generated script: `max_tokens={MLX_MAX_TOKENS}` so the
   emitted script contains the literal integer.
2. Audit the rest of the emitted script for any other bare parent-scope names (there are none
   currently besides `MLX_MAX_TOKENS`, but grep the generated block to be sure — fix the class,
   not just the instance).

### Test plan (Bug 2)

- Unit: render the generated script (refactor the script body into a builder that returns the
  string, or assert on the file contents) and assert it contains `max_tokens=8192` (the literal)
  and does **not** contain the bare token `MLX_MAX_TOKENS`. This is the regression that would
  have caught the NameError without needing a real model load.
- Break-the-code check: temporarily revert to the bare name, confirm the assertion goes red.

---

## Order of work

Fix Bug 2 first (one-line + test; restores the last-resort fallback so the tool can degrade),
then Bug 1 (per-model prompt sizing + host plumbing + stale fallback id).

---

## Resolution (implemented)

**Bug 2** — `lib/mlx_lib.py`: the generated script now interpolates the literal
(`max_tokens={MLX_MAX_TOKENS}` in the f-string → `max_tokens=8192` in the emitted script).
Regression test `test_generated_script_interpolates_max_tokens` in `tests/test_mlx_lib.py`
captures the emitted script and asserts the literal is present and the bare name is absent.
Break-the-code check confirmed the test goes red with the old code.

**Bug 1** — `twitter/summarize.py`:
- Added `_context_window_for_model` / `_ctx_chars_for_model`. `foundation` is sized against
  `FOUNDATION_CONTEXT_WINDOW` (env `TWITTER_FOUNDATION_CONTEXT_WINDOW`, default 4096) instead of
  the 8192 server window. The output reserve is capped at half the window so a small window does
  not collapse the input budget to zero.
- `summarize_with_llm` and `call_fn` size the prompt **per model** (foundation now gets ~6 KB /
  8 tweets and returns HTTP 200 instead of the previous 500 context overflow).
- `get_available_models(base_url)` now honors `--host` / `$OLLAMA_BASE_URL`.
- Default `TWITTER_FALLBACK_MODELS` no longer ships the stale `qwen3.6-35b-a3b-mxfp4` id; the
  fallback list is filtered against the live `/v1/models` list (keeping `foundation` as the
  on-device last resort) so a stale id no longer wastes a request/timeout cycle.

Verified: `foundation` returns 200 on a right-sized prompt; MLX fallback no longer NameErrors;
`tests/test_twit_summarize.py` and `tests/test_mlx_lib.py` green; ruff clean.

## MLX route research (corrected conclusion)

The Osaurus-installed models are **not** opaque mlx-swift blobs. They are standard
HF-structured checkpoints (`config.json`, `tokenizer.json`, `model.safetensors.index.json`):

| Model | `model_type` | Quant |
|---|---|---|
| `gemma-4-E4B-it-8bit`, `gemma-4-12B-it-MXFP8` | `gemma4` / `gemma4_unified` | MXFP8 |
| `Qwen3.6-*-MXFP8-MTP`, `Qwen-AgentWorld-*-MXFP8` | `qwen3_5_moe` | MXFP8 |

Stock `mlx_lm` (text-only) rejects them: `ValueError: Received 126 parameters not in model`
(the vision/audio towers + MoE heads it cannot construct). The package that CAN build these
architectures is **`mlx-vlm` from git main** (NOT PyPI `0.6.4`, which predates PR #1523 — the
audio-conv-weight layout fix these checkpoints need).

BLOCKER (verified live): `mlx-vlm@git` HEAD calls `load()` which builds a
processor (`Gemma4Processor`) that does **not exist in any installable `transformers`**
(>=5.14.0 is the floor mlx-vlm declares, and 5.14.0/5.14.1 both lack it). So `load()`
fails at the processor step with `ModuleNotFoundError: Gemma4Processor` under `rtk uv run
--with mlx --with torch --with "mlx-vlm @ git+..."`. An earlier generate that appeared to
work was a transient uv-cache state; it is NOT reproducible. **Conclusion: the on-device
mlx-vlm path is best-effort and currently blocked by an upstream mlx-vlm/transformers
version drift. The Osaurus server is the reliable primary.**

What IS proven working (live):
- `summarize_with_llm` against the running Osaurus server (`:1337`) produces a real
  structured summary from `gemma-4-e4b-it-8bit` (headers + bullets). Server path = verified.
- The fallback cascade reaches the mlx-vlm stage and degrades gracefully: dependency
  errors are detected (`is_vlm_dependency_error`) and reported clearly instead of
  crashing on a decode/import traceback.

Decision (user, corrected): Osaurus server first; on failure restart+retry the server;
only if the server is down do we try the Python mlx-vlm route as a last resort. Invoke
`mlx-vlm` via the **uv git+https URL** (floating HEAD). Implementation in `lib/mlx_vlm.py`
(`probe_mlx_vlm_loadable`, `call_mlx_vlm`, `find_*_mlx_vlm_model`, `is_vlm_dependency_error`)
and `twitter/summarize.py` (`_direct_mlx_fallback` → vlm stage then lm stage; `mlx_fn`
= server restart+retry first, then mlx-vlm).

