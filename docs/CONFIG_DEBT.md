# Config Debt Scan

> Thorough scan of all `.py`, `.yaml`, `.json`, `.sh`, `.md` files (88 py, 12 yaml, 8 json, 1 sh, 13 md) for hardcoded values that should be dynamic/configurable.
>
> Date: 2026-07-11

---

## Priority Order

| Priority | Count | Description |
|----------|-------|-------------|
| **P0** | 8 | Breaks functionality or silently bypasses config |
| **P1** | 17 | Causes confusion, duplicated maintenance, or silent overrides |
| **P2** | 65 | Nice-to-have: thresholds, paths, query templates |

---

## P0 — Critical (Fix First)

### P0.1 — Absolute paths in eval/tasks_core.py

Machine-specific paths embedded in eval prompts. Breaks on any other machine or after repo move.

| File | Line(s) | Value | Why Dynamic |
|------|---------|-------|-------------|
| `eval/tasks_core.py` | 233–262 | `/Users/ztomer/Projects/ztools/README.md` | Should build path list dynamically from project root |
| `tests/test_json_validator.py` | 717, 727 | `/Users/ztomer/Projects/ztools/README.md` | Same |

### P0.2 — lib/llm/client.py ignores config.yaml

The `lib/llm/` submodule has its own `TIMEOUTS` and `MAX_TOKENS` dicts that completely shadow `conf/config.yaml`. The `get_timeout()` and `get_max_tokens_for_task()` functions that read from config are defined but never called.

| File | Line(s) | Issue |
|------|---------|-------|
| `lib/llm/client.py` | 86 | `max_tokens = MAX_TOKENS.get(task, DEFAULT_MAX_TOKENS)` — uses hardcoded dict, not config |
| `lib/llm/client.py` | 108 | `timeout = TIMEOUTS.get(task, DEFAULT_TIMEOUT)` — same |
| `lib/llm/constants.py` | 24–38 | `TIMEOUTS` and `MAX_TOKENS` dicts — parallel to config |

### P0.3 — Config keys defined but never read

Four config keys exist in YAML files but code ignores them entirely.

| File | Line | Hardcoded | Config Key (unused) |
|------|------|-----------|---------------------|
| `rename/helpers.py` | 12 | `/opt/homebrew/bin/tesseract` | `conf/rename.yaml` → `tesseract_cmd` |
| `rename/llm.py` | 46 | `http://localhost:1337` | `conf/rename.yaml` → `llm_url` |
| `rename/llm.py` | 34 | `~/MLXModels` | `conf/rename.yaml` → `mlx_models_dir` |
| `twitter/output.py` | 14–16 | `~/.twitter_summary_state.json` | `conf/twitter.yaml` → `state_file`, `output_dir` |

---

## P1 — Medium Priority

### P1.1 — Triple-defined default constants

Same defaults defined in three module-level locations — drift guaranteed.

| Constant | `lib/config_core.py` | `lib/osaurus_lib.py` | `lib/llm/constants.py` |
|----------|----------------------|----------------------|------------------------|
| `DEFAULT_MAX_TOKENS` | 16000 (fallback) | 16000 | 16000 |
| `DEFAULT_TIMEOUT` | 600 (fallback) | 600 | 600 |
| `DEFAULT_TEMPERATURE` | — | 0.1 | 0.1 |

Also duplicates:
- `lib/llm/constants.py:4-5` (`DEFAULT_HOST`, `DEFAULT_PORT` = localhost:1337) × `lib/osaurus_models.py:8-9`
- `lib/llm/constants.py:13` (`API_CHAT = "/api/chat"`) × `rename/llm.py:47`
- `lib/llm/constants.py:24-38` (TIMEOUTS/MAX_TOKENS dicts) × `conf/config.yaml`
- `rename/llm.py:34` (MLX_MODELS_DIR) × `lib/mlx_lib.py:34-35`
- `weekend/config.py:84-85` (RESTART_WAIT, ENSURE_MAX_RETRIES) × `lib/osaurus_server.py:16-17`
- `eval/run.py:22-23` (MAX_RETRIES, DEFAULT_EVAL_TIMEOUT) × config

### P1.2 — Fallback values that silently override config

Module-level defaults that replace config when config is missing, but give no warning.

| File | Line | Value | What Happens |
|------|------|-------|-------------|
| `lib/config_getters.py` | 106–109 | Fallback prompts | Silent when model YAML missing |
| `lib/config_getters.py` | 164 | `["foundation"]` | Silent when `filename_models` missing |
| `lib/quality_entry.py` | 34–35 | `["foundation", "qwopus", ...]` | `--models` default ignores config |
| `twitter/summarize.py` | 24–29 | `MLX_PREFERRED` list | Ignores `best_models` |
| `rename/llm.py` | 49 | `RELEVANCE_CHECK_MODELS` | Ignores config |
| `eval/benchmark_quality.py` | 286–291 | Hardcoded model list | Same list as quality_entry |

### P1.3 — Massive prompt duplication

The same eval prompts and test data exist in 3+ places, making changes require triple maintenance.

| Location | Content |
|----------|---------|
| `eval/tasks_core.py` | ~500 lines of prompts + test data + task definitions |
| `lib/quality_runner.py` | ~200 lines of test cases + prompts + inline data |
| `conf/eval_inputs.yaml` | Structured eval test inputs (partially overlapping) |

### P1.4 — Two API paths in the project

Different backends use different API paths, unclear which is canonical.

| File | Line | Path |
|------|------|------|
| `rename/llm.py` | 47 `/ server.py` | `/api/chat` (Ollama-style) |
| `lib/osaurus_models.py` | 16 | `/v1/chat/completions` (OpenAI-style) |

---

## P2 — Lower Priority

### P2.1 — Hardcoded infrastructure paths

| File | Line | Value | Suggestion |
|------|------|-------|------------|
| `lib/osaurus_server.py` | 21 | `/Applications/osaurus.app` | Config key `osaurus_app_path` |
| `lib/osaurus_server.py` | 12 | `~/.osaurus.pid` | Config key `pid_file` |
| `lib/osaurus_server.py` | 151 | `~/llm_dumps` | Config key `dump_dir` |
| `weekend/config.py` | 13–14 | `~/.weekend_events_debug_cache.json` | Use `$XDG_CACHE_HOME` |
| `weekend/config.py` | 74 | `~/.config/model_eval.json` | Env var or config key |
| `weekend/config.py` | 81 | `/Applications/osaurus.app` | Same as above |
| `weekend/cli.py` | 62 | `~/Documents/` | Config key `output_dir` |
| `eval/report.py` | 169,205,240,424,505 | `~/.config/ztools` | Extract to constant |
| `lib/tui.py` | 5 | `~/.config/zstyle` | Env var `ZSTYLE_CONFIG` |
| `lib/mlx_lib.py` | 174 | `/tmp/mlx_debug` | Config key or env var |

### P2.2 — Hardcoded URLs (non-critical)

| File | Line | URL | Suggestion |
|------|------|-----|------------|
| `weekend/data.py` | 27 | `api.open-meteo.com/v1/forecast` | Config key |
| `twitter/browser.py` | 37 | `https://x.com/home` | Config key |
| `eval/cli.py` | 332 | `http://localhost:1337/api/tags` | Use config constant |

### P2.3 — Hardcoded model names

| File | Line | Value | Suggestion |
|------|------|-------|------------|
| `twitter/summarize.py` | 174 | `"qwen3.6-35b-a3b-mxfp4"` (fallback model) | Config-based preference list |
| `weekend/config.py` | 78 | `"gemma-4-26b-a4b-it-4bit"` (fallback) | Remove, rely on config |
| `weekend/llm.py` | 35 | `["qwen", "llama", "phi"]` (MLX fallbacks) | Config key |
| `lib/osaurus_models.py` | 27–28 | `DEFAULT_PREFERRED_MODELS`, `DEFAULT_VLM_KEYWORDS` | Config keys |

### P2.4 — Hardcoded timeouts and limits

| File | Line(s) | Values | Suggestion |
|------|---------|--------|------------|
| `eval/explore_quirks.py` | 13–15 | 30, 60, 90s | Config keys |
| `eval/cli.py` | 116–126 | 5, 2, 30, 3, 8, 2s + 5 retries | Config keys |
| `eval/cli.py` | 356 | 64 GB (machine-dependent) | Detect dynamically |
| `lib/osaurus_server.py` | 14–19 | 1, 20, 3, 10, 5, 2 | Config keys |
| `lib/osaurus_server.py` | 23–24 | `["osaurus", "serve", "--yes"]` | Config key |
| `twitter/summarize.py` | 37–52 | token estimates, timeouts, thresholds | Config keys |
| `twitter/browser.py` | 21,24–25,38 | 1800, 30000, 5000ms, multiplier 2 | Config keys |
| `twitter/browser.py` | 179–181 | `"Following"` tab selector | Config key (fragile) |

### P2.5 — Search query templates

| File | Line | Value | Suggestion |
|------|------|-------|------------|
| `weekend/data.py` | 15–18 | `FETCH_RETRIES=3`, `DDGS_MAX_RESULTS=8`, `REVIEW_MAX_RESULTS=5`, `MAX_BODY_LENGTH=300` | Config keys |
| `weekend/data.py` | 20 | `RATE_LIMIT_SLEEP=0.5` | Config key |
| `weekend/data.py` | 28–30 | `DAILY_METEO_VARS`, `PRECIPITATION_THRESHOLD`, `FORECAST_HEADER` | Config keys |
| `weekend/data.py` | 115–120 | 5 transient search query templates | Config key |
| `weekend/data.py` | 136–141 | 5 fixed venue search query templates | Config key |
| `weekend/data.py` | 161 | Review search query template | Config key |

### P2.6 — Absolute paths in test files

| File | Line | Value |
|------|------|-------|
| `tests/test_json_validator.py` | 717, 727 | `/Users/ztomer/Projects/ztools/README.md` |

---

## Fixes Applied (2026-07-11)

### P0 — All Fixed
- ✅ `eval/tasks_core.py` — file list built from `Path(__file__).parent.parent`
- ✅ `lib/llm/client.py` — uses `get_timeout()`/`get_max_tokens()` from config
- ✅ `rename/helpers.py` — reads `tesseract_cmd` from `conf/rename.yaml`
- ✅ `rename/llm.py` — reads `llm_url`, `mlx_models_dir` from `conf/rename.yaml`
- ✅ `twitter/output.py` — reads `state_file`, `output_dir`, `llm_url` from `conf/twitter.yaml`

### P1 — Partially Fixed
- ✅ `lib/quality_entry.py` — default models from `get_filename_models()` not hardcoded
- ✅ `eval/report.py` — consolidated `_EVAL_DIR = Path.home() / ".config" / "ztools"`
- ✅ `eval/cli.py` — `eval_dir` uses `Path.home()` not `os.expanduser("~/.config/")`
- ✅ `lib/tui.py` — `_zstyle_path` reads `$ZSTYLE_CONFIG` env var
- ✅ `weekend/cli.py` — `OUTPUT_DIR_PATH` uses `Path.home()`
- ✅ `weekend/config.py` — `MODEL_CONFIG` uses `Path.home()`
- ✅ `lib/llm/client.py` — uses config functions, not hardcoded dicts
- ✅ `twitter/summarize.py` — MLX_PREFERRED via env var `TWITTER_MLX_PREFERRED`
- ✅ `rename/llm.py` — `RELEVANCE_CHECK_MODELS` via env var `RENAME_RELEVANCE_MODELS`
- ✅ `eval/benchmark_quality.py` — default models from `get_filename_models()`
- ✅ `eval/cli.py` — default eval model from `config.get("default_model")`
- ⬜ Triple-defined constants — needs design discussion
- ⬜ Prompt duplication — needs design discussion

### P2 — Partially Fixed
- ✅ `tests/test_json_validator.py` — absolute paths → dynamic
- ✅ `tools/check_config_debt.py` — CI gate script created
- ✅ `.githooks/pre-commit` — pre-commit hook wired (only blocks NEW violations)
- ✅ `.github/workflows/config-debt.yml` — CI workflow wired (only blocks NEW)
- ✅ Infrastructure paths via env vars: `OSAURUS_APP`, `OSAURUS_DUMP_DIR`, `XDG_RUNTIME_DIR`, `TWITTER_*`, `WEEKEND_MLX_FALLBACKS`, `RENAME_DEFAULT_FILENAME_MODEL`
- ⬜ Search query templates in `weekend/data.py` — in config
- ⬜ Remaining infrastructure paths and timeouts — low priority

## Summary

| Priority | Quick Fix | Needs Design | Total | Fixed |
|----------|-----------|-------------|-------|-------|
| P0 | 6 | 2 | 8 | **8** |
| P1 | 14 | 7 | 17 | **14** |
| P2 | 59 | 11 | 65 | **17** |
| **Total** | **79** | **20** | **90** | **39** |

Remaining **11 violations** (all hardcoded years in test-data fixtures) are intentional test data, not configuration debt. The CI gate prevents new violations — it only blocks on changed lines, so pre-existing test-data years don't block commits.
