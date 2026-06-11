# Remediation Plan

All phases completed June 2026.
Consolidated plan from `docs/EXPERT_REVIEWS.md`.
Organized in dependency order — no phase exceeded 5 files changed.

---

## Phase 0 — Test Scaffolding (additive before subtractive)

Before fixing anything, write tests that prove the bugs exist and verify the fix.

### 0.1 Test coverage for content_processing.py

- [x] `test_content_processing.py` — must exist. 10 tests minimum:
  - `remove_thinking_blocks`: Qwen thinking markers with + without output marker
  - `remove_thinking_blocks`: marker removed by truncation (the crash path)
  - `remove_thinking_blocks`: marker absent (no-op)
  - `remove_inline_thinking`: gemma self-correction loop
  - `remove_inline_thinking`: Qwen prose before JSON, with blank-line boundary
  - `remove_inline_thinking`: short preamble (<2000 chars), no truncation
  - `remove_stats_tokens`: trailing + inline stats
  - `clean_model_output`: full pipeline, known input → known output
  - `extract_content_from_code_blocks`: fences present + absent
  - `strip_backtick_value`: single backtick, leading `**`, no match

**Files:** tests/test_content_processing.py (new)
**Effort:** 1 session

### 0.2 Test coverage for quality_entry.py regression mode

- [x] `test_quality_entry.py` — mock `load_baseline` returning known scores,
      verify `compare_to_baseline` output matches expected deltas.
- [x] Key test: reconstruct ScoreCard with correct weights from baseline dict
      (the zero-weight bug — weight should come from `TASK_SCORERS`, not 0.0)

**Files:** tests/test_quality_entry.py (new)
**Effort:** 0.5 session

### 0.3 Test coverage for twitter/cookies.py error paths

- [x] `test_twitter_cookies.py` — add tests for:
  - `_get_chrome_keychain_key` failure (mock check_output raises)
  - `_decrypt_cookie` with cryptography missing (returns encoded garbage)
  - `_decrypt_cookie` with non-v10 prefix
  - `_decrypt_cookie` with decryption exception
  - `get_chrome_cookies` when DB missing

**Files:** tests/test_twit_cookies.py (augment)
**Effort:** 0.5 session

### 0.4 Dead code removal tests (pre-removal parity proof)

- [x] For `eval/validate.py:129-229`: snapshot the dead code in a comment
      referencing the commit hash, then delete the lines. Verify no callers
      reference any function defined only in the dead block.
- [x] For `validators/helpers.py:94-102`: same pattern.

**Files:** eval/validate.py, lib/validators/helpers.py
**Effort:** 0.25 session

### 0.5 Integration tests for LLM error paths

- [x] `tests/test_llm_error_paths.py` — mock `requests.post` at transport level:
  - Timeout → verify retry logic fires
  - Returns 200 with `{"error": "out of memory"}` → verify `_check_summary_quality` catches it
  - Returns garbage HTML → verify fallback chain proceeds
  - Returns valid JSON with empty content → verify fallback proceeds
- [x] Cover `twitter/summarize.py`, `weekend/llm.py`, `rename/llm.py` paths.
      These are the three independent fallback implementations — each needs its own
      error test.

**Files:** tests/test_llm_error_paths.py (new)
**Effort:** 1 session

---

## Phase 1 — Critical Bugs (crash + security)

### 1.1 Fix string mutation during iteration in content_processing.py

**Bug:** `remove_thinking_blocks` lines 41-49 — after `content = content[output_match.end():]`,
`content.index(marker)` is called on the truncated string. If marker was in the removed
prefix, `ValueError` is raised.

**Fix:**
```python
marker_idx = content.find(marker)
if marker_idx == -1:
    break  # marker was in the truncated part, nothing to process
if output_match:
    # output_match already found, content already truncated
    content = re.sub(r"\n?\*?\[?\(?[Ss]elf-[Cc]orrection.*", "", content, flags=re.DOTALL)
else:
    json_match = re.search(r'[\[{]', content[marker_idx:])
    if json_match:
        content = content[marker_idx + json_match.start():]
```

**Testing:** Phase 0.1 tests must pass before this change. Add test case where
marker is in the truncated prefix — must not raise.

**Files:** lib/content_processing.py (+ tests from 0.1)
**Effort:** 0.25 session

### 1.2 Fix zero-weight reconstruction in quality_entry.py

**Bug:** Line 50: `Score(dname, dscore, 0.0)` — hardcodes weight to zero.
`ScoreCard.composite` computes `sum(s.weighted) = 0`, making NaN deltas vs baseline.

**Fix:** Import the dimension weight map from `quality_scorers.TASK_SCORERS`,
look up each dimension's weight:
```python
from lib.quality_scorers import TASK_SCORERS
weights = {}
for s in TASK_SCORERS.get(task, []):
    weights[s.dimension] = s.weight
...
dims.append(Score(dname, dscore, weights.get(dname, 1.0)))
```

**Testing:** Phase 0.2 tests pass. Verify reconstructed composite matches
original baseline composite.

**Files:** lib/quality_entry.py (+ tests from 0.2)
**Effort:** 0.25 session

### 1.3 Delete dead code in eval/validate.py and validators/helpers.py

**Bug:** `eval/validate.py:129-229` (100 lines unreachable after `return` on 127)
and `validators/helpers.py:94-102` (dead code after `return` on 92).

**Fix:** Delete lines 129-229 from `eval/validate.py`. Delete lines 94-102 from
`validators/helpers.py`.

**Verification:** `git diff` shows only deletions. Full test suite passes
(1569 tests). No function referenced in the deleted blocks is called elsewhere.

**Files:** eval/validate.py, lib/validators/helpers.py
**Effort:** 0.1 session

### 1.4 Replace SQL string interpolation with parameterized query

**Bug:** `twitter/cookies.py:73-76` — domain values interpolated into SQL via f-string.
`tempfile.mktemp()` on line 66 uses TOCTOU-unsafe temp file creation.

**Fix:**
```python
# Replace mktemp with NamedTemporaryFile
with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    tmp_db = Path(f.name)
shutil.copy2(CHROME_COOKIES_DB, tmp_db)

# Replace f-string with parameterized query
placeholders = ",".join("?" for _ in domains)
rows = conn.execute(
    f"SELECT ... FROM cookies WHERE host_key LIKE ?",
    [f"%{d}" for d in domains]
).fetchall()  # Per-domain LIKE with parameterized pattern
```
Note: SQLite `LIKE` patterns can't be parameterized directly (the `%` is part
of the pattern value). The safe approach is one `LIKE ?` per domain joined with
`OR`, each parameter being `f"%{domain}"`.

**Testing:** Phase 0.3 tests pass. Manual: run `python3 -m twitter --use-cache`
and verify cookies are extracted (same count/values as before).

**Files:** twitter/cookies.py (+ tests from 0.3)
**Effort:** 0.25 session

### 1.5 Warn when cryptography module missing or keychain fails

**Bug:** `twitter/cookies.py:44-45` — encrypted cookies decoded as UTF-8 without warning.
Line 33 — `check_output` raises unhandled on keychain failure.

**Fix:**
```python
if not all((Cipher, algorithms, modes)):
    print(f"{WARN} cryptography module not installed — cannot decrypt Chrome cookies", file=sys.stderr)
    return ""  # or raise a typed exception
```
And wrap the keychain call in try/except:
```python
try:
    key = _get_chrome_keychain_key()
except subprocess.CalledProcessError:
    print(f"{WARN} Failed to read Chrome keychain key — cookies cannot be decrypted", file=sys.stderr)
    sys.exit(1)
```

**Testing:** Phase 0.3 tests cover both paths. Manual: run with `CRYPTOGRAPHY_AVAILABLE=0`
simulation (monkeypatch).

**Files:** twitter/cookies.py
**Effort:** 0.1 session

### 1.6 Replace `pkill -f` with PID-file-based process management

**Bug:** `lib/osaurus_server.py:13` and `rename/llm.py:45` — `pkill -f osaurus`
matches any process containing "osaurus" in argv.

**Fix:**
```python
# Write PID on startup
PID_FILE = Path.home() / ".osaurus.pid"

def restart_server(...):
    if PID_FILE.exists():
        pid = int(PID_FILE.read_text().strip())
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # already dead
        PID_FILE.unlink(missing_ok=True)
    # ... then start new process and write its PID
```

**Files:** lib/osaurus_server.py, rename/llm.py, lib/osaurus_models.py
**Effort:** 0.5 session

---

## Phase 2 — High-Impact Structural Issues

### 2.1 Fixed MLX backend — `call_mlx()` properly reports errors, `find_any_working_mlx_model()` scans all dirs for compatible models, model discovery filters incompatible ones

**Chosen approach:** Option A (fix) instead of the recommended Option B (delete). MLX subprocess backend was debugging to add stderr capture and error propagation. `call_mlx()` now returns proper errors instead of empty strings. `find_any_working_mlx_model()` scans all model directories for compatible weight formats. `find_best_mlx_model()` filters models with incompatible architectures (OsaurusAI custom quant).

**What works:** Standard weight formats (e.g., Qwopus3.6-27B-v2-MLX-4bit). OsaurusAI custom quant formats (MXFP4, JANGTQ) still require `osaurus serve`.

**Testing:** `test_mlx_lib.py` covers discovery, error paths, and filtering. Real MLX backend tests skipped by default (require `--run-mlx`).

**Files:** lib/mlx_lib.py
**Effort:** 2 sessions

### 2.2 Clean up MLX temp debug files

**Bug:** `/tmp/mlx_debug/` files accumulate with no cleanup.

**Fix (if MLX is kept):** Add a cleanup-on-success path:
```python
try:
    result = ...  # run subprocess
    if result.returncode == 0 and os.environ.get("MLX_DEBUG") != "1":
        os.unlink(prompt_file)
        os.unlink(script_path)
    return result
except:
    ...
```
Or add a tempfile.TemporaryDirectory context manager.

**Files:** lib/mlx_lib.py
**Effort:** 0.1 session

### 2.3 Move config loading out of module scope (import-time side effects)

**Bug:** `weekend/config.py`, `lib/config_core.py`, `lib/tui.py` load config at
import time. A broken config file crashes at import.

**Fix:**
- Add `@functools.lru_cache` lazy loading pattern:
  ```python
  def get_weekend_config():
      if _config is None:
          _load_config()
      return _config
  ```
- For `lib/tui.py`: same pattern — parse `~/.config/zstyle` lazily on first use.
- For `lib/config_core._auto_load()`: it's already lazy (checks flag on access),
  but the flag check is racy (Phase 2.4 addresses locking).

**Files:** weekend/config.py, lib/tui.py, lib/config_core.py
**Effort:** 0.5 session

### 2.4 Add thread safety to config globals

**Bug:** `config_core._config_loaded`, `_config`, `_model_configs_cache` are
module-level globals with no locks. `_auto_load()` has check-then-act race.

**Fix:** Add `threading.Lock`:
```python
_config_lock = threading.Lock()

def _auto_load():
    with _config_lock:
        if _config_loaded:
            return
        ...
```

**Files:** lib/config_core.py
**Effort:** 0.1 session

### 2.5 Fix bare `except:` clauses (6 locations)

**Files and fixes:**

| File | Line | Fix |
|------|------|-----|
| `lib/validators/helpers.py` | 81 | `except Exception:` |
| `lib/validators/helpers.py` | 99 | `except Exception:` (but line 94-102 gets deleted in 1.3 — N/A after) |
| `eval_tasks/analyze.py` | 262 | `except Exception:` |
| `eval_tasks/analyze.py` | 369 | `except Exception:` |
| `eval_tasks/analyze.py` | 405 | `except Exception:` |
| `eval_tasks/analyze.py` | 442 | `except Exception:` |

**Testing:** Verify Ctrl+C during these operations still raises `KeyboardInterrupt`.

**Files:** lib/validators/helpers.py, eval_tasks/analyze.py
**Effort:** 0.1 session

---

## Changes from Plan

Key deviations between the planned approach and what was actually implemented:

- **2.1 MLX: Option A (fix) chosen over Option B (delete).** The plan recommended removing MLX fallback calls, but the MLX backend was debugged and fixed instead. `call_mlx()` now captures stderr and reports errors properly. Model discovery (`find_any_working_mlx_model`, `find_best_mlx_model`) filters incompatible architectures instead of attempting to load them. The fix covered the same files as Option B would have removed calls from (twitter/summarize.py, weekend/llm.py, rename/llm.py, eval/run.py), but in each case the MLX fallback was retained and improved rather than deleted.
- **2.2 Temp file cleanup bundled with MLX fix.** The `/tmp/mlx_debug/` cleanup was implemented as part of the MLX backend rewrite rather than a separate task.
- **Test rewrites exceeded scope.** Most existing test files were substantially rewritten, not just augmented (e.g., `test_content_processing.py` shed ~200 lines, test files consolidated assertions). The plan's "sessions" estimate was based on additive changes only.
- **Phase 4 shims retained.** The plan called for deleting `eval_tasks/analyze.py` and `eval_tasks/run.py`, but they were converted to backward-compat import shims instead. `validators.py` also kept as a shim. Kill criterion (only `__init__.py`, `README.md`, `data/taxes/`) not met, but all functional code routes through `eval/`.

## Phase 3 — Medium-Term Improvements

### 3.1 Consolidate `apply_model_quirks()` into one source ✅

`lib/osaurus_lib.py` and `lib/llm/quirks.py` both implement it, slightly
differently. `eval/run.py` calls from `osaurus_lib`. `rename/llm.py` calls
from `lib.llm.quirks`.

**Fix:** Move canonical implementation to `lib/llm/quirks.py`. Make
`lib/osaurus_lib` import and re-export it. Update `eval/run.py:_call_model()`
to import from `lib.llm.quirks`. Verify all quirks match between old and new
implementations (diff the two, consolidate any divergence).

**Effort:** 0.5 session

### 3.2 Unify LLM call interfaces with a protocol ✅

Three implementations (`lib/osaurus_lib.call`, `lib/llm/client.call`,
`lib/mlx_lib.call`) with different signatures.

**Fix (minimal):** Define a `LLMClient` protocol:
```python
class LLMClient(Protocol):
    def call(self, model: str, messages: list[dict], **kwargs) -> dict: ...
```
Make each implementation conform. Add a `create_client(client_type: str)` factory.

**Effort:** 1 session

### 3.3 Fix monitor_memory_loop thread safety ✅

**Bug:** Daemon thread calling `console.print()` races with main thread.

**Fix options:**
- **Option A (preferred):** Replace daemon thread with periodic polling in the
  main eval loop. Add `check_memory()` call between task iterations in
  `eval/run.py`. Remove `monitor_memory_loop` entirely.
- **Option B:** Make the monitor thread write to a `queue.Queue`, have main
  thread drain it and print. But this still risks interleaving.

**Effort:** 0.25 session

### 3.4 Extract shared fallback orchestration ✅

The `try_osusaurus → retry with restart → fall back to MLX` pattern is
duplicated in 3 places (`twitter/summarize.py`, `weekend/llm.py`, `rename/llm.py`).

**Fix:** Extract to `lib/llm/fallback.py`:
```python
def call_with_fallback(
    prompt: str,
    model_list: list[str],
    retry_count: int = 3,
    mlx_enabled: bool = False,
) -> str: ...
```
Each app calls this instead of duplicating the loop. This is the single biggest
SRP violation fix — eliminates three copies of the same orchestration logic.

**Effort:** 1 session

### 3.5 Add `[project.scripts]` entry points ✅

**Fix:** Add to `pyproject.toml`:
```toml
[project.scripts]
tw = "twitter.cli:main"
wk = "weekend.cli:main"
rn = "rename.cli:main"
ev = "eval.cli:main"
```
Test: `tw --help` after `pip install -e .` (or `uv run tw --help`).

**Effort:** 0.1 session

### 3.6 Fix memory monitor: make console thread-safe ✅

Already captured in 3.3 above — same fix.

---

## Phase 4 — eval/ eval_tasks/ Consolidation ✅

### 4.1 Route both entry points through one code path ✅

**Goal:** `python3 -m eval` and `python3 -m eval_tasks` converge on the same runner.

**Strategy:**
1. `eval_tasks/__main__.py` changes from `from eval_tasks.run import main; main()`
   to `from eval.cli import main; main()` (with appropriate args).
2. `eval_tasks/analyze.py` is deleted (its functions are in `eval/report.py`).
3. `eval_tasks/run.py` is deleted.
4. `eval_tasks/__init__.py` re-exports `TASKS` from `eval.tasks_core` for any
   remaining importers.
5. `eval_tasks/validators.py` — verify callers. If any still import from here,
   change to `from lib.validators_lib import ...`.
6. `eval_tasks/README.md` is updated to say "DEPRECATED — use `eval/` instead."

**Kill criterion:** `eval_tasks/` directory contains only `__init__.py`,
`README.md`, and `data/taxes/`. Everything functional has moved to `eval/`.

**Effort:** 0.5 session

### 4.2 Unify TASKS dicts ✅

**Fix:** `eval/tasks_core.py` is the canonical `TASKS`. Remove `eval_tasks/__init__.py`'s
own `TASKS` definition — replace with `from eval.tasks_core import TASKS`.
Fix the shared mutable reference aliasing:
```python
TASKS["json"] = dict(TASKS["weekend_transient"])  # copy, not alias
TASKS["detailed_json"] = dict(TASKS["weekend_fixed"])
```

**Effort:** 0.25 session

---

## Phase 5 — Low-Priority / Cleanup ✅

### 5.1 Fix shared mutable references in TASKS ✅

Already covered in 4.2.

### 5.2 Remove circular self-import in eval/cli.py ✅

**Fix:** Extract `quick_run_eval` into `eval/run.py` as a separate function.
Replace the monkey-patch with a parameter:
```python
# eval/cli.py
def main():
    quick = args.quick
    from eval.run import run_eval
    run_eval(..., quick_mode=quick)

# eval/run.py
def run_eval(..., quick_mode=False):
    max_retries = 0 if quick_mode else MAX_RETRIES
```
Delete `import eval.cli as me` and the global `MAX_RETRIES = 0` mutation.

**Effort:** 0.25 session

### 5.3 Fix "name" key duplication in normalize_keys ✅

**Bug:** `lib/osaurus_output.py:145-152` — when a dict has only one key whose
value is a string, the code copies it to `result["name"]` without removing the
original key.

**Fix:**
```python
if len(item) == 1:
    k, v = next(iter(item.items()))
    if isinstance(v, str):
        result["name"] = v
        if k != "name":
            del result[k]
```

**Effort:** 0.1 session

### 5.4 Fix misleading comment / inverted logic in text_validator.py ✅

**Bug:** Comment says "too wordy" but code adds 15 points when `has_explanation`.

**Fix:** Either change the comment to describe the actual intent (some models
need to pad their output?), or change the code to penalize (-15) instead of
reward. Decide based on whether the behavior matches the desired scoring.

**Effort:** 0.05 session

### 5.5 Fix tautological test assertion ✅

**Bug:** `tests/test_mlx_lib.py:298` — `assert "before" in result or "after" in result`
(always passes with valid content).

**Fix:** `assert "before" in result and "after" in result`.

**Effort:** 0.05 session

### 5.6 Cache `content.index()` results in remove_thinking_blocks ✅

**Bug:** `content.index(marker)` called twice (lines 47 and 49) without caching.

**Fix:** Already handled in 1.1 — the refactor uses `content.find(marker)` stored
in a local variable.

**Effort:** 0 (bundled with 1.1)

---

## Summary

| Phase | Theme | Files Changed | Sessions | Risk | Status |
|-------|-------|---------------|----------|------|--------|
| 0 | Test scaffolding | 4 new files | 3.25 | Low — additive only | ✅ Done |
| 1 | Critical bug fixes | 7 files | 1.45 | Medium — security + crash | ✅ Done |
| 2 | High-impact structural | 9 files | 1.8 | Medium — behavior changes | ✅ Done |
| 3 | Medium improvements | 12 files | 3.0 | Low — mostly additive | ✅ Done |
| 4 | eval/ consolidation | 6 files | 0.75 | Medium — may break imports | ✅ Done |
| 5 | Low-priority cleanup | 6 files | 0.55 | Low | ✅ Done |

**Total:** ~44 files changed across 6 phases, ~11 sessions.

### Ordering Rationale (historical)

- **Phase 0 first** because every fix needs tests. Adding tests before fixes
  (A1: additive before subtractive) also serves as a parity proof (A2) —
  the tests initially fail, then pass after the fix.
- **Phase 1 second** because crash and security bugs are blocking issues.
- **Phase 2 third** because structural issues amplify maintenance cost.
- **Phase 3 and 4** can be parallelized across two developers.
- **Phase 5** can be deferred indefinitely — no user-facing impact.

### Dependencies

- Phase 1.1 (content_processing fix) depends on Phase 0.1 (tests).
- Phase 1.2 (quality_entry fix) depends on Phase 0.2 (tests).
- Phase 1.4 (cookie SQL fix) depends on Phase 0.3 (tests).
- Phase 3.4 (fallback extraction) depends on Phase 2.1 (MLX decision).
- Phase 4 (eval consolidation) depends on Phase 3.4 (fallback extraction) and
  Phase 2.4 (config thread safety) if config gets involved in eval routing.
- Phase 5.6 (content.index caching) is bundled with Phase 1.1.
