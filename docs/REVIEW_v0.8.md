# v0.8 Code/Arch/Design Review

Persona panel: John Carmack (engineering), Mitchell Hashimoto (ops),
Robert C. Martin / Uncle Bob (test debt), Linus Torvalds (structure).

Generated 2026-07-11 after v0.8.3.

---

## John Carmack — performance, build order, engineering

### HIGH — Three `Console()` instances

`eval/run.py:80` / `eval/cli.py:110` / `eval/report.py:20`

Each module allocates its own Rich `Console` at import time. Minimal memory
waste, but the real cost: tests monkey-patch ONE instance and miss output
routed to another. The `weekend/output.py` instance also forces
`force_terminal=True, force_interactive=True` — see Hashimoto finding.

**Fix:** one shared `Console()` in `lib/tui`, imported everywhere else.

### MED — Circular dependency cli → run → failures → run

`eval/run.py:330` lazy-imports `eval.cli.print_memory_usage`
`eval/failures.py:30` lazy-imports `eval.run.DEFAULT_EVAL_TIMEOUT`

Resolved only by lazy imports. Any code path that calls `run_eval()` before
`cli` is initialized, or `_classify_failure()` before `run` is initialized,
crashes at runtime. Fragile.

**Fix:** extract `print_memory_usage` as a callback parameter of `run_eval()`
(remove the import). Move `DEFAULT_EVAL_TIMEOUT` to a shared constants module.

### MED — `flush_between_models()` is a closure inside `main()`

`eval/cli.py:311-343`

Cannot be tested in isolation. Forces 15+ `patch()` calls per test.

**Fix:** extract to a module-level function accepting `args` as parameter.

### LOW — Boilerplate `print_*` functions in `report.py`

All four `print_*` functions follow: save console → replace → try/finally → restore.
Repeated at every call site.

**Fix:** extract a `_with_console()` context manager.

---

## Mitchell Hashimoto — operator experience, config foot-guns, silent failures

### HIGH — `lib/tui.py` reads `~/.config/zstyle` silently

`lib/tui.py`

If the file is missing or malformed, all TUI output silently degrades
(WARN defaults to `"!"`, STEP to some other fallback). No error message.
The Kare-style library (`tui/lib.py` from the template) was never deployed;
what exists is a 22-line skeleton with no `NO_COLOR` awareness, no `isatty()`
check, and no functional helpers (`info`, `ok`, `err`, `warn`, `die`, `section`, `hr`).

**Fix:** fall back to hardcoded Kare glyphs with a `logger.debug()` message.
Bootstrap the full TUI library via `~/projects/scripts/init_repo.sh`.

### HIGH — `weekend/output.py` forces terminal mode

`weekend/output.py:16-18`

```python
Console(force_terminal=True, force_interactive=True)
```

Forces terminal features even when stdout is piped or `NO_COLOR` is set.
ANSI codes leak into redirected files.

**Fix:** remove both flags — Rich detects terminal mode automatically.

### MED — `_eval_inputs_cache` never invalidated

`lib/config_tasks.py:11`

Module-level mutable cache. A developer edits `conf/eval_inputs.yaml` and the
stale cache silently serves old data. No invalidation mechanism.

**Fix:** add file-mtime check or expose `clear_eval_inputs_cache()`.

### MED — `lib/config_tasks.py` imports from `eval/` via try/except

`lib/config_tasks.py:68-73`

```python
try:
    from eval.validate import validate_file_summary
except ImportError:
    def validate_file_summary(data, source_text=""):
        from lib.validators_lib import validate_summary
        return validate_summary(data)
```

On ImportError, silently falls back to a degraded validator. The operator
never knows. Also an architecture inversion (see Linus finding).

**Fix:** invert the dependency — `eval/` should import from `lib/`. Fail
loudly on ImportError if the dependency is truly required.

### LOW — `eval/__init__.py` megamodule

Re-exports every sub-module function into one flat namespace. `from eval import X`
can get something from anywhere. Refactoring any sub-module risks breaking
callers that rely on the flat surface.

**Fix:** export only the public API surface explicitly; prefix everything
else with `_`.

---

## Robert C. Martin (Uncle Bob) — duplication, test debt, misleading names

### HIGH — `quality_weekend_scorers.py` has ZERO tests

`lib/quality_weekend_scorers.py` (205 lines, 5 scoring dimensions, 5 scorers)

No test file anywhere references `_score_weekend_completeness`,
`_score_weekend_weather_match`, `_score_weekend_age_match`,
`_score_weekend_source_grounding`, or `_score_weekend_exclusions`.
A regression in weather-matching or age-overlap logic silently produces
wrong eval scores.

**Fix:** add `tests/test_quality_weekend_scorers.py` with 3-5 cases per scorer
(happy path, edge, failure).

### MED — THREE copies of the GENERIC filename set

1. `lib/quality_scorers.py:81-82` — inside `_score_filename_format`
2. `lib/quality_scorers.py:470-471` — inside `score_output`
3. `lib/validators/text_validator.py:49-50` — as `GENERIC_FILENAMES`

All three defs: `{"filename.txt", "file.txt", "text.txt", ...}`. They will
drift.

**Fix:** extract to a shared constant in one module.

### MED — Dead import: `ScoreCard` in `quality_weekend_scorers.py`

`lib/quality_weekend_scorers.py:5`

```python
from lib.quality_models import Score, ScoreCard, TestCase, _str, _lower
```

`ScoreCard` is never used in the file body (all functions return `Score`).

**Fix:** remove `ScoreCard` from the import.

### LOW — Tests import `ScoreCard` from wrong module

`tests/test_eval_run_integration.py:257,269,658,667`

```python
from lib.quality_entry import ScoreCard
```

Should be `from lib.quality_models import ScoreCard`.

**Fix:** fix the import path.

### LOW — `_str` / `_lower` imported in every scorer module

`lib/quality_scorers.py:5` and `lib/quality_weekend_scorers.py:5`

Trivial one-liners (`str(x) if x is not None else ""`). The import adds a
dependency leaf from every scorer to `quality_models`.

**Fix:** inline both, or move to a `_utils.py` if used more widely.

---

## Linus Torvalds — taste, structure, layering

### HIGH — Architecture inversion: `lib/` imports from `eval/`

`lib/config_tasks.py:69`

```python
from eval.validate import validate_file_summary
```

The `lib` directory is the foundation layer — it should depend on NOTHING
outside `lib`. This inverts the dependency arrow. `eval` is an application
layer that should import from `lib`, never the reverse.

**Fix:** move `_safe_format_prompt` and task-building logic to `eval/` where
it belongs. Keep `lib/config_core.py` as the pure config-loading leaf.

### HIGH — Mutable globals mutated cross-module

`eval/cli.py:247`

```python
eval.run.DEFAULT_EVAL_TIMEOUT = args.timeout
```

Side-effect mutation of another module's global constant. If `main()` runs
twice in the same process (as in tests), the timeout leaks across invocations.

**Fix:** thread `timeout` as a function parameter; never mutate globals.

### MED — `TASK_SCORERS` registry is a hardcoded switch statement

`lib/quality_scorers.py:455`

```python
TASK_SCORERS = {
    **TASK_SCORERS_WEEKEND,
}
```

Assembled at import time. Adding a new task family requires modifying this
central dict. Also, the keys (`"weekend_transient"`, `"weekend_fixed"`) and
scorer lists are identical in both `quality_scorers.py` and
`quality_weekend_scorers.py` — duplicate key structure.

**Fix:** use a decorator-based registration (`@registers("task_name")`) so a
new task-group module registers itself without touching the central dict.

### MED — `eval/__init__.py` flattens all boundaries

Re-exports every function from every sub-module. Nothing is private. `from eval import X`
gives no hint which sub-module `X` lives in. Refactoring is harder.

**Fix:** export only the surface needed by `eval/__main__.py`.

### LOW — `quality_weekend_scorers.py` / `quality_scorers.py` duplicate key structure

Both define `TASK_SCORERS` / `TASK_SCORERS_WEEKEND` with the same two keys
and the same five-scorer list. When one changes, the other must be updated.

**Fix:** have one define task-type differences only; the common base lives
in one place.

---

## Cross-cutting: Config format migration (YAML + JSON → TOML)

The project previously used **YAML + JSON** for config. **All config is now TOML:**

| Format | Files | Purpose |
|---|---|---|
| TOML | 12 files | All app configs, model configs, eval inputs |
| JSON | 3 files | Eval/phase/extract signals (auto-generated, machine-written) |

**Migration completed:**
- `conf/config.toml`, `conf/twitter.toml`, `conf/rename.toml`, `conf/weekend.toml`
- `conf/eval_inputs.toml`, `conf/models/*.toml` (7 model configs)
- `lib/config_toml.py` uses stdlib `tomllib` (Python 3.11+) — no PyYAML dependency
- `load_config()` auto-resolves `.yaml` → `.toml` via `.with_suffix(".toml")` for backwards compat

---

## Fix Status (v0.8.4 / v0.8.5)

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | Dead import `ScoreCard` in `quality_weekend_scorers.py` | MED | FIXED |
| 2 | Tests import `ScoreCard` from wrong module | LOW | FIXED |
| 3 | `weekend/output.py` forced terminal flags | HIGH | FIXED |
| 4 | TUI lib: `~/.config/zstyle` fallback + `NO_COLOR` + helpers | HIGH | FIXED |
| 5 | GENERIC_FILENAMES dedup (3 copies → 1) | MED | FIXED |
| 6 | `_eval_inputs_cache` invalidation (expose `clear_eval_inputs_cache()`) | MED | FIXED |
| 7 | Architecture inversion: remove try/except ImportError fallback | HIGH | FIXED |
| 8 | Circular dependency: `print_memory_usage` as callback parameter | MED | FIXED |
| 9 | Mutable globals: remove `eval.run.DEFAULT_EVAL_TIMEOUT = ...` | HIGH | FIXED |
| 10 | `flush_between_models()` closure → module-level function | MED | FIXED |
| 11 | Three `Console()` instances → one shared in `lib/tui.py` + `capture_console()` context manager | HIGH | FIXED |
| 12 | `TASK_SCORERS` decorator registry + dedup weekend keys | MED | FIXED |
| 13 | `eval/__init__.py` export surface (trim to public API only) | LOW | FIXED |
| 14 | Config migration YAML/JSON → TOML | MED | FIXED |
| 15 | Tests for `quality_weekend_scorers.py` (47 tests, found + fixed 2 bugs) | HIGH | FIXED |
| 16 | Flaky `test_quality_runner.py::test_query_model_success` | LOW | FIXED |
