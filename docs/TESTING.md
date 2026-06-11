# Testing Patterns and Findings

**Updated: June 2026** — captures the test patterns, mock infrastructure, and bugs found during the multi-round test audit. All phases of the remediation plan complete.

---

## Overview

- **1515 pass, 3 skip** (real MLX backend, real Osaurus server).
- **96% coverage** total. All modules ≥ 95%, most at 100%.
- All tests use mocked LLM providers — **no real model calls in CI**.

Run all: `python3 -m pytest tests/`
Run single file: `python3 -m pytest tests/test_twit_browser.py -v`
Run single test: `python3 -m pytest tests/test_file.py::test_name -v`
Coverage: `python3 -m pytest --cov=. --cov-report=term`

Use `tests/` for a full regression check before commits. Use a specific file during development to iterate faster on one component.

---

## Mock Infrastructure (`lib/testing.py`)

`MockLLM` is the canonical fake. Patches three modules at test time:

| Module | Patched functions |
|--------|------------------|
| `lib.osaurus_lib` | `call`, `call_llm_api`, `get_models`, `is_server_running`, `get_best_model`, `check_llm_availability`, `ensure_server`, `panic_dump`, `_extract_json_only` |
| `lib.llm.client` | `call`, `get_models`, `is_server_running` |
| `lib.mlx_lib` | `call`, `call_mlx`, `find_text_mlx_model`, `find_mlx_model`, `process_mlx_content` |
| `lib.config` | `get_model_prompts_all`, `build_tasks_from_model` |

`mock_llm` pytest fixture (in `tests/conftest.py`) auto-applies and tears down all three.

**Default content per task** is defined in `_default_content_for(task)`:

- `json` / `weekend_transient`: 2 items (Spring Festival, Indoor Coding Workshop)
- `weekend_fixed` / `detailed_json`: 2 items (Vaughan Sports Arena, High Park)
- `filename` / `image_rename`: literal string `"mock_test_filename"`
- `summarize`: markdown with 3 bullet points
- `file_summary`: 4 items with real-ish paths

### Key Pattern: Patch Local Bindings

`from lib.osaurus_lib import call` creates a local reference in the importing module. Patching `lib.osaurus_lib.call` does **not** affect modules that already imported it. Solution: use `patch.object(target_module, "call", mock.call)`.

```python
def test_something(mock_llm):
    import eval_run as er
    with patch.object(er, "call", mock_llm.call):
        result = er.run_eval(...)
```

`tests/conftest.py` also captures `_REAL_MLX_FUNCTIONS` at import time so the real `lib.mlx_lib` functions can be retrieved before any patches overwrite them. Use the `real_mlx_functions` fixture for tests that need to call real MLX logic with fake inputs.

---

## Rules for Writing Tests

### 1. Every test must have a non-tautological assertion

A test that just calls a function and asserts `mock_x.assert_called()` is a **smoke test** — it would pass even if the function had a bug. Always verify a concrete outcome.

```python
# BAD — smoke test
def test_foo():
    some_function()
    mock_x.assert_called()

# GOOD — verifies real behavior
def test_foo():
    result = some_function()
    assert result == expected_value
    assert "expected" in captured_output
```

**Acceptable weak forms** (use only as a floor, not the sole assertion):
- `isinstance(x, (int, float))` + `0 <= x <= 100` (range checks when the exact value is hard to predict)
- `mock_x.assert_called_once_with(expected_args)` (verifies args, not just that it was called)
- `assert "substring" in captured_output` (verifies observable behavior)

### 2. Test the real code, not a re-implementation

If you `mock` the function under test AND replicate the logic in the test, the test will pass even if the real code has the bug. **Use `exec` to run the real `__main__` block.**

```python
def test_main_block(monkeypatch):
    import re
    import textwrap
    with open("eval/benchmark_quality.py") as f:
        source = f.read()
    match = re.search(
        r'if __name__ == "__main__":\n((?:    .*\n)+)',
        source,
    )
    main_source = textwrap.dedent(match.group(1))
    code = compile(main_source, "<benchmark_quality __main__>", "exec")
    monkeypatch.setattr(sys, "argv", ["benchmark_quality", "m1", "--quiet"])
    monkeypatch.setattr(benchmark_quality, "run_benchmark", mock_run)
    exec(code, benchmark_quality.__dict__)
    mock_run.assert_called_once_with(["m1"], verbose=False)
```

### 3. Verify specific numbers, not just "≥ 0"

```python
# BAD — would pass for score=1 or score=100
assert score >= 0

# GOOD — catches real bugs
assert score == 100
assert score == pytest.approx(2.4286, abs=0.01)
```

When the exact value is hard to predict, use the **range pattern** with a documented reason: `assert 0 <= score <= 100` is acceptable as a floor, but stronger is better.

### 4. Use the real scorer/validator when possible

If the function under test calls a scorer, let the **real** scorer run on a hand-crafted input. The test exercises the actual scoring logic, not a mock that returns whatever you told it to.

```python
def test_run_benchmark_scoring_correctness(self, mock_llm):
    from benchmark_quality import FILENAME_CASES
    perfect_outputs = [
        "login_error_invalid_credentials.png",
        "summer_festival_2024_central_park.txt",
        # ... one perfect output per FILENAME_CASE
    ]
    # Real score_filename runs on each case
    with patch.object(bq, "get_model_prompt", return_value="p"), \
         patch.object(bq, "query_model", side_effect=qm_side_effect):
        bq.run_benchmark(["m1"], verbose=True)
    # Verify the aggregated score
    assert mock_summary.call_args.args[2] == 100.0  # avg_human
```

### 5. Don't trust your math — run the function to learn the score

Test value assertions should be **discovered**, not guessed. Run the function in isolation first to know the exact score, then assert it.

```python
# In a one-off Python invocation:
from lib.validators.text_validator import validate_summary
print(validate_summary("@user1 @user2 hello"))  # (20, '...')
# Then in the test:
assert score == 20  # not 10, not "≥ 0"
```

---

## Patterns for Hard-to-Test Code

### Captured Closures (e.g. `handle_response` in `twit_browser`)

The handler is registered via `page.on("response", handler)` and only fires when the browser emits a response event. To test the handler:

1. Capture the handler from `page.on.side_effect = lambda event, handler: captured.append(handler)`.
2. Set `page.evaluate.side_effect` to invoke the captured handler on the first call. The scroll loop calls `page.evaluate(...)` once per scroll.
3. Assert based on whether the loop ran once or all `MAX_SCROLLS` times (handler set `oldest_seen` → loop broke early).

```python
def eval_side_effect(*args, **kwargs):
    if len(captured_handler) == 0:
        return None
    fake_response = MagicMock()
    fake_response.url = "https://x.com/api/graphql/HomeLatestTimeline"
    fake_response.json.return_value = tweet_data
    captured_handler[0](fake_response)
    return None
mock_page.evaluate.side_effect = eval_side_effect
# ... run with MAX_SCROLLS=10
assert mock_page.evaluate.call_count == 1  # loop broke after 1 scroll
```

### Subprocess / AppleScript Calls

`flush_between_models` in `eval/run.py` calls `subprocess.run(["osascript", "-e", 'quit app "osaurus"'])` and `subprocess.run(["open", "-n", "-a", "osaurus"])`. Test by:

```python
with patch("subprocess.run") as mock_subprocess, ...:
    model_eval.main()
sub_calls = [str(c) for c in mock_subprocess.call_args_list]
assert any("osascript" in s and "quit" in s for s in sub_calls)
assert any("open" in s and "osaurus" in s for s in sub_calls)
```

### MLX Model Discovery via Fake Filesystem

`test_mlx_lib.py` uses a `fake_mlx_dir(tmp_path)` fixture that creates real directories and `config.json` files for model discovery tests — no mocking of `Path.iterdir()` or file existence checks. This exercises the real filesystem scanning logic:

```python
@pytest.fixture
def fake_mlx_dir(tmp_path):
    models = tmp_path / "MLXModels"
    models.mkdir()
    qwen = models / "qwen-7b-fp16"
    qwen.mkdir()
    (qwen / "config.json").write_text(json.dumps({"context_length": 8192}))
    (models / "no-config").mkdir()
    return models
```

Use with `real_mlx_functions` fixture to run real discovery functions against fake dirs:

```python
def test_find_mlx_model_top_level(self, mock_llm, fake_mlx_dir, real_mlx_functions):
    real = real_mlx_functions["find_mlx_model"]
    result = real("qwen", mlx_dir=fake_mlx_dir)
    assert result == fake_mlx_dir / "qwen-7b-fp16"
```

### `ddgs` Mock Pattern

---

## Bugs Found During Test Audits

### 1. `--quiet` was treated as a model name (FIXED)

**File:** `eval/benchmark_quality.py:344-347`

```python
# BROKEN
verbose = "--quiet" not in sys.argv
models = sys.argv[1:] if len(sys.argv) > 1 else None

# FIXED
quiet = "--quiet" in sys.argv
models = [a for a in sys.argv[1:] if a != "--quiet"] or None
verbose = not quiet
```

The original code included `--quiet` in the models list when iterating. Locked in by `test_benchmark_quality_runner.py::test_main_block_with_models_and_quiet` which now uses `exec` to test the real `__main__` block.

### 2. `_parse_transient` alt-keys branch is unreachable dead code (DOCUMENTED)

**File:** `weekend/cli.py:149-157`

`alt_items` is filtered using the same keys as `valid_items`, so both are empty simultaneously. Documented by `test_alt_keys_are_dead_code` with a comment explaining the unreachable branch.

### 3. `_score_item` is normalized (NOT just summed) — easy to miss

**File:** `weekend_llm.py:182`

```python
return min(round(score / 2.0, 1), 5.0)
```

Plus a `+0.5` location bonus for `len(location) > 5`. Tests that use raw scores like `assert score == 8.5` will fail. Correct expected values: 0/0.2/0.7/1.2/1.7/4.5 (with perfect inputs).

### 4. `validate_summary` user-mention scoring is non-linear

**File:** `lib/validators/text_validator.py`

- 1 user = 15 pts (10 base + 5)
- 2 users = 20 pts (10 base + 5 + 5)
- 3 users = 25 pts (10 base + 5 + 5 + 5)

Not the linear `5 * count` some tests assumed.

### 5. `validate_file_summary` 40% real paths is a boundary

**File:** `lib/validators/text_validator.py:207-213`

```python
if real_paths >= len(items) * 0.7:
    ...  # full credit
elif real_paths >= len(items) * 0.4:
    ...  # partial credit, no failure msg
else:
    failures.append(f"unrealistic paths ...")
```

40% exactly is partial credit (no failure msg). < 40% triggers "unrealistic" failure. Tests at exactly 2/5 = 40% should not assert "unrealistic" in msg.

### 6. `extract_json` returns keys, not values, for flat dicts

**File:** `lib/osaurus_output.py`

```python
extract_json('{"name": "x", "location": "y"}')  # returns ["name", "location"]
```

Not `[{"name": "x", "location": "y"}]`. Locked in by `test_dict_extraction` which now asserts the exact list.

### 7. `test_top_level_exception_caught` requires a non-dict body

**File:** `twitter/summarize.py`

```python
data = {"data": "not a dict"}  # string has no .get()
# data.get("data", {}).get("home", {}) → AttributeError
# try/except catches it → returns []
```

The branch can only be exercised with a non-dict value inside the response. Tests must use this pattern to hit the error path.

---

## Test Categories

| Category | File pattern | Strategy |
|----------|-------------|----------|
| Validator unit | `test_*_validator.py` | Direct function call, assert exact score |
| Scorer unit | `test_benchmark_quality*.py` | Real scorer on hand-crafted input, verify aggregated output |
| Integration | `test_*_integration.py` | Patch LLM, exercise real `main()` flow, capture stdout |
| Browser | `test_twit_browser.py` | Mock Playwright, inject responses via `page.evaluate.side_effect` |
| CLI / argparse | `test_*_main.py`, `test_*_branches.py` | `monkeypatch.setattr(sys, "argv", ...)` + capture stdout |
| `__main__` block | `test_benchmark_quality_runner.py` | `exec` the real block source with `compile()` |
| Config / getters | `test_config*.py` | Patch `Path` to point to `tmp_path` fixtures |

---

## Coverage Discipline

The `--cov` target is `.` (project root). The audit script used to find uncovered lines:

```bash
python3 -m pytest --cov=. --cov-report=term-missing
```

Uncovered lines are usually:
- `if __name__ == "__main__":` blocks (tested separately via `exec`)
- Defensive `except` branches that never fire in normal flow
- Subprocess `osascript` calls (covered via `subprocess.run` mock)

When a test reaches an `if __name__ == "__main__":` block via `exec`, it counts as covered for the real block source.

---

## Audit Checklist for New Tests

Before merging a test, verify:

1. [ ] Has at least one non-tautological `assert` (or `pytest.raises`)
2. [ ] Asserts a specific value or captures `stdout` for a substring
3. [ ] Does not mock the function under test (or uses `exec` to run the real block)
4. [ ] Uses the `mock_llm` fixture or the `MockLLM` provider to avoid real LLM calls
5. [ ] Patches local bindings via `patch.object(module, "name", mock.name)` if importing
6. [ ] Closes over real scorer/validator when possible
7. [ ] Documents any non-obvious patterns in the test docstring

---

## See Also

- `docs/MODEL_QUIRKS.md` — model behavior quirks, best prompts
- `lib/testing.py` — MockLLM implementation
- `tests/conftest.py` — shared fixtures
- `CLAUDE.md` — project rules

## New Modules (Phase 3)

- **`lib/llm/fallback.py`** — shared fallback orchestration (`call_with_fallback`), extracted from the three duplicate implementations in `twitter/summarize.py`, `weekend/llm.py`, `rename/llm.py`.
- **`lib/llm/protocol.py`** — `LLMClient` protocol class unifying the three call interfaces (`lib.osaurus_lib.call`, `lib.llm/client.call`, `lib.mlx_lib.call`).
