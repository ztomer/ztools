# Testing Patterns and Findings

**Updated: July 2026** — test patterns, mock infrastructure, the autouse structural gates, and the bugs each was added to prevent.

---

## Overview

- **2120 pass**, 4 skipped, 7 xfailed, at 95.11% coverage (2026-08-12).
- Tests use **mocked LLM providers** — no real model calls in CI.
- The OCR files (`test_img_helpers.py`, `test_image_renamer.py`) run **with** coverage like
  everything else. They were excluded for a numpy C-extension crash that the lazy-import fix
  cured; the exclusions outlived it and held the suite below its own floor — see
  **Coverage Limitations** below.

**Run what the gate runs.** `pytest references/tests/` passes in a developer
environment that the pre-push gate does not have: the gate runs the whole
rootdir, with coverage against a 95% floor, and with the LLM server pointed at a
dead port. A green `pytest references/tests/` therefore proves very little — it
misses coverage regressions entirely, and misses anything that only shows when
the server is unreachable. Before pushing, run the gate's own command:

```bash
OLLAMA_BASE_URL=http://127.0.0.1:1 \
MLX_MODELS_DIR=/tmp/nonexistent-mlx-models-dir-for-testing \
  .venv/bin/pytest --cov --cov-report=term-missing --cov-fail-under=95 .
```

Local venv must carry the same test deps CI installs, or gates pass locally
that fail in CI:

```bash
uv pip install --python .venv/bin/python pytest pytest-cov pytest-asyncio
```

Run all: `python3 -m pytest references/tests/`
Run single file: `python3 -m pytest references/tests/test_twit_browser.py -v`
Run single test: `python3 -m pytest references/tests/test_file.py::test_name -v`
Full coverage — no exclusions (see **Coverage Limitations**; the numpy crash that once
required them no longer reproduces, and excluding those two files put the suite under its
own floor):
```bash
python3 -m pytest references/tests/ \
  --cov=rename.llm --cov=rename.helpers \
  --cov=twitter.summarize --cov=twitter.cookies \
  --cov-report=term-missing
```

---

## Mock Infrastructure (`lib/testing.py`)

`MockLLM` is the canonical fake. Patches three modules at test time:

| Module | Patched functions |
|--------|------------------|
| `lib.osaurus_lib` | `call`, `call_llm_api`, `get_models`, `is_server_running`, `get_best_model`, `check_llm_availability`, `ensure_server`, `panic_dump`, `_extract_json_only` |
| `lib.llm.client` | `call`, `get_models`, `is_server_running` |
| `lib.mlx_lib` | `call`, `call_mlx`, `find_text_mlx_model`, `find_mlx_model`, `process_mlx_content` |
| `lib.config` | `get_model_prompts_all`, `build_tasks_from_model` |

`mock_llm` pytest fixture (in `references/tests/conftest.py`) auto-applies and tears down all three.

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

`references/tests/conftest.py` also captures `_REAL_MLX_FUNCTIONS` at import time so the real `lib.mlx_lib` functions can be retrieved before any patches overwrite them. Use the `real_mlx_functions` fixture for tests that need to call real MLX logic with fake inputs.

---

## Structural Gates (conftest, autouse — you cannot opt out by forgetting)

Two autouse fixtures in `references/tests/conftest.py` make whole classes of test misbehavior
impossible rather than relying on each test to remember. Both were added after the
failure they prevent actually happened.

### `no_real_browsers_or_cookies`

Pins the browser backend to chromium and stubs cookie discovery for every test.

**Why:** `test_twitter_browser_no_playwright` patched only `sync_playwright`. Once
camoufox became the preferred backend, that patch stopped covering the launch path —
the test launched a **real Firefox and read the developer's actual x.com session**
during a unit-test run. Backend selection is not something each test should have to
know about.

Opt out with `@pytest.mark.real_cookie_discovery` only when the test *is* the cookie
reader (`references/tests/test_twit_cookies.py` sets it module-wide via `pytestmark`).

### `_saved_outputs_stay_in_tmp`

Redirects `EVAL_OUTPUT_DIR` at a tmp dir for the whole session.

**Why:** `run_eval` now saves each model's raw answer under `~/.config/ztools/outputs`
so a scorer can be questioned without re-running the model. Every existing test that
calls `run_eval` with a fake model started writing there too — the suite left
`outputs/m`, `outputs/m1` and `outputs/mock-model` in the developer's own config
directory within minutes of the feature landing (2026-08-12).

**What this teaches beyond the one bug:** `_tracked_config_stays_clean` could not have
caught it. That gate digests `conf/` and `docs/`, and this escape went to `$HOME`. **A
sandbox gate only covers the directories it was told about, so any new persistence path
needs its own redirect on the same commit that introduces the write.** Redirect via the
environment variable rather than a module attribute — that is the seam production reads,
and a value-import of the path would slip a module patch.

`test_the_suite_cannot_write_into_the_real_config_dir` asserts the redirect is in force,
because a sandbox nobody verifies is one that can quietly stop applying. Proven by
disabling the redirect and watching it go red.

### Why `git push` looked like it was hanging (2026-08-12)

Every `git push` appeared to hang for minutes and was written off as a credential problem
through several sessions. It was not. **`.githooks/pre-push` runs the whole suite with
coverage**, which takes about three minutes — the "hang" was the gate doing its job with no
output until it finished.

It then failed, every time, for a reason no diff could fix: `_tracked_config_stays_clean`
digests `conf/*.json`, a model sweep was running in another terminal, and `ev` rewrites
`conf/eval_signals.json` after every task. The suite was green (2130 passed, 95.11%) and
the push still exited 1.

Fixed by excluding the three signals files from the digest, which loses nothing:
`_signals_files_stay_clean` already redirects all three for the whole session, so a change
to the real file *cannot* have come from a test. The specific gate covers them; the broad
one was only adding a false positive.

Diagnostic worth reusing: `git ls-remote` with `-c credential.helper=` returns in under a
second. If that works and `push` "hangs", the network is fine and something local — a
hook — is doing the waiting.

### `_signals_files_stay_clean`

Redirects `eval.run.EVAL_SIGNALS_PATH` and `weekend.llm.PHASE_SIGNALS_PATH` at a tmp
dir for the whole session.

**Why:** both modules persist learned per-model timeouts into `conf/eval_signals.json`
and `conf/phase_signals.json`, which are tracked. Every `pytest` run rewrote them and
left the working tree dirty, so `git status` was never trustworthy after a test run.
Production reads `EVAL_SIGNALS_DIR` / `PHASE_SIGNALS_DIR` from the environment.

**The pattern:** when a test escapes its sandbox (network, filesystem, real
credentials, a browser), fix it in `conftest.py` as an autouse gate with a named
opt-out marker — not by patching the one test that surfaced it. The next test to make
the same mistake won't get its own review.

---

## Rules for Writing Tests

### 0. Prove a new test can fail

A test you have only ever seen pass is not evidence. Break the behavior it covers —
delete the guard, invert the condition, remove the conversion — confirm it goes red,
then restore. Do this before you trust any new assertion.

```bash
# Example from the scroll stop-conditions work
sed -i '' 's/if stagnant >= STAGNANT_SCROLL_LIMIT:/if False:/' references/twitter/browser.py
pytest references/tests/test_twit_browser.py -q     # expect: red
git checkout -- references/twitter/browser.py
find references -name __pycache__ -type d -exec rm -rf {} +   # see below
pytest references/tests/test_twit_browser.py -q     # expect: green
```

**Clear `__pycache__` when you restore.** Python validates cached bytecode by
(mtime, size) at second granularity. A one-character mutation — `>= 60` to
`>= 30` — does not change the file size, and mutate-run-restore usually happens
inside the same second, so the restored file can look unchanged to the import
system while the interpreter keeps running the MUTATED bytecode. The symptom is
maddening: `inspect.getsource` shows the correct source, evaluating the same
expression by hand gives the correct answer, and the running function does
something else. `dis.dis(fn)` shows the truth (`LOAD_CONST (30)` against a file
that reads `60`). This cost an hour once; clearing the cache costs nothing.

Mutations that were verified to turn the suite red: disabling stagnation detection,
disabling the runtime budget, disabling drain mode, removing the millisecond-expiry
conversion, removing the session-cookie check, removing root-bounce detection,
removing signed-in-profile preference, and inverting the Following-tab branch.

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

### pytesseract / numpy C Extension Crash

`import pytesseract` → `import numpy` → `_multiarray_umath` C extension crashes under pytest-cov with `ImportError: cannot load module more than once per process`. This is a known macOS + numpy + `sys.settrace` interaction.

**Fix (lazy import + mock injection):** Make pytesseract a lazy import inside the two functions that use it, with an optional `pytesseract` parameter so tests pass a mock directly:

```python
# rename/helpers.py
def extract_first_line(image_path: Path, pytesseract=None) -> Optional[str]:
    if pytesseract is None:
        pytesseract = _get_tesseract()  # lazy import inside here
    if pytesseract is None:
        return None
    ...
```

Tests pass a `MagicMock` instead of using `patch`:

```python
def test_returns_first_line(self):
    mock_pt = MagicMock()
    mock_pt.image_to_string.return_value = "First line\nSecond line"
    with patch("rename.helpers.Image.open", return_value=MagicMock()):
        result = extract_first_line("test.png", pytesseract=mock_pt)
    assert result == "First line"
```

This avoids importing pytesseract under coverage entirely.

**The `--ignore` flags outlived the fix (found 2026-08-12).** Before the lazy import, the
two OCR test files were excluded from every coverage run. That fix removed the crash — both
files now run clean under `--cov` (41 and 19 tests) — but the exclusions stayed in CLAUDE.md
and here, and nobody re-checked. They were not free: those files carry ~95 statements of
`rename/helpers.py` (39% → 90% with them) and `rename/cli.py` (60% → 99%), so the documented
pre-push gate measured modules whose tests it had just discarded and scored **94.09%** against
its own 95% floor. **The gate could not pass at HEAD, for anyone**, which is how a gate becomes
decorative. Without the ignores: 95.11%, 2120 passed.

Generally: a workaround for a bug is a claim about the world that stops being true when the
bug is fixed. When a root-cause fix lands, go delete the workarounds it obsoletes — and if a
coverage floor is involved, remember that excluding a *test* file silently subtracts its
module's coverage from a denominator that still counts that module.

### MLX Function Testing (`query_mlx_for_filename`)

MLX-dependent functions in `rename/llm.py` mock the MLX layer entirely:

```python
def test_first_model_succeeds(self):
    from rename.llm import query_mlx_for_filename
    with patch("rename.llm.find_mlx_model", return_value=Path("/tmp/test.mlx")), \
         patch("rename.llm.find_any_working_mlx_model", return_value=None), \
         patch("rename.llm.call_mlx", return_value="my_cool_file"), \
         patch("rename.llm.process_mlx_content", side_effect=lambda x: x), \
         patch("rename.llm.PROMPT_TEXT_TO_FILENAME", "Text: {text}"), \
         patch("rename.llm.FILENAME_MODELS", ["test-model"]), \
         patch("rename.llm.MLX_MODELS_DIR", Path("/tmp")):
        result = query_mlx_for_filename("some text")
    assert result == "my_cool_file"
```

Key: mock both the MLX discovery (`find_mlx_model`, `find_any_working_mlx_model`) AND the MLX response (`call_mlx`, `process_mlx_content`). Also mock module-level config constants (`FILENAME_MODELS`, `PROMPT_TEXT_TO_FILENAME`, `MLX_MODELS_DIR`).

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

### 8. `query_mlx_for_filename` lines 147/150 are unreachable dead code

**File:** `rename/llm.py:146-150`

```python
words = re.findall(r'[a-z]+', content)
if not words:
    continue                         # ← catches empty at 139-140

content = '_'.join(words[:6])        # ← join of alpha-only words

if not re.match(r"^[a-z_]+$", content):  # ← always matches (dead)
    continue

if not any(c.isalpha() for c in content): # ← always True (dead)
    continue
```

`re.findall(r'[a-z]+')` extracts only lowercase-alpha sequences. If any found, the join of those with `_` always passes both guards. These lines have been documented but left in place for safety.

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

Per-module coverage for key packages (excluding numpy-crashing test files):

| File | Coverage | Remaining gaps |
|------|----------|----------------|
| `twitter/summarize.py` | **100%** | — |
| `twitter/cookies.py` | **94%** | L18-21: cryptography ImportError guard (unreachable when crypto IS installed) |
| `rename/llm.py` | **99%** | L147, L150: unreachable dead code (regex `re.findall(r'[a-z]+', ...)` guarantees match) |
| `rename/helpers.py` | 21%+ | OCR functions excluded by numpy crash; non-OCR logic tested via `test_img_llm.py` imports |

### Coverage Limitations

1. **numpy + pytest-cov crash**: `ImportError: cannot load module more than once per process` when coverage traces through `import numpy`. Affects `rename/helpers.py` (pytesseract → numpy). **Workaround**: lazy import + mock injection (see pattern above). OCR tests (`test_img_helpers.py`, `test_image_renamer.py`) run without `--cov`.

2. **Unreachable dead code**: `rename/llm.py` lines 147/150 are logically unreachable — `re.findall(r'[a-z]+', content)` at line 138 guarantees `^[a-z_]+$` matches and alpha chars exist. These will be removed in a future cleanup pass.

### Audit Script

```bash
python3 -m pytest references/tests/ \
  --ignore=references/tests/test_img_helpers.py \
  --ignore=references/tests/test_image_renamer.py \
  --cov=rename.llm --cov=rename.helpers \
  --cov=twitter.summarize --cov=twitter.cookies \
  --cov-report=term-missing
```

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
- `references/tests/conftest.py` — shared fixtures
- `CLAUDE.md` — project rules

## New Modules (Phase 3)

- **`lib/llm/fallback.py`** — shared fallback orchestration (`call_with_fallback`), extracted from the three duplicate implementations in `twitter/summarize.py`, `weekend/llm.py`, `rename/llm.py`.
- **`lib/llm/protocol.py`** — `LLMClient` protocol class unifying the three call interfaces (`lib.osaurus_lib.call`, `lib.llm/client.call`, `lib.mlx_lib.call`).

---

## Bugs Found While Adding Tests (July 2026)

### 1. Empty string is substring of every string — `_score_weekend_exclusions`

`lib/quality_weekend_scorers.py:_score_weekend_exclusions` checked `loc in e` for
exclusion matching, where `loc` is the lowered location string from each item.
If an item had no `"location"` key (or empty location), `loc = ""` and `"" in e`
is always `True` in Python — causing every item without a location to match
every exclusion.

**Fix:** guard both sides of the substring check:
```python
if any((name and e in name) or (name and name in e) for e in exclude):
```
Same pattern applied to the location check and the source-grounding scorer.

### 2. Case mismatch in `_score_weekend_source_grounding` source name comparison

The function lowered item `name` to `_lower(item.get("name", ""))` but compared
against raw (un-lowered) source names from the reference:
```python
source_names = _parse_reference(case).get("source_item_names", [])
if any(sn in name for sn in source_names): ...
```

`"Central Park" in "central park visit"` is `False`. Lowered the source names
at parse time:
```python
source_names = [_lower(sn) for sn in source_names_list]
```

---

## Bugs Found While Adding Tests (July 2026, twitter/camoufox)

Every one of these was invisible to a green test suite and only surfaced by running
the real path. That is the point of `prove-before-claim`: a passing suite tells you
the code does what the tests say, not that the feature works.

### 3. A swallowed `add_cookies` hid a 100% failure rate

`twitter/browser.py` wrapped per-cookie injection in `except Exception: pass`. Newer
Firefox builds (Zen included) store `moz_cookies.expiry` in **milliseconds**;
Playwright rejects that value outright ("only -1 or a positive number for the unix
timestamp in s"). All 17 x.com cookies were rejected, silently, and x.com served its
logged-out page — which scrolls like any other page, so the symptom presented as a
hung scroll loop, not an auth error.

Two fixes, and the second matters more than the first: normalize the unit
(`normalize_expiry`, dividing by 1000 above a plausible-seconds bound), **and stop
swallowing** — `_inject_cookies` now reports the browser's own rejection reason and
aborts if the session cookie did not survive. A per-item `except: pass` around a loop
must always be paired with a check on the aggregate outcome.

### 4. Guest cookies are not a session

A jar full of `guest_id` / `__cf_bm` / `gt` looks like a login and is not. Tests that
mocked cookies as `[{"name": "x"}]` passed happily against code that required a real
session. Fixture realism matters: `SIGNED_IN_COOKIES` now carries `auth_token` + `ct0`
alongside guest cookies, and `GUEST_ONLY_COOKIES` exists specifically to prove the
rejection path.

### 5. A warning printed on the success path

The "Could not locate 'Following' tab" warning lived *inside* `if following_tab:` —
so it fired on every successful click and never on a real failure. Nothing tested the
message, so it survived indefinitely. When a branch exists only to emit a log line,
assert on the log line (`TestFollowingTabReporting` covers both directions).

### 6. A test id that no longer exists still "fails"

Re-running a specific test id after the file was split (`test_image_renamer.py` →
`+ test_rename_cli_main.py`) can report a failure for a test that is no longer there,
because `.pytest_cache` and assertion-rewrite bytecode outlive the source. Before
concluding a failure is stale or a ghost, check `git log -- <file>` — a concurrent
worktree may simply have fixed and moved it. Clear caches with
`rm -rf .pytest_cache references/tests/__pycache__` when a result looks impossible.

### 7. `core.hooksPath` is absolute, so hooks resolve the wrong tree

The pre-commit hook derived its root from `__file__`, which always pointed at the main
checkout. Committing from a linked worktree therefore ran every gate against main's
index — the 500-line check measured main's copy of a file, not the staged one. Resolve
the tree under test with `git rev-parse --show-toplevel`; keep the hook's own path only
for tooling that lives in the main checkout.

---

## Report content-class cases (`references/tests/test_report_class_cases.py`, August 2026)

Stage 0 of G3. One case per weakness class in `docs/REPORT_WEAKNESS_CLASSES.md`, run
against real shipped `tw` / `wk` reports rather than eval fixtures.

**Pattern — `xfail(strict=True)` as the open-defect ledger.** Each case asserts the
*correct* behaviour, so it fails today. `strict=True` means that when Stage 1 fixes the
class the test XPASSes and pytest turns that into a **failure**, forcing the marker to be
removed in the same change. A class therefore cannot be quietly fixed-and-forgotten, nor
quietly left open. This is a structural gate, not a reminder.

Proven end-to-end on 2026-08-02: fixing C8's TOML nesting flipped
`test_C8_declared_exclusions_reach_production` from XFAIL to `FAILED [XPASS(strict)]`,
and the marker was removed.

**Rules for these cases:**
- The checkers live in `eval/report_classes.py` as pure functions over report text. No LLM,
  no network — they run in CI in under a second.
- Every checker needs a **passing** counter-test (`test_checkers_pass_on_a_clean_report`).
  A checker that can only fail proves nothing.
- A class with thin evidence gets `strict=False` and a comment saying why — see C11, whose
  mechanism is real but was never reproduced. Do not promote it without a probe.
- The `wk` fixture is the real 2026-07-31 plan. The `tw` fixtures are synthetic
  reproductions: the real ones carry the user's private timeline and are not vendored.
  `test_real_*_exhibit_the_catalogued_classes` re-runs the checks against the real files
  when present and skips in CI.

## Scorer calibration: the reference is the control (`test_factual_coverage.py`, August 2026)

`summarize_factual_coverage` was failed by all five models that produced real results
(16, 11, 16, 33, 27 — max 33/100). Identical failure across independent models read as a
prompt weakness. It was the scorer: `validate_factual_coverage` matched each key fact as an
exact case-insensitive substring, while the prompt orders the model to reword ("use
narrative verbs and connecting phrases"). It scored verbatim **copying** and called it
coverage.

**Pattern — score the SOURCE against its own extraction targets.** The timeline contains
every key fact by construction, so anything under 100 is a defect with no model involved.
It scored 94: `'Amazon launches drone delivery in Toronto'` is not a substring of its own
source line (`'...in select Toronto neighborhoods'`), so no output could ever match it.
That number was available before any model ran. `test_every_key_fact_is_reachable_from_its_own_source`
keeps it, and it is the gate that fails the next time someone edits a fact or a tweet.

Second control: an ideal answer **in the form the task demands** — all 18 topics, reworded
rather than copied. It scored 5/100. Kept as `test_paraphrase_is_credited_and_still_separated_from_partial_coverage`,
which also asserts a ≥50-point gap from a 4-topic partial, because a metric that credits
paraphrase must not credit everything.

**Bug found by the repair, twice, both too generous.** Matching tokens against raw text:
`'GPT-5'` reduces to `('gpt', '5')` and `'5' in text` is true of every timestamp, so it
scored as covered by all 27 tweets. A regex word boundary still matched the `.5` inside
`Gemini 2.5`. The same looseness flagged a clean summary as repeating a hoax — `'100'` and
`'000'` from "layoffs of 100,000 employees" both matched inside "1000+ qubits". **Run both
sides through the same tokenizer and compare token to token.** A loose matcher is wrong in
both directions at once.

**The test that let this through was vacuous.** `validate_factual_coverage(output, source_text="", key_facts=None)`
— the old assertion passed the fact list *positionally*, landing it in `source_text`, so
`key_facts` was `None` and the validator returned 100 without reading the output.
`assert fc == 100` held for reasons unrelated to coverage. When a validator takes two
same-typed optional parameters, **pass them by keyword in tests**; the positional slip is
invisible and permanent.

### Bug found by writing these: a health check that only lists models

`ev` reports `Server: OK` from `/v1/models`, which answers instantly even when the MLX
backend cannot serve a single token. On 2026-08-02 every MLX model timed out at 300s while
`foundation` answered in 1s, and `ev` spent 600s per task discovering this. **Probe one
trivial completion per model before trusting an eval run** — and treat "server up" and
"server can serve" as different assertions.

## Proving a test can fail: purge `__pycache__` first (August 2026)

"Break the code, watch the test go red, restore, watch it go green" is the rule. The
restore step has a trap that makes the whole check unreliable.

CPython validates a cached `.pyc` against the source's **(mtime, size)** only. Edit a
constant from `16000` to `32000` and the file size does not change -- both are five
characters -- so the only thing standing between you and a stale cache is the mtime.
That is enough to bite:

    edit 32000 -> 16000, run tests   3 failed      (correct, the break worked)
    restore 16000 -> 32000, run      3 failed      (WRONG: still executing the .pyc)
    rm -rf __pycache__, run          22 passed     (the truth)

For ten minutes the evidence said a correct fix was broken. The same mechanism
already produced wrong mutation-testing numbers earlier in the same session --
`tools/mutate.py` carries `purge_bytecode()` for exactly this -- so treat it as a
property of the interpreter, not a one-off.

**Do this** when breaking code deliberately:

```bash
find references -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest <target> -q
```

`-B` and `PYTHONDONTWRITEBYTECODE=1` stop the run WRITING a cache; neither stops it
READING one that already exists. The `rm -rf` is the part that matters.

Signature to recognise: a test that fails while the source you just read plainly
contains the passing value. Verify with
`python -c "import m; print(m.CONST)"` -- when that disagrees with the file, it is
the cache, not your reasoning.

## Testing a machine-wide lock, and three ways the tests can lie (August 2026)

The GPU/osaurus lock (`lib/gpu_lock.py` + `tools/gpu_lock.sh`) is guarded by
`test_gpu_lock.py`, `test_gpu_lock_call_sites.py` and `test_gpu_lock_shell.py`. Four
patterns from writing them generalise to anything that touches shared machine state.

**1. A lock is only testable through an explicit path seam, and the seam needs its own
gate.** The lock's whole correctness rests on `/tmp/mac-osaurus-gpu.lock` being
machine-wide, so tests must not touch it: a real eval may be holding it right now, and a
test that acquires would block it — the exact harm the lock exists to prevent, caused by
the tests for it. `ZTOOLS_GPU_LOCK_DIR` redirects it, an autouse conftest fixture points
it at `tmp_path` per test, and two tests hold the seam honest — one asserts the DEFAULT is
still the machine-wide path, one asserts the redirect is actually in force. Without the
second, every other test in the file could be silently exercising the real lock and
passing for the wrong reason.

**2. `patch("subprocess.run")` reaches every caller in the process, not the one under
test.** Patching an attribute of a shared stdlib module always does. Three separate
failures came from this in one change:

- An existing test used `patch("subprocess.run", side_effect=Exception("cmd error"))` to
  exercise one error path. The moment the lock started shelling out to `ps` on the same
  code path, that test failed on a collaborator it never meant to touch.
- `patch.object(srv.subprocess, "run")` broke the lock's liveness probe, so it reported
  every owner as an impostor, and the test passed or failed for reasons unrelated to its
  subject.
- Patching only `Popen` breaks it too — `subprocess.run` is *implemented on* `Popen`.

Use a **pass-through spy**: record the commands you care about, delegate the rest to the
real function captured at import time.

```python
_REAL_RUN = subprocess.run

def _spy_run(calls):
    def run(cmd, *args, **kwargs):
        parts = cmd if isinstance(cmd, (list, tuple)) else [cmd]
        if parts and str(parts[0]) == "ps":      # the collaborator, not the subject
            return _REAL_RUN(cmd, *args, **kwargs)
        calls.append(" ".join(str(p) for p in parts))
        return MagicMock(returncode=0, stdout="")
    return run
```

**3. Never hand code-under-test a PID that points at the test runner.** The shell fixture
stubbed `pgrep` to echo `os.getpid()`, which reads as harmless until you remember what
`osaurus_one.sh` does with that number: `stop_all` SIGTERMs and then SIGKILLs it. The
first mutation run that removed the lock from the script reached `stop_all` and killed
pytest (`exit 143`, no test output, looks like a hang). Spawn a disposable process
(`subprocess.Popen(["sleep", "600"])`) and hand over ITS pid.

**4. Gate the CLASS, then calibrate the gate both ways.** Guarding the two known
`quit app "osaurus"` sites leaves the failure mode alive — the next tool that quits a slow
server reintroduces it, and every existing test stays green because none of them knows the
new site exists. `TestNoUnguardedServerMutationCanBeAdded` scans the repo for every command
that stops or starts the server and requires each file to be on an allowlist with its
guard. Two traps in writing it:

- **Substrings match prose.** `"osaurus serve"` also matches "osaurus servers" and
  "osaurus serves from MODELS_DIR", which dragged four innocent files onto the allowlist.
  Word-anchor the patterns (`\bosaurus serve\b`).
- **Narrowing a scan can silently disable it.** Stripping `#` comment lines is right (a
  comment cannot execute) but is exactly the kind of edit that turns a gate into a no-op.
  So assert BOTH directions on the same command: executable line caught, commented-out
  line ignored.

The scan found two real defects on its first run, neither of which any existing test
covered: `eval/run_weekend_eval.sh` restarted the server with a bare `osaurus serve &` on
any failed curl — no check for an existing one, so a transient failure left TWO servers —
and two user-facing messages advised starting a server by hand, contradicting the
documented rule and bypassing the lock.

**Do not edit the tree while a coverage run is in flight.** One run in this change
reported 94.66% against a 95 floor and looked like a regression; four other runs of the
same tree reported 95.26% with byte-identical per-file tables. The outlier was measured
while source files were being edited in another window — coverage read a tree that moved
underneath it. Same class as the hazard `tools/sweep_models.sh` documents for its own
source (bash reads a script lazily by byte offset). Before believing a coverage number
that surprises you, re-run it on a quiet tree and diff the TABLES, not just the totals:
the totals cannot tell a real regression from a contaminated measurement, and the per-file
diff answers it in one line.

**Mutation coverage.** All eighteen deliberate breaks of the lock (both languages, both
call sites, the entry-point wiring, and the class gate itself) were caught by a NAMED
failing test, not merely by a red run. A red run with no `FAILED` line is a crash or a
hang, not a gate doing its job — check for the name, not the exit code.
