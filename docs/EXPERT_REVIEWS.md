# Expert Reviews

## Susan Kare (Iconography, Apple Macintosh)

> "Your symbol system is coherent — ·/✗/! reads like a unified icon family. That's good. But then you staple on Rich Progress spinners, === banners, and emoji. Imagine Helvetica next to Comic Sans. Pick one visual language."

**Verdict:** B+. Symbols right, ornamentation wrong.

### Fixes
- Kill Rich Progress in weekend package
- Kill `===` decorative borders
- Kill the `N/A` column — wasted pixels
- Weather: readable sentence, not raw data dump

---

## Dieter Rams (Industrial Design, Braun)

> "Less but better — your progress spinner shows 5 tasks with timing down to the second. That's not better, it's more. The N/A score column is dishonest: you're showing data you don't have. The weather line is not thorough — you just concatenated raw floats."

**Verdict:** C+. Functional but noisy. Every character must earn its place.

### Fixes
- Replace progress with 5 simple `·` lines that complete silently
- Kill the `N/A` column
- Format weather: `Fri 27°C (rain), Sat 20°C (heavy rain), Sun 18°C (rain)`
- Remove `===` header — adds nothing

---

## Linus Torvalds (Linux Kernel)

> "Taste matters. Renaming 42 files prints 42 lines of old_name -> new_name — that's noise. Show me the diff, not the full path twice. And Foundation as default model? Either it's good enough to be default or it isn't. Don't hide behind weak defaults."

**Verdict:** C. Too much noise, wrong model.

### Fixes
- Collapse renames: `· 42 renamed (_1 suffix stripped), 0 skipped, 0 errors`
- Show `✗` lines only for failures
- Change default model from foundation to something competent
- Rip out Progress infrastructure for 5 sequential calls

---

## Uncle Bob (Clean Architecture)

> "You're violating SRP all over the place. weekend/cli.py::main() fetches data, calls LLMs, and formats output. weekend/output.py exists but the main function still has console.print calls. Output formatting is a detail — domain logic shouldn't know about console or Progress."

**Verdict:** D. Business logic knows about terminal rendering.

### Fixes
- Push all `console.print()` calls into `*_output.py` modules
- Main functions return data structures; presenters handle rendering
- `lib/tui.py` is good but not enough if callers scatter output throughout business logic

---

## Uncle Bob (Clean Architecture — Second Pass)

> "I see you split the monoliths. Good. But splitting a mess into smaller files gives you a directory of mess. The real problems are structural, not cosmetic."

### 1. Three LLM Clients, Zero Interfaces

> "You have `lib/osaurus_lib.call()`, `lib/llm/client.call()`, and `lib/mlx_lib.call()`. They do the same thing: send a prompt, get a response. They have different argument orders, different return shapes, different error handling. The application code in `twitter/summarize.py`, `weekend/llm.py`, and `rename/llm.py` each re-implements the fallback chain — try Osaurus, retry with restart, fall back to MLX. That's the same pattern copy-pasted three times. A dependency inversion would collapse this into one interface and one fallback strategy, configurable per app."

**Verdict:** D. Three parallel implementations, zero abstractions, repeated orchestration.

### 2. Global Mutable Config

> "`config_core._config` is a module-level dict. `_config_loaded` is a module-level flag. `_model_configs_cache` is another global cache. Every test that touches config must call `reset_config()` or it poisons the next test. This isn't 'shared state' — it's a hidden global variable you're afraid to admit exists. A Config object — passed, not imported — would eliminate an entire class of test bugs and make the dependency explicit."

**Verdict:** F. Globals disguised as a config system.

### 3. The Shim Tax

> "Every module split creates a backward-compatibility shim that re-exports the sub-modules. `lib/config.py` is a shim. `lib/quality.py` is a shim. `lib/validators_lib.py` is a shim. `lib/osaurus_lib.py` is a shim. That's four files that exist only to say `from X import *`. Shim files are a migration tool, not an architecture. Either commit to the new structure and update the imports, or don't split. Half-measures accumulate as cruft."

**Verdict:** C-. Migration debt with no migration deadline.

### 4. Two Eval Systems, Neither Complete

> "`eval/` and `eval_tasks/` both define `TASKS` dicts. Both have `run.py`. Both have reporting. `eval_tasks/analyze.py` is a copy-paste of `eval/report.py` — 474 lines and 475 lines of near-identical formatting code. You can't maintain two parallel implementations. Pick one, delete the other, and route all traffic through it."

**Verdict:** F. The worst kind of duplication — structural, not accidental.

### 5. SRP at Package Scale

> "The `eval/cli.py` module does argument parsing, model discovery, task construction, memory monitoring, flush-between-models orchestration, and result reporting. That's six responsibilities in one function (`main()`). The package split gave you smaller files, but each file still does too much. `cli.py` should parse arguments and call a use case. The use case should not know about `console.print()`, Rich tables, or CSV export."

**Verdict:** D+. Package split was mechanical, not architectural.

---

## Linus Torvalds (Kernel Hacker — Code Review)

> "Good taste is not aesthetic. It's about not doing stupid shit. Let me show you the stupid shit."

### 1. Dead Fallback Code

> "`lib/mlx_lib.py` has 400+ lines of subprocess management, model discovery, prompt formatting — and per your own docs, it's broken. The subprocess returns empty. But every fallback chain in every application still calls it. `twitter/summarize.py` spends time on `find_mlx_model() + find_best_mlx_model()`. `weekend/llm.py` calls `find_text_mlx_model()`. `rename/llm.py` calls `query_mlx_for_filename()`. All of which return nothing useful. There are two options: either make it work, or rip it out and stop lying to yourself. A broken fallback path that adds latency to every call is worse than no fallback."

**Verdict:** F. Shipping dead code that slows down the working path.

### 2. Models as Magic Strings

> "You have model names like `qwen3.6-35b-a3b-mxfp4` as string literals scattered through `twitter/summarize.py`, `weekend/llm.py`, and `rename/llm.py`. When a model name changes, you need a grep + regex + find-and-replace across the whole tree. The config system (`conf/models/`) already has the right idea — model configs in YAML — but then the fallback lists are hardcoded arrays in Python files. `FILENAME_MODELS` in `rename/llm.py`. `SUMMARIZE_MODELS` pattern in `twitter/summarize.py`. If these lived in config, adding a model would be a YAML change, not a code change."

**Verdict:** D. Taste: every magic string is a future diff you haven't written yet.

### 3. eval_tasks/ vs eval/ is Not Duplication — It's Confusion

> "Same thing, two directories with nearly the same files. If I land on this codebase and need to run an eval, which do I use? `python3 -m eval` or `python3 -m eval_tasks`? The `__main__.py` files don't explain the difference. The README in `eval_tasks/` says 'How to add tasks' but doesn't say why this directory exists when `eval/` already has tasks. I'd bisect this to a refactor that never finished. Clean up your mess or annotate it."

**Verdict:** F. Two directories, one purpose, zero documentation on the difference.

### 4. No Script Entry Points

> "`python3 -m twitter`, `python3 -m weekend`, `python3 -m rename`, `python3 -m eval` — everyone types all 12 characters every time. There are zero `[project.scripts]` entries in `pyproject.toml`. Four tools, zero shell commands. `tw`, `wk`, `rn`, `ev` would each save 9 keystrokes. It's a trivial addition and it's not there. Small omission, but it tells me nobody actually runs these tools from a terminal day-to-day."

**Verdict:** C. Detail-oriented failure.

### 5. Test Coverage of the Wrong Thing

> "58 test files, 96% coverage. Nice number. But every single test mocks the LLM layer. You have zero tests that verify what happens when the Osaurus server returns garbage JSON, or times out, or returns empty, or returns HTML instead of JSON. The error-handling code in `eval/run.py`, `twitter/summarize.py`, `weekend/llm.py` — all the try/except/retry logic — is completely untested with real conditions. You tested the happy path with mocks. The unhappy path is where bugs live."

**Verdict:** C-. Coverage number is a vanity metric when the hard parts aren't tested.

### 6. ApplyModelQuirks — Two Implementations

> "`apply_model_quirks()` exists in `lib/osaurus_lib.py` and in `lib/llm/quirks.py`. Same function, offset slightly, maintained independently. If I fix a quirk for qwen in one, the other is wrong until someone remembers. The model quirk logic belongs in one place — called by all transport layers. You already have it right in `eval/run.py:_call_model()` which calls `apply_model_quirks()`. But `twitter/summarize.py` and `weekend/llm.py` call it from `osaurus_lib`. The `rename/llm.py` calls it from `lib.llm.quirks`. Two copies of the same logic, diverging over time."

**Verdict:** D. Technical debt with a ticking clock.

---

## Uncle Bob (Clean Code — Bug Hunt)

> "I told you about structure last time. Now I opened the files. You have *actual bugs* — not style problems, not architecture debates. These will fail at runtime."

### 1. Mutating Content While Iterating Over It

> "`content_processing.py` at lines 42-49 — you slice `content` with `content = content[output_match.end():]`, then call `content.index(marker)` on the *new* string. If the marker was in the part you cut off, you get `ValueError`. You also call `content.index(marker)` twice — once to search, once to slice — without caching the result. You perform surgery on a patient, then look for the appendix in the bucket. This will crash on real LLM output."

**Verdict:** F. Text-processing 101: don't mutate what you're iterating.

### 2. Dead Code That References Undefined Variables

> "`eval/validate.py` lines 129-229 — that's 100 lines of code the `return` on line 127 guarantees nobody will ever reach. This isn't 'commented-out' or 'legacy'. It will crash if called, because it references `data_str` and `data_lower` that were local to the earlier scope and died on the return. `lib/validators/helpers.py` lines 94-102 has the same disease — dead code referencing `content`, which doesn't exist in that scope. You aren't 'keeping code for later.' You're maintaining corpse."

**Verdict:** F. Dead code is a liability. Delete it or it will be resuscitated by accident.

### 3. A Regression-Detection System That Always Reports Regression

> "`lib/quality_entry.py` line 50: `Score(dname, dscore, 0.0)`. The third argument is `weight`, and you set every dimension's weight to zero. The composite score becomes `weighted_sum / total_weight = 0.0 / 0.0 = NaN`. When `compare_to_baseline` checks deltas, it computes `NaN - previous_score = NaN` — which, in a delightful artifact of IEEE 754, means *every comparison reports regression*. `--regression-only` mode is a random-number generator that always says 'bad.' You probably introduced this during the module split and never tested it end-to-end."

**Verdict:** F. The quality system's headline feature is a lie.

### 4. SQL Injection in Cookie Extraction

> "`twitter/cookies.py` line 73: `conn.execute(f"SELECT ... FROM cookies WHERE {domain_clauses}")`. String interpolation into SQL. Yes, the current callers pass hardcoded strings. But the pattern is wrong. One day someone will say 'support custom domains' and the world will get a space in `host_key`. And `tempfile.mktemp()` on line 66 — that's a TOCTOU race. Between generating the name and copying the file, a symlink could redirect your write anywhere. You copied the Chrome cookie database to a predictable temp file, then queried it insecurely."

**Verdict:** D. Two security patterns you should never, ever use. In the same function.

### 5. Credential Leakage Without Warning

> "`twitter/cookies.py` lines 44-45: if the `cryptography` module isn't installed, encrypted cookie values are decoded as `utf-8` with `errors='replace'`. The AES ciphertext becomes garbled Unicode. No warning. No log. The caller sends this garbage to Twitter as the cookie value. Authentication silently fails, and the user sees 'no tweets found' with zero indication that Chrome cookies couldn't be decrypted."

**Verdict:** D. Silent failure disguising itself as a feature.

### 6. Import-Time Config Evaluation

> "`weekend/config.py` runs `load_weekend_config()` at module level. `lib/config_core.py` runs `_auto_load()` on import. `lib/tui.py` reads and parses `~/.config/zstyle` at module level. Three different files evaluate potentially-missing-or-malformed configuration at import time. A broken `weekend.yaml` or malformed `zstyle` file crashes the entire tool at import, not at the call site. Configuration loading belongs in a factory, not in module scope."

**Verdict:** D. Import-time side effects in three places. `import weekend` should not raise `FileNotFoundError`.

---

## Linus Torvalds (Kernel Hacker — Bug Hunt)

> "I said 'good taste' last time. Forget taste. You have bugs that would make me reject a kernel driver patch."

### 1. `pkill -f` is Not a Surgical Tool

> "`lib/osaurus_server.py` line 13 and `rename/llm.py` line 45: `subprocess.run(["pkill", "-f", "osaurus"])`. The `-f` flag matches against the *full* process argument list. Any process with 'saurus' in its argv gets killed — including unrelated Python scripts that happen to import a module with 'saurus' in the name, your shell's command history grep for 'osaurus', or literally anything running on the machine. `pkill -f osaurus` is a chainsaw. You should be using PID files or process groups."

**Verdict:** F. This kills the wrong process. Ship it and your users will learn what `SIGTERM` feels like for no reason.

### 2. A 30-Minute Subprocess That Catches Ctrl+C

> "`lib/mlx_lib.py` line 186: `except Exception as e:` with `timeout=1800` (30 minutes) at line 175. If a user hits Ctrl+C during model inference, Python raises `KeyboardInterrupt` — which is a `BaseException`, not `Exception`. So that's fine, Ctrl+C escapes. But the `except` on line 186 also catches `SystemExit` and `GeneratorExit`. And the subprocess module raises `TimeoutExpired` (an `Exception` subclass), which this catch block will handle by logging and then falling through to return `None`. But wait — line 173 says the timeout is 1800 seconds. A user waits 30 minutes for a timeout that could have been caught earlier. And the stdout output check at line 180 returns output even when `returncode != 0`. This function has never seen a production edge case it couldn't mishandle."

**Verdict:** D+. One good decision (not catching BaseException) surrounded by bad ones.

### 3. Self-Import as a Monkey-Patch Vehicle

> "`eval/cli.py` line 295: `import eval.cli as me`. Inside `eval/cli.py`. You're importing the module you're currently in to get a reference to its own namespace, then replacing `me.run_eval` with a different function. This works in CPython because `import` returns the partially-loaded module if it's already in `sys.modules`. But it's fragile, confusing, and suggests you should have separated the runner from the CLI. It's the import equivalent of `goto`."

**Verdict:** D. Works by accident, confuses by design.

### 4. Function Returns the Wrong Type on the Except Path

> "`lib/validators/text_validator.py` — in several scoring functions, the happy path returns a `tuple[list[str], int]` but the 'we can't even score this' path returns `(["can't score"], 0.0)`. The `int` vs `float` mismatch is harmless in Python but indicates the author wasn't sure what the return type should be. `lib/validators/helpers.py:81` and `:99` — bare `except:` blocks that catch `KeyboardInterrupt`, making Ctrl+C during validation impossible."

**Verdict:** C. Inconsistent types signal copy-paste coding.

### 5. Accumulating Debug Files

> "`lib/mlx_lib.py` lines 137-141: every call to `call_mlx` creates `/tmp/mlx_debug/prompt_{uuid}.txt` and `/tmp/mlx_debug/script_{uuid}.py`. Never deleted. UUID-based naming means they never collide. Also never deleted. On a system running evals daily, `/tmp/mlx_debug/` accumulates indefinitely. The user has no opt-out. These files contain the full prompt text — potentially including personal data from weekend plans or Twitter timelines."

**Verdict:** D. A data leak that grows without bound.

### 6. Thread-Unsafe Console In a Daemon Thread

> "`eval/cli.py` lines 144-159: `monitor_memory_loop` spawns a `daemon=True` thread that calls `console.print()` in a 5-second loop. `Rich.Console` is not thread-safe. Concurrent `console.print()` from this thread and the main thread's output produces interleaved terminal garbage. And because it's a daemon thread, it's killed unceremoniously when the process exits — no cleanup, no final memory reading."

**Verdict:** D. Racing the terminal one daemon thread at a time.

---

## Findings (Ranked by Impact)

| # | Priority | Item | Area | Expert | Status |
|---|----------|------|------|--------|--------|
| 1 | **CRITICAL** | Fix string mutation during iteration in `remove_thinking_blocks` | lib/content_processing | Uncle Bob | ✅ |
| 2 | **CRITICAL** | Fix zero-weight reconstruction in `quality_entry.py` — `--regression-only` is broken | lib/quality | Uncle Bob | ✅ |
| 3 | **CRITICAL** | Kill dead code in `eval/validate.py:129-229` and `validators/helpers.py:94-102` | eval, lib | Uncle Bob | ✅ |
| 4 | **CRITICAL** | Replace SQL string interpolation with parameterized query in `twitter/cookies.py` | twitter | Uncle Bob | ✅ |
| 5 | **CRITICAL** | Replace `pkill -f` with PID-file-based process management | lib/osaurus, rename | Torvalds | ✅ |
| 6 | **HIGH** | Warn when cryptography not installed / cookies returned as garbage | twitter | Uncle Bob | ✅ |
| 7 | **HIGH** | Clean up `/tmp/mlx_debug/` files or add opt-out / TTL | lib/mlx | Torvalds | ✅ |
| 8 | **HIGH** | Replace `tempfile.mktemp` with `NamedTemporaryFile(delete=False)` | twitter | Uncle Bob | ✅ |
| 9 | **HIGH** | Add error handling for keychain read failure in cookie extraction | twitter | Uncle Bob | ✅ |
| 10 | **HIGH** | Remove bare `except:` clauses (catch KeyboardInterrupt) in 6 locations | eval_tasks, lib | Both | ✅ |
| 11 | **MEDIUM** | Move config loading out of module scope into init/lazy functions | weekend, lib | Uncle Bob | ✅ |
| 12 | **MEDIUM** | Make monitor_memory_loop thread-safe or move to main-loop polling | eval/cli | Torvalds | ⬜ |
| 13 | **MEDIUM** | Cache `content.index()` results; fix double-eval in generator | lib/content_processing | Uncle Bob | ✅ |
| 14 | **MEDIUM** | Use `requests.Session` as context manager everywhere | All apps | Uncle Bob | ⬜ |
| 15 | **MEDIUM** | Add integration tests for LLM error/edge cases | tests | Torvalds | ⬜ |
| 16 | **LOW** | Fix shared mutable references in `TASKS["json"] = TASKS["weekend_transient"]` | eval/tasks | Uncle Bob | ⬜ |
| 17 | **LOW** | Remove `eval/cli.py:295` circular self-import | eval/cli | Torvalds | ⬜ |
| 18 | **LOW** | Fix `"name"` key duplication in `normalize_keys` | lib/osaurus_output | Uncle Bob | ⬜ |
| 19 | **LOW** | Fix misleading comment / inverted logic in `text_validator.py` "wordy" bonus | lib/validators | Uncle Bob | ⬜ |
| 20 | **LOW** | Fix tautological test assertion in `test_mlx_lib.py:298` | tests | Torvalds | ⬜ |
