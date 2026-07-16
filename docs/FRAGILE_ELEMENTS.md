# ZTools Code Fragility Review

This document provides a systematic review of the `ztools` codebase to identify fragile elements—areas that are prone to failure under runtime changes, environment differences, or API updates. It classifies findings by severity and proposes concrete mitigations.

---

## 📊 Priority Grid of Fragile Elements

| Priority | Area | File | Description | Impact | Status (v0.9.7) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **P0 (Critical)** | Platform / OS | [cookies.py](file:///Users/ztomer/Projects/ztools/twitter/cookies.py) | Hardcoded Chrome database paths and macOS keychain CLI dependency (`security`). | Breaks completely on non-macOS systems or custom profiles. | **Resolved** (Scans all profiles, handles non-macOS keychains) |
| **P0 (Critical)** | Process / Subprocess | [mlx_lib.py](file:///Users/ztomer/Projects/ztools/lib/mlx_lib.py) | Dynamic code generation and script writing under `/tmp/mlx_debug/` run via a sub-shell wrapper. | Execution state leakage; vulnerable to file write errors or missing venv dependencies. | **Mitigated** (Escapes path parameters dynamically) |
| **P1 (High)** | Browser Automation | [browser.py](file:///Users/ztomer/Projects/ztools/twitter/browser.py) | Brittle GraphQL API endpoint matching and UI selectors (e.g., `"Following"` tab). | Silently collects 0 tweets or crashes if Twitter updates its layout/localization. | **Resolved** (Localized term scanning + ordinal fallback) |
| **P1 (High)** | Browser Automation | [browser.py](file:///Users/ztomer/Projects/ztools/twitter/browser.py) | Hardcoded timeouts and arbitrary thread sleeps (`INITIAL_PAGE_WAIT`, `TAB_SWITCH_WAIT`). | Leads to race conditions on slow connections or unnecessary waiting on fast connections. | **Resolved** (Dynamic wait_for_selector timers) |
| **P1 (High)** | Error Handling | [llm.py](file:///Users/ztomer/Projects/ztools/rename/llm.py) | Silent swallowing of broad exceptions via bare/generic `except Exception:` blocks. | Obscures actual HTTP connection failures, JSON decoding issues, or timeouts. | **Resolved** (Detailed warning logging configured) |
| **P2 (Medium)** | Architecture | [client.py](file:///Users/ztomer/Projects/ztools/lib/llm/client.py) | Persistent global mutable state (`_session`) for HTTP connection pooling. | Potential connection leakage or socket reuse issues; complicates test isolation. | **Resolved** (Self-healing reset_session on error & teardown) |
| **P2 (Medium)** | Configuration | Multiple | Duplicate defaults for model constants (e.g., `DEFAULT_MAX_TOKENS`, `DEFAULT_PORT`). | Leads to configuration drift and inconsistent model invocation bounds. | **Resolved** (Centralized defaults imported from constants.py) |

---

## 🔍 Detailed Findings

### P0 — Critical (Immediate Functional Impact)

#### 1. Platform and profile dependencies in Chrome cookie extraction
* **File:** [cookies.py:24-30](file:///Users/ztomer/Projects/ztools/twitter/cookies.py#L24-L30)
* **Vulnerability:**
  - `CHROME_COOKIES_DB` is hardcoded to the default macOS path: `Library/Application Support/Google/Chrome/Default/Cookies`.
  - The decryption key is extracted using the macOS-only command: `["security", "find-generic-password", "-w", "-s", "Chrome Safe Storage"]`.
* **Fragility:** If a user runs this on Linux or Windows, or has Chrome installed in a non-standard path, or is using a secondary profile (e.g., `Profile 1` instead of `Default`), cookie collection fails instantly.
* **Mitigation:**
  - Resolve the database path dynamically by scanning common locations or allowing the profile folder to be configured in `conf/twitter.yaml`. Add a cross-platform helper for keychain queries (e.g., using python libraries like `keyring`).

#### 2. Dynamic file-writing and subshell execution for MLX models
* **File:** [mlx_lib.py:168-210](file:///Users/ztomer/Projects/ztools/lib/mlx_lib.py#L168-L210)
* **Vulnerability:** 
  - `call_mlx()` writes an ad-hoc python script (`script_{uid}.py`) and prompt file to the filesystem, then executes it using `subprocess.run` inside `rtk uv run --with mlx --with mlx-lm`.
* **Fragility:**
  - Any disk write issue or missing python environment will block execution.
  - The generated script is prone to path-handling errors if the folder paths contain special characters or spaces.
  - The subprocess timeout is exceptionally high (`MLX_GEN_TIMEOUT = 1800` / 30 minutes), which can cause the client to hang indefinitely if the subprocess gets stuck.
* **Mitigation:**
  - Use python's internal multiprocessing or import the MLX libraries directly in-process when run in the correct virtual environment, rather than wrapping shell scripts.
  - Reduce execution timeout limits to a reasonable maximum (e.g., 2–3 minutes).

---

### P1 — High (Operational Instability)

#### 3. Brittle Playwright UI selectors and title matching
* **File:** [browser.py:172-183](file:///Users/ztomer/Projects/ztools/twitter/browser.py#L172-L183)
* **Vulnerability:**
  - Detects login state using `page.title().lower()` string matches: `("log in", "login", "sign in", "signin")`.
  - Locates the Following tab using a text-based selector: `page.locator('[role="tab"]', has_text="Following").first`.
* **Fragility:**
  - If a user's browser is localized in another language (e.g., French, German), neither the login keywords nor the `"Following"` text will match, failing the tab switch.
  - Social media markup is highly volatile; tag roles and class names change frequently.
* **Mitigation:**
  - Check network responses or cookies instead of title text to confirm authentication. Fall back gracefully if the `"Following"` selector cannot be found, rather than letting Playwright wait for a hard timeout.

#### 4. Arbitrary static delays and scroll limits
* **File:** [browser.py:21-30](file:///Users/ztomer/Projects/ztools/twitter/browser.py#L21-L30)
* **Vulnerability:**
  - `INITIAL_PAGE_WAIT = 3` and `TAB_SWITCH_WAIT = 2` are hardcoded seconds.
  - The scroll loop pause is set to a static 1.8 seconds.
* **Fragility:**
  - On slow network connections, 3 seconds might not be enough to load the DOM, causing subsequent steps to fail.
  - On fast connections, this wastes execution time.
* **Mitigation:**
  - Replace static `sleep()` calls with Playwright's native `wait_for_selector()` or event-based listeners (e.g., waiting for the GraphQL timeline responses).

#### 5. Silent exception swallowing in LLM call loops
* **File:** [llm.py:85-109](file:///Users/ztomer/Projects/ztools/rename/llm.py#L85-L109) and [llm.py:117-157](file:///Users/ztomer/Projects/ztools/rename/llm.py#L117-L157)
* **Vulnerability:**
  - Iterates over `RELEVANCE_CHECK_MODELS` and `FILENAME_MODELS` inside a `try/except Exception:` block without logging the exception.
* **Fragility:** If a model fails because of a malformed prompt, server authentication error, or connection refusal, the code silently continues to the next model. The developer has no insight into why fallbacks are occurring.
* **Mitigation:**
  - Catch specific exceptions (e.g., `requests.exceptions.Timeout`, `requests.exceptions.ConnectionError`, `json.JSONDecodeError`).
  - Log failures with `logger.warning` or `logger.exception` to capture traceback context.

---

### P2 — Medium (Maintenance & Architecture)

#### 6. Global HTTP sessions without connection state checks
* **File:** [client.py:46-55](file:///Users/ztomer/Projects/ztools/lib/llm/client.py#L46-L55)
* **Vulnerability:**
  - Employs a global `_session` variable to reuse TCP connections across HTTP posts.
* **Fragility:**
  - If a connection is closed by the server, requests can fail with stale pool errors.
  - Tests require explicit setup/teardown logic to clear `_session`, which is prone to pollution.
* **Mitigation:**
  - Wrap connection pooling inside a client manager object that handles session lifetime and recreates sockets upon connection drops.

#### 7. Duplicated default constants (Config Drift)
* **Files:** `lib/config_core.py`, `lib/osaurus_lib.py`, `lib/llm/constants.py`
* **Vulnerability:**
  - Default connection details, model timeouts, and token limits are defined as literals in multiple source files.
* **Fragility:** Updating a default in one file (e.g., changing the default port from `1337` to `11434`) leaves the other files with outdated defaults, causing subtle bugs.
* **Mitigation:**
  - Keep a single source of truth for fallback constants in `lib/llm/constants.py` and import them everywhere, or ensure all modules read strictly from the central YAML configuration.

---

## 🛠️ Summary Action Items

1. **Robust Browser Locales:** Update [browser.py](file:///Users/ztomer/Projects/ztools/twitter/browser.py) to parse localization-independent selectors or use element indicators (e.g., data attributes or tab order).
2. **Detailed Logging:** Replace bare `except:` clauses in [llm.py](file:///Users/ztomer/Projects/ztools/rename/llm.py) with structured logging of fallback triggers.
3. **Configuration Verification:** Run `python3 tools/check_config_debt.py` to ensure no new hardcoded dependencies or configs leak into the codebase.
