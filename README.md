# ZTools

Local LLM tools for Osaurus/Ollama.

## Prerequisites

| Requirement | Notes |
|------------|-------|
| **[Osaurus](https://osaurus.ai/)** server | **Hard runtime dependency.** ZTools talks to an Osaurus (or Ollama-compatible) server at `http://localhost:1337`. Install once with `brew install --cask osaurus`, then start it (`osaurus serve` or launch the app). |
| **uv** | Installed automatically as a Homebrew dependency; builds the tool venv |
| **Python 3.13** | Provided by the formula |

## Install (Homebrew)

```bash
# 1. Osaurus server (hard dependency) — macOS 15+, Apple Silicon
brew install --cask osaurus
osaurus serve &>/dev/null &   # or open the Osaurus.app GUI

# 2. ZTools commands
brew tap ztomer/tap
brew install ztomer/tap/ztools
```

Installs five `PATH` commands behind a `uv`-managed venv:

| Command | Module | Notes |
|---------|--------|-------|
| `twitter-summarize` (`twitter`) | `rust/` | High-performance native Rust Twitter summarizer |
| `weekend-plan` (`weekend`) | `rust/` | High-performance native Rust weekend planner |
| `image-renamer` (`rename_images`) | `rust/` | High-performance native Rust image renamer |
| `model-eval` (`oeval`) | `rust/` | High-performance native Rust model quality benchmark |
| `ab_test` | `bin/ab_test` | Side-by-side performance benchmark comparing Rust vs Python references |
| `ztools` | `rust/` | Native Rust launcher for the four tools |

## Quick Start (from a checkout)

```bash
# Dashboard (TUI)
./ztools                           # or: uv run python3 -m tui.app

# Weekend planner
python3 -m weekend

# Twitter summarizer (needs uv for playwright)
uv run -m twitter

# Image renamer (needs uv for vision deps)
uv run -m rename /path/to/images

# Evaluate models
python3 -m eval --quick
python3 -m eval --task file_summary --quick
```

**Entry points** (via `pyproject.toml` `[project.scripts]`):
```
ztools → tui.app (Textual dashboard)
tw     → python3 -m twitter
wk     → python3 -m weekend
rn     → python3 -m rename
ev     → python3 -m eval
```
After `pip install -e .`: `ztools`, `tw --help`, `wk --help`, etc.

**Config location.** Every tool reads the bundled `conf/` directory, resolved
the same way in a checkout and in an installed wheel. Set `ZTOOLS_CONF=<dir>`
to point them at a different one — it fails loudly if the directory does not
exist, rather than silently falling back to built-in defaults. Per-user
overrides layer on top from `~/.config/ztools/config.toml`.

## The Tools

### Weekend Planner

```bash
python3 -m weekend
python3 -m weekend --model qwen3.6-35b-a3b-mxfp4
python3 -m weekend --skip-web
```

Fetches weather → searches local events/venues → 4-phase LLM pipeline (condense weather → extract sources → draft ideas → structure JSON).

---

### Twitter Summarizer

```bash
uv run -m twitter
uv run -m twitter --since 24h
uv run -m twitter --use-cache
uv run -m twitter --model foundation
uv run -m twitter --login        # only if no browser is signed in to x.com
uv run -m twitter --debug        # show the browser window
```

Finds your x.com session → scrolls the Following timeline in a headless browser →
LLM extracts key facts → markdown briefing.

**Session discovery.** It looks for an `auth_token` cookie across every installed
browser, Firefox-family first (Zen, Firefox, LibreWolf, Waterfox — unencrypted, no
keychain prompt), then Chromium-family (Chrome, Chromium — decrypted via the
`Chrome Safe Storage` keychain entry), and reports which one it used. Guest cookies
are not a session: if no browser is signed in it says so and exits rather than
scrolling a logged-out page. `--login` opens a window so you can sign in yourself
into a persistent camoufox profile (`~/.twitter-camoufox-profile`); your password
never passes through ztools.

**Browser backend.** Defaults to [camoufox](https://camoufox.com/) (anti-detect
Firefox) and falls back to Playwright chromium, naming the reason. One-time browser
download:

```bash
uv run python3 -m camoufox fetch     # ~310 MB
```

| Env var | Default | Purpose |
|---------|---------|---------|
| `TWITTER_BROWSER_BACKEND` | `auto` | `auto` \| `camoufox` \| `chromium`. Naming one makes a launch failure loud instead of falling back. |
| `TWITTER_MAX_RUNTIME_S` | `300` | Wall-clock budget for the scroll loop |
| `TWITTER_STAGNANT_SCROLL_LIMIT` | `8` | Give up after N scrolls with no new tweets and no page movement |
| `TWITTER_MAX_SCROLLS` | `1200` | Hard scroll ceiling |
| `TWITTER_SCROLL_PAUSE_MS` | `1800` | Pause between scrolls |
| `TWITTER_CAMOUFOX_HUMANIZE` | `0` | Animated cursor. Off by default — it breaks the Following-tab click. |
| `TWITTER_PROFILE_DIR` | `~/.twitter-camoufox-profile` | Where `--login` stores the session |

Ctrl+C stops the scroll and still summarizes what was collected; press it twice to
quit immediately.

---

### Image Renamer

```bash
uv run -m rename ~/Desktop/screenshots
```

OCR (pytesseract) or Vision LLM → snake_case filename.

---

### Dashboard (TUI)

```bash
ztools                    # installed via Homebrew
./ztools                  # from a checkout (project root)
```

Textual TUI wrapping all four tools. `Tab` between panes, `Enter` to run, `Ctrl+C` to quit.

Does not replace the standalone tools — each still works independently.

---

### Model Evaluator

```bash
python3 -m eval                    # full benchmark
python3 -m eval --quick             # single run, no retries
python3 -m eval --task weekend_fixed
python3 -m eval --model qwen3.6-35b-a3b-mxfp4
```

**Tasks:** `weekend_transient`, `weekend_fixed`, `summarize`, `filename`, `file_summary`, `taxes_anomalies`, `taxes_audit_readiness`, `taxes_synthesis`, `taxes_yoy_narrative`, `taxes_qa`, `taxes_slip_qa`, plus the mixed-signal, adversarial and vision variants — 30 in total; `python3 -m eval` prints the count it actually loaded.

The last three are scored by arithmetic and set membership rather than a keyword
rubric, against a `grounding` block shipped inside each snapshot. `taxes_yoy_narrative`
is the one that ranks; `taxes_slip_qa` saturates and is a hallucination GATE, like
`image_real`.

Adaptive timeout: p95 latency per (model, task) × 1.5. Persisted in `conf/eval_signals.json`.

Quality checks:
- Source matching (detects hallucination), item detail & JSON validation
- Code-pattern detection (file_summary: filename inference vs actual file reading)
- Tax-domain grounding (counts T1135 / Form 106 / box 38 mentions)
- GT-leak detection (score=0 if filed-return totals appear in output)

Taxes tasks (ported 2026-05-17): three real cross-border prompts from [ztomer/Taxes](https://github.com/ztomer/Taxes), sanitized (bucketed $, no PII) in `eval_tasks/data/taxes/`. Harder than other tasks — 2.7-7.5kB user prompts, dense domain context.

---

## Best Models by Task

| Task | Best Model | Notes |
|------|-----------|-------|
| weekend_transient | foundation | 8s, clean JSON |
| weekend_fixed | foundation | 100% |
| summarize | foundation | Clean ## headers |
| filename | foundation | Follows schema |
| file_summary | foundation | Code-pattern validation (44%) |
| vlm | gemma-4-26b-a4b-it-mxfp4 | Vision tasks |

Model quirks: `docs/MODEL_QUIRKS.md`

## Architecture

The Python packages live under `references/`; `conf/`, `tui/`, `docs/`,
`eval_tasks/` and `bin/` sit at the repo root. An installed wheel flattens the
packages into site-packages, so code never derives a shipped-resource path from
its own `__file__` — it goes through `lib.paths` (`conf_path`,
`eval_tasks_path`, `repo_path`).

```
references/
lib/                     # Shared infrastructure
├── tui.py               # Output helpers (· ! ✗)
├── osaurus_lib.py       # Server API, JSON extraction
├── osaurus_server.py    # Server lifecycle (PID-based restart)
├── mlx_lib.py           # Local Apple Silicon MLX fallback
├── content_processing.py# Clean LLM output (thinking, stats)
├── quality_*.py         # Quality scoring models + runners
├── validators_lib.py    # Source matching, hallucination detection
├── config_core.py       # Config loading (lazy, thread-safe)
├── paths.py             # conf/ + eval_tasks/ + repo-root resolution
├── logging_config.py    # Structured logging
├── testing.py           # MockLLM infrastructure
├── llm/                 # LLM client, protocol, fallback, quirks
│   ├── client.py        # Core LLM client
│   ├── protocol.py      # LLMClient protocol
│   ├── fallback.py      # Shared fallback orchestration
│   ├── quirks.py        # Model quirks (canonical source)
│   └── parsing.py       # JSON extraction, output cleaning
└── validators/          # Validator implementations
    ├── helpers.py       # Shared validation helpers
    ├── text_validator.py# Text/entity validation
    ├── json_validator.py# JSON structure validation
    ├── taxes_validator.py# Tax-domain validators (rubric-scored)
    └── taxes_grounded.py # Tax-domain validators (grounding-scored)

tui/                     # Textual TUI dashboard (ztools command)
├── app.py               # Dashboard entry point + scheduler tab

twitter/                 # Twitter summarizer
├── cli.py               # CLI entry point
├── browser.py           # Playwright browser automation
├── cookies.py           # Chrome cookie extraction
├── output.py            # Markdown output formatting
└── summarize.py         # LLM summarization + MLX fallback

weekend/                 # Weekend planner
├── cli.py               # CLI entry point
├── config.py            # Weekend-specific config
├── data.py              # Weather + events data fetching
├── prompts.py           # LLM prompt templates
├── llm.py               # LLM orchestration + MLX fallback
└── output.py            # HTML/markdown output

rename/                  # Image renamer
├── cli.py               # CLI entry point
├── helpers.py           # OCR, text processing
└── llm.py               # LLM + VLM calls + MLX fallback

eval/                    # Model evaluator
├── cli.py               # CLI entry point
├── run.py               # Eval runner
├── tasks_core.py        # Task definitions
├── report.py            # Report generation
├── failures.py          # Failure classification
├── validate.py          # Output validation
├── benchmark_output.py  # Benchmark formatting
└── benchmark_quality.py # Benchmark quality scoring
```

## Run Tests

```bash
pytest references/tests/
pytest references/tests/ -v           # verbose
pytest references/tests/ -k weekend   # run specific test file

# With coverage — OCR tests must be excluded (numpy's C extension crashes
# under pytest-cov), then run separately without --cov:
pytest references/tests/ --ignore=references/tests/test_img_helpers.py --ignore=references/tests/test_image_renamer.py --cov
pytest references/tests/test_img_helpers.py references/tests/test_image_renamer.py
```

Tests never reach the network, launch a browser, or read your real cookies —
`references/tests/conftest.py` enforces that for every test. See `docs/TESTING.md`.

**Key test files:**
- `references/tests/test_quality_entry.py` — Score reconstruction, baseline comparison
- `references/tests/test_content_processing.py` — Thinking block removal
- `references/tests/test_twit_cookies.py` — Cookie extraction error paths
- `references/tests/test_twit_browser.py` — Backend selection, scroll stop conditions, logged-out detection
- `references/tests/test_signal_handling.py` — Ctrl+C drain mode, cleanup ordering
- `references/tests/test_img_llm.py` — LLM server restart, MLX fallback
- `references/tests/test_mlx_lib.py` — Model discovery, execution
- `references/tests/test_weekend_*.py` — Weekend planner output, config, LLM
- `references/tests/test_twit_*.py` — Twitter summarizer output, browser, cookies
- `references/tests/test_model_eval*.py` — Eval runner, reports

Eval results stored in `~/.config/ztools/`. To track:

```bash
git add -f ~/.config/ztools/eval_results.json ~/.config/ztools/eval_history.json
```