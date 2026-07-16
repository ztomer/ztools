# ZTools

Run local LLMs to automate real-world tasks — no API keys, no cloud, no privacy concerns.

## What is this?

ZTools is a suite of productivity scripts powered by local LLMs. They run entirely on your machine via the Osaurus server (Ollama-compatible), handling tasks like:

- **Planning your weekend** — finds family-friendly activities based on weather and local events
- **Summarizing your Twitter feed** — distills your timeline into a factual briefing
- **Renaming screenshots** — generates descriptive filenames from OCR or vision models
- **Evaluating models** — tests which local models work best for your use case

## Prerequisites

| Requirement | Notes |
|------------|-------|
| **[Osaurus](https://osaurus.ai/)** server | **Hard runtime dependency.** ZTools talks to an Osaurus (or Ollama-compatible) server at `http://localhost:1337`. Install once with `brew install --cask osaurus`, then start it (`osaurus serve` or launch the app). |
| **uv** | Installed automatically as a Homebrew dependency; builds the tool venv |
| **Python 3.13** | Provided by the formula |

> **Osaurus is required.** ZTools does not bundle or download models — it is a
> thin client that drives a local LLM server. If the server is not running, the
> tools fail (or auto-restart the Osaurus app if it is installed at
> `/Applications/osaurus.app`).

## Install (Homebrew)

```bash
# 1. Osaurus server (hard dependency) — macOS 15+, Apple Silicon
brew install --cask osaurus
osaurus serve &>/dev/null &   # or open the Osaurus.app GUI

# 2. ZTools commands
brew tap ztomer/tap
brew install ztomer/tap/ztools
```

This installs four commands on your `PATH`, each backed by a `uv`-managed venv:

| Command | Module | Notes |
|---------|--------|-------|
| `weekend` | `weekend` | Family weekend planner |
| `twitter` | `twitter` | Twitter timeline summarizer (needs `twitter --install-browser` once) |
| `oeval` | `eval` | Local model evaluator |
| `rename_images` | `rename` | OCR/vision screenshot renamer (operates on `$PWD`) |

Each wrapper self-heals: if its venv is ever removed it rebuilds it on next run.

## Quick Start (from a checkout)

```bash
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

Shim scripts at root (`twitter_summarizer.py`, etc.) still work for backward compat.

**Entry points** (via `pyproject.toml` `[project.scripts]`):
```
tw    → python3 -m twitter
wk    → python3 -m weekend
rn    → python3 -m rename
ev    → python3 -m eval
```
After `pip install -e .`: `tw --help`, `wk --help`, etc.

## The Tools

### Weekend Planner

Generates a family-friendly weekend itinerary.

```bash
python3 -m weekend
python3 -m weekend --model qwen3.6-35b-a3b-mxfp4
python3 -m weekend --skip-web
```

**What it does:** Fetches weather forecast → searches for local events/venues → runs a 4-phase LLM pipeline per section (condense weather → extract sources → draft ideas → structure JSON). Adapts batch size and timeout per model — values persisted to `conf/phase_signals.json` so learned optimizations carry across runs.

---

### Twitter Summarizer

Turns your Twitter/X timeline into a structured briefing.

```bash
uv run -m twitter
uv run -m twitter --use-cache
uv run -m twitter --model foundation
uv run -m twitter --since 24h
```

**What it does:** Opens Chrome via Playwright → scrolls your timeline → LLM extracts key facts → outputs markdown briefing.

---

### Image Renamer

Generates descriptive filenames for screenshots and photos.

```bash
uv run -m rename ~/Desktop/screenshots
```

**What it does:** Runs OCR (pytesseract) or Vision LLM → LLM generates a clean snake_case filename.

---

### Model Evaluator

Tests which local models perform best on your actual prompts.

```bash
python3 -m eval                    # full benchmark
python3 -m eval --quick             # single run, no retries
python3 -m eval --task weekend_fixed
python3 -m eval --model qwen3.6-35b-a3b-mxfp4
```

**Tasks:** `weekend_transient`, `weekend_fixed`, `summarize`, `filename`, `file_summary`, `taxes_anomalies`, `taxes_audit_readiness`, `taxes_synthesis`

Adaptive: tracks p95 latency per (model, task) and sets timeout = p95 × 1.5. Learned values persist in `conf/eval_signals.json`.

**Quality Checks:**
- Source matching (detects hallucination)
- Item details validation
- JSON structure validation
- Code-pattern detection (file_summary: detects filename inference vs actual file reading)
- Cross-border tax-domain grounding (taxes_*: counts T1135 / Form 106 / box 38 / quarterly-tax mentions)
- GT-leak detection (taxes_*: drops score to 0 if filed-return totals appear in output)

**Taxes tasks (ported 2026-05-17):** three real cross-border tax-prep prompts from [github.com/ztomer/Taxes](https://github.com/ztomer/Taxes), sanitized (dollar amounts bucketed, no PII) and vendored in `eval_tasks/data/taxes/`. Substantially harder than the other tasks — 2.7-7.5kB user prompts, dense domain context, expect specific finding codes. Good for filtering "useful for a real workload" from "can summarize four bullets." Regenerate from the source repo via `python scripts/snapshot_eval_prompts.py --year YYYY --sanitize` and copy `*.sanitized.json` over.

---

## File Summary Validation

The `file_summary` task tests if models actually read file content vs inferring from filenames.

| Check | Points | Detection |
|-------|--------|-----------|
| `##` headers | 20 | Structure compliance |
| Length >= 500 | 20 | Effort indicator |
| Python code patterns | 20 | `.py` files: `def `, `class `, `import ` |
| Markdown patterns | 12 | `.md` files: lists, links, headers |
| YAML patterns | 3 | `.yaml` files: key-value syntax |
| Line variance | 8 | Variety in summary lengths |

**Why?** Models that output "A Python script for..." instead of "def plan_weekend(), validate_json(), async call()" score low.

---

## Best Models by Task

| Task | Best Model | Notes |
|------|-----------|-------|
| weekend_transient | foundation | Fast (8s), clean JSON |
| weekend_fixed | foundation | 100%, reliable |
| summarize | foundation | Fast, clean ## headers |
| filename | foundation | Fast, follows schema |
| file_summary | foundation | Code-pattern validation (44%) |
| vlm | gemma-4-26b-a4b-it-mxfp4 | Vision tasks |

See `docs/MODEL_QUIRKS.md` for detailed model-specific quirks and known issues.

## Architecture

```
lib/                     # Shared infrastructure
├── osaurus_lib.py       # Server API, JSON extraction
├── osaurus_server.py    # Server lifecycle (PID-based restart)
├── mlx_lib.py           # Local Apple Silicon MLX fallback
├── content_processing.py# Clean LLM output (thinking, stats)
├── quality_*.py         # Quality scoring models + runners
├── validators_lib.py    # Source matching, hallucination detection
├── config_core.py       # Config loading (lazy, thread-safe)
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
    └── taxes_validator.py# Tax-domain validators

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

## Development Tools

### Model Quirks Explorer

Discover which prompts work best for a model:

```bash
python3 -m eval.explore_quirks foundation
python3 -m eval.explore_quirks qwen3.6-35b-a3b-mxfp4
```

Tests:
- Simple JSON extraction
- No preamble/markdown prompts
- Schema-strict prompts
- Source matching (detects hallucination)

### Run Tests

```bash
pytest tests/
pytest tests/ -v           # verbose
pytest tests/ -k weekend   # run specific test file
```

**Key test files:**
- `tests/test_quality_entry.py` — Score reconstruction, baseline comparison
- `tests/test_content_processing.py` — Thinking block removal
- `tests/test_twit_cookies.py` — Cookie extraction error paths
- `tests/test_img_llm.py` — LLM server restart, MLX fallback
- `tests/test_mlx_lib.py` — Model discovery, execution
- `tests/test_weekend_*.py` — Weekend planner output, config, LLM
- `tests/test_twit_*.py` — Twitter summarizer output, browser, cookies
- `tests/test_model_eval*.py` — Eval runner, reports

### Eval Results

Eval results are stored in `~/.config/ztools/` (outside the repo). To track them:

```bash
# Track eval results (intentional, not automatic)
git add -f ~/.config/ztools/eval_results.json
git add -f ~/.config/ztools/eval_history.json
```

## Requirements

- **Osaurus or Ollama** server running on port 1337
- **Models** installed (`osaurus pull <model>`)
- **uv** for scripts with extra dependencies (twitter, image_renamer)