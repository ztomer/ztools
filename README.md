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
| **[Osaurus](https://github.com/osaurus) or [Ollama](https://ollama.com)** | Server running at `http://localhost:1337` |
| **Python 3.11+** | For most scripts |
| **uv** | Required for scripts with browser/vision dependencies |

## Quick Start

```bash
# Weekend planner
python3 weekend_planner.py

# Twitter summarizer (needs uv for playwright)
uv run twitter_summarizer.py

# Image renamer (needs uv for vision deps)
uv run image_renamer.py /path/to/images

# Evaluate models
python3 model_eval.py --quick
python3 model_eval.py --task file_summary --quick  # test file reading quality
```

## The Tools

### Weekend Planner

Generates a family-friendly weekend itinerary.

```bash
python3 weekend_planner.py
python3 weekend_planner.py --model qwen3.6-35b-a3b-mxfp4  # use specific model
python3 weekend_planner.py --skip-web  # use cached search results
```

**What it does:** Fetches weather forecast → searches for local events/venues → uses LLM to filter and rank activities.

---

### Twitter Summarizer

Turns your Twitter/X timeline into a structured briefing.

```bash
uv run twitter_summarizer.py
uv run twitter_summarizer.py --use-cache      # skip fetching, use last run
uv run twitter_summarizer.py --model foundation
uv run twitter_summarizer.py --since 24h      # tweets from last 24 hours
```

**What it does:** Opens Chrome via Playwright → scrolls your timeline → LLM extracts key facts → outputs markdown briefing.

---

### Image Renamer

Generates descriptive filenames for screenshots and photos.

```bash
uv run image_renamer.py ~/Desktop/screenshots
```

**What it does:** Runs OCR (pytesseract) or Vision LLM → LLM generates a clean snake_case filename.

---

### Model Evaluator

Tests which local models perform best on your actual prompts.

```bash
python3 model_eval.py                    # full benchmark
python3 model_eval.py --quick             # single run, no retries
python3 model_eval.py --task weekend_fixed  # test specific task
python3 model_eval.py --model qwen3.6-35b-a3b-mxfp4  # test specific model
```

**Tasks:** `weekend_transient`, `weekend_fixed`, `summarize`, `filename`, `file_summary`

**Quality Checks:**
- Source matching (detects hallucination)
- Item details validation
- JSON structure validation
- Code-pattern detection (file_summary: detects filename inference vs actual file reading)

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
lib/
├── osaurus_lib.py        # Server API, JSON extraction, normalization
├── mlx_lib.py           # Local Apple Silicon MLX models
├── content_processing.py  # Clean LLM output (markdown, thinking)
├── validators_lib.py     # Quality scoring (source matching)
├── config.py            # Centralized config
└── logging_config.py   # Structured logging
```

## Development Tools

### Model Quirks Explorer

Discover which prompts work best for a model:

```bash
python3 explore_model_quirks.py foundation
python3 explore_model_quirks.py qwen3.6-35b-a3b-mxfp4
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

**Test Coverage:**
- `test_validators.py` — scoring logic, source matching
- `test_parse.py` — JSON extraction, markdown stripping, year fixes
- `test_weekend.py` — weekend planner output
- `test_content_processing.py` — thinking block removal
- `test_twitter.py` — twitter output cleaning

## Requirements

- **Osaurus or Ollama** server running on port 1337
- **Models** installed (`osaurus pull <model>`)
- **uv** for scripts with extra dependencies (twitter, image_renamer)