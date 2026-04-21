# ZTools

Run local LLMs to automate real-world tasks — no API keys, no cloud, no privacy concerns.

## What is this?

ZTools is a suite of productivity scripts powered by local LLMs. They run entirely on your machine via the Osaurus server (Ollama-compatible), handling tasks like:

- **Planning your weekend** — finds family-friendly activities based on weather and local events
- **Summarizing your Twitter feed** — distills your timeline into a factual briefing
- **Renaming screenshots** — generates descriptive filenames from OCR or vision models
- **Evaluating models** — tests which local models work best for your use case (NOT a full eval!)

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
python3 model_eval.py --task json        # test specific task
python3 model_eval.py --model gemma-4    # test specific model
```

**Tasks:** `json`, `detailed_json`, `summarize`, `filename`

---

## Configuration

All settings live in `conf/config.yaml`.

```yaml
# Default LLM
llm_url: http://localhost:1337
default_model: qwen3.6-35b-a3b-mxfp4

# Task-specific models
best_models:
  summarize: qwen3.6-35b-a3b-mxfp4
  json: qwen3.6-35b-a3b-mxfp4
  detailed_json: qwen3.6-35b-a3b-mxfp4
  filename: foundation
  vlm: gemma-4-26b-a4b-it-4bit

# Model-specific prompts (for fine-tuning output)
summarize_prompts:
  qwen3.6: "Output the summary. Use ## headers for topics.\n\n<timeline>\n{}\n</timeline>\n\nSummarize the timeline. Include your analysis."
  foundation: "Output the summary. Use ## headers for topics.\n\n<timeline>\n{}\n</timeline>\n\nSummarize the timeline."
```

### Model-Specific Tuning

Different models need different prompts. See `MODEL_QUIRKS.md` for detailed learnings — it covers which models work best for each task and why.

## Architecture

```
lib/
├── osaurus_lib.py      # Server API, retries, fallback logic
├── mlx_lib.py          # Local Apple Silicon MLX models
├── content_processing.py # Clean LLM output (markdown, thinking blocks)
├── validators_lib.py    # Evaluation scoring logic
└── config.py           # Centralized config from config.yaml
```

## Development

```bash
# Install
pip install -e .

# Lint
ruff check .

# Test (run after any code changes to catch regressions)
pytest tests/
pytest tests/ -v          # verbose output
pytest tests/ -k weekend  # run specific test file
```

The test suite covers:
- `test_content_processing.py` — thinking block removal (qwen/gemma)
- `test_weekend.py` — JSON extraction and normalization
- `test_twitter.py` — twitter summarizer output cleaning
- `test_validators.py` — scoring logic validation

## Requirements

- **Osaurus or Ollama** server running on port 1337
- **Models** installed (`osaurus pull <model>`)
- **uv** for scripts with extra dependencies (twitter, image_renamer)
