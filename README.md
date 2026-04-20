# ZTools - Local LLM Utilities for Osaurus

ZTools is a collection of pragmatic, real-world Python scripts powered by local Large Language Models (LLMs) via the Osaurus server or local MLX inference. These tools are built to run entirely locally, preserving privacy while automating daily workflows.

## Prerequisites

1. **Osaurus Server** - Must be running (check at http://localhost:1337)
2. **Python 3.12+** - Most scripts work with `python3`
3. **uv** - Some scripts need uv for dependency management (see below)

## The Tools

### 1. Weekend Planner (`weekend_planner.py`)
Autonomously generates a weekend itinerary for your family.
- **How it works:** Fetches weather from Open-Meteo and uses DuckDuckGo search to find events/venues. Uses LLM to filter based on family constraints.
- **Requires:** `ddgs` package for search
- **Run:**
  ```bash
  python3 weekend_planner.py
  # Or with specific model:
  python3 weekend_planner.py --model qwen3.6-35b-a3b-mxfp4
  ```
- **Options:** `--skip-web` to use cached data, `--model` to specify model

### 2. Twitter Summarizer (`twitter_summarizer.py`)
Fetches your X/Twitter home timeline and summarizes with local LLM.
- **How it works:** Uses playwright + Chrome cookies to scroll timeline. LLM generates distilled summary.
- **Requires:** `uv` for playwright dependencies
- **Run:**
  ```bash
  uv run twitter_summarizer.py
  # Use cached tweets (faster iteration):
  uv run twitter_summarizer.py --use-cache
  # Specify model:
  uv run twitter_summarizer.py --model foundation
  ```
- **Options:** `--use-cache`, `--model`, `--since 24h`

### 3. Image Renamer (`image_renamer.py`)
Renames images based on OCR or visual description.
- **How it works:** Uses pytesseract (OCR) or VLM to describe image, LLM generates filename.
- **Requires:** `uv` for vision dependencies
- **Run:**
  ```bash
  uv run image_renamer.py /path/to/images
  ```

### 4. Model Evaluator (`model_eval.py`)
Tests models against real prompts from the tools above.
- **Run:**
  ```bash
  python3 model_eval.py
  # Quick test single model/task:
  python3 model_eval.py --model qwen3.6 --task json --quick
  ```
- **Options:** `--model`, `--task`, `--quick`, `--debug`

## Configuration

Edit `conf/config.yaml` for model selection and task-specific prompts:

```yaml
llm_url: http://localhost:1337
default_model: qwen3.6-35b-a3b-mxfp4

best_models:
  summarize: qwen3.6-35b-a3b-mxfp4
  json: qwen3.6-35b-a3b-mxfp4
  # ...

# Model-specific prompts (April 2026 learnings)
summarize_prompts:
  qwen3.6: "Output the summary... Include your analysis."
  foundation: "Output the summary..."
```

See `MODEL_QUIRKS.md` for detailed model-specific learnings.

## Architecture & Libraries

All tools share a common, robust library architecture found in `lib/`:
- **`osaurus_lib.py`**: Handles API communication with the Osaurus server, including self-healing automated retries (`max_retries`) and fallback logic.
- **`mlx_lib.py`**: Executes local Apple Silicon MLX models via subprocesses when the server is unavailable or when specialized small models are preferred.
- **`content_processing.py`**: Sanitizes raw LLM outputs by aggressively stripping markdown blocks and `<think>` reasoning tags.
- **`validators_lib.py`**: Contains strict evaluation logic for `model_eval.py`, heavily unit-tested via `pytest`.
- **`logging_config.py`**: Provides comprehensive debug logging.

## Installation & Setup

ZTools requires Python 3.11+. Dependencies are managed via `pyproject.toml` or dynamically injected using `uv run`.

1. Ensure the Osaurus server is running (`localhost:1337`) or you have local MLX models downloaded.
2. Install project dependencies:
   ```bash
   pip install -e .
   # or use uv for standalone scripts:
   uv run <script_name.py>
   ```

## Development & Testing

ZTools adheres to strict code quality standards:
- **Linting:** 100% compliant with `ruff check .`
- **Testing:** 100% unit test coverage for validators (`pytest tests/`)
