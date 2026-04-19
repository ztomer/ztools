# ZTools - Local LLM Utilities for Osaurus

ZTools is a collection of pragmatic, real-world Python scripts powered by local Large Language Models (LLMs) via the Osaurus server or local MLX inference. These tools are built to run entirely locally, preserving privacy while automating daily workflows.

## The Tools

### 1. Weekend Planner (`weekend_planner.py`)
Autonomously generates a weekend itinerary for your family.
- **How it works:** Fetches the weekend weather forecast from Open-Meteo and uses DuckDuckGo search to find current local events and venues. It then uses a local LLM to filter these activities based on your family's constraints (e.g., ages, weather logic, previously visited places).
- **Configuration:** Personalize your family's details, location, and excluded places in `conf/weekend.yaml`.
- **Usage:**
  ```bash
  python3 weekend_planner.py
  ```

### 2. Twitter Summarizer (`twitter_summarizer.py`)
Fetches your X/Twitter home timeline via browser automation and summarizes it with a local LLM.
- **How it works:** Uses `playwright` and your local Chrome session cookies (macOS Keychain) to scroll and extract tweets from your home timeline. It then feeds the timeline to an LLM to generate a distilled, factual markdown summary.
- **Usage:**
  ```bash
  uv run twitter_summarizer.py
  ```

### 3. Image Renamer (`image_renamer.py`)
Renames images based on their actual text content or visual description.
- **How it works:** Uses `pytesseract` (OCR) to extract text from images (like screenshots) or local Vision Language Models (VLMs) to describe the image. An LLM then generates a concise, snake_case filename under 50 characters.
- **Usage:**
  ```bash
  uv run image_renamer.py /path/to/images
  ```

### 4. Model Evaluator (`model_eval.py`)
Tests and scores your locally installed models against the *actual real-world prompts* used by the tools above.
- **How it works:** Evaluates models on their ability to strictly follow instructions, generate valid JSON, and extract concise filenames. Includes automated retries for resilience.
- **Usage:**
  ```bash
  python3 model_eval.py
  ```

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
