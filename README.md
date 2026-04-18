# ZTools - Local LLM Tools

Local LLM utilities for Osaurus server (localhost:1337).

## Location

`~/Projects/ztools/`

## Quick Start

```bash
# Weekend planner - generate family activities
weekend

# Twitter/X summarizer - fetch timeline and summarize  
twitter

# Rename images by OCR text content
rename_images /path/to/images

# Model evaluator - test models on standard tasks
oeval
```

## Environment

Requires Osaurus server running on `localhost:1337`.

Models are configured in `osaurus_lib.py`:
- `BEST_MODELS` - recommended per task type
- `TIMEOUTS` - timeout seconds per task

## Dependencies

Each command uses `uv run --with` for on-demand deps:

| Alias | Dependencies |
|-------|-------------|
| weekend | ddgs, beautifulsoup4, rich |
| twitter | playwright, cryptography, requests, rich |
| rename_images | pillow, pytesseract, requests |
| oeval | rich, requests |

## Library Usage

```python
from osaurus_lib import call, call_with_prompt, get_best_model

# Simple call
result = call_with_prompt("gemma-4-26b-a4b-it-4bit", "Hello", "think")

# With JSON extraction
result = call_with_prompt("gemma-4-26b-a4b-it-4bit", "List 3 items", "json")
print(result["parsed"])      # parsed JSON
print(result["quality_score"])  # 0-100 score

# Check server
from osaurus_lib import is_server_running, get_models
print(get_models())  # available models
```

## Files

| File | Purpose |
|------|----------|
| osaurus_lib.py | Generic LLM utilities |
| model_eval.py | Model evaluator |
| generate_weekend.py | Weekend planner |
| twitter_summary.py | Twitter summarizer |
| rename_images_by_content.py | Image renamer |

## Configuration (YAML)

Each tool has a config file. Extract to override defaults:

| Config | Purpose |
|--------|----------|
| config.yaml | Shared config (llm_url, default_model) |
| weekend.yaml | Family info (children ages, visited places) |
| twitter.yaml | Twitter settings (output_dir, max_scrolls) |
| rename.yaml | OCR/VLM paths and model preferences |