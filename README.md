# ZTools

High-performance native Rust local LLM tools for [Osaurus](https://osaurus.ai/).

ZTools provides standalone command-line tools for Twitter timeline summarization, family weekend planning, image renaming, and model evaluation benchmarks—running entirely locally on Apple Silicon.

Detailed architectural deep-dive is available in [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Prerequisites

| Requirement | Notes |
|------------|-------|
| **[Osaurus](https://osaurus.ai/)** server | **Hard runtime dependency.** ZTools communicates with an Osaurus (or OpenAI-compatible) server at `http://localhost:1337`. Start it via `osaurus serve &` or the Osaurus macOS menu app. |
| **Rust toolchain** | Required when building from source (`cargo` / `rustc`). |

---

## Installation (Homebrew)

```bash
# 1. Start Osaurus server (macOS 15+, Apple Silicon)
brew install --cask osaurus
osaurus serve &>/dev/null &

# 2. Install ZTools via Homebrew tap
brew tap ztomer/tap
brew install ztomer/tap/ztools
```

Installs native Rust binaries directly onto your `PATH`:

| Command | Alias | Description |
|---------|-------|-------------|
| `twitter` | `twitter-summarize` | Scrapes following timeline & generates executive summary |
| `weekend` | `weekend-plan` | Curates family weekend plans with weather & seasonal events |
| `rename_images` | `image-renamer` | Renames screenshots/photos using OCR and Vision LLMs |
| `oeval` | `model-eval` | Benchmarks local LLM performance & accuracy across 30 tasks |
| `ab_test` | — | Performance and parity benchmark comparing Rust vs Python references |
| `ztools` | — | Unified native binary dispatcher for all subcommands |

---

## Quick Start (from a checkout)

Build the release binaries:

```bash
./build.sh
```

Run any tool directly from `./bin/`:

```bash
# Weekend planner
./bin/weekend

# Twitter summarizer (live browser scraping + GPU summarization)
./bin/twitter

# Replay previous Twitter scrape from cache
./bin/twitter --use-cache

# Image renamer
./bin/rename_images ~/Desktop/screenshots

# Model evaluation benchmark
./bin/oeval
```

---

## The Tools

### 1. Twitter Summarizer (`twitter`)

```bash
twitter                  # Scrape live timeline & summarize
twitter --use-cache      # Summarize cached tweets from last run
twitter --since 24h      # Fetch tweets from the last 24 hours
twitter --fetch-only     # Collect and cache tweets without summarizing
twitter --login          # Open browser window to sign in to x.com
twitter --debug          # Show browser window and verbose output
```

- **Session Discovery**: Automatically extracts authenticated `auth_token` and `ct0` cookies from Zen Browser, Firefox, LibreWolf, or Chrome.
- **Anti-Detect Headless Scraping**: Launches Camoufox to scroll the Following timeline and capture live GraphQL tweets.
- **Semantic Clustering**: Clusters related tweets via local embeddings before prompt synthesis to create structured topic categories.

### 2. Weekend Planner (`weekend`)

```bash
weekend
weekend --location "Vaughan/Toronto" --ages "13,10,6"
weekend --md-out ~/Documents/weekend_plan.md
```

- **Weather-Aware**: Fetches 3-day forecasts from Open-Meteo REST API.
- **Dual-Source Scraping**: Scrapes seasonal festivals and activities from DuckDuckGo (with automatic fallback to DuckDuckGo Lite).
- **4-Phase LLM Pipeline**: Condenses weather → Extracts candidate snippets → Drafts itinerary → Structures validated JSON.
- **Rule Enforcement**: Drops unsourced rows, ensures in-window dates, and applies venue exclusions from `conf/weekend.toml`.

### 3. Image Renamer (`rename_images`)

```bash
rename_images ~/Desktop/screenshots
rename_images /path/to/photos --dry-run
rename_images /path/to/photos --vlm-model qwen3.8-27b-8bit
```

- **Prompt Injection Defense**: Wraps OCR text in `<<<BEGIN_UNTRUSTED_DOCUMENT` delimiters to prevent prompt hijacking.
- **Vision Fallback**: Uses OpenAI-compatible base64 data URIs for images lacking OCR text.
- **Sanitization**: Generates clean, descriptive snake_case filenames.

### 4. Model Evaluator (`oeval`)

```bash
oeval                    # Run full model benchmark
oeval --model qwen3.8-27b-8bit
```

- Evaluates models against 30 automated task suites (financial analysis, adversarial resistance, JSON extraction, entity grounding).
- Uses kernel-level GPU locking (`gpu_lock`) to synchronize access across concurrent tasks.

---

## Testing & Quality Gates

Run the full native Rust test suite (186 unit tests, 8 integration tests, 5 model eval tests, 2 HTTP mock tests):

```bash
cd rust
cargo test
cargo clippy --all-targets -- -D warnings
```

Git pre-commit and pre-push hooks (`.githooks/`) enforce quality gates automatically:
- **Pre-commit**: Runs Emoji gate, file size gate, Ruff linting, Python syntax check, and Rust Clippy + Tests on staged files.
- **Pre-push**: Runs complete Python test suite with 95% coverage requirement and full Rust test suite.

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