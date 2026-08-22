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

**Tasks:** `weekend_transient`, `weekend_fixed`, `summarize`, `filename`, `file_summary`, `taxes_anomalies`, `taxes_audit_readiness`, `taxes_synthesis`, `taxes_yoy_narrative`, `taxes_qa`, `taxes_slip_qa`, plus the mixed-signal, adversarial and vision variants — 30 in total; oeval prints the count it loaded.

- Evaluates models against 30 automated task suites (financial analysis, adversarial resistance, JSON extraction, entity grounding).
- Uses kernel-level GPU locking (`gpu_lock`) to synchronize access across concurrent tasks.

---

## Testing & Quality Gates

The production implementation is Rust; the Python `references/` tree is kept solely as the
A/B parity layer. Both stacks are gated.

### Rust (production)

```bash
cd rust
cargo test                          # 464 tests: 403 unit + 61 integration
cargo clippy --all-targets -- -D warnings
cargo llvm-cov --summary-only       # coverage, floor 94% lines
```

Key suites:
- `tests/model_resolve_http.rs` — missing-model substitution over the wire (dead tag → roster → one retry, surfaced reason, quirk re-derivation)
- `tests/reasoning_retry.rs` — a REASONING overrun retries with a raised budget against a mock that only answers once it is raised
- `tests/transport_http.rs`, `tests/signals_prefill.rs` — wire format, stream guard, learned timeouts, capability recording
- `tests/validator_parity.rs` — prints fixture verdicts from the RUST validators; `references/tests/test_rust_validator_parity.py` asserts they match the PYTHON validators byte-for-byte

**Coverage**: enforced on every push by `.githooks/pre-push` (`cargo llvm-cov --fail-under-lines 94`; current ~94.8%). The residual uncovered code is live-process spawning (`login_live`, `collect_tweets_live` — real Camoufox browser), environment-absent branches, and assertion panic-format arms. The floor may only move up; re-baselining requires a stated reason in the diff.

### Python (parity reference)

```bash
OLLAMA_BASE_URL=http://127.0.0.1:1 MLX_MODELS_DIR=/tmp/nonexistent \
  .venv/bin/pytest --cov --cov-fail-under=95 .
```

2,791 tests at 95%+ coverage. The suite structurally forbids launching real browsers or reading real cookie stores.

### Git hooks (`.githooks/`)
- **Pre-commit**: Emoji gate, file size gate, `#[allow]` ban, Ruff linting, Rust Clippy + tests.
- **Pre-push**: full Python suite with the 95% coverage floor, full Rust suite, and the Rust llvm-cov 94% line floor.

GitHub Actions CI is disabled — the local pre-push gate runs a strict superset of what it checked. `release.yml` was removed with it: the Homebrew formula update lives in `tools/release.sh` (below).

Eval results live in `~/.config/ztools/` (`eval_results.csv`, `eval_history.json`, `eval_signals.json`, and the raw-answer archive under `outputs/`). To track:

```bash
git add -f ~/.config/ztools/eval_results.json ~/.config/ztools/eval_history.json
```

---

## Release

```bash
tools/release.sh            # bump patch from the latest tag (v2.1.7 -> v2.1.8)
tools/release.sh 2.2.0      # explicit version
```

The script syncs BOTH manifests (`pyproject.toml` + `rust/Cargo.toml`, so the binary and the sdist never disagree), tags HEAD, pushes, computes the GitHub tarball's SHA256, and updates the Homebrew tap formula in `ztomer/homebrew-tap`. Requires `gh` authenticated.