# ZTools Architecture

## Overview

**ZTools** is a high-performance, native Rust toolkit designed for local LLM workflows on macOS (Apple Silicon) communicating with a local **Osaurus** server (`http://localhost:1337`) or OpenAI-compatible inference servers.

The project began as Python utilities and has been completely ported to a unified native Rust binary (`ztools`) with thin launcher shims (`twitter`, `weekend`, `rename_images`, `oeval`, `ab_test`).

```
                              ┌──────────────────────────────────┐
                              │            ztools CLI            │
                              │ (Clap Parser in rust/src/cli.rs) │
                              └────────────────┬─────────────────┘
                                               │
             ┌──────────────────┬──────────────┴─────┬──────────────────┐
             ▼                  ▼                    ▼                  ▼
    ┌─────────────────┐ ┌───────────────┐ ┌──────────────────┐ ┌───────────────┐
    │     Twitter     │ │    Weekend    │ │  Image Renamer   │ │  Model Eval   │
    │   Summarizer    │ │    Planner    │ │ (OCR / Vision)   │ │  (Benchmark)  │
    └────────┬────────┘ └───────┬───────┘ └────────┬─────────┘ └───────┬───────┘
             │                  │                  │                   │
             └──────────────────┼──────────────────┴───────────────────┘
                                ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                   Shared Core Infrastructure                     │
    │  • Config & Model Hierarchy (rust/src/config.rs)                 │
    │  • Semantic Embeddings & Clustering (rust/src/ztools/embeddings) │
    │  • GPU Concurrency Locking (rust/src/ztools/eval/gpu_lock.rs)    │
    │  • Model Health & Shard Inspection (model_health.rs)             │
    │  • Osaurus HTTP Client (/v1/chat/completions, /v1/embeddings)    │
    └──────────────────────────────────┬───────────────────────────────┘
                                       ▼
    ┌──────────────────────────────────────────────────────────────────┐
    │                      Osaurus Inference Server                    │
    │             http://localhost:1337 (Apple Silicon / MLX)          │
    └──────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
ztools/
├── bin/                    # Standalone executable launchers (twitter, weekend, etc.)
├── conf/                   # Shared prompt templates and benchmark configs
│   ├── config.toml         # Benchmark rankings and model slots ([best_models])
│   ├── prompts.toml        # Canonical LLM prompts across tools
│   └── weekend.toml        # Excluded venues and default activity seeds
├── docs/                   # Developer documentation and quality specs
│   ├── PORT_PARITY.md      # Parity ledger and benchmark comparisons
│   ├── RUST_PORT_PLAN.md   # Architectural roadmap and port milestones
│   └── MODEL_QUIRKS.md     # Observed model quirks and workarounds
├── references/             # Historical Python implementations used for A/B testing
├── rust/                   # Native Rust crate (ztools)
│   ├── Cargo.toml          # Rust dependencies (reqwest, serde, clap, chrono, etc.)
│   └── src/
│       ├── main.rs         # Application entry point
│       ├── cli.rs          # Clap CLI definitions
│       ├── cli_ztools.rs   # Subcommand dispatch logic
│       ├── config.rs       # Dynamic TOML config loader & model fallbacks
│       └── ztools/         # Tool subsystem implementations
│           ├── twitter/    # Browser scraping, embedding clustering, summarization
│           ├── weekend/    # Weather API, DuckDuckGo scraper, 4-phase LLM pipeline
│           ├── rename/     # OCR sanitization, prompt injection defense, VLM naming
│           ├── eval/       # GPU locks, benchmark runners, validation suites
│           ├── embeddings.rs # Semantic embedding vector calculations
│           ├── model_eval.rs # Quality evaluation test cases
│           └── weekend_cache.rs # Exclusion rules and cached GTA venues
└── .githooks/              # Local quality gates (pre-commit & pre-push)
```

---

## Subsystem Architecture

### 1. Twitter Timeline Summarizer (`rust/src/ztools/twitter/`)

The Twitter summarizer scrapes your authenticated Following timeline, extracts key insights, deduplicates content across languages and emojis, and formats an executive briefing.

```
 [User Session Discovery]
   (Zen / Firefox / Chrome SQLite Cookie Stores)
             │
             ▼
 [Live Browser Collector (browser.rs)]
   (Headless Camoufox / Playwright -> GraphQL Interception)
             │
             ▼
 [Deduplication & Character-Safe Truncation]
   (Unicode-safe char iterators, normalize signatures)
             │
             ▼
 [Semantic Clustering (embeddings.rs)]
   (Cosine similarity over /v1/embeddings -> Topic Groups)
             │
             ▼
 [Prompt Construction & GPU Inference]
   (conf/prompts.toml -> Osaurus /v1/chat/completions)
             │
             ▼
 [Markdown Output]
   (~/Documents/twitter_summaries/YYYY-MM-DD_HHMM_summary.md)
```

- **Session Discovery**: Automatically extracts active `auth_token` and `ct0` cookies from Zen Browser, Firefox, LibreWolf, or Google Chrome.
- **Headless Camoufox Scraping**: Runs an anti-detect Firefox instance to scroll the timeline and intercept live GraphQL tweet batches.
- **Semantic Clustering**: Clusters related tweets using local embeddings before prompt synthesis to ensure high topical coherence.
- **Resilience**: Features character-safe UTF-8 signature trimming, 3-second embedding timeouts, non-blocking stdin handling, and `--use-cache` replay.

---

### 2. Weekend Planner (`rust/src/ztools/weekend/`)

The weekend planner generates family weekend itineraries tailored for kids, combining live weather forecasts, real-time event web scraping, curated GTA venues, and strict rule enforcement.

```
 [Open-Meteo API]              [DuckDuckGo HTML + Lite Scraper]
 (Weather Forecast)            (Seasonal Events & Festivals)
         │                                    │
         └─────────────────┬──────────────────┘
                           ▼
               [4-Phase LLM Pipeline]
   Phase 1: Weather Condensation (phases.rs)
   Phase 2: Source Extraction & Filtering (phases.rs)
   Phase 3: Activity Drafting (phases.rs)
   Phase 4: Structured JSON Synthesis (phases.rs)
                           │
                           ▼
               [Enforcement & Validation (enforce.rs)]
   • Recency Gate (drops events outside window)
   • Exclusion Gate (filters venues from conf/weekend.toml)
   • Region Evidence Gate (positive GTA token matching)
   • Weather Consistency Gate (Indoor/Outdoor label checks)
                           │
                           ▼
               [Formatted Console & Markdown Table (format.rs)]
```

- **Dual-Source Scraping**: Queries Open-Meteo REST API for precise weekend weather and DuckDuckGo (with automatic fallback to DuckDuckGo Lite) for local events.
- **4-Phase LLM Pipeline**: Progressively refines raw search text into validated JSON items, falling back to monolithic extraction if any phase stalls.
- **Enforcement Rules**: Drops unsourced rows (anti-hallucination), reconciles day names with ISO dates, and enforces user exclusion lists.

---

### 3. Image Renamer (`rust/src/ztools/rename/`)

Sanitizes and generates descriptive snake_case filenames for screenshots, photos, and documents using local OCR and Vision-Language Models (VLM).

- **Prompt Injection Defense**: All OCR-extracted text is wrapped inside `<<<BEGIN_UNTRUSTED_DOCUMENT` delimiters to prevent adversarial text inside images from hijacking LLM instructions.
- **Vision Model Fallback**: If an image contains no readable text, sends an OpenAI-compatible base64 data URI to the configured VLM (`qwen3.8-27b-8bit`).
- **Sanitization Heuristics**: Strips code fences, prefixes, file extensions, and limits names to 6 words / 50 characters.

---

### 4. Model Evaluator & GPU Lock (`rust/src/ztools/eval/`)

Automates regression testing and leaderboard scoring of local LLMs against 30 challenging task suites (tax analysis, synthesis, adversarial resistance, JSON formatting).

- **GPU Concurrency Lock (`gpu_lock.rs`)**: Uses kernel-level file locking (`flock`) on `/tmp/osaurus_gpu.lock` to ensure only one process uses the GPU during benchmarks, automatically reclaiming dead locks.
- **Model Health Probe (`model_health.rs`)**: Inspects model directory shards offline before loading, detecting broken MTP speculative drafting weights, missing `.safetensors` parts, and corrupt downloads.

---

## Quality Gates & Git Hooks

Local verification is enforced before code reaches GitHub CI:

- **`.githooks/pre-commit`**:
  - Emoji gate (permits only Kare icons: `→ ✓ ✗ ⚠ ↔ ↑ ↓`).
  - Python linting (`ruff`) and syntax check.
  - Rust Clippy (`cargo clippy --all-targets -- -D warnings`).
  - Rust unit and integration test suite (`cargo test`).
- **`.githooks/pre-push`**:
  - Full Python test suite with 95% coverage requirement.
  - Full Rust test suite (186 unit tests, 8 integration tests, 5 model eval tests, 2 HTTP mock tests).
