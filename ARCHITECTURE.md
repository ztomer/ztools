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
             ↓                  ↓                    ↓                  ↓
    ┌─────────────────┐ ┌───────────────┐ ┌──────────────────┐ ┌───────────────┐
    │     Twitter     │ │    Weekend    │ │  Image Renamer   │ │  Model Eval   │
    │   Summarizer    │ │    Planner    │ │ (OCR / Vision)   │ │  (Benchmark)  │
    └────────┬────────┘ └───────┬───────┘ └────────┬─────────┘ └───────┬───────┘
             │                  │                  │                   │
             └──────────────────┼──────────────────┴───────────────────┘
                                ↓
    ┌──────────────────────────────────────────────────────────────────┐
    │                   Shared Core Infrastructure                     │
    │  • Config & Model Hierarchy (rust/src/config.rs)                 │
    │  • Semantic Embeddings & Clustering (rust/src/ztools/embeddings) │
    │  • GPU Concurrency Locking (rust/src/ztools/eval/gpu_lock.rs)    │
    │  • Model Health & Shard Inspection (model_health.rs)             │
    │  • Osaurus HTTP Client (/v1/chat/completions, /v1/embeddings)    │
    └──────────────────────────────────┬───────────────────────────────┘
                                       ↓
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
│   ├── ROADMAP.md          # Forward-looking backlog (port complete; open items: none)
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
             ↓
 [Live Browser Collector (browser.rs)]
   (Headless Camoufox / Playwright -> GraphQL Interception)
             │
             ↓
 [Deduplication & Character-Safe Truncation]
   (Unicode-safe char iterators, normalize signatures)
             │
             ↓
 [Semantic Clustering (embeddings.rs)]
   (Cosine similarity over /v1/embeddings -> Topic Groups)
             │
             ↓
 [Prompt Construction & GPU Inference]
   (conf/prompts.toml -> Osaurus /v1/chat/completions)
             │
             ↓
 [Markdown Output]
   (~/Documents/twitter_summaries/YYYY-MM-DD_HHMM_summary.md)
```

- **Session Discovery**: Automatically extracts active `auth_token` and `ct0` cookies from Zen Browser, Firefox, LibreWolf, or Google Chrome.
- **Headless Camoufox Scraping**: Runs an anti-detect Firefox instance to scroll the timeline and intercept live GraphQL tweet batches.
- **Semantic Clustering**: Clusters related tweets using local embeddings before prompt synthesis to ensure high topical coherence.
- **Resilience**: Features character-safe UTF-8 signature trimming, 3-second embedding timeouts, non-blocking stdin handling, and `--use-cache` replay.
- **Heading-guarded output** (`summary_section_for`): the `## …` preamble is only prepended when the model body is non-empty *and* carries no heading of its own. The model opens its own briefings with `## Executive Summary`; an unconditional preamble once produced an empty `## Summary` section above it on every stored page.

---

### 2. Weekend Planner (`rust/src/ztools/weekend/`)

The weekend planner generates family weekend itineraries tailored for kids, combining live weather forecasts, real-time event web scraping, curated GTA venues, and strict rule enforcement.

```
 [Open-Meteo API]              [DuckDuckGo HTML + Lite Scraper]
 (Weather Forecast)            (Seasonal Events & Festivals)
         │                                    │
         └─────────────────┬──────────────────┘
                           ↓
               [4-Phase LLM Pipeline]
   Phase 1: Weather Condensation (phases.rs)
   Phase 2: Source Extraction & Filtering (phases.rs)
   Phase 3: Activity Drafting (phases.rs)
   Phase 4: Structured JSON Synthesis (phases.rs)
                           │
                           ↓
               [Enforcement & Validation (enforce.rs)]
   • Recency Gate (drops events outside window)
   • Exclusion Gate (filters venues from conf/weekend.toml)
   • Region Evidence Gate (positive GTA token matching)
   • Weather Consistency Gate (Indoor/Outdoor label checks)
                           │
                           ↓
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

- **GPU Concurrency Lock (`gpu_lock.rs`)**: machine-wide mutual exclusion at `/tmp/mac-osaurus-gpu.lock`, acquired by atomic `mkdir` (macOS ships no `flock(1)`). Dead-owner locks are reclaimed via PID + process start time (a recycled PID cannot impersonate its predecessor); the wedge ceiling measures PROGRESS through heartbeats, so an honest multi-hour sweep never loses the lock while a hung one does. A waiter that cannot get the lock fails and names the holder.
- **Transport pipeline (`transport.rs`)**: mirrors Python's request path exactly — model quirks applied inside the call (a substituted model re-derives them), streamed attempt under a reasoning-overrun guard, blocking fallback, and missing-model substitution (`model_resolve.rs`): on an HTTP 404 naming a dead tag, the roster is fetched (disk-corroborated), a stand-in is retried ONCE, and the substitution is surfaced in the result.
- **Runner (`runner.rs`)**: per-task loop with failure classification (`failures.rs`: INFRA / TIMEOUT / PARSE / FORMAT / CONTENT / REASONING), retry-token escalation for reasoning overruns, infra abandonment, learned per-task timeouts and p95 signal recording behind `run_eval_with_signals`.
- **Oversize refusal (`oversize.rs`)**: a model whose weights exceed 80% of reclaimable memory — or a machine already paging — is refused before measuring (`EVAL_ALLOW_OVERSIZE=1` overrides deliberately).
- **Config-resolved budgets & timeouts (`budgets.rs`, `signals.rs`)**: `[max_tokens]` / `[timeouts]` tables from `conf/config.toml`; per-model caps from `conf/models/<family>.toml`, family resolved from the architecture recorded in eval_signals before falling back to name matching.
- **Model Health Probe (`model_health.rs`)**: inspects model directory shards offline before loading, detecting broken MTP speculative drafting weights, missing `.safetensors` parts, and corrupt downloads.
- **Task data**: canonical snapshots live in `eval_tasks/data/taxes/` and are shared byte-for-byte with the Python reference; validator agreement is enforced every push by the CI parity gate (`rust/tests/validator_parity.rs` + `references/tests/test_rust_validator_parity.py`).

---

## Quality Gates & Git Hooks

Local verification is enforced before code reaches GitHub CI:

- **`.githooks/pre-commit`**:
  - Emoji gate (permits only Kare icons: `→ ✓ ✗ ⚠ ↔ ↑ ↓`).
  - File size gate — 500 lines, **Python and Rust alike, no exemption for tests**
    (`tools/check_file_size.py`). Python-only until 2026-08-23, which is how
    `json_validator.rs` reached 1126 lines under a green hook; a test-file exemption
    existed briefly after that and was removed 2026-08-24 — an oversized test file is
    split the same as production (sibling `test_*.py` files, or a Rust `#[path=...] mod`).
  - `#[allow]` ban across Rust source (a suppression is a defect, not a configuration).
  - Python linting (`ruff`) and syntax check.
  - Rust Clippy (`cargo clippy --all-targets -- -D warnings`) and test suite.
- **`.githooks/pre-push`**:
  - Full Python parity suite with a 95% coverage floor (`pytest --cov-fail-under=95`).
  - Full Rust test suite (403 unit + 61 integration tests).
  - Rust coverage floor: `cargo llvm-cov --fail-under-lines 94` (~94.8% current; the residual is live-browser process spawning and env-absent branches, itemized in the coverage report).
