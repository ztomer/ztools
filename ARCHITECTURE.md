# ZTools Architecture

## Overview

**ZTools** is a high-performance, native Rust toolkit designed for local LLM workflows on macOS (Apple Silicon) communicating with a local **Osaurus** server (`http://localhost:1337`) or OpenAI-compatible inference servers.

The project began as Python utilities and is now a single native Rust binary (`ztools`) that dispatches on `argv[0]`; `twitter`, `weekend`, `rename_images`, `oeval` and their long forms are symlinks to it, installed by `install.sh` and the Homebrew formula. `bin/ab_test` is the smoke harness. The Python tree was retired 2026-09-13; nothing on the product path runs an interpreter.

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
├── .gatesrc                 # THE gate step list (GOH_CI_STEPS); coverage floor 95
├── .githooks/               # Local quality gates (pre-commit & pre-push)
├── bin/                     # ab_test — smoke + parity harness for the installed binary
├── conf/                    # Shared prompt templates, benchmark config, learned signals
│   ├── config.toml          # Benchmark rankings and model slots ([best_models])
│   ├── prompts.toml         # Canonical LLM prompts across tools
│   ├── weekend.toml         # Excluded venues and default activity seeds
│   ├── eval_inputs.toml, eval_vision.toml, rename.toml, twitter.toml
│   ├── eval_signals.json    # Per-model learned rates (JSON: outside the 500-line cap)
│   └── models/<family>.toml # Per-model caps, resolved from eval_signals
├── docs/                    # Developer documentation and quality specs
│   ├── TESTING.md           # Test patterns, mock infrastructure, coverage rules
│   ├── MODEL_QUIRKS.md      # Observed model quirks and workarounds
│   ├── ROADMAP.md           # Forward-looking backlog: phases L, G, M, R by
│   │                        # item ID; contract gated by tools/tests/test_roadmap.py
│   ├── PORT_PARITY.md       # Parity ledger and benchmark comparisons
│   └── BUGS_twitter_llm_fallback.md, EVALUATION_WORKFLOW.md,
│       LLM_GUIDANCE_AND_TEACHER_EVAL.md, DEEP_MODEL_VS_GEMINI_EVALUATION.md,
│       REPORT_WEAKNESS_CLASSES.md, eval_calibration_2026-07-11.md,
│       eval_baseline.json
├── eval_tasks/data/         # Task corpora (the taxes snapshots)
├── pyproject.toml           # ruff (line-length 100) + pytest: testpaths, the
│                            # sandboxed_server_script marker, unknown marks fatal
├── routines.toml            # Routine/pipeline definitions (+ routines-twitter.toml)
├── tests/fixtures/          # Parity fixtures + the frozen Python verdicts the Rust
│                            # goldens assert: collect_parity, validator_parity,
│                            # weekend_parity
├── tools/                   # Dev gates only; nothing here ships
│   ├── gate.sh              # Declares the toolchains, delegates — holds no gate logic
│   ├── gpu_lock.sh          # The bash half of the machine-wide GPU lock
│   ├── osaurus_one.sh       # Enforce exactly one server, under that lock
│   ├── release.sh           # This repo's release specifics; the kit owns sequencing
│   ├── sweep_models.sh      # The long model sweep
│   ├── rerun_truncated.sh   # Resume a truncated sweep
│   ├── upgrade_tap.py       # Manual/local tap bump (the kit owns the release one)
│   └── tests/               # pytest over the shell tooling; conftest.py guards it
├── rust/                    # Native Rust crate (ztools)
│   ├── Cargo.toml           # THE version source (reqwest, serde, clap, chrono, ...)
│   ├── src/
│   │   ├── main.rs          # Application entry point
│   │   ├── lib.rs
│   │   ├── cli.rs           # Clap CLI definitions
│   │   ├── cli_ztools.rs    # Subcommand dispatch logic
│   │   ├── config.rs        # Dynamic TOML config loader & model fallbacks
│   │   ├── manifest.rs      # Path helpers
│   │   ├── units.rs         # Narrowing conversions that are decisions, not incidents
│   │   └── ztools/          # Tool subsystem implementations
│   │       ├── twitter/     # Browser scraping, cookie stores, embedding clustering
│   │       ├── weekend/     # Weather API, DDG→Bing→Brave search, 4-phase LLM pipeline
│   │       ├── rename/      # OCR sanitization, prompt injection defense, VLM naming
│   │       ├── eval/        # GPU locks, benchmark runners, validators, watchdog
│   │       ├── store.rs     # Store directories and read-side access
│   │       ├── status.rs    # Status for the `routines` harness
│   │       └── embeddings.rs, model_eval.rs, model_health.rs, llm.rs, weekend_cache.rs
│   └── tests/               # Integration suites, incl. gpu_lock_shell_parity.rs
└── vendor/camoufox-rs/      # Third-party crate carried in-tree (excluded from the gate)
```

---

## Subsystem Architecture

### 1. Twitter Timeline Summarizer (`rust/src/ztools/twitter/`)

The Twitter summarizer scrapes your authenticated Following timeline, extracts key insights, deduplicates content across languages and emojis, and formats an executive briefing.

```
 [User Session Discovery]
   (Firefox-family SQLite cookie stores: Zen / Firefox / LibreWolf / Waterfox)
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

- **Session Discovery**: Reads `auth_token` and `ct0` from any **Firefox-family** profile
  under `~/Library/Application Support` — `cookies.rs:89-96` lists `zen`/`Zen`,
  `Firefox`, `LibreWolf`/`librewolf` and `Waterfox`/`waterfox`, in that preference
  order. **Chrome is NOT read**: the port dropped that fallback deliberately
  (`cookies.rs:251-254`) and a Chrome-only user gets a refusal naming
  `twitter --login` (`native.rs:382`), never a silent downgrade. An earlier version
  of this doc claimed Chrome support, which the code has never had.
- **Headless Camoufox Scraping**: Runs an anti-detect Firefox instance to scroll the timeline and intercept live GraphQL tweet batches.
- **Semantic Clustering**: Clusters related tweets using local embeddings before prompt synthesis to ensure high topical coherence.
- **Resilience**: Features character-safe UTF-8 signature trimming, 3-second embedding timeouts, non-blocking stdin handling, and `--use-cache` replay.
- **Heading-guarded output** (`summary_section_for`): the `## …` preamble is only prepended when the model body is non-empty *and* carries no heading of its own. The model opens its own briefings with `## Executive Summary`; an unconditional preamble once produced an empty `## Summary` section above it on every stored page.

---

### 2. Weekend Planner (`rust/src/ztools/weekend/`)

The weekend planner generates family weekend itineraries tailored for kids, combining live weather forecasts, real-time event web scraping, curated GTA venues, and strict rule enforcement.

```
 [Open-Meteo API]              [DuckDuckGo → Bing → Brave search]
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
   • Suitability Gate (suitability.rs: ages from [[children]],
     listing-page titles, fixed venues repeated as events)
                           │
                           ↓
               [Fit Score (score.rs): age share + weather]
                           │
                           ↓
               [Formatted Console & Markdown Table (format.rs)]
```

- **Dual-Source Scraping**: Queries Open-Meteo REST API for precise weekend weather and a three-engine search (DuckDuckGo, then Bing, then Brave — each consulted only when the one before walls or empties the query) for local events. Every engine's verdict per query is recorded and the plan says when a bot wall starved it.
- **4-Phase LLM Pipeline**: Progressively refines raw search text into validated JSON items, falling back to monolithic extraction if any phase stalls.
- **Enforcement Rules**: Drops unsourced rows (anti-hallucination), reconciles day names with ISO dates, enforces user exclusion lists, and drops rows unfit for the family (`suitability.rs`).
- **Window**: Friday to Sunday, extended to a statutory Monday from the `[location] province` rule table (`holidays.rs`); the planner and `ztools status` share `plan_window`.
- **Model**: the configured `weekend_model`, verbatim. A model the server's roster does not list is `ModelHealth::NotInstalled`, named in the plan's degraded banner; there is no family or first-listed substitute.

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
- **Task data**: the task table is `eval/tasks.rs` (pinned row-for-row to the retired Python table by `tasks_tests.rs`); taxes snapshots live in `eval_tasks/data/taxes/`; the validators' agreement with the Python originals is a golden (`rust/tests/validator_parity.rs` over `tests/fixtures/validator_parity/expected_python_verdicts.json`).

---

## Quality Gates & Git Hooks

**One step list is the gate of record** — `GOH_CI_STEPS` in `.gatesrc`, run by
`tools/gate.sh --full`, `make ci`, `tools/release.sh`, the pre-push hook, and
GitHub Actions (`.github/workflows/ci.yml`, macOS on Apple silicon, every push to
`main` and every pull request; it clones gates_of_heck at `main` exactly as the
hooks delegate to it). The step list is in `.gatesrc` and nowhere else:

- **`.githooks/pre-commit`** delegates to `gates_of_heck/gates/structural.sh --staged`
  (emoji gate with the Kare icon set, 500-line cap for tracked files whose suffix is
  one of `.rs .py .swift .c .h .cpp .hpp .cc .m .mm .kt .java .go .ts .tsx .js .jsx
  .sh .bash .rb` — so markdown/JSON/TOML are outside it, conflict markers, shell lint,
  secrets).
- **`.githooks/pre-push`** runs the full step list on the commit being pushed, in a
  clean worktree (`$GOH_DIR/gates/push_gate.sh`). The Rust suite runs once, inside the
  coverage step (every lib, bin and test target, `--all-features`), which applies a
  95% floor and per-file floors (`tools/coverage_floors.jsonc`). Not
  `cargo llvm-cov --fail-under-lines`: the house
  gate takes ONE lcov export PER TEST TARGET and unions them with a CGU-normalising
  merger (`gates_of_heck/gates/coverage_gate.sh:212`), because a plain
  `cargo llvm-cov` emits one record per generic instantiation and reads several
  points lower (95.65% merged vs ~92% plain, 2026-09-14).
