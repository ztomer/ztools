# Architecture & Implementation Roadmap — `ztools` & `routines`

_The consolidated engineering roadmap for porting the Python reference implementation in `~/Projects/ztools` to the native Rust static binary in `~/Projects/routines`, synchronizing the 30-task model evaluation matrix, enforcing strict quality gates, and conducting deep behavioral A/B testing._

---

## 1. Overview & Architectural Goals

- **Single Static Binary Goal**: Escape the Python `venv`/`uv` startup latency and runtime dependency by shipping all `ztools` capabilities (`tw`, `wk`, `rn`, `ev`) inside the native Rust binary `routines`.
- **Zero-Drift & Behavioral Parity**: Eliminate parallel pipeline divergences by aligning prompts, sanitizers, timestamp formats, and model matrices across Python and Rust, gated by automated A/B tests.
- **Strict Quality Standard**:
  - **Python**: `ruff check .`, `ruff format --check .`, `pytest --cov --cov-fail-under=95 .`.
  - **Rust**: `cargo clippy --all-targets --all-features -- -D warnings`, `tools/check_no_allow.py` (zero `#[allow]` attributes permitted), `cargo fmt --all -- --check`, `tools/check_no_emoji.py` (Kare glyphs only `→ ✓ ✗ ⚠ ↔ ↑ ↓`), `tools/check_file_length.py` (≤ 400 lines/file), and `cargo llvm-cov --fail-under-lines 95`.

---

## 2. 30-Task Benchmark Leaderboard & Best Model Matrix

Synchronized across `conf/config.toml` (`[best_models]`) and `routines/src/config_ztools.rs`:

| Consumer Slot | Assigned Best Model | Quality Score | Latency | Key Strengths & Justification |
|---|---|---|---|---|
| **`json`** (Weekend & JSON Suite) | **`qwen3.8-27b-8bit`** | **100%** | ~40s | 100% on all 7 weekend/json tasks (`weekend_transient`, `weekend_fixed`, `weekend_transient_mixed`, `weekend_fixed_mixed`, `weekend_transient_schema`, `json`, `detailed_json`). Zero schema parse failures. |
| **`filename`** (Image Renaming) | **`gemma-4-e2b-it-8bit`** | **100%** | **0.2s** | 100% filename quality + 100% on `filename_injection` (resists adversarial prompt injections in OCR text). Ultra-fast interactive latency. |
| **`summarize`** (Twitter & Reports) | **`gemma-4-e2b-it-8bit`** | **89.5%** group | **35s** | Top adversarial robustness: 100% on `summarize_contradiction`, 67% on `summarize_factual_accuracy` (beats 12B/27B models that parrot planted falsehoods). |
| **`think`** (Structured Fallback & Taxes) | **`ornith-1.0-35b-jang_4m`** / **`qwen3.8-27b-8bit`** | **100%** / **88%** | ~60s | 100% on `file_summary`, `taxes_qa`, `taxes_slip_qa`; 84% on `taxes_synthesis`. |
| **`vlm`** (Vision Renamer) | **`qwen3.8-27b-8bit`** | **100%** | ~8s | 100% on `image_real`, `image_rename`, and `image_rename_mixed`. Clean 8-bit quantization with high decode throughput (17 tok/s). |

---

## 3. Subsystem Implementation & Porting Phases

### Phase 1: Security, Packaging Health & Configuration
1. **Broken Model & Packaging Defect Detection (`src/ztools/model_health.rs`)**:
   - Offline inspection of `~/MLXModels/<org>/<model>` and HF cache.
   - Detect unsupported MTP (Multi-Token Prediction) shards (`*mtp*.safetensors`) when `runtime_available = false`.
   - Parse `model.safetensors.index.json` to verify all weight shards exist on disk.
   - Detect incomplete download artifacts (`*.incomplete`, `*.lock`).
   - Decode thrashing guard: flag and refuse models decoding under `THRASHING_DECODE_TOKENS_PER_SEC` (1.0 tok/s).
2. **Untrusted Content Framing (`src/ztools/image_renamer.rs`)**:
   - Wrap OCR image text in `<untrusted_content>...</untrusted_content>` tags.
   - Instruct LLM to strictly ignore instructions embedded within untrusted tags (`filename_injection` defense).
   - Sanitize conversational artifacts (`"Here is the filename:"`), code fences, and file extensions.
3. **Model Matrix Configuration (`src/config_ztools.rs`)**:
   - Update fallback defaults to match the 30-task benchmark winners.
   - Support optional dynamic loading from `conf/config.toml` `[best_models]`.

### Phase 2: Prompts, Timestamps & Schema Parity
1. **Twitter Summarizer Parity (`src/ztools/twitter.rs`)**:
   - Synchronize prompt instructions with `references/eval/tasks_prompts.py: TWITTER_PROMPT`.
   - Format tweet timestamps as `%b %d %H:%M` in prompt payloads to prevent date dropping at the LLM boundary (C2a fix).
2. **Weekend Planner Schemas & Exclusions (`src/ztools/weekend/`)**:
   - Align event JSON schema to include `start_date`, `end_date`, `price`, `day`, `weather` (C2b fix).
   - Match candidate events against exclusion patterns from `conf/weekend.toml` (C8 fix).

### Phase 3: Deep Behavioral A/B Testing & Gate Verification
1. **Automated A/B Parity Suite (`bin/ab_test`)**:
   - **CLI Surface Parity**: Verify subcommands, help strings, and exit codes.
   - **Defect Probing Parity**: Pass mock model fixtures (clean, broken MTP, missing index shards, incomplete downloads) to both Python and Rust, asserting identical diagnostic verdicts.
   - **Security Parity**: Pass adversarial injection prompts to both engines, asserting identical sanitized filenames.
   - **Date & Payload Parity**: Verify tweet timestamps and weekend event dates match across both engines.
2. **Continuous Quality Gate Verification**:
   - Run `rust_gate.sh` (`cargo clippy -- -D warnings`, `cargo fmt --check`, `check_no_allow.py`).
   - Run `make gate` (`check_no_emoji.py`, `check_file_length.py`, `check_status_symbol_contract.py`).
   - Enforce ≥ 95% line coverage in both `routines` (`cargo llvm-cov`) and `ztools` (`pytest --cov`).
