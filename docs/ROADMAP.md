# Architecture & Implementation Roadmap — `ztools`

_Behavioral A/B testing, quality gates, and the parity roadmap for the native Rust ztools binary._

---

## 1. Overview & Architectural Goals

- **Single Static Binary Goal**: Escape the Python `venv`/`uv` startup latency and runtime dependency by shipping all ztools capabilities (`twitter-summarize`, `weekend-plan`, `image-renamer`, `model-eval`) inside the native Rust binary `ztools` in this repo.
- **Zero-Drift & Behavioral Parity**: Eliminate parallel pipeline divergences by aligning prompts, sanitizers, timestamp formats, and model matrices across both implementations, gated by automated A/B tests using `bin/ab_test`.
- **Strict Quality Standard**:
  - **Rust**: `cargo clippy --all-targets --all-features -- -D warnings`, `tools/check_no_allow.py` (zero `#[allow]` attributes permitted), `cargo fmt --all -- --check`, `tools/check_no_emoji.py` (Kare glyphs only `→ ✓ ✗ ⚠ ↔ ↑ ↓`), `tools/check_file_length.py` (≤ 400 lines/file), and `cargo llvm-cov --fail-under-lines 95`.
  - **Python reference** (`references/`): preserved for A/B parity verification only; production runs execute Rust code.

---

## 2. 30-Task Benchmark Leaderboard & Best Model Matrix

Synchronized across `conf/config.toml` (`[best_models]`) and read dynamically by the Rust binary via `with_ztools_best_models()` and `with_shared_prompts()` on startup:

| Consumer Slot | Assigned Best Model | Quality Score | Latency | Key Strengths & Justification |
|---|---|---|---|---|
| **`json`** (Weekend & JSON Suite) | **`qwen3.8-27b-8bit`** | **100%** | ~40s | 100% on all 7 weekend/json tasks (`weekend_transient`, `weekend_fixed`, `weekend_transient_mixed`, `weekend_fixed_mixed`, `weekend_transient_schema`, `json`, `detailed_json`). Zero schema parse failures. |
| **`filename`** (Image Renaming) | **`gemma-4-e2b-it-8bit`** | **100%** | **0.2s** | 100% filename quality + 100% on `filename_injection` (resists adversarial prompt injections in OCR text). Ultra-fast interactive latency. |
| **`summarize`** (Twitter & Reports) | **`gemma-4-e2b-it-8bit`** | **89.5%** group | **35s** | Top adversarial robustness: 100% on `summarize_contradiction`, 67% on `summarize_factual_accuracy` (beats 12B/27B models that parrot planted falsehoods). |
| **`think`** (Structured Fallback & Taxes) | **`ornith-1.0-35b-jang_4m`** / **`qwen3.8-27b-8bit`** | **100%** / **88%** | ~60s | 100% on `file_summary`, `taxes_qa`, `taxes_slip_qa`; 84% on `taxes_synthesis`. |
| **`vlm`** (Vision Renamer) | **`qwen3.8-27b-8bit`** | **100%** | ~8s | 100% on `image_real`, `image_rename`, and `image_rename_mixed`. Clean 8-bit quantization with high decode throughput (17 tok/s). |

---

## 3. Subsystem Implementation & Parity Status

All 10 roadmap items are completed and verified through A/B testing with the Python reference:

- [x] **1. Broken Model & Packaging Defect Detection** — Ported to `rust/src/ztools/model_health.rs`.
- [x] **2. Best Model Matrix & Dynamic Configuration** — Synchronized with 30-task benchmark winners. `with_ztools_best_models()` dynamic loader from `~/.config/ztools/config.toml` or `conf/config.toml`.
- [x] **3. Image Renamer Security & Untrusted Framing** — Ported to `rust/src/ztools/rename/`. `clean_filename`, `is_meaningful_text`, `is_non_human_readable`, `is_generic_name`, word-boundary truncation, VLM vision path with OpenAI-style content parts (NOT Ollama `images` key).
- [x] **4. Twitter Summarizer Prompt & Timestamp Parity (C2a fix)** — Synchronized with `TWITTER_PROMPT`. Timestamps formatted as `%b %d %H:%M`.
- [x] **5. Weekend Planner Schema & Exclusion Filtering (C2b, C8 fixes)** — Aligned JSON schema, token-subset + containment matching, C8 seasonal-event exception.
- [x] **5b. Weekend constraint suite (C5 weather, C4 constant columns, C3 window, C7 provenance)** — Full `enforce.py` constraint suite ported to `rust/src/ztools/weekend/` in canonical order.
- [x] **5c. Weekend 4-phase pipeline (extract → draft → refine → structure) + supply prioritisation** — Phase templates, extract_sources, prioritise_in_window, in_window_count all ported with same date scanner.
- [x] **6. Greedy decoding across all LLM callers** (temperature 0.0) — deterministic reproducible leaderboard outputs.
- [x] **7. Derived request timeouts** from measured cold-start / prefill / decode rates.
- [x] **8. Eval validator + content cleaning parity** — Rust `validate.rs` and `clean.rs` ported from Python. Every regex proved-fail-first.
- [x] **9. Twitter Live Timeline Browser Scraping Parity** — Camoufox anti-detect Firefox automation with session discovery, embedding clustering, UTF-8 safe signature truncation, non-blocking stdin handling, caching.
- [x] **10. Resilient DuckDuckGo Event Scraping & Git Hook Quality Gates** — Dual HTML snippet parsers, DDG Lite fallback, `cargo clippy -D warnings` and `cargo test` quality gates.

---

## 4. A/B Test & Verification

**`bin/ab_test --functional`** runs test fixtures through both Rust and Python, asserting identical diagnostic verdicts, sanitized filenames, and prompt payloads. All 4 tools pass with 3.6-4.7x Rust speedup.

- **Rust quality gates**: `cargo clippy --all-targets -D warnings` and the 500-line cap per file prevent code rot in the primary implementation.
- **Prompt surface**: `conf/prompts.toml` is the canonical home, read by both sides; drift-gate test `test_twitter_prompt_matches_shared_conf` enforces byte-identical Rust fallback.
- **Model choice**: Both Rust and Python read from `conf/config.toml [best_models]`. Rust binary loads dynamic config via `with_ztools_best_models()` on startup.
- **Eval validators + cleaning**: Rust `validate.rs` and `clean.rs` are faithful ports from Python with proved-fail-first parity.

---

## 5. Structural Fix (Standing Hazard)

The "parallel reimplementation" failure mode is addressed by:

1. **Shared surface in shared config** — Prompts (`conf/prompts.toml`) and model choice (`conf/config.toml [best_models]`) are the single source of truth read by both sides. Editing one file updates both the Rust binary and Python reference.

2. **Automated A/B test harness** — `bin/ab_test --functional` runs test fixtures through both Rust and Python, asserting identical diagnostic verdicts, sanitized filenames, and prompt payloads. Catches divergence the day it happens.

3. **Rust quality gates** — `cargo clippy --all-targets -D warnings` and the 500-line cap per file in `~/Projects/ztools/rust` prevent code rot in the primary implementation.