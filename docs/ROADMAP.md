# Architecture & Implementation Roadmap — `ztools`

_Forward-looking: items 1–4 are the active roadmap. Items 5–10 are completed and will be pruned to git history. Python `references/` is preserved for A/B parity verification only; production runs execute Rust code._

---

## 1. Broken Model & Packaging Defect Detection
Ported to `rust/src/ztools/model_health.rs`. Probes model health at startup: detects unsupported MTP shards, verifies weight shards exist on disk, flags incomplete downloads, and refuses models that decode under thrashing conditions. Structural gate: `tools/check_no_allow.py` prohibits `#[allow]` attributes.

## 2. Best Model Matrix & Dynamic Configuration
Synchronized with 30-task benchmark winners. `with_ztools_best_models()` dynamic loader from `~/.config/ztools/config.toml` or `conf/config.toml`. Consumer slots: `json` → `qwen3.8-27b-8bit`, `filename` → `gemma-4-e2b-it-8bit`, `summarize` → `gemma-4-e2b-it-8bit`, `think` → `ornith-1.0-35b-jang_4m`/`qwen3.8-27b-8bit`, `vlm` → `qwen3.8-27b-8bit`. Both Rust and Python read from shared `conf/config.toml [best_models]`.

## 3. Image Renamer Security & Untrusted Framing
Ported to `rust/src/ztools/rename/`. `clean_filename`, `is_meaningful_text`, `is_non_human_readable`, `is_generic_name`, word-boundary truncation. VLM vision path with OpenAI-style content parts (NOT Ollama `images` key). Resists adversarial prompt injections in OCR text. All behaviors proved-fail-first against mutants.

## 4. Twitter Summarizer Prompt & Timestamp Parity (C2a fix)
Synchronized with `TWITTER_PROMPT`. Timestamps formatted as `%b %d %H:%M`. Canonical prompt home: `conf/prompts.toml`. Drift-gate test `test_twitter_prompt_matches_shared_conf` enforces byte-identical Rust fallback. Both Rust and Python read from shared config surface.

---

### Rust Port Status (2026-08-20)
The Rust binary `ztools` is the primary implementation; Python `references/` (~23.8k LOC) is preserved for A/B parity verification only. All 10 roadmap items are completed and verified through A/B testing with the Python reference.

**Covered tools** (Rust ≈ 2.6k production LOC):
- `twitter-summarize` — prompt sync, greedy decoding, timestamp parity
- `weekend-plan` — 4-phase pipeline, C3-C8 enforcement, supply prioritisation
- `image-renamer` — clean filename, VLM vision path, untrusted framing
- `model-eval` — data-driven `EvalTask`/`Check` enum, content cleaning

**Key infrastructure ported**: `eval/samples.rs` (median-of-5 clean estimation), `eval/watchdog.rs` (stall detection), `eval/gpu_lock.rs` (GPU lock), `eval/memory.rs` (thrashing detection), `eval/completeness.rs` (truncated-run detection), `eval/discrimination.py` (gate/ranking split), `eval/tasks_core.py` (RANKING_TASKS/GATE_TASKS).

**A/B test harness**: `bin/ab_test --functional` runs test fixtures through both Rust and Python, asserting identical diagnostic verdicts, sanitized filenames, and prompt payloads. All 4 tools pass with 3.6-4.7x Rust speedup.

---

### Cross-Cutting Fixes (completed)
- **Completeness derived by diffing tasks**: `run_eval` now returns `{"results", "expected_tasks", "completed_tasks", "truncated_reason", "complete"}` rather than a bare list. Partial runs are marked `(partial)` in stats and excluded from historical averages.
- **Truncated runs refused**: `save_historical_results` records reason; `load_historical_stats` filters incomplete entries with `excluded` count visible.
- **Oversize refusal with seam**: reads psutil `available` (not `Pages free`); hard refusal with `--allow-oversize` override; both directions testable on machine.
- **Sample-clean gate consults GPU lock**: `machine_is_uncontended()` now checks `gpu_lock.foreign_holder()` in addition to swap/compressor pressure.

---

### Items 5–10 — Completed (pruned to git history)
All 10 roadmap items were verified through A/B testing. Items 5–10 and their detailed A/B verification records have been moved to git history; their defect classes are recorded in git history alongside this repo's port work.

---

_Behavioral A/B testing, quality gates, and the parity roadmap for the native Rust ztools binary. Production runs execute Rust code; Python `references/` is preserved for A/B parity verification only._

**(Items 5–10 detailed records and A/B test matrices have been pruned to git history per house rule #13; see git log for full history.)**