# Architecture & Implementation Roadmap — `ztools`

_Forward-looking: items 1–4 are the active roadmap. Items 5–10 are completed and will be pruned to git history._

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

_Behavioral A/B testing, quality gates, and the parity roadmap for the native Rust ztools binary. Production runs execute Rust code; Python `references/` is preserved for A/B parity verification only._