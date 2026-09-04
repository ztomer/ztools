# Python ↔ Rust parity ledger

The Rust port is the primary implementation. The Python reference under `references/`
exists for parity verification only — it is no longer the active code path.

## Current state

- **Rust** (`rust/src/ztools/`) — primary implementation, native static binary,
  no Python dependency. Entry points: `twitter-summarize`, `weekend-plan`,
  `image-renamer`, `model-eval`.
- **Python** (`references/`) — reference implementation only. Used for A/B testing
  and parity verification. Entry points `tw`/`wk`/`rn`/`ev` no longer executed;
  the Rust binaries in `~/Projects/ztools/bin/` replace them.

The Python code under `references/` is preserved for parity comparison but is not
run in production. All four tools execute Rust code.

## Model selection

The Rust binary reads model choices from shared TOML config, matching Python:

| Slot | Python `conf/config.toml` | Rust (from config) |
|---|---|---|
| `think` | `ornith-1.0-35b-jang_4m` | `ornith-1.0-35b-jang_4m` |
| `json` / Weekend | `qwen3.8-27b-8bit` | `qwen3.8-27b-8bit` |
| `summarize` / Twitter | `gemma-4-e2b-it-8bit` | `gemma-4-e2b-it-8bit` |
| `filename` / Renaming | `gemma-4-e2b-it-8bit` | `gemma-4-e2b-it-8bit` |
| `vlm` / Vision | `qwen3.8-27b-8bit` | `qwen3.8-27b-8bit` |

Both sides read from `conf/config.toml` `[best_models]`. The Rust binary loads
dynamic `[best_models]` via `with_ztools_best_models()` and shared prompts via
`with_shared_prompts()` on startup (no `--config` flag needed when running from
the project directory). This was merged 2026-08-19 — prior to this, the Rust
binary had hardcoded defaults that could drift from the shared config.

## Prompt surface

The canonical prompt surface is `conf/prompts.toml`, read by both sides:

- **Rust**: Embedded fallback copy kept byte-identical to `conf/prompts.toml` by
  drift-gate test `test_twitter_prompt_matches_shared_conf`. The `with_shared_prompts()`
  loader reads from `~/.config/ztools/prompts.toml` or `conf/prompts.toml` at runtime.
- **Python**: `references/eval/tasks_prompts.py` `TWITTER_PROMPT` composes
  `load_prompt("twitter", "summarize")` from the same file, wrapping the timeline
  fixture into the shared block.

Editing `conf/prompts.toml` updates both sides — the gate enforces synchronization.

## Eval validators + content cleaning

The Rust eval pathway (`rust/src/ztools/eval/`) is a faithful port of the Python
reference:

- `validate.rs` ports `eval/validate.py`'s `validate_file_summary` — list/dict/raw-string
  branches, filename-echo guard, header bonus; multiply-form thresholds so a 4-file
  list scores 100 only with >= 4 detailed descriptions
- `clean.rs` ports `lib/content_processing.py`'s cleaning chain:
  `remove_thinking_blocks`, `remove_inline_thinking`, `remove_stats_tokens`,
  `remove_markdown_blocks`, `extract_content_from_code_blocks`,
  `clean_model_output`

Every ported regex was proved-fail-first by neutering (THINK_RE, gemma correction
loop, stats chain, code-block extraction; generic description branch; cleaning-before-scoring
wiring). The tag trap (`<thinking>` full word vs `<channel\|>thought`) is shared
between both sides — fixtures must carry the exact reference bytes.

## Resolved & Ported into Rust (The Parity Roadmap)

All 10 roadmap items are completed, verified through A/B testing with the Python
reference:

- [x] **1. Broken Model & Packaging Defect Detection** — Ported to
  `rust/src/ztools/model_health.rs`. Detects unsupported MTP shards, missing index
  shards, and incomplete downloads.
- [x] **2. Best Model Matrix & Dynamic Configuration** — Synchronized with 30-task
  benchmark winners. `with_ztools_best_models()` dynamic loader from
  `~/.config/ztools/config.toml` or `conf/config.toml`.
- [x] **3. Image Renamer Security & Untrusted Framing** — Ported to
  `rust/src/ztools/rename/`. `clean_filename`, `is_meaningful_text`,
  `is_non_human_readable`, `is_generic_name`, word-boundary truncation, VLM vision
  path with OpenAI-style content parts (NOT Ollama `images` key).
- [x] **4. Twitter Summarizer Prompt & Timestamp Parity (C2a fix)** — Synchronized
  with `TWITTER_PROMPT`. Timestamps formatted as `%b %d %H:%M`.
- [x] **5. Weekend Planner Schema & Exclusion Filtering (C2b, C8 fixes)** — Aligned
  JSON schema, token-subset + containment matching, C8 seasonal-event exception.
- [x] **5b. Weekend constraint suite (C5 weather, C4 constant columns, C3 window,
  C7 provenance)** — Full `enforce.py` constraint suite ported to
  `rust/src/ztools/weekend/` in canonical order: provenance → exclusion → window →
  weather → constant columns.
- [x] **5c. Weekend 4-phase pipeline (extract → draft → refine → structure) +
  supply prioritisation** — Phase templates, extract_sources, prioritise_in_window,
  in_window_count all ported with same date scanner. Weather precedes pipeline.
- [x] **6. Greedy decoding across all LLM callers** (temperature 0.0) — deterministic
  reproducible leaderboard outputs.
- [x] **7. Derived request timeouts** from measured cold-start / prefill / decode rates.
- [x] **8. Eval validator + content cleaning parity** — Rust `validate.rs` and `clean.rs`
  ported from Python. Every regex proved-fail-first.
- [x] **9. Twitter Live Timeline Browser Scraping Parity** — Camoufox anti-detect
  Firefox automation with session discovery across browsers, embedding clustering,
  UTF-8 safe signature truncation, non-blocking stdin handling, caching.
- [x] **10. Resilient DuckDuckGo Event Scraping & Git Hook Quality Gates** — Dual HTML
  snippet parsers, DDG Lite fallback, `cargo clippy -D warnings` and `cargo test`
  quality gates.

## Closed divergences

These were previously tracked divergences but have been resolved through the parity
roadmap above:

1. **Summarizer prompt duplication** — Resolved: `conf/prompts.toml` is the canonical
   home, drift-gate test enforces byte-identical Rust fallback.

2. **Model selection drift** — Resolved: Both Rust and Python read from
   `conf/config.toml [best_models]`. Rust binary loads dynamic config via
   `with_ztools_best_models()` on startup.

3. **Eval Python-only** — Resolved: Rust `validate.rs` and `clean.rs` are ported
   from Python reference with proved-fail-first parity. Python `references/` remains
   as reference only.

## Structural fix (standing hazard)

The "parallel reimplementation" failure mode is addressed by:

1. **Shared surface in shared config** — Prompts (`conf/prompts.toml`) and model choice
   (`conf/config.toml [best_models]`) are the single source of truth read by both sides.
   Editing one file updates both the Rust binary and Python reference.

2. **Automated A/B test harness** — `bin/ab_test --functional` runs test fixtures
   through both Rust and Python, asserting identical diagnostic verdicts, sanitized
   filenames, and prompt payloads. Catches divergence the day it happens.

3. **Rust quality gates** — `cargo clippy --all-targets -D warnings` and the 500-line
   cap per file in `~/Projects/ztools/rust` prevent code rot in the primary
   implementation.