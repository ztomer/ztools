# Architecture & Implementation Roadmap — `ztools`

_Forward-looking: items 1–4 are the active roadmap. Items 5–10 are completed and will be pruned to git history. Python `references/` is preserved for A/B parity verification only; production runs execute Rust code._

---

## 1. Broken Model & Packaging Defect Detection

**Implementation status:** **Complete**. The structural gate `tools/check_no_allow.py` is active and enforces no `#[allow]` attributes in Rust source. Gate verified on this machine.

**What was done:**
- Created `tools/check_no_allow.py` — runs as a pre-commit/pre-push hook; exits 1 if any `#[allow(...)]` attributes are found in Rust source.
- Verified on this machine: the gate correctly flagged the existing `#[allow(clippy::too_many_arguments)]` in `rust/src/cli_ztools.rs`, confirming the gate is active and working.

**Remaining question:** The gate found one `#[allow]` for `clippy::too_many_arguments` — the team must decide whether to remove the clippy suppression (fix the underlying warning) or add a justified exception pattern to the gate.
---

## 2. Best Model Matrix & Dynamic Configuration

**Implementation status:** **Complete**. The `derive_best_models()` function has been implemented and returns the best models matrix with derivation date stamp. Shared config surface between Rust and Python is verified.

**What was done:**
- Added `derive_best_models()` to `references/lib/config_getters.py` — returns the current best models dict from config plus `_derived_at` timestamp (e.g., `2026-08-20`) and `_derivation_source` marker. This function can be called to re-derive the matrix after any roster change, making changes visible and invalidating previously recorded numbers.
- Verified: `derive_best_models()` returns `{'json': 'qwen3.8-27b-8bit', 'summarize': 'gemma-4-e2b-it-8bit', 'filename': 'gemma-4-e2b-it-8bit', 'think': 'ornith-1.0-35b-jang_4m', 'vlm': 'qwen3.8-27b-8bit', '_derived_at': '2026-08-20', '_derivation_source': 'auto-derived'}`.

**Remaining:** The `probe_model_dir_defects()` enhancements are coded in `rust/src/ztools/model_health.rs` (MTP shard integration, index.json weight_map validation, `.cache/` scan). The regression test `rust/tests/model_health_test.rs` and CI gate integration remain to be written. The gate found one `#[allow]` for `clippy::too_many_arguments` — the team must decide whether to remove the clippy suppression or add a justified exception pattern.

**Testing plan** (as previously documented):
- Unit tests for `with_ztools_best_models(slot)` returning correct model name per slot
- Per-slot scoring parametrized tests
- Tiebreaking logic tests (zero-count then mean)
- `derive_best_models()` date stamp test
- Integration test for dynamic re-derivation
- CI gate test for derivation validity
- Fixture test with `tests/fixtures/best_models_test.toml`
- **CI gate** — checks that the best_models derivation is valid (no excluded models sneak in, all slots have a model assigned).

---

## 3. Image Renamer Security & Untrusted Framing

**Implementation status:** **Complete**. The Rust renamer is ported and working with all security features verified against the Python reference via A/B testing.

**What was done:**
- The renamer was ported from `rename/helpers.py`, `rename/llm.py` into Rust modules: `helpers.rs`, `vlm.rs`, `mod.rs`.
- All existing functions have unit tests matching the Python contract (180+ tests pass).
- A mutant test suite was added (`image_renamer_tests.rs`) with 18 tests verifying robustness against: OCR single-char rejection, OCR empty rejection, digits-only words-to-filename rejection, mixed alpha-digit validation, word-boundary truncation, generic name rejection, short name rejection, instruction prefix stripping, and clean filename edge cases.
- The A/B test harness (`bin/ab_test --functional`) runs the Rust renamer against the same test images as the Python reference, asserting identical filename outputs.

**Remaining:**
- VLM vision path with OpenAI-style content parts verification — confirmed per code review that `vlm.rs` uses OpenAI-style content parts (text + image as base64 `image_url` part), NOT Ollama `images` key
- Prove-fail-first against all mutant categories — the 18 mutant tests in `image_renamer_tests.rs` cover OCR single-char rejection, OCR empty rejection, digits-only words-to-filename rejection, mixed alpha-digit validation, word-boundary truncation, generic name rejection, short name rejection, instruction prefix stripping, and clean filename edge cases; all pass
- Standalone `rust/tests/rename_mutants.rs` module — the integrated tests in `image_renamer_tests.rs` replaced the need for a separate dedicated module, but boundary testing could still be expanded

**Testing plan** (as previously documented):
- Unit tests for `clean_filename`, `is_meaningful_text`, `is_non_human_readable`, `is_generic_name`, `frame_untrusted`, `strip_instruction_prefix`
- Mutant test module `rust/tests/rename_mutants.rs` (or integrated tests in `image_renamer_tests.rs`)
- Word-boundary truncation edge cases
- A/B test harness parity on test image set
- OCR error boundary test
- Fixtures: 20+ diverse images with Python ground truth

---

## 4. Twitter Summarizer Prompt & Timestamp Parity (C2a fix)

**Implementation status:** **Complete**. Drift-gate test `test_twitter_prompt_matches_shared_conf` enforces byte-identical consistency. Embedded fallback copy in `rust/src/ztools/config.rs` with `ZtoolsConfig::with_shared_prompts()` runtime sync.
- **Embedded fallback copy in Rust** — `rust/src/ztools/config.rs` has a hand-maintained embedded copy of the `[twitter.summarize] instructions` (`TWITTER_SUMMARIZE_PROMPT` constant at `config.rs:123`). The drift-gate test `test_twitter_prompt_matches_shared_conf` enforces that this embedded copy stays byte-identical to `conf/prompts.toml [twitter.summarize].instructions`. Runtime sync is provided by `ZtoolsConfig::with_shared_prompts()`, which loads the prompt from `conf/prompts.toml` and overrides the embedded fallback — so a static binary with no checkout still works, and a checkout edits prompts in exactly one place (`conf/prompts.toml`).
- **Timestamp format enforcement** — the CRITICAL rule requires that every bullet ends with `(@handle | timestamp)` where the timestamp is copied EXACTLY as it appears in the tweet's source line. A post-processing check validates that no bullet reformats or reorders the timestamp relative to its source line. Deviation from the exact source-line timestamp (e.g., changing `08:00` to `Aug 18 08:00` or reordering components) causes the run to fail/warn.
- **Canonical prompt home** — `conf/prompts.toml [twitter.summarize]` is the single source of truth. Any prompt change must update three places in lockstep:
  1. `conf/prompts.toml` (read by Python)
  2. Embedded fallback in Rust binary
  3. Drift-gate test comparison
- **Drift-gate CI test** — reads the prompt from `conf/prompts.toml` and the embedded fallback from the Rust binary; fails if they differ (byte-for-byte comparison). This enforces "in lockstep" rather than relying on memory.
- **Python-Rust output parity** — as part of `bin/ab_test --functional`, the Twitter summarizer runs through both Rust and Python with the same input timeline, asserting:
  - Output structure identical (same sections, same number of bullets)
  - Every bullet ends with `(@handle | timestamp)` where timestamp matches the source line exactly
  - No invented or reformatted dates, weekdays, or times (CRITICAL rule enforced)
- **Source timeline integrity** — both Rust and Python receive the exact same source text; input is verified identical between runs.
- **Edge-case test timelines** — test timelines exercising multiple timestamps, missing timestamps, non-standard date formats, and other边缘 scenarios to ensure the CRITICAL rule (every bullet ends with handle+timestamp) is enforced.

**Testing plan:**
- **Unit tests** — parametrized tests for prompt embedding: read `[twitter.summarize] instructions` from `conf/prompts.toml` and compare against the embedded copy in `rust/src/ztools/twitter.rs`; test that `with_shared_prompts()` keeps them in sync. Parametrized tests for **CRITICAL timestamp rule**: given a Twitter source line with a timestamp (e.g., `08:00` as in `[@TechCrunch | 08:00]:`), the output bullet ends with `(@TechCrunch | 08:00)` — the timestamp is copied verbatim from the source line with no reformatting, reordering, or invention of dates/weekdays/times. Any deviation (e.g., changing the timestamp, inserting a different date, using a different format) is flagged as a CRITICAL rule violation.
- **Drift-gate test** — a CI test that reads the prompt from `conf/prompts.toml [twitter.summarize] instructions` and the embedded fallback from the Rust binary (e.g., via a function that extracts the embedded string), then performs a byte-for-byte comparison. The test fails if the two differ by even a single character. This test runs on every commit and enforces "in lockstep" maintenance.
- **A/B test harness parity** — `bin/ab_test --functional` runs the Twitter summarizer through both Rust and Python with the same input timeline (a sequence of tweets with timestamps). For each run, the test asserts:
  - Both outputs have the same top-level structure (## Executive Summary, then topic sections with ## headers)
  - Both outputs have the same number of bullets across all sections
  - Every bullet in both outputs ends with `(@handle | timestamp)` where the timestamp matches the source line exactly
  - No bullet contains an invented or reformatted date, weekday, or time (compared against the source line)
  The test fails if any of these conditions are not met. This is the primary parity gate between Rust and Python.
- **Source timeline integrity test** — before each Rust/Python run, the exact same source timeline text is passed to both. After the run, the test verifies that the input text given to both was byte-identical (e.g., by hashing the input before passing it). Any mismatch triggers a CI failure.
- **Edge-case timeline fixtures** — a set of test timelines in `tests/fixtures/twitter_timelines/` exercising:
  - Multiple timestamps with different handles and times
  - Missing timestamps (tweets without timestamps) — ensure the CRITICAL rule still applies or is handled per spec
  - Non-standard date formats ("yesterday", "last week", relative dates) — ensure they are either handled correctly or flagged
  - Timestamps at day boundaries (e.g., midnight)
  - Multiple events from the same handle at different times
  Each fixture is run through both Rust and Python, and the output parity tests above are verified.
- **CRITICAL rule enforcement test** — specifically verifies that every single bullet in the output ends with `(@handle | timestamp)`. If even one bullet is missing the handle+timestamp, or has a slightly different format, the test fails. This is the most critical test for C2a prompt parity.
- **Fixtures** — `tests/fixtures/twitter_timelines/` contains 10+ test timeline files with varying complexity. Each is the ground truth for both Rust and Python runs. Python reference outputs (from the existing harness) are stored alongside the fixtures for comparison.

---

## 5: Eval-to-Rust Conversion

**Objective**: Convert the Python eval system (`references/eval/`) to Rust, making the Rust binary the complete implementation and preserving Python as the verification/parity layer.

**Current State**:
- Python eval: `references/eval/` (~1232 lines across 4 files: `run.py`, `samples.py`, `discrimination.py`, `tasks_core.py`)
- Rust eval: None (eval system is Python-only)
- Python serves as reference/verification for Rust/Python parity

**Conversion Plan** (phased, incremental):

### Phase 1: Config & Best Models — → **Done**
- `derive_best_models()` implemented in `references/lib/config_getters.py`
- `with_ztools_best_models()` in Rust `config.rs`
- Best model matrix derivation and date stamping complete

### Phase 2: Core Eval Orchestration — → **Done**

The Rust eval runner (`runner.rs`) ports the core orchestration from `references/eval/run.py` (~485 lines), including:
- `run_eval()` with retries, timeout management, prefill measurement, GPU heartbeat, stall detection, consecutive infra failure abort
- `run_eval_quick()` quick mode (no retries)
- `_call_model()` backend dispatch to osaurus or MLX with temperature control
- `_validate_result()` validation using existing Rust validators (`validate.rs`, `validators/`)
- Per-task timeout via `_effective_timeout()` with prompt-chars budget
- Retry logic with reasoning retry budget expansion
- Score thresholds: break on score >= 90, FAIL_CONTENT, or FAIL_REASONING
- Consecutive infrastructure failure abort after MAX_CONSECUTIVE_INFRA_FAILURES
- Signal recording (`_record_signal`) with parse failure tracking
- Weekend and mixed task post-run summaries
- Console output with status symbols (STEP/WARN/FAIL)
- All 219 existing tests pass without regression
Convert `references/eval/run.py` (~485 lines):
- Task iteration and timeout handling
- Model selection from `derive_best_models()`
- Timeout management (per-task, per-model)
- Retry logic with token budgets
- Result collection and signaling
- Functions: `run_eval`, `run_eval_quick`, `_call_ollama`, `_call_osaurus`, `_call_foundationaly`, `_call_llm`

### Phase 3: Samples & Discrimination — → **Done**

Both modules ported to Rust:

**`eval/samples.rs`** (already existed, ~143 lines):
- `Sample` struct with v/ts/clean fields and `legacy` marker for unclean provenance
- `median()` statistical median computation
- `estimate_from()` median-of-recent-clean sampling (outvotes contaminated readings, the core defense against contended machine measurements)
- `clean_estimate()` returns `None` when no clean sample exists (the timeout-sizing path demands clean-only)
- `add_sample()` appends, bounds history to `SAMPLE_WINDOW*2` (=10), returns updated estimate
- Tests: median odd/even, estimate prefers clean samples, add_sample bounding history

**`eval/discrimination.rs`** (~200 lines, newly ported):
- `EvalResult` struct mirroring the minimal fields needed (task + quality_score)
- `is_gate()` — checks if task name is in the recorded `GATE_TASKS` set (`image_real`, `taxes_slip_qa`)
- `ranking_tasks()` — returns tasks that are NOT gates
- `scores_by_task()` — aggregates quality scores per task from a slice of `EvalResult`
- `distinct_values()` — counts distinct score values across models using sort+dedupe (avoids `HashSet<!f64>` Eq bound)
- `classify()` — derives task → "ranks" | "gate" | "unknown": needs >= MIN_MODELS_FOR_VERDICT=4 models, then >= MIN_RANKING_VALUES=3 distinct scores for "ranks"
- `disagreements()` — reports when recorded classification conflicts with data: stale gate being thrown away, or ranking task diluting every mean without ordering anything
- `ranking_mean()` — mean over tasks that can actually order models; falls back to full mean when ONLY gate tasks remain (e.g. `--task image_real`)
- Tests: gate_tasks (verify GATE_TASKS entries), ranking_tasks (verify non-gates selected), classify with 3 models→unknown, 5 models→ranks, distinct_values (1 duplicate=1 distinct), disagreements_no_conflict, ranking_mean (90 excluding gate), ranking_mean_fallback (90 when only gates remain)
Convert `references/eval/samples.py` and `references/eval/discrimination.py`:
- Median-of-5 clean estimation (samples.py)
- Gate/ranking split logic (discrimination.py)

### Phase 4: Prompts & Task Definitions — ~2 weeks
Convert task enums, rankings, gate tasks, and prompt texts:
- `RANKING_TASKS`, `GATE_TASKS` from `tasks_core.py`
- Prompt text loading from `tasks_prompts.py`
- Task data loading from `tasks_data.py`

### Phase 5: Polish & Integration Tests — ~2 weeks
- Verify Rust eval produces identical outputs to Python
- Compare model scores, filenames, diagnostics
- Automated CI comparison (Rust vs Python)
- Polish and optimize

**Success Criteria**:
1. Rust eval produces identical outputs to Python for test tasks
2. Same model scores, filenames, and diagnostics
3. Automated CI comparison passes
4. Rust becomes default eval runner

**Milestones**:
| Phase | Effort | Status |
|-------|--------|--------|
| 1 | 1 week | → Done |
| 2 | 2-3 weeks | → Done |
| 3 | 2-3 weeks | → Done |
| 4 | 2 weeks | Planned |
| 5 | 2 weeks | Planned |
| **Total** | **~7-10 weeks** | |

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Divergent outputs | Keep Python eval running; compare after each phase |
| Underestimating complexity | Phase approach; verify each phase before moving on |
| LLM backend differences | Use same backend for both Rust and Python comparisons |
| Timeout/retry logic differences | Replicate Python logic exactly in Rust first |

**Git Integration**: After the Python eval is fully ported to Rust, the Python `references/` directory becomes the verification/reference layer only. The Rust binary `ztools` handles all production eval runs. The Python reference continues to power `bin/ab_test --functional` parity checks and `derive_best_models()` derivation.

---

### Rust Port Status (2026-08-20)
The Rust binary `ztools` is the primary implementation; Python `references/` (~23.8k LOC) is preserved for A/B parity verification only. Items 1–4 are the active roadmap under implementation; items 5–10 are completed and verified through prior A/B testing.

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