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
## Test-debt triage (Phase 4 item 7) — 2026-09-13

Per-file verdicts for the 139 files under `references/tests/`, produced by
inspection (header, imports, test names) cross-checked against the Rust side
(`rust/src/**`, `rust/tests/*.rs`). Totals at triage: **COVERED 59 /
RE-EXPRESS 40 / DELETE 40**; current: **COVERED 62 / RE-EXPRESS 38 /
DELETE 39** (one triage error corrected DELETE→RE-EXPRESS then closed with
two others; 5 more rows partially covered but still counted RE-EXPRESS
until their remainders land). Spot-verified (sampled Rust test names exist
verbatim;
`test_weekend.py` confirmed to target JSON extraction, not the planner).
RE-EXPRESS rows are execution items: port the case fail-first at cutover time.
Only `test_llm.py`, `test_mlx.py`, `test_gemma.py` need a live server/GPU.

### Weekend (18: COVERED 9 / RE-EXPRESS 5 / DELETE 4)

| file | verdict | evidence |
|---|---|---|
| test_weekend.py | COVERED | json_validator_tests/shape.rs::test_extract_list_from_dict_prefers_known_keys_in_order |
| test_weekend_config.py | DELETE | retired Python debug-cache files + osaurus.app wrapper plumbing |
| test_weekend_data.py | DELETE | retired ddgs DDGS.text/429-retry/scrape_review_score (corpus kernel now pinned by weekend parity gate) |
| test_weekend_followup.py | COVERED | weekend_followup_tests.rs::follow_aggregators_fetches_directory_pages_and_bounds_the_tally |
| test_weekend_integration.py | RE-EXPRESS | missing mocked main/_fetch_data/skip-web end-to-end pin |
| test_weekend_llm_1.py | RE-EXPRESS (partial: alt-key normalization covered 2026-09-13 by `normalize_llm_item` + 2 tests in weekend_parse_tests.rs; still open: per-model TOML field_mapping chain, get_llm_json retry/panic_dump pins) |
| test_weekend_llm_2.py | RE-EXPRESS | missing phase timeout-learning (widen/floor/accumulate) pin |
| test_weekend_output.py | COVERED | weekend_parse_tests.rs::test_render_weekend_plan_gorgeous_fixed_activities |
| test_weekend_planner_branches.py | RE-EXPRESS | missing parse fixed/transient edge pins (alt-keys, weather-filter) |
| test_weekend_supply.py | COVERED | float_top.rs::in_window_lines_float_to_the_top_marked_but_nothing_is_removed |
| test_quality_weekend_scorers.py | DELETE | retired Python quality-harness weekend scorers, no Rust eval home |
| test_report_class_c4.py | RE-EXPRESS | missing unknown-word sentinel normalization pin |
| test_report_class_cases.py | DELETE | retired frozen-artifact xfail catalogue + private real samples |
| test_report_class_fixes.py | COVERED | weekend_enforce_tests.rs (exclusions/dates/defects gates) |
| test_report_class_region.py | COVERED | weekend_filter_tests.rs::a_foreign_city_beats_any_local_token |
| test_routines_status.py | COVERED | status.rs::status_flags_a_stale_plan_and_hollow_transient |
| test_constant_columns.py | COVERED | defects.rs::the_shipped_target_age_constant_is_flagged |
| test_empty_tables.py | COVERED | weekend_tests.rs::test_format_weekend_plan_empty_and_populated |

### Twitter (12: COVERED 4 / RE-EXPRESS 4 / DELETE 4)

| file | verdict | evidence |
|---|---|---|
| test_twitter.py | RE-EXPRESS (partial: extract_thinking/merge_thinking_with_summary covered 2026-09-13 in `twitter/mod.rs` + 5 tests incl. mock-server wiring into `run_summary`; verified Python keeps the raw `<thinking>` block in `cleaned`, mirrored exactly. Still open: prompt-building smoke test is a tautology with no product code — DELETE candidate at cutover) |
| test_twitter_summarizer.py | RE-EXPRESS (partial: `--fetch-only` now persists the debug cache 2026-09-13 via `save_tweets_json` + round-trip tests. Still open: main clean/use-cache orchestration) |
| test_twit_browser_1.py | COVERED | browser_parse.rs::test_parse_tweets_from_mock_graphql, now incl. `id_str` capture (added 2026-09-13: `Tweet.id` with serde default for old caches; verified same-input-same-IDs against the Python parser) |
| test_twit_browser_2p1.py | DELETE | retired Playwright collect_tweets_via_browser orchestration |
| test_twit_browser_2p2.py | DELETE | retired Playwright scroll/handler/dedup orchestration |
| test_twit_browser_3.py | COVERED | collect.rs scroll_stops_*/logged_out_matrix |
| test_twit_browser_4.py | COVERED | cookies_tests.rs normalize_expiry_bounds/read_firefox_cookies_filters_like_python |
| test_twit_cookies.py | DELETE | retired Chrome keychain/AES Cookies-DB reader |
| test_twit_output.py | RE-EXPRESS (partial: clean_folder + `--clean` covered 2026-09-13 by `store.rs` (3 tests incl. deterministic delete-failure via subdir trick) wired to the store dir; glob/dotfile/case semantics verified against Python. Still open: last_run state, bat plumbing, write_markdown shape) |
| test_twit_summarize.py | COVERED 2026-09-13 — timeout-estimate (`twitter/budget.rs`), per-attempt branch (`handle_model_output`), selection + chain construction (`twitter/fallback.rs`), and now the retry loop itself: `twitter/chain.rs` runs the intended model then the `conf/twitter.toml [fallback]` models (roster-resolved, `TWITTER_FALLBACK_MODELS` override), records provenance and writes the C9 degraded banner into the artifact; 13 tests + the stub-server dispatch test. Restart and direct-MLX tiers retired (ops-side / no Rust counterpart) |
| test_twit_summarize_mlx_fallback.py | DELETE | retired local mlx/mlx-vlm fallback tier |
| test_routines_twitter_status.py | COVERED | store.rs newest_md_picks_the_latest_by_mtime_not_name |

### Eval core (20: COVERED 11 / RE-EXPRESS 5 / DELETE 4)

| file | verdict | evidence |
|---|---|---|
| test_eval_failures.py | COVERED | eval/failures.rs::infra_timeout_and_reasoning_are_distinguished |
| test_eval_outputs.py | COVERED | eval/outputs.rs::save_output_writes_header_body_and_reasoning_to_the_seamed_dir |
| test_eval_report.py | RE-EXPRESS (partial: score-stats/error-rates/failure-groups covered 2026-09-13 by `eval/report_metrics.rs` + 7 tests; still open: token-estimates/verbosity need per-outcome content capture in `TaskOutcome`, a serialization decision) |
| test_eval_report_extra_1.py | DELETE — rich `print_*` summary tables were the Python presenter; the Rust report renders through `eval/report.rs` (`render_eval_report`, `render_historical_trends`), covered by `tests/model_eval.rs` |
| test_eval_report_extra_2.py | COVERED (partial) 2026-09-13 — CSV in `report_csv.rs`; history in `report.rs::{save_historical_results, load_historical_stats, render_historical_trends}`. Python's `diff_from_last_run` presenter is not ported (B4 deferral: the trend table already shows the per-model delta) |
| test_eval_run_integration_1.py | COVERED | tests/eval_runner.rs::perfect_output_scores_ok_without_retry |
| test_eval_run_integration_2.py | COVERED | task_loader_tests::graded_score_dispatches_every_graded_variant |
| test_eval_run_integration_3.py | COVERED | tests/eval_runner.rs::consecutive_infra_failures_abandon_the_model_early |
| test_eval_server_restart.py | DELETE | osaurus_one.sh/osascript restart plumbing retired, no Rust counterpart |
| test_eval_task_sources.py | COVERED 2026-09-13 — `tasks_tests::the_source_is_the_prompt_the_model_was_shown` pins that every source-gated validator (summary, mixed summary, attribution, fabrication, injection) is fed the prompt as its source; `filename` is fed the raw input |
| test_eval_tasks_core.py | COVERED 2026-09-13 — the task table is `eval/tasks.rs` (24 rows incl. `json`/`detailed_json` aliases), pinned row-for-row to the Python table (name, roles, `parse_json`, validator) by `tasks_tests.rs`; `_extract_items_from_text` (table/bullet fallback parser) was a `run_validate`-only helper with no roster consumer — retired |
| test_eval_validate.py | COVERED | eval/validate.rs::all_detailed_scores_100 |
| test_eval_watchdog.py | COVERED | eval/watchdog.rs::test_watchdog_detects_stall |
| test_adversarial_tasks.py | COVERED | validators/adversarial.rs::test_validate_no_fabrication_catches_lures |
| test_benchmark_output.py | DELETE | retired benchmark rich presenter superseded by model_eval render fns |
| test_benchmark_quality.py | DELETE | retired benchmark heuristic scorers superseded by validators/ |
| test_benchmark_quality_runner.py | DELETE | retired benchmark query/run loop superseded by runner::run_eval |
| test_completeness.py | COVERED | eval/completeness.rs::a_run_that_reported_every_task_is_complete |
| test_detailed_json_boundaries.py | COVERED | detailed_score.rs::test_validate_detailed_json_all_four_source_caps |
| test_discrimination.py | COVERED | eval/discrimination.rs::test_ranking_mean |

### Validators / quality / taxes (25: COVERED 12 / RE-EXPRESS 4 / DELETE 9)

| file | verdict | evidence |
|---|---|---|
| test_faithfulness_validators.py | COVERED 2026-09-13 — validators in `eval/validators/faithfulness.rs` (11 tests) and now WIRED: `weekend_transient_schema` / `summarize_contradiction` / `filename_leak` rows of `eval/tasks.rs` via `Graded::{StrictSchema,NoContradiction,NoLeak}` |
| test_g3_edge_cases.py | RE-EXPRESS | prompt_render refusal + fetch/extract/ensure_server branches unported |
| test_name_matching_boundaries.py | COVERED | shape.rs::test_names_match_containment_and_token_overlap |
| test_quality_entry.py | DELETE | retired lib/quality_* benchmark harness |
| test_quality_models.py | DELETE | retired lib/quality_models Score/ScoreCard harness types |
| test_quality_report.py | DELETE | retired lib/quality_report baseline harness |
| test_quality_runner.py | DELETE | retired lib/quality_runner suite harness |
| test_quality_scorers_1.py | DELETE | retired lib/quality_scorers filename/summarize scorers |
| test_quality_scorers_2.py | DELETE | retired lib/quality_scorers structure/specificity/file scorers |
| test_factual_coverage.py | COVERED | text_match.rs::test_identifying_tokens_extracts_acronyms_and_digits |
| test_roster_corroboration.py | COVERED | disk.rs corroboration_accepts_disk_configs_or_documented_windows_only |
| test_scorer_discrimination.py | COVERED | basics.rs::test_validate_detailed_json_detects_generic_location |
| test_self_correcting_samples.py | COVERED | samples.rs::test_estimate_prefers_clean_samples |
| test_summary_scorer_boundaries.py | COVERED 2026-09-13 — summary arms by `eval/validators/summary.rs`, filename arms by `eval/validators/filename.rs`, file-summary arms by `eval/validate.rs` |
| test_task_winnability.py | COVERED | mixed_signal.rs::test_validate_mixed_signal_requested_count_caps_recall_target |
| test_taxes_grounded.py | COVERED | taxes_grounded_tests.rs::test_yoy_narrative_exact_reconciliation_scores_full |
| test_taxes_validator.py | COVERED | taxes_rubric.rs::grounding_scores_follow_the_rubric_ladder |
| test_text_validator_1.py | COVERED 2026-09-13 — filename arms by `eval/validators/filename.rs` (20 tests), summary arms by `summary_arm_tests.rs` (10 exact-score cross-checks); dict/None input coercion is Python-typing-specific with no Rust meaning (`&str` signature) |
| test_text_validator_2.py | COVERED | validate.rs generic_description_counted + attribution catches_swapped_author |
| test_json_validator_1.py | COVERED | shape.rs::test_extract_list_from_dict_prefers_known_keys_in_order |
| test_json_validator_2.py | COVERED | json_score.rs::test_validate_json_count_bands |
| test_json_validator_3.py | COVERED | mixed_signal.rs::test_validate_mixed_signal_counts_tp_and_fp |
| test_validator_ordering.py | COVERED 2026-09-13 — all four arms pinned in `eval/validators/ordering_tests.rs` (summary, filename, file-summary, grounded-vs-invented JSON at better than 2:1) |
| test_validators.py | DELETE | retired validators_lib shim (surviving json covered in json_validator_tests) |
| test_validators_helpers.py | DELETE | retired validators.helpers module, no Rust counterpart |

### Config / infra / rename (24: COVERED 6 / RE-EXPRESS 10 / DELETE 8)

| file | verdict | evidence |
|---|---|---|
| test_config.py | RE-EXPRESS | model field-mapping/top-keys TOML lookup untested in Rust |
| test_config_core_edges.py | RE-EXPRESS | auto-load ConfigurationError + user-config overlay untested |
| test_config_core_getters_1.py | RE-EXPRESS | Task enum/auto-load state/getter semantics untested |
| test_config_core_getters_2.py | RE-EXPRESS | versioned model-config cache/TOML fallback chain unpinned |
| test_config_debt_checks.py | DELETE | tools gate helper stays Python per roadmap; nothing to port |
| test_config_tasks.py | COVERED | task_loader_tests.rs::test_load_taxes_tasks_from_real_data_dir + poisoned_file_fails |
| test_cli_ux.py | RE-EXPRESS | server-down guidance copy untested in Rust |
| test_content_processing.py | COVERED | eval/clean.rs::removes_think_tags + siblings |
| test_coverage_boost.py | DELETE | coverage filler; surviving behaviors covered at owning modules |
| test_capabilities.py | RE-EXPRESS | vision probing + disk-based memory estimate unported |
| test_foundation_lib.py | DELETE | Apple Foundation Models bridge unported, macOS-only |
| test_logging_config.py | DELETE | Python get_logger wrapper with no Rust counterpart |
| test_memory_reading.py | RE-EXPRESS | reclaimable-available + compression-rate semantics unported |
| test_paths.py | DELETE | installed/checkout layout scan is packaging plumbing |
| test_parse.py | RE-EXPRESS | extract_json/normalize/filter/fix-years helpers untested in Rust |
| test_parse2.py | RE-EXPRESS | _extract_json_only think-block/fence cases untested in Rust |
| test_rename_cli_main.py | COVERED | tests/cli_dispatch_ztools.rs::image_renamer_dry_run_proposes_names_without_touching_files + apply_renames_the_files |
| test_image_renamer.py | COVERED | image_renamer_tests.rs::test_scan_and_rename + VLM branch + dedupe |
| test_img_helpers.py | COVERED | rename/helpers.rs::clean_filename_matches_python_contract + siblings |
| test_img_llm_1.py | COVERED | image_renamer_tests.rs::test_query_llm_filename_mock_server + test_query_vlm_for_filename_mock_server_sends_data_uri |
| test_img_llm_2.py | RE-EXPRESS | MLX filename fallback chain untested |
| test_tui_parameters.py | DELETE | GLOBAL_OVERRIDES TUI plumbing; no Rust override surface |
| test_file_size_gate.py | DELETE | tools gate stays Python per roadmap; nothing to port |
| test_testing.py | DELETE | MockLLM harness; Rust tests use per-test mock HTTP servers |

### LLM / server plumbing (18: COVERED 9 / RE-EXPRESS 4 / DELETE 5)

| file | verdict | evidence |
|---|---|---|
| test_lib_llm.py | COVERED | quirks::qwen_system_prompts_get_the_json_trigger + transport_http::blocking_call_extracts_content_and_parses_json + clean::full_pipeline |
| test_llm.py | COVERED + NEEDS-LIVE | transport_http::blocking_call_extracts_content_and_parses_json (live smoke only) |
| test_llm_protocol.py | DELETE | retired multi-backend factory (Mlx/Generic arms) |
| test_osaurus_lib_1.py | COVERED | quirks tests + transport_http::blocking_call_reports_http_error_as_data |
| test_osaurus_lib_2.py | COVERED | transport response_format json_object + clean::full_pipeline |
| test_osaurus_models.py | RE-EXPRESS | URL-builder/server-check/VLM-select unit behavior missing |
| test_osaurus_output_1.py | COVERED | clean::full_pipeline + qwen_marker/code_block helpers |
| test_osaurus_output_2.py | RE-EXPRESS | merge/filter/year-fix/text-normalize item pipeline has no Rust test |
| test_osaurus_restart.py | DELETE | retired lib/osaurus_server.py lifecycle |
| test_osaurus_server.py | DELETE | retired lib/osaurus_server.py lifecycle |
| test_no_live_server_gate.py | DELETE | Python-only socket-block harness gate, not product behavior |
| test_server_contention.py | RE-EXPRESS | per-task two-server announcement + name-not-cmdline counting missing |
| test_signal_handling.py | RE-EXPRESS | drain-mode interrupt/cleanup-ordering semantics unported |
| test_streaming_guard.py | COVERED | transport_http::overrun_guard_aborts_reasoning_past_budget_with_no_content |
| test_transport_perimeter.py | DELETE | Python-only raw-POST perimeter gate over .py packages |
| test_oversize_and_gpu_gate.py | COVERED | oversize::oversize_headroom_branches_are_exact + gpu_lock acquire tests |
| test_reasoning_overrun.py | COVERED | reasoning_retry::a_reasoning_overrun_retries_with_a_raised_budget_and_then_passes |
| test_prefill_measurement.py | COVERED | signals_prefill::prefill_probe_sends_nonce_led_filler_and_records_capabilities |

### Models / misc (22: COVERED 8 / RE-EXPRESS 8 / DELETE 6)

| file | verdict | evidence |
|---|---|---|
| test_model_caps_probe.py | COVERED | model_resolve_tests/disk.rs discovery/context/generative + signals_prefill probe |
| test_model_defects.py | COVERED | model_health_tests.rs defect cases + model_eval.rs broken-model refusal |
| test_model_eval.py | DELETE | retired eval/cli.py helper units; surviving guards covered by oversize.rs |
| test_model_eval_main_1p1.py | COVERED | tests/model_eval.rs filter/sweep/report + eval_runner.rs retry/scoring |
| test_model_eval_main_1p2.py | COVERED | eval_runner.rs quality-vs-infra + oversize_refusal + render_eval_report |
| test_model_eval_main_2.py | DELETE | retired flush-via-osaurus_one.sh restart; Rust holds GPU lock whole sweep |
| test_model_resolve.py | RE-EXPRESS | missing startup stale-slot audit (substitute/fetch covered) |
| test_model_substitution_retry.py | COVERED | tests/model_resolve_http.rs retry/quirks/switch-off + roster.rs negatives |
| test_mlx.py | DELETE + NEEDS-LIVE | skipped live-MLX test; direct-MLX path retired |
| test_mlx_lib.py | DELETE | retired direct-MLX subprocess fallback |
| test_mlx_vlm.py | DELETE | retired mlx-vlm subprocess path; Rust VLM goes via server image_url |
| test_gemma.py | DELETE + NEEDS-LIVE | skipped live-server test; no live-model tests in port |
| test_gpu_lock.py | COVERED | gpu_lock_tests.rs acquire/reclaim/foreign_holder/inherit + inline |
| test_gpu_lock_call_sites.py | RE-EXPRESS | missing entry-holds-lock + heartbeat-per-task assertions |
| test_gpu_lock_shell.py | RE-EXPRESS | shell impl stays ops-side; script wiring has no Rust test |
| test_vision_payload.py | COVERED | image_renamer_tests.rs VLM data_uri + quirks.rs multimodal passthrough |
| test_vision_task.py | COVERED 2026-09-13 — `eval/vision.rs`: fixtures as data (`conf/eval_vision.toml`), flat-fill PNG renderer (decoded and probed in tests), data-URI content parts on `ChatMessage.images`, `validate_image_description` (full/synonym/partial/blind/empty cases); wired as the `image_real` row |
| test_untrusted_framing.py | RE-EXPRESS | missing slot-ordering + override-merge (framing covered) |
| test_prompts_conf.py | COVERED | config.rs::test_twitter_prompt_matches_shared_conf + layering tests |
| test_rust_prompt_parity.py | COVERED 2026-09-13 — the generated Rust constants are canonical now; the surviving contract (eval prompt == production `conf/prompts.toml` instructions + fixture timeline) is pinned by `eval/prompts/mod.rs` tests (red-proven against a conf edit) |
| test_rust_validator_parity.py | COVERED 2026-09-13 — `rust/tests/validator_parity.rs` asserts every fixture verdict against `expected_python_verdicts.json`, frozen from the Python validators' last run; fixture↔golden completeness + mutation calibration tests |
| test_rust_weekend_parity.py | COVERED 2026-09-13 — `rust/tests/weekend_parity.rs` asserts corpus/candidates/aggregator payloads against `expected_python_payloads.json`, frozen from the Python pipeline's last run; kernel-rule reach + mutation calibration tests |
