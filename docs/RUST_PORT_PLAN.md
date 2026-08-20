# Rust Port of ZTools — Status, Gaps, and Bridge Plan

Status as of 2026-08-19. Deliverable: a single forward-looking plan for (1) moving
the Rust port out of `routines` and into this repo, and (2) closing the feature
gaps against the Python reference.

**Phase 0 (relocation) is DONE.** The port now lives in this repo at `rust/`
(binary `ztools`), the `routines` tree has no ztools references left, and the
launch/A-B plumbing points at the new crate. The rest of this plan is the
forward-looking bridge work.

---

## 1. Where the Rust code lives today — and why that is wrong

The Rust port is **not in this repo**. It is embedded in the `routines` scheduler
crate at `~/Projects/routines`:

```
~/Projects/routines/src/
├── ztools/                 # the four tools + helpers
│   ├── mod.rs              #  46 L
│   ├── twitter.rs          # 301 L  twitter-summarize
│   ├── embeddings.rs       # 107 L  tweet clustering (nomic-embed-text)
│   ├── weekend/            # 727 L  weekend-plan
│   │   ├── mod.rs          #  330
│   │   ├── fetch.rs        #  164
│   │   └── format.rs       #  233
│   ├── weekend_cache.rs    # 353 L  cached activities + region/exclusion filters
│   ├── image_renamer.rs    # 220 L  image-renamer
│   ├── model_eval.rs       # 241 L  model-eval
│   ├── model_health.rs     # 194 L  broken-model / packaging defect detection
│   └── *_tests.rs          # ~930 L
├── config_ztools.rs        # 172 L  ZtoolsConfig (models, timeouts, paths)
└── cli_ztools.rs           # 134 L  subcommand dispatch
```

Total ≈ 2,600 lines of production Rust, compiled into the `routines` binary as
four subcommands (`cli_args.rs:183-216`), served by `bin/` shims and shell
functions that exec `routines twitter-summarize` etc. (`build.sh` in this repo,
`~/.zshrc:154-158`).

**Why this is the wrong home:** `routines` is a scheduler/health-monitor
repository. ZTools is a separate product with its own repo, its own config, its
own gate. Every ztools change currently lives in someone else's repo, is gated
by `routines`' 400-line cap and its `cargo llvm-cov` run, and — per
`docs/PORT_PARITY.md` — the two codebases have already drifted. The port must be
a crate **inside this repo** so the tool and its parity debt live together.

### Coupling to routines (how hard extraction is)

Good news: the ztools modules are nearly self-contained. The complete list of
routines internals they touch:

| Dependency | Location | Cost to cut |
|---|---|---|
| `crate::manifest::expand_tilde` | `manifest.rs:132` — 9-line `~` expander | copy inline (~10 L) |
| `crate::config::ZtoolsConfig` | `config_ztools.rs` — fully standalone struct | move as-is |
| `crate::config::Config.ztools` field | `config.rs:210` | drop; new crate owns its config |
| `crate::cli::{Cli, Cmd}` + clap dispatch | `cli_args.rs`, `cli_run.rs:381-392` | replace with crate-local clap |
| `crate::ztools` module tree | `lib.rs:34` | move wholesale |

Everything else (reqwest, serde, chrono, comfy-table, toml, dirs) is already a
normal Cargo dependency. **Extraction is a mechanical move + a fresh `main.rs`.**

---

## 2. What the port covers today (honest scorecard)

| Tool | Rust does | Python also does |
|---|---|---|
| **twitter-summarize** | dedupe → embed-cluster → prompt → osaurus call → md file | browser collection, cookie decrypt, `--login`, MLX fallback, `--since`/state, Ctrl+C drain, rich markdown/provenance |
| **weekend-plan** | DDG scrape → weather → cached activities → scoring → table | 4-phase LLM pipeline, `enforce.py` constraint suite, supply prioritisation, review scores, caches, `--skip-web`, per-model field mapping |
| **image-renamer** | clean filename stem → optional LLM → rename | OCR (tesseract), VLM for textless images, relevance check, `--pattern`, `--no-ocr`, MLX fallback |
| **model-eval** | 4 hardcoded test cases + health gate + latency | 30 tasks with snapshots, real validators (taxes/json/text/vision/adversarial), quality scorers, adaptive timeouts, sample median, GPU lock, watchdog, report classes |
| **shared infra** | `model_health.rs`, `embeddings.rs`, minimal config | server lifecycle, LLM client/fallback/quirks/parsing, content cleaning, signal handling, Kare TUI, paths, logging |

Scale: Rust ≈ **2.6k** production LOC vs Python reference ≈ **23.8k** LOC
(`references/`, excluding tests). The port is the happy-path core of each tool —
the machinery that makes them robust, safe, and observable is still Python-only.

---

## 3. Gap inventory (Python-only features the port lacks)

### 3.1 twitter — gap list
- **Collection**: `twitter/browser.py` (441 L), `browser_launch.py`, `browser_parse.py` — camoufox/playwright scroll loop, stagnation/budget limits (`budget.py`), logged-out detection. Rust reads a JSON file/stdin/cache only. Native replacement exists (Phase 3): `camoufox-rs` drives the Camoufox Firefox binary over Juggler (`navigate`/`evaluate`/`screenshot`), so no Python browser wrapper is needed.
- **Cookies & session**: `cookies.py` (Chrome keychain decrypt), `cookies_firefox.py`, `session.py` (`--login`). Native Rust: `decrypt-cookies` (Chrome/Edge/Firefox, tested incl. macOS) or `cookie-scoop` (Chrome/Firefox/Safari, macOS keychain via `security`).
- **Resilience**: MLX fallback when the server is down (`summarize.py`, `lib/mlx_lib.py`).
- **Run state**: `--since` resolution, `last_run` state, `--use-cache`, `--fetch-only`, `--clean` (`cli.py`).
- **Signal handling**: Ctrl+C drain (`lib/signal_handling.py`).
- **Output fidelity**: `output.py` header/format, provenance/attribution.
- **Prompt**: duplicated in `twitter.rs:105` vs `eval/tasks_prompts.py` `TWITTER_PROMPT` — edits apply to half the runs.

### 3.2 weekend — gap list
- **Pipeline**: Python is a 4-phase LLM pipeline — condense weather → extract sources → draft ideas → structure JSON (`weekend/phases.py`, `prompts.py`, `llm.py`). Rust is a single extract-from-scrape call.
- **Code-side enforcement** (`weekend/enforce.py`, 492 L): drop unsourced rows, drop excluded places, drop events outside the weekend window, reconcile day↔dates, correct weather labels, flag constant columns. Rust has only `filter_exclusions` + `flag_constant_columns`.
- **Supply awareness** (`weekend/supply.py`): in-window count + prioritisation; Rust doesn't count in-window candidates, so a thin plan reads as a model problem.
- **Venues**: `fetch_fixed_venues` + `scrape_review_score` (`data.py`) — Rust uses a hardcoded activity list.
- **Caching / offline**: events+venues caches, `--skip-web`, `--use-cache`.
- **Per-model field mapping** (`get_model_field_mapping`), foundation fallback.

### 3.3 image-renamer — gap list
- **OCR**: `rename/helpers.py` — tesseract text extraction from the actual image. Rust only cleans the filename *stem*; it never reads image content. Native replacement (Phase 3): pure-Rust engines — `ocrs` (robertknight), `rusto-rs`/`ocr-rs` (PaddleOCR on Alibaba MNN), or `oar-ocr` (PP-OCRv6 + Candle, CPU/Metal).
- **VLM**: `lib/mlx_vlm.py`, `rename/llm.py` — vision LLM for images with no usable text. **Ported** (Phase 2 item 3): `rename/vlm.rs` sends base64 data-URI content parts to osaurus; no MLX fallback yet.
- **Relevance gate** (`--force`), `--pattern`, `--max-length`, `--no-ocr`, `--test`, MLX fallback.
- **Security**: untrusted framing exists in Rust (`image_renamer.rs:21`), but with no OCR path it guards an empty channel.

### 3.4 model-eval — gap list
- **Tasks**: 30 tasks with bundled snapshot data (`eval/tasks_core.py`, `tasks_prompts.py`, `eval_tasks/`) vs 4 hardcoded prompts.
- **Validators**: `lib/validators/*` — taxes grounding (rubric + grounding-scored), JSON structure, text/entity, vision, adversarial, attribution, hallucination/source-matching. Rust has `validate_file_summary` (`eval/validate.rs`) and a data-driven `Check` enum (contains / json-array-length / file-summary threshold) on 5 tasks.
- **Measurement integrity**: `eval/samples.py` median-of-5, `capabilities.py`, `prefill.py`/`signals.py` adaptive timeouts, `watchdog.py`, `memory.py`, `eval/memory` rate signal, GPU lock (`lib/gpu_lock.py`), contamination handling — none in Rust.
- **Reporting**: `report_classes*.py`, failure classification, `benchmark_quality.py`, discrimination/completeness analysis.
- **Model quirks**: `lib/llm/quirks.py` + `eval/explore_quirks.py` ("Output JSON now." for qwen3.6, etc.).

### 3.5 Shared infrastructure missing entirely in Rust
- `lib/osaurus_server.py` — server lifecycle (start/restart, PID guard)
- `lib/osaurus_lib.py`, `lib/osaurus_output.py`, `lib/osaurus_degrade.py` — API wrapper, output cleaning, degrade-with-reason
- `lib/mlx_lib.py`, `lib/mlx_vlm.py`, `lib/foundation_lib.py` — on-device fallbacks
- `lib/content_processing.py` — thinking-block removal, stats stripping (ported: `eval/clean.rs`, Phase 2 item 4)
- `lib/llm/{client,protocol,fallback,quirks,parsing,streaming,constants}.py`
- `lib/signal_handling.py`, `lib/tui.py` (Kare style), `lib/paths.py`, `lib/logging_config.py`
- `lib/model_resolve.py`, `lib/model_caps.py`, `lib/gpu_lock.py`

---

## 4. Bridge plan

Ordering rule (house standards): **foundation before features, gates before
drift, parity before replacement.** Do not grow features inside `routines` —
everything below happens after extraction, in this repo.

### Phase 0 — Relocate the port into this repo (prerequisite for everything) — DONE 2026-08-19

1. Created a single crate in this repo: `rust/` (binary `ztools`, lib+bin split
   so the modules stay `pub`-testable without dead-code warnings). Deps as listed
   in §4, plus `tempfile` (dev).
2. Moved `ztools/*` modules, `config.rs` (was `config_ztools.rs`), `cli.rs`,
   `cli_ztools.rs`, `manifest.rs` (inline `expand_tilde`) in; wrote `main.rs` and
   re-exports so `ztools::weekend` etc. resolve at the crate root.
3. Cut ztools out of `routines`: dropped `lib.rs` mod, `cli_args.rs` variants,
   `cli_run.rs` dispatch, `Config.ztools`, the module tree, `config_ztools.rs`,
   `cli_ztools.rs`, and the `comfy-table` dep; fixed the four test-file `Config`
   constructions; removed the three integration test files that referenced the
   cut API. `routines` is scheduler-only and its full gate is green.
4. Rewired launch: `build.sh` builds `rust/` and writes the `bin/` shims to
   resolve the crate binary via `cargo metadata` (never a hardcoded target path);
   `bin/ab_test` points at the `ztools` binary and runs `cargo test` in the
   crate; `docs/PORT_PARITY.md` and `README.md` updated.
5. Config seam: the binary gained a global `--config <path>` flag that loads a
   flat `ZtoolsConfig` TOML and skips the dynamic `[best_models]` override — so
   tests and CI can point URLs at stubs without reaching the operator's real
   config. Default (no `--config`) keeps `with_ztools_best_models()`.

Also fixed a pre-existing defect the move surfaced: `fetch_duckduckgo_events`
(weekend) had gained a cache fallback + empty-input guard (routines commit
`61c03d7`) that fabricated stale-dated events and broke two tests at HEAD. Reverted
to the pre-regression contract — the model is always called; unreachable search
and model yield nothing, never invented events. The weekend CLI test needed the
same fix to exercise the real LLM extraction path.

Exit criteria (met): the four commands work from a checkout without `routines`;
`grep -r ztools ~/Projects/routines/src` is empty; `bin/ab_test` runs green
against the new binary; `cargo test` in `rust/` (94 tests) and the full routines
gate both pass.

### Phase 1 — Shared config & prompt surface (kill the drift class)

Status: **partially done — twitter summarize prompt shared, in progress.**

1. ~~Move prompt texts into `conf/` read by both Rust and Python~~ — the twitter
   summarize instruction block is DONE: `conf/prompts.toml`
   `[twitter.summarize]` is canonical; Python composes `TWITTER_PROMPT` from it
   (`references/eval/tasks_prompts.py`, fixture timeline stays as eval data);
   Rust embeds a fallback (`config.rs`) and layers the file at runtime
   (`with_shared_prompts()`). **Open:** weekend schemas, rename task restatement.
2. Extend `ZtoolsConfig` to load the full `conf/config.toml` `[best_models]`
   matrix (already scaffolded as `with_ztools_best_models`), plus weekend
   exclusions and eval signal thresholds. — open.
3. **single-source-version check** — DONE for the twitter prompt:
   `test_twitter_prompt_matches_shared_conf` in `rust/src/config.rs` fails if the
   embedded fallback drifts from `conf/prompts.toml` (proven to fail, then
   reverted). Reuse the pattern for the weekend/rename prompts as they land.

### Phase 2 — Parity without new risk (bring the cheap, high-value Python logic over)

1. **twitter**: port `budget.py`/scroll-stop conditions as pure functions;
   port `output.py` markdown template; port `--since`/state logic; port signal
   drain semantics. (Browser collection is Phase 3; its Juggler client and
   cookie reader are native Rust, so the whole tool stays Python-free.)
2. **weekend**: port `enforce.py` constraint suite (all pure string/date checks);
   port `supply.py` in-window counting; port the 4-phase prompt pipeline.
   - **C8 exclusion enforcement DONE** (`rust/src/ztools/weekend/enforce.rs`):
     typographic-punctuation folding, token-subset matcher with connector and
     possessive handling, seasonal-event exceptions, `drop_excluded_places`
     wired into `weekend_cache` and the `weekend_plan` dispatch. Ported test
     suite proved-fail-first against the old weaker matcher (the "Sky Zone
     Toronto" interpolated case failed before the port).
   - **C5 weather labels + C4 constant-columns DONE**: `correct_weather_labels`
     (indoor/outdoor markers) and the suspect-conjunction
     `flag_constant_columns` (label/alias based, case- and space-insensitive,
     "rows kept" notes) ported and wired into the dispatch; `WeekendEvent` now
     carries `start_date`/`end_date`/`weather`/`duration` so enforcement sees
     the raw parsed values. Both test suites proved to fail when neutered.
   - **C3 window + day-reconcile DONE**: `find_dates_in` ported to
     `rust/src/ztools/weekend/dates.rs` (ISO + named-month shapes, explicit-year
     wins, durations excluded, shared with the enforcer so a future
     prioritiser cannot drift); `window_overlap`, `drop_events_outside_window`
      and `reconcile_day_with_dates` wired into the dispatch. Window tests proved
      to fail when `window_overlap` was neutered.
    - **C7 provenance gate DONE**: `fetch_duckduckgo_events` now returns the
      fetched corpus alongside the events; `row_is_sourced` (>= 0.6 of the name's
      significant words present in the normalised corpus) and
      `drop_unsourced_rows` ported to `enforce.rs` and run FIRST in the dispatch,
      before any shape check. Tests proved to fail when `row_is_sourced` was
      forced false.
    - **supply in-window counting DONE**: `prioritise_in_window` /
      `in_window_count` / `mentions_window` ported to
      `rust/src/ztools/weekend/supply.rs` using the SAME `find_dates_in` scanner
      the enforcer uses, wired into `fetch_duckduckgo_events`: the corpus is
      stably partitioned (marked `[THIS WEEKEND]` lines floated, nothing
      removed) before the model sees it, and the operator sees
      `in_window/total mention a date this weekend` -- the number that
      separates a supply problem from a model problem. Tests proved to fail
      when `mentions_window` was forced false.
    - **4-phase prompt pipeline DONE**: `weekend/prompts.rs` (verbatim port of
      the extract/draft/refine/structure templates + the C1-checking `render`)
      and `weekend/phases.rs` (`extract_sources` with adaptive batching,
      `draft_activities`, `refine_draft`, `structure_to_json`, `condense_weather`,
      plus `PlanContext` so the year/date-range/ages/exclusions cannot disagree
      across phases) replace the single-shot prompt in `fetch_duckduckgo_events`,
      with the old monolithic prompt retained as the fallback when the draft
      phase stalls (the Python original does the same). Weather is fetched
      BEFORE the pipeline so the draft can condition on it, matching the Python
      ordering. Phase tests prove the degrade-not-starve fallbacks
      (proved-fail-first against a neutered pass-through and a neutered refine
      fallback). Phase 2 (weekend) is complete.
3. **rename** — DONE (2026-08-20): `helpers.py` text cleaning ported verbatim
   (prefix/fence stripping is `strip_instruction_prefix`'s job; `clean_filename`
   glues dots exactly as Python; plain lowercase hex is NOT non-human-readable)
   and the VLM call path ported to the osaurus vision API via OpenAI content
   parts with a base64 data URI — NOT the Ollama `images` key, which osaurus
   silently drops. New `image_renamer_vlm_model` config gates the vision path;
   meaningless stems fall back to a clean of the stem when no vision model is
   configured. All ported behaviors proved-fail-first. `image_renamer.rs` split
   into `rename/{mod,helpers,vlm}.rs` (500-line cap).
4. **eval** — DONE (2026-08-20): `validate.py` (`validate_file_summary`) ported
   verbatim to `eval/validate.rs` (list/dict/raw-string branches, filename-echo
   guard, header bonus; thresholds use multiply-form so a 4-file list cannot
   score 100 on 3 detailed descriptions) and `content_processing` cleaning ported
   to `eval/clean.rs` (thinking-block, inline COT, stats, markdown, code-block
   extraction — `  thinking`/` response` tags byte-identical to Python, verified
   by hexdump because the display layer renders angle brackets into spaces).
   `model_eval.rs` is now data-driven (`EvalTask`/`Check` enum: contains, json
   array length, file-summary threshold) with `eval_model` cleaning raw output
   before checks. All clean.rs regexes (THINK, gemma loop, stats, code block)
   proved fail-first, plus the validate generic-desc branch and the
   cleaning-before-scoring wiring (`eval_model_cleans_thinking_before_scoring`).
   Loading the 30 `eval_tasks/` tasks is left open (data-driven harness in
   place).
5. Extend `bin/ab_test --functional` to cover every ported behavior (assert
   identical verdicts on both sides) — DONE (2026-08-20) for eval: the parity
   block runs the Python clean + validate on the same fixtures the Rust tests
   assert, and fails if either side drifts. Note: the fixture must use the real
   `  thinking`/` response` tag bytes — the angle-bracket form does not match
   the reference regex and the check goes red.

Exit criteria: the A/B harness runs green on the new behaviors; both
implementations agree on the shared surface.

### Phase 3 — The native-only wins (why the port exists)

The point of the port is one static binary, no venv, no Python startup. The two
formerly-Python seams now have native Rust replacements, so the port stays
**Python-free end to end** — the only external dependency is the Camoufox
Firefox binary itself (C++-level stealth lives in the binary, not in a driver):
1. **twitter collection**: `camoufox-rs` (github.com/9prodhi/camoufox-rs) drives
   the Camoufox Firefox binary over the Juggler protocol via `-juggler-pipe` —
   process launch, null-delimited JSON framing, `Browser`/`BrowserContext`/
   `MainFrame` wrappers, `navigate`/`evaluate`/`screenshot`, 60s request
   deadlines. This replaces `twitter/browser.py` + `browser_launch.py` and the
   Playwright wrapper entirely; the scroll loop, stagnation/budget limits and
   logged-out detection port as pure logic (Phase 2 item 1). Cookies come from
   `decrypt-cookies` (Chrome keychain + Firefox `cookies.sqlite`), replacing
   `cookies.py` / `cookies_firefox.py`. The pipeline seam stays the JSON cache
   already read by `twitter.rs`.
2. **model-eval measurement**: port the GPU-lock (`/tmp/mac-osaurus-gpu.lock`),
   watchdog heartbeat, and median-of-5 sampling to Rust — the integrity logic is
   pure; only the osaurus probe is I/O.
3. **rename OCR**: replace tesseract with a pure-Rust engine — `ocrs`
   (robertknight, Apache-2.0, ONNX models), `rusto-rs`/`ocr-rs` (PaddleOCR on
   MNN, sub-second Apple Silicon), or `oar-ocr` (PP-OCRv6 + Candle). Rust owns
   naming/security on extracted text, as today; the VLM path (`rename/vlm.rs`)
   is already native.

Exit criteria: scheduled jobs (`routines.toml`) can run the Rust binary for
summarize/rename/plan, and the twitter collection path runs native (Camoufox
binary + Juggler client + native cookie reader) — no Python anywhere in the
port.

### Phase 4 — Full evaluator parity (largest, lowest priority)

Port the real scorers/validators that rank models: taxes grounding, adversarial,
attribution, quality scorers, report classes. This is the biggest lift
(~1,000+ LOC of validator logic) and only matters if `model-eval` is a product
surface. Defer unless interactive `oeval` is used over the Python one.

### Phase 5 — Decommission

- When the A/B harness shows parity on the shared surface for N consecutive
  weeks, move `references/` Python tool entry points behind the Rust binary
  (like `bin/ztools` already does). No browser/OCR seam remains to preserve —
  the port is Python-free end to end.
- Prune completed plan work to git history (house rule: one backlog file).

---

## 5. Decisions to lock during Phase 0 (open questions)

1. **Crate layout**: single `rust/` crate, module-per-tool — decided. (Locked for
   ~2.6k LOC; revisit only if `lib/` helpers grow past one file.)
2. **Shared config mechanism**: TOML files in `conf/` (both sides parse) is the
   low-friction option; a JSON schema + codegen is overkill. Lock the file format
   in Phase 0 so prompts/best-models live in exactly one place.
3. **What happens to `routines` subcommands**: hard-drop — decided and done. The
   `routines` binary is scheduler-only; `bin/ztools` shims are the migration path.
4. **Rust gate inside this repo**: mirror `routines`' gate — `cargo fmt`,
   `clippy -D warnings`, the 500-line cap, no-emoji — wired into ztools' existing
   pre-commit/CI so the port obeys this repo's rules (see `polyglot-gate-parity`).
   The crate gate today: `cargo test` (94 tests) + `cargo clippy --all-targets
   -D warnings`; coverage on the Rust half is a follow-up, not yet wired.

---

## 6. Verification strategy

- **Behavioral A/B** (`bin/ab_test --functional`): same fixtures → same verdicts,
  Rust vs Python, for every ported behavior. Prove each new parity test can fail
  before trusting it green.
- **User-POV**: run the real `weekend-plan` / `twitter-summarize` against the
  live server; the Rust output must not regress the markdown/table shape users
  rely on (scheduled jobs read it).
- **Relocation proof**: done — `grep -r ztools ~/Projects/routines/src` is empty;
  the four commands work with `routines` uninstalled from PATH (verified via the
  `bin/` shims against the crate build).
- **Gate parity**: ztools CI runs the Rust gate exactly as CI (local gate must
  equal CI gate — a weaker local gate ships red).