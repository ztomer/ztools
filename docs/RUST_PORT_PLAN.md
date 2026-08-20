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
- **Collection**: `twitter/browser.py` (441 L), `browser_launch.py`, `browser_parse.py` — camoufox/playwright scroll loop, stagnation/budget limits (`budget.py`), logged-out detection. Rust reads a JSON file/stdin/cache only.
- **Cookies & session**: `cookies.py` (Chrome keychain decrypt), `cookies_firefox.py`, `session.py` (`--login`).
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
- **OCR**: `rename/helpers.py` — tesseract text extraction from the actual image. Rust only cleans the filename *stem*; it never reads image content.
- **VLM**: `lib/mlx_vlm.py`, `rename/llm.py` — vision LLM for images with no usable text.
- **Relevance gate** (`--force`), `--pattern`, `--max-length`, `--no-ocr`, `--test`, MLX fallback.
- **Security**: untrusted framing exists in Rust (`image_renamer.rs:21`), but with no OCR path it guards an empty channel.

### 3.4 model-eval — gap list
- **Tasks**: 30 tasks with bundled snapshot data (`eval/tasks_core.py`, `tasks_prompts.py`, `eval_tasks/`) vs 4 hardcoded prompts.
- **Validators**: `lib/validators/*` — taxes grounding (rubric + grounding-scored), JSON structure, text/entity, vision, adversarial, attribution, hallucination/source-matching. Rust checks are string `contains`.
- **Measurement integrity**: `eval/samples.py` median-of-5, `capabilities.py`, `prefill.py`/`signals.py` adaptive timeouts, `watchdog.py`, `memory.py`, `eval/memory` rate signal, GPU lock (`lib/gpu_lock.py`), contamination handling — none in Rust.
- **Reporting**: `report_classes*.py`, failure classification, `benchmark_quality.py`, discrimination/completeness analysis.
- **Model quirks**: `lib/llm/quirks.py` + `eval/explore_quirks.py` ("Output JSON now." for qwen3.6, etc.).

### 3.5 Shared infrastructure missing entirely in Rust
- `lib/osaurus_server.py` — server lifecycle (start/restart, PID guard)
- `lib/osaurus_lib.py`, `lib/osaurus_output.py`, `lib/osaurus_degrade.py` — API wrapper, output cleaning, degrade-with-reason
- `lib/mlx_lib.py`, `lib/mlx_vlm.py`, `lib/foundation_lib.py` — on-device fallbacks
- `lib/content_processing.py` — thinking-block removal, stats stripping
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

Per `PORT_PARITY.md` "standing hazard", the structural fix is shared config, not
parallel copies:
1. Move prompt texts (`TWITTER_PROMPT`, weekend schemas, rename task restatement)
   into `conf/` (e.g. `conf/prompts.toml`) read by both Rust and Python.
2. Extend `ZtoolsConfig` to load the full `conf/config.toml` `[best_models]`
   matrix (already scaffolded as `with_ztools_best_models`), plus weekend
   exclusions and eval signal thresholds.
3. Add a **single-source-version check**: a test that greps the Rust binary's
   embedded prompt against `conf/prompts.toml` and fails on drift (structural
   gate, not a "remember to" note).

### Phase 2 — Parity without new risk (bring the cheap, high-value Python logic over)

1. **twitter**: port `budget.py`/scroll-stop conditions as pure functions;
   port `output.py` markdown template; port `--since`/state logic; port signal
   drain semantics. (No browser yet — collection stays Python until Phase 3.)
2. **weekend**: port `enforce.py` constraint suite (all pure string/date checks);
   port `supply.py` in-window counting; port the 4-phase prompt pipeline.
3. **rename**: port `helpers.py` text cleaning (pure); port VLM call path to the
   osaurus vision API (no local MLX yet).
4. **eval**: port `validate.py` JSON validator + `content_processing` cleaning
   (thinking-block removal); load the 30 tasks from `eval_tasks/` instead of
   hardcoded cases.
5. Extend `bin/ab_test --functional` to cover every ported behavior (assert
   identical verdicts on both sides).

Exit criteria: the A/B harness runs green on the new behaviors; both
implementations agree on the shared surface.

### Phase 3 — The native-only wins (why the port exists)

The point of the port is one static binary, no venv, no Python startup. Land
the features that only make sense native:
1. **twitter collection**: a Rust scroll driver for camoufox? No — collection
   needs a real browser; keep it Python-side via the launcher (exec
   `uv run -m twitter --fetch-only` then feed JSON to the Rust summarizer).
   The pipeline seam is the JSON cache already read by `twitter.rs`.
2. **model-eval measurement**: port the GPU-lock (`/tmp/mac-osaurus-gpu.lock`),
   watchdog heartbeat, and median-of-5 sampling to Rust — the integrity logic is
   pure; only the osaurus probe is I/O.
3. **rename OCR**: keep tesseract via `uv run` seam (same pattern as twitter
   collection); Rust owns naming/security once text is extracted.

Exit criteria: scheduled jobs (`routines.toml`) can run the Rust binary for
summarize/rename/plan; Python stays only for browser/OCR seams.

### Phase 4 — Full evaluator parity (largest, lowest priority)

Port the real scorers/validators that rank models: taxes grounding, adversarial,
attribution, quality scorers, report classes. This is the biggest lift
(~1,000+ LOC of validator logic) and only matters if `model-eval` is a product
surface. Defer unless interactive `oeval` is used over the Python one.

### Phase 5 — Decommission

- When the A/B harness shows parity on the shared surface for N consecutive
  weeks, move `references/` Python tool entry points behind the Rust binary
  (like `bin/ztools` already does), keep Python only for browser/OCR seams.
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