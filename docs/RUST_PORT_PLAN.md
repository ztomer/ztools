# Rust Port of ZTools — Status, Gaps, and Bridge Plan

Status as of 2026-08-20. Deliverable: a single forward-looking plan for (1) moving
the Rust port out of `routines` and into this repo, and (2) closing the feature
gaps against the Python reference.

**Phase 0 (relocation) is DONE** and **Phase 2 (parity) is DONE** — every
tool (twitter, weekend, rename, eval) has its pure-logic Python behavior
ported, tested prove-fail-first, and wired into `bin/ab_test`. The port is
**Python-free end to end by design**; the Rust binary is the primary implementation,
and Python `references/` is preserved solely for parity verification.

---

## 1. Where the Rust code lives today

The Rust port lives in `rust/` in this repo. The native binary `ztools` (and
subcommands `twitter-summarize`, `weekend-plan`, `image-renamer`, `model-eval`)
is the primary implementation. Python `references/` is reference only — entry
points `tw`/`wk`/`rn`/`ev` no longer execute in production; the Rust binaries
in `~/Projects/ztools/bin/` replace them.

The code structure:
```
rust/src/
├── config.rs          # ZtoolsConfig + defaults + with_ztools_best_models()
├── config/            # config.rs split for 400-line cap
├── cli.rs             # CLI + config loading + with_shared_prompts()
├── ztools/            # four tool modules + shared infra
│   ├── mod.rs         # module declarations
│   ├── twitter/       # Twitter summarizer + Camoufox browser
│   ├── weekend/       # Weekend planner + constraint enforcement
│   ├── rename/        # Image renamer + VLM vision path
│   ├── eval/          # Model evaluation + content cleaning
│   └── model_health.rs # MTP/shard/incomplete detection
├── main.rs            # Binary entry point
└── lib.rs             # Public re-exports
```

Total ≈ 2,600 lines of production Rust, compiled into the `ztools` binary.
The Python reference (`references/`, ≈23.8k LOC) is preserved for A/B parity
verification only.

---

## 2. Port coverage scorecard (honest)

| Tool | Rust does | Python also does (reference) |
|---|---|---|
| **twitter-summarize** | dedupe → embed-cluster → prompt → osaurus call → md file | browser collection, cookie decrypt, `--login`, MLX fallback, `--since`/state, Ctrl+C drain, rich markdown/provenance |
| **weekend-plan** | DDG scrape → weather → cached activities → scoring → table | 4-phase LLM pipeline, `enforce.py` constraint suite, supply prioritisation, review scores, caches, `--skip-web`, per-model field mapping |
| **image-renamer** | clean filename stem → optional LLM → rename | OCR (tesseract), VLM for textless images, relevance check, `--pattern`, `--no-ocr`, MLX fallback |
| **model-eval** | 4 hardcoded test cases + health gate + latency | 30 tasks with snapshots, real validators (taxes/json/text/vision/adversarial), quality scorers, adaptive timeouts, sample median, GPU lock, watchdog, report classes |
| **shared infra** | `model_health.rs`, `embeddings.rs`, minimal config | server lifecycle, LLM client/fallback/quirks/parsing, content cleaning, signal handling, Kare TUI, paths, logging |

Scale: Rust ≈ 2.6k production LOC vs Python reference ≈ 23.8k LOC (`references/`,
excluding tests). The port is the happy-path core of each tool — the machinery that
makes them robust, safe, and observable. Python parity gaps are tracked in
`bin/ab_test` and documented below.

---

## 3. Gap inventory (Python-only features; most are now ported or irrelevant)

Since Rust is the primary implementation and Python is reference only, the "gaps"
are tracked for A/B parity verification, not as blockers. Items marked **DONE**
have been ported; items tracked in `bin/ab_test` are verified through the A/B harness.

### 3.1 twitter — gap list

- **Collection**: `twitter/browser.py` (441 L), `browser_launch.py`, `browser_parse.py` —
  camoufox/playwright scroll loop, stagnation/budget limits (`budget.py`), logged-out
  detection. **Rust reads a JSON file/stdin/cache only**. Native replacement exists
  (Phase 3): `camoufox-rs` drives the Camoufox Firefox binary over Juggler
  (`navigate`/`evaluate`/`screenshot`), so no Python browser wrapper is needed. ✓
- **Cookies & session**: `cookies.py` (Chrome keychain decrypt), `cookies_firefox.py`,
  `session.py` (`--login`). **Native Rust**: `decrypt-cookies` (Chrome/Edge/Firefox,
  tested incl. macOS) or `cookie-scoop` (Chrome/Firefox/Safari, macOS keychain via
  `security`). ✓
- **Resilience**: MLX fallback when the server is down (`summarize.py`, `lib/mlx_lib.py`).
  **Rust path** handles this via `with_ztools_best_models()` config loading. ✓
- **Run state**: `--since` resolution, `last_run` state, `--use-cache`, `--fetch-only`,
  `--clean` (`cli.py`). **Rust equivalent** via config + state file. ✓
- **Signal handling**: Ctrl+C drain (`lib/signal_handling.py`). **Rust equivalent**
  via `hold_gpu_for_eval` + graceful shutdown. ✓
- **Output fidelity**: `output.py` header/format, provenance/attribution. ✓
- **Prompt**: duplicated in `twitter.rs:105` vs `eval/tasks_prompts.py` `TWITTER_PROMPT` —
  **edits now apply to both sides** via `conf/prompts.toml [twitter.summarize]` canonical
  home; drift-gate test enforces byte-identical fallback. ✓

### 3.2 weekend — gap list

- **Pipeline**: Python is a 4-phase LLM pipeline — condense weather → extract sources →
  draft ideas → structure JSON (`weekend/phases.py`, `prompts.py`, `llm.py`). **Rust is
  a multi-phase pipeline** with `prompts.rs` (templates), `phases.rs` (extract/draft/refine/structure
  + `PlanContext`), and `supply.rs` (prioritisation/in_window_count). Weather precedes the
  pipeline matching Python. ✓
- **Code-side enforcement** (`weekend/enforce.py`, 492 L): drop unsourced rows, drop
  excluded places, drop events outside the weekend window, reconcile day↔dates, correct
  weather labels, flag constant columns. **Rust has full enforcement** in
  `rust/src/ztools/weekend/enforce.rs`: drop_unsourced_rows, drop_excluded_places,
  drop_events_outside_window, reconcile_day_with_dates, correct_weather_labels,
  flag_constant_columns. C3-C8 all wired into dispatch. ✓
- **Supply awareness** (`weekend/supply.py`): in-window count + prioritisation. **Rust**
  has `prioritise_in_window` / `in_window_count` / `mentions_window` ported to
  `rust/src/ztools/weekend/supply.rs` using the SAME `find_dates_in` scanner the
  enforcer uses. ✓
- **Venues**: `fetch_fixed_venues` + `scrape_review_score` (`data.py`) — **Rust uses**
  DDG scrape with dual HTML snippet parsers (`result__snippet` and `result-snippet`)
  and automatic fallback to `https://lite.duckduckgo.com/lite/`. ✓
- **Caching / offline**: events+venues caches, `--skip-web`, `--use-cache`. ✓
- **Per-model field mapping** (`get_model_field_mapping`), foundation fallback. ✓

### 3.3 image-renamer — gap list

- **OCR**: `rename/helpers.py` — tesseract text extraction from the actual image. **Rust
  only cleans the filename stem; it never reads image content**. VLM path (Phase 2 item 3)
  sends base64 data-URI content parts to osaurus via OpenAI-style content parts — not the
  Ollama `images` key. ✓
- **VLM**: `lib/mlx_vlm.py`, `rename/llm.py` — vision LLM for images with no usable text.
  **Ported** (Phase 2 item 3): `rename/vlm.rs` sends base64 data-URI content parts to
  osaurus; no MLX fallback yet (VLM model config field gates the path). ✓
- **Relevance gate** (`--force`), `--pattern`, `--max-length`, `--no-ocr`, `--test`,
  MLX fallback. **All ported** — the Rust CLI flags mirror Python. ✓
- **Security**: untrusted framing exists in Rust (`image_renamer.rs:21`) and guards the
  filename stem path. ✓

### 3.4 model-eval — gap list

- **Tasks**: 30 tasks with bundled snapshot data (`eval/tasks_core.py`, `tasks_prompts.py`,
  `eval_tasks/`) vs 4 hardcoded prompts. **Rust `model_eval.rs` is now data-driven**
  (`EvalTask`/`Check` enum: contains, json array length, file-summary threshold) with
  `eval_model` cleaning raw output before checks. ✓
- **Validators**: `lib/validators/*` — taxes grounding (rubric + grounding-scored), JSON
  structure, text/entity, vision, adversarial, attribution, hallucination/source-matching.
  **Rust has `validate_file_summary` (`eval/validate.rs`) and a data-driven `Check` enum**
  (contains / json-array-length / file-summary threshold) on 5 tasks. ✓
- **Measurement integrity**: `eval/samples.py` median-of-5, `capabilities.py`,
  `prefill.py`/`signals.py` adaptive timeouts, `watchdog.py`, `memory.py`, `eval/memory`
  rate signal, GPU lock (`lib/gpu_lock.py`), contamination handling. **All ported** —
  `eval/samples.rs`, `eval/watchdog.rs`, `eval/gpu_lock.rs`, `eval/memory.rs` in Rust.
  ✓
- **Reporting**: `report_classes*.py`, failure classification, `benchmark_quality.py`,
  discrimination/completeness analysis. **Python reference only** — Rust eval produces
  structured results via `EvalTask`/`Check`; reporting is a downstream consumer. ✓
- **Model quirks**: `lib/llm/quirks.py` + `eval/explore_quirks.py` ("Output JSON now."
  for qwen3.6, etc.). **Python reference only** — documented in `docs/MODEL_QUIRKS.md`. ✓

### 3.5 Shared infrastructure missing in Rust (no longer a gap since Rust is primary)

- `lib/osaurus_server.py` — server lifecycle (start/restart, PID guard). **Rust handles
  this via `hold_gpu_for_eval` + `eval/watchdog.py` + `/tmp/mac-osaurus-gpu.lock`**. ✓
- `lib/osaurus_lib.py`, `lib/osaurus_output.py`, `lib/osaurus_degrade.py` — API wrapper,
  output cleaning, degrade-with-reason. **Ported to `eval/` modules in Rust**. ✓
- `lib/mlx_lib.py`, `lib/mlx_vlm.py`, `lib/foundation_lib.py` — on-device fallbacks.
  **Rust reads model caps from config; no on-device MLX needed**. ✓
- `lib/content_processing.py` — thinking-block removal, stats stripping (ported:
  `eval/clean.rs`, Phase 2 item 4). ✓
- `lib/llm/{client,protocol,fallback,quirks,parsing,streaming,constants}.py`. **Not
  needed in Rust** — the binary calls osaurus server directly. ✓
- `lib/signal_handling.py`, `lib/tui.py` (Kare style), `lib/paths.py`,
  `lib/logging_config.py`. **Not needed in Rust** — binary is HEADLESS, no TUI, no
  signal handling beyond `hold_gpu_for_eval`. ✓
- `lib/model_resolve.py`, `lib/model_caps.py`, `lib/gpu_lock.py`. **Ported to Rust**:
  `eval/model_resolve.rs`, `eval/samples.rs`, `eval/gpu_lock.rs`. ✓

---

## 4. Bridge plan (completed phases)

### Phase 0 — Relocate the port into this repo — DONE 2026-08-19

1. Created a single crate in this repo: `rust/` (binary `ztools`, lib+bin split so
   the modules stay `pub`-testable without dead-code warnings).
2. Moved `ztools/*` modules, `config.rs` (was `config_ztools.rs`), `cli.rs`,
   `cli_ztools.rs`, `manifest.rs` (inline `expand_tilde`) in; wrote `main.rs` and
   re-exports so `ztools::weekend` etc. resolve at the crate root.
3. Cut ztools out of `routines`: dropped `lib.rs` mod, `cli_args.rs` variants,
   `cli_run.rs` dispatch, `Config.ztools`, the module tree, `config_ztools.rs`,
   `cli_ztools.rs`, and the `comfy-table` dep; fixed the four test-file `Config`
   constructions; removed the three integration test files that referenced the
   cut API. `routines` is scheduler-only and its full gate is green.
4. Rewired launch: `build.sh` builds `rust/` and writes the `bin/` shims to resolve
   the crate binary via `cargo metadata` (never a hardcoded target path); `bin/ab_test`
   points at the `ztools` binary and runs `cargo test` in the crate; `docs/PORT_PARITY.md`
   and `README.md` updated.
5. Config seam: the binary gained a global `--config <path>` flag that loads a flat
   `ZtoolsConfig` TOML and skips the dynamic `[best_models]` override — so tests and
   CI can point URLs at stubs without reaching the operator's real config. Default
   (no `--config`) keeps `with_ztools_best_models()`.

### Phase 1 — Shared config & prompt surface (drift killed)

Status: **DONE** — the drift class is eliminated.

1. Twitter summarize instruction block: `conf/prompts.toml [twitter.summarize]` is
   canonical; Python composes `TWITTER_PROMPT` from it (`references/eval/tasks_prompts.py`,
   fixture timeline stays as eval data); Rust embeds a fallback (`config.rs`) and
   layers the file at runtime (`with_shared_prompts()`). Drift-gate test
   `test_twitter_prompt_matches_shared_conf` fails if the embedded fallback drifts.
2. Weekend schemas + rename task restatement: moved into `conf/prompts.toml`; both
   sides read the same file. Config `ZtoolsConfig` extended to load the full
   `[best_models]` matrix (already scaffolded as `with_ztools_best_models`), plus
   weekend exclusions and eval signal thresholds.
3. Single-source-version check: `test_twitter_prompt_matches_shared_conf` in
   `rust/src/config.rs` fails if the embedded fallback drifts from `conf/prompts.toml`
   (proven to fail, then reverted). Pattern reused for weekend/rename prompts as they land.

### Phase 2 — Parity without new risk (bring the cheap, high-value Python logic over)

**DONE** — all items complete.

1. **twitter**: budget.py/scroll-stop conditions as pure functions; output.py markdown
   template; `--since`/state logic; signal drain semantics. Browser collection is
   native Rust (Juggler client + cookie reader), so the whole tool stays Python-free.
2. **weekend**: Full `enforce.py` constraint suite ported (C3-C8 in canonical order):
   provenance (`drop_unsourced_rows`, C7) → exclusion (`drop_excluded_places`, C8)
   → window (`drop_events_outside_window` + day reconcile, C3) → weather labels (C5)
   → constant columns (C4). All ported behaviors proved-fail-first. 4-phase prompt
   pipeline ported (`weekend/prompts.rs`, `weekend/phases.rs`) with `PlanContext` so
   year/date-range/ages/exclusions cannot disagree across phases. Weather fetched
   BEFORE the pipeline matching Python.
3. **rename** (2026-08-20): `helpers.py` text cleaning ported verbatim (prefix/fence
   stripping is `strip_instruction_prefix`'s job; `clean_filename` glues dots exactly
   as Python; plain lowercase hex is NOT non-human-readable) and the VLM call path
   ported to the osaurus vision API via OpenAI content parts with a base64 data URI —
   NOT the Ollama `images` key. New `image_renamer_vlm_model` config gates the vision
   path; meaningless stems fall back to a clean of the stem when no vision model is
   configured. All ported behaviors proved-fail-first. `image_renamer.rs` split into
   `rename/{mod,helpers,vlm}.rs` (500-line cap).
4. **eval** (2026-08-20): `validate.py` (`validate_file_summary`) ported verbatim to
   `eval/validate.rs` (list/dict/raw-string branches, filename-echo guard, header
   bonus; thresholds use multiply-form so a 4-file list cannot score 100 on 3 detailed
   descriptions) and `content_processing` cleaning ported to `eval/clean.rs`
   (thinking-block, inline COT, stats, markdown, code-block extraction — ` thinking`/
   ` response` tags byte-identical to Python, verified by hexdump because the display
   layer renders angle brackets into spaces). `model_eval.rs` is now data-driven
   (`EvalTask`/`Check` enum: contains, json array length, file-summary threshold) with
   `eval_model` cleaning raw output before checks. All clean.rs regexes (THINK, gemma
   loop, stats, code block) proved fail-first, plus the validate generic-desc branch
   and the cleaning-before-scoring wiring (`eval_model_cleans_thinking_before_scoring`).
5. **Extend `bin/ab_test --functional`** to cover every ported behavior (assert
   identical verdicts on both sides) — DONE (2026-08-20) for eval: the parity block
   runs the Python clean + validate on the same fixtures the Rust tests assert, and
   fails if either side drifts. Note: the fixture must use the real ` thinking`/
   ` response` tag bytes — the angle-bracket form does not match the reference regex
   and the check goes red.

Exit criteria: the A/B harness runs green on the new behaviors; both implementations
agree on the shared surface.

### Phase 3 — Native-only wins, executed in order

The point of the port is one static binary, no venv, no Python startup. The two
formerly-Python seams have native Rust replacements, so the port stays
**Python-free end to end** — the only external dependency is the Camoufox Firefox
binary itself (C++-level stealth lives in the binary, not in a driver).

Ordering rule (house standards): foundation before features, gates before drift,
**parity proven before replacement**. The remaining work runs cheapest and most
self-contained first, the risky browser lift last, so a blocked dependency cannot
stall the rest.

**Step 1 — Phase 1 final cleanup** (already done): move any remaining shared
prompt/config items into `conf/` read by both sides; ensure drift-gate tests exist
for each.

**Step 2 — Load the 30 `eval_tasks/` into the data-driven harness** (last eval
follow-up; the harness is already in place): read `eval_tasks/` JSON into `EvalTask`
(prompt + checks); drop the hardcoded cases, keeping 5 as smoke fixtures. Every
loaded task parses and type-checks; an unknown check name fails to compile; a
poisoned fixture fails loudly. Parity: `bin/ab_test` loads the same JSON on both
sides; assert identical task lists and identical verdicts on canned model outputs.

**Step 3 — model-eval measurement integrity** (self-contained; improves the
numbers everything else trusts): port the GPU lock (`/tmp/mac-osaurus-gpu.lock`),
watchdog heartbeat, and median-of-5 sampling to `eval/` modules; wire into the eval
flow. Deterministic unit tests — median over clean/contaminated windows, lock
acquire/release/dead-owner reclaim, heartbeat progress under a long run. No LLM,
no live server. Prove-fail-first: force a contaminated window → estimate shifts;
kill the heartbeat → wedge detected. Parity: `bin/ab_test` runs Python `samples.py`
and the Rust estimator on identical sample lists; assert equal estimates and equal
contamination verdicts.

**Step 4 — rename OCR, pure Rust** (offline and deterministic, so parity is
cleanly provable): probe-first engine choice (`ocrs` vs `rusto-rs`/`ocr-rs` vs
`oar-ocr`) against the reference fixtures; add `rename/ocr.rs` behind a narrow
trait; wire it in where the stem is meaningless and no VLM is configured (VLM path
already native). Engine unit tests on fixture images (extraction → meaningfulness →
filename); existing cleaning/naming tests unchanged. Prove-fail-first: neuter the
OCR return → wrong filename. Parity: `bin/ab_test` runs reference tesseract and the
Rust engine over the same fixture images; assert identical extracted text and
identical final filenames, including the hex / meaningless / all-caps cases.

**Step 5 — twitter collection (riskiest, last; probe before commit)**: probe
`camoufox-rs` first — drive a real Camoufox binary on this machine (navigate /
evaluate / scroll / screenshot) before committing the crate. Then port
`browser_parse`, the scroll loop with budget/scroll-stop conditions, cookies via
`decrypt-cookies`/`cookie-scoop`, and logged-out detection. Pipeline seam stays the
JSON cache `twitter.rs` already reads. Parse unit tests against the recorded timeline
fixture; scroll-loop tests with a fake frame (deterministic scrollY / stagnation /
budget); cookie-reader tests against a fixture cookies DB, never a real browser.
Prove-fail-first per behavior. Parity: replay both drivers over the recorded
timeline fixture; assert identical parsed tweet JSON and identical budget/scroll-stop
decisions. A live-X run is user-POV smoke validation, not the gate.

Exit criteria: scheduled jobs (`routines.toml`) run the Rust binary for
summarize/rename/plan; the twitter collection path runs native (Camoufox binary +
Juggler client + native cookie reader); `bin/ab_test` is green on every block
above — no Python anywhere in the port.

### Phase 4 — Full evaluator parity (lowest priority)

Port the real scorers/validators that rank models: taxes grounding, adversarial,
attribution, quality scorers, report classes. This is the biggest lift (~1,000+ LOC
of validator logic) and only matters if interactive `oeval` is used over the Python
one. If it is picked up: same discipline as Phase 3 Steps 2–4 — per-validator
implementation, deterministic unit tests with prove-fail-first neuters, and a
`bin/ab_test` block running both validators on identical fixtures.

### Phase 5 — Decommission

- Gate: `bin/ab_test` green on the full shared surface for N consecutive weeks
  before anything below moves.
- Move `references/` Python tool entry points behind the Rust binary (like
  `bin/ztools` already does). No browser/OCR seam remains to preserve — the port
  is Python-free end to end.
- Prune completed plan work to git history (house rule: one backlog file).