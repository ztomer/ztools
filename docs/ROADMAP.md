# Architecture & Implementation Roadmap — `ztools`

_Forward-looking backlog only; completed plans are pruned to git history (house rule #13).
The 2026-08-21 "port COMPLETE" record below this plan was retired 2026-09-13: it claimed
`references/` survives for parity only, which is wrong. The Rust binary shells out to
`references/twitter/cli.py` for login + live timeline collect (`twitter/browser.rs`,
`pyenv::apply_pythonpath`), and `routines.toml` runs `references/routines_status.py` on
schedule. This file now plans the full retirement of the Python implementation._

---

## Goal: zero Python at runtime

Production runs the static Rust binary with no interpreter, no `.venv`, no `uv`.
`tools/*.py` dev gates STAY Python (house precedent: `gates_of_heck` checkers stay
Python; gates are not the product). Everything else under `references/` and
`eval_tasks/` goes, with its behavior either ported or explicitly deferred (B4).

## Work inventory (what the Python still owns)

| # | Python owner | LOC | Rust replacement | Difficulty |
|---|---|---|---|---|
| 1 | `twitter/browser.py` collect loop: scroll, tab click, GraphQL response capture (`HomeTimeline` / `HomeLatestTimeline`), stop conditions | ~440 + `browser_parse.py` 81 (parse already ported: `browser_parse.rs`) | `camoufox-rs` driver behind the existing `LiveBrowserCollector` runner seam (`twitter/browser.rs`) | High — live-site behavior, needs A/B |
| 2 | `twitter/session.py` + `browser_launch.py`: persistent profile, headed `--login`, session-cookie check | ~120 + 121 | `camoufox-rs` launch with `profile_dir` + headless toggle; `get_cookies()` for the session check; `rusqlite` read of `cookies.sqlite` for `has_saved_session` without launch | Medium |
| 3 | `twitter/cookies.py` (+ `cookies_firefox.py`): session-cookie name, Chrome-store fallback | ~217 + 120 | Session-cookie check via `get_cookies()`; persistent-profile design already removed the Chrome-keychain path — confirm no caller needs it, then drop (do not port keychain AES) | Low-Medium |
| 4 | Weekend search: inline `ddgs` script (`weekend/mod.rs::collect_snippets_external`) + `weekend/followup.py` aggregator fetch (bs4) | script ~30 + followup | Direct HTTP via existing `reqwest`: DDG html endpoint for snippets; `weekend/fetch.rs` extended for bounded aggregator fetch + text extraction (no bs4 in Rust; extract from raw HTML) | Medium |
| 5 | `references/routines_status.py` (160): newest-plan hollow/stale check via `eval.report_classes` parsers | **DONE** (2b): `ztools/status.rs` + `weekend/report.rs` ports; A/B-identical against Python on real plans | Low-Medium |
| 6 | `eval_tasks/` (141): `run.py`/`analyze.py`/`validators.py` + `data/*/*.json` rubrics | **DONE** (2b): `run.py`/`validators.py`/`analyze.py`/`__main__.py` deleted (no importers, verified by rg); `__init__.py` + `README.md` + `data/` remain (kill-criterion shim, documented) | Low |
| 7 | `references/tests/` (138 files, ~34k LOC) | tests | Triage per file: covered-by-Rust-test / re-express / delete-with-code (see Phase 4) | High-effort, low-risk |
| 8 | `pyproject.toml` packaging + `[project.scripts]` (`tw`/`wk`/`rn`/`ev`) + `.venv` + `uv.lock` | config | Delete at cutover; `camoufox`/`playwright`/`ddgs`/`cryptography` Python deps die with it | Trivial, LAST |

## Phase 0 — Spikes (time-boxed, B3: two sessions each, then decide)

RESULTS 2026-09-13 — all three GREEN (mechanism proven in `/tmp/s1spike`, kept out
of the repo; rerun on demand). S1's kill criterion did NOT fire: proceed.

- **S1 GREEN:** launch/headless/persistent-profile, navigate, evaluate, network capture,
  body fetch, cookie round-trip, clean shutdown, relaunch — all via `camoufox-rs`
  against the pip-fetched Camoufox binary (v152.0.4-beta.29). Three integration
  findings the real port must honor: (a) Juggler `responseReceived` params are FLAT
  (no `response` envelope, no url) — correlate `requestId → url` from
  `requestWillBeSent`, filter on the HomeTimeline/HomeLatestTimeline URL there;
  (b) `get_response_body` returns RAW bytes — add brotli/gzip decode (Playwright
  decoded transparently; x.com serves `br`); (c) profile-lock contention exits 0
  with empty stderr — detect (second launch while first alive) and hard-error,
  never silently run sessionless.
- **S2 GREEN (mechanics):** `LaunchConfig.headless` toggle + `profile_dir` persist
  (`cookies.sqlite` written, readable via sqlite3, relaunch connects). x.com-specific
  remainder (headed login UX, session-cookie check via `get_cookies()`) is
  straightforward — needs the user's live login at Phase 3, not more spiking.
- **S3 GREEN:** POST `https://html.duckduckgo.com/html/` (`q`/`b`/`l` form, browser UA)
  returns 9–10 parsed results on all five probe queries; snippet bodies are
  same-quality event corpus as `ddgs` (less nav chrome). GET is challenged — POST
  only. Parity bar for Phase 3 is corpus quality, NOT URL-set equality (overlap
  1–5/8 after `uddg` unwrap; ranking varies per request on both sides).
- **Binary resolution (Phase 1 must-fix):** `camoufox-rs` defaults only to
  `~/.cache/camoufox/camoufox`, which misses the macOS pip location
  (`~/Library/Caches/camoufox/browsers/official/*/Camoufox.app/Contents/MacOS/camoufox`).
  The port resolves `CAMOUFOX_BIN` → pip cache glob → `~/.cache` fallback, in that
  order, and fails loud naming every path tried (`pyenv` discipline).

- **S1: GraphQL capture via `camoufox-rs`.** Response listener on `HomeTimeline` /
  `HomeLatestTimeline` returning bodies identical to the Playwright capture for one
  fixed account + time window. Primitives confirmed present (`set_request_interception`,
  `get_response_body`, `evaluate`, tab/mouse dispatch).
- **S2: Headed login + persistent profile.** `profile_dir` launch headed, manual sign-in,
  relaunch headless, `get_cookies()` shows the session cookie; `cookies.sqlite` marker
  readable via `rusqlite` without launch.
- **S3: DDG without `ddgs`.** `reqwest` GET against the DDG html endpoint returns
  title/body/href triples matching the helper for five fixed queries; one bounded
  aggregator fetch extracts event text without bs4.
- **Kill criterion (B1):** if S1 cannot see response bodies, STOP the whole plan and keep
  the Python collector — a blind rewrite of the riskiest path is how the timeline
  silently goes empty. Record the finding (D3) and do Phases 2b/3 only (items 4–8).

## Phase 1 — Foundation (A4)

DONE 2026-09-13: `vendor/camoufox-rs` vendored from the **app_updates copy** (NOT
routines upstream — the app_updates copy carries the two macOS-compat patches the
spike ran against: `pipe` for `pipe2`, stdout-based readiness sentinel; the
routines copy has neither and does not compile on macOS). Noted as tech debt: the
two vendor copies have drifted with no direction recorded; reconciling them is a
separate job for whoever owns `camoufox-rs`. `vendor/` stays gitignored per house
convention (local-only, like app_updates). Deps added: `camoufox` (path),
`rusqlite` (bundled), `brotli` + `flate2` (GraphQL bodies arrive `br`-encoded;
Playwright decoded transparently). `twitter/browser_bin.rs`: `CAMOUFOX_BIN` →
macOS pip cache glob (newest wins) → `~/.cache` fallback, hard error naming every
path tried; 4 unit tests (fail-first proven by mutation), clippy clean, 480/480
lib tests green, live `resolve()` proven against this machine via throwaway test.

- Vendor `camoufox-rs` (copy from `app_updates/vendor/camoufox-rs`, note the fork
  relationship in its README the way the Camoufox binary resolution is shared:
  `CAMOUFOX_BIN` / `~/.cache/camoufox/camoufox`, same order as `browser_launch.py`).
- Add `rusqlite` (bundled) + nothing else: `reqwest` already present.
- Resolve the `profile_dir` + `headless` config surface once (`twitter.toml` keys
  `TWITTER_PROFILE_DIR`, `TWITTER_LOGIN_*` honored as-is so the cutover is drop-in).

## Phase 2 — Port behind seams (A1: additive, Python still default)

PROGRESS 2026-09-13:
- `twitter/browser_bin.rs` (Phase 1, done): binary resolution + 4 tests.
- `twitter/collect.rs` (2a-i, done): scroll stop-state machine, char-safe dedup
  post-pass (exact + content keys, oldest-first), logged-out check — 10 tests,
  fail-first proven by mutation, clippy clean.
- `twitter/cookies.rs` (2a-i, done): Firefox-family roots (zen/Zen/Firefox/
  LibreWolf/Waterfox, canonicalized dedupe), WAL-safe copy-then-read via
  `rusqlite`, container/empty/suffix filtering, ms-expiry normalize, session
  select with guest fallback — 6 tests + live read of the real Zen profile
  (session present). Chrome keychain read: deferred (see item 3).
- `twitter/native.rs` (2a-ii, done): `scratch_dir` profile Builder +
  Collector impl; scroll loop handles JSON-number evals (new `eval_value`);
  `ScrollLimits::from_env_or_default()` honors `TWITTER_MAX_RUNTIME_S`.
- `twitter/capture.rs` (2a-ii, done): request interception catches timeline
  channels; `decode_body` sniffs first (JSON passthrough / gzip magic `1f 8b` →
  brotli last) — x.com headers claim gzip but bodies are br, found live.
- 2a LIVE PROOF 2026-09-13: `TWITTER_COLLECTOR=native` fetched 4 tweets on the
  real Zen profile (`→ ✓ 4 tweets fetched and cached.`). 506→509 lib tests,
  clippy 0 warnings, fmt clean.
- `weekend/search.rs` (2b, done): direct DDG html POST (`q`/`b`/`l` form, Safari
  UA — the spike's green path) replaces the `ddgs` subprocess; `SearchResult`
  triples; challenge/transport failures reported, never silent-empty. 4 tests.
- `weekend/followup.rs` (2b, done): port of `followup.py` — aggregator-marker
  detection via `normalize_for_match`, bounded fetch (3 pages, 8s, 4000 chars),
  raw-HTML tag stripping (no bs4), `- [{title}] {line}` candidate lines. 6 tests.
- `weekend/fetch.rs` corpus (2b, done): `Vec<SearchResult>` threads, dedupe by
  title+body, region evidence on title+body, lines are `- {title or "Event"}: {body}`,
  `follow_aggregators` output appended. Corpus test still green on the snippet-only
  mock (empty titles → `- Event:` fallback). `mod.rs` now 354 lines (< 500 cap).
- LIVE PROOF 2026-09-13: `weekend-plan` against real DDG end-to-end — most queries
  parsed (a per-query bot wall still challenges a few; those report and the rest
  carry the corpus), `→ Followed 3 event listing page(s)`, `→ Candidates: 4/75`.
- `weekend/report.rs` (2b, done): port of `eval.report_classes.py`'s read side —
  `parse_tables` (Python dict.setdefault semantics: key emitted on first header/row,
  in first-use order), `transient_rows`/`fixed_rows`, `parse_window_from_filename`
  (kept byte-for-byte incl. the year-boundary start>end quirk; `%B %d %Y` via chrono).
  5 tests (window name shape, separator-skip, section select, unparseable rejects).
- `ztools/status.rs` (2b, done): port of `routines_status.py` — `upcoming_weekend`
  (Math#1: `today - (weekday-4)` on Python weekday() Mon=0), newest `weekend_plan_*.md`
  by mtime, hollow/stale → `attention`, missing store → `unknown`, JSON per
  STATUS_CONTRACT. Reads `WEEKEND_OUTPUT_DIR` else `~/Documents` (NOT the `weekend_plans/`
  store dir; matches `weekend.cli.OUTPUT_DIR_PATH`). Wired as `ztools status` (clap
  `Cmd::Status`, `cli_ztools::status`). 5 tests, env-mutating ones `#[serial_test::serial]`.
- LIVE PROOF 2026-09-13: `cargo run -- status` vs `.venv/bin/python references/routines_status.py`
  on the real `~/Documents` plans — **semantic JSON diff = identical** (byte-diff only in
  key order; serde_json lacks `preserve_order`, Python dicts keep insertion order).
  Both read `weekend_plan_August_14_to_August_16_2026.md`, both say
  `state: attention` (stale + hollow) with the same `items`, `window`, counts,
  `ran_at`, `upcoming_weekend`. Invariant tests mutation-calibrated (state flags
  inverted → both red).
- `eval_tasks` deletion (2b, done): `run.py`/`validators.py`/`analyze.py`/`__main__.py`
  removed — no importer anywhere (rg-verified); `__init__.py` + `README.md` + `data/`
  rubrics stay per the shim's own kill criterion. `pytest` paths/taxes/grounded 76
  passed, `cargo test --lib task_loader` 16 passed.
- Gate repaired (needed for `tools/gate.sh --full` green — all three pre-existing at
  HEAD, none introduced by 2b): (1) `twitter/cookies.rs` 505 lines (GOH cap) → tests
  split to sibling `cookies_tests.rs` via `#[path]` (289 + 216); (2)
  `test_rust_prompt_parity` — generated `eval/prompts/*.rs` were stale (old `r"..."`
  raw strings the parity regex can't parse); regenerated via
  `PYTHONPATH=references tools/gen_rust_prompts.py`, 27 pass; (3)
  `test_the_pre_commit_hook_calls_check_file_size` — asserted the pre-GOH inlined hook;
  re-expressed as hook-delegates-to-`structural.sh` + `.gatesrc` `GOH_MAX_LINES=500`
  (the delegation only enforces the cap when the repo wires it).
  Full gate 2805 passed / 0 failed.
- NEXT: `twitter/session.rs` (profile dir, saved-session check, headed login),
  then the `camoufox-rs` collect driver behind the `LiveBrowserCollector`
  runner seam, env-flagged, Python default.

- **2a (needs S1+S2 green):** `camoufox-rs` collector behind the `LiveBrowserCollector`
  runner seam + login/session port. Env-flag flip per path (`TWITTER_COLLECTOR=rust`
  style), Python remains default until A/B passes.
- **2b (independent of S1):** weekend search → `reqwest` (item 4), `ztools status`
  subcommand (item 5), `eval_tasks` code deletion (item 6). These can land while 2a
  is still in review.

## Phase 3 — Parity proof (A2) before any cutover

- Extend `bin/ab_test`: Rust-collect vs Python-collect on the same account/window —
  tweet-ID set equality, not summary similarity (summaries are model-sampled; IDs are
  deterministic). Prove the harness red first: point it at a stale cache and watch it
  fail (E3).
  2026-09-13 update: comparator + fixtures landed (`run_collect_id_parity` in
  `bin/ab_test`, `tests/fixtures/collect_parity/` — green on identical sets,
  red-proofed on divergent ones). Prerequisite gaps closed: `Tweet.id`
  captured end-to-end in Rust, and `--fetch-only` now actually writes the
  shared debug cache (its message claimed it did). Live run pending a quiet
  box (9GB swap held by other tenants as of this writing).
  2026-09-13 update 2: BOTH legs ran live despite the load (Rust 49 tweets +
  Python 47 via camoufox per your direction, 100% ID coverage both sides,
  user cache preserved) but ID sets barely overlap (3 shared) on the same
  account + 24h window. Diagnosis: same authors, same span, monotonic
  full-window coverage both sides = different feed rankings, i.e. at least
  one leg silently collected For You instead of Following (a failure mode
  Python's own comments document). Both sides click the tab but neither
  verifies captured traffic came from the Following endpoint. Required before
  the A/B can mean anything: filter or assert recorded responses to the
  Following endpoint so a wrong-feed run fails loudly.
- `routines.toml` flip to `ztools status` only after one full daily cycle emits an
  identical page (diff the JSON, not eyeballs).
- Weekend: **DONE 2026-09-13**: `clean_search_results` extracted in
  `rust/src/ztools/weekend/fetch.rs`, byte-exact port of
  `references/weekend/data.py::_clean_search_results` (title-only dedupe with
  `rstrip(".,!?:; ")` char-set semantics, 300-char body truncation, empty-title
  drop, region evidence on `"{title} {body}"`). Proven by
  `rust/tests/weekend_parity.rs` + `references/tests/test_rust_weekend_parity.py`
  over `tests/fixtures/weekend_parity/` (corpus kernel, `as_candidate_lines`
  over one aggregator page, `looks_like_aggregator` flags) — byte-compared,
  mutation-calibrated (truncation removal goes red, revert goes green).
  Fixtures are region-consensus by construction; the region-filter divergence
  (Rust foreign-city blocklist vs Python whitelist-only) is a pinned design
  decision (`a_foreign_city_beats_any_local_token`), not drift.
  2026-09-13 update: region place-data moved OUT of code into
  `conf/weekend.toml [region]` (`in_region`, `foreign`) — both pipelines read
  the same file, no bare "york" (measured 9/9 bare-york-only live hits were
  York-UK/PA junk), compounds only. Remaining divergence surface shrank to
  London-Ontario / Chicago-with-Toronto / ON-postal-only cases.

## Phase 4 — Cutover (subtractive, F1: one commit per bullet)

1. `routines.toml` → binary.
2. Delete `references/` (runtime code + tests triaged per below).
3. Delete `eval_tasks/*.py` (keep `data/` rubrics), `pyproject.toml` Python packaging +
   `[project.scripts]`, `.venv`, `uv.lock`, root `ztools` wrapper if still stale.
4. Gate becomes Rust-only (`.gatesrc` already is — no Python step runs there; confirm
   `tools/gate.sh --full` green at HEAD before and after).
5. Correct `PORT_PARITY.md` + `ARCHITECTURE.md` end-state statements.

## Test-debt triage (item 7, runs inside Phase 4)

For each of the 138 files: **covered** (name the Rust test) / **re-express** (port the
case, fail-first) / **delete-with-code** (behavior retired, say which). The validator
parity gate's Python half (`test_rust_validator_parity.py`) is re-expressed as Rust
fixture assertions before its deletion — the byte-agreement property survives, the
interpreter does not. Anything that cannot be classified in one pass is an honest
deferral (B4) with its name recorded here, not a silent drop.

Ledger (2026-09-13, all 139 files classified, spot-verified):
`docs/PORT_PARITY.md` section "Test-debt triage" — COVERED 59 / RE-EXPRESS 40 /
DELETE 40. Only `test_llm.py`, `test_mlx.py`, `test_gemma.py` need a live
server/GPU. Biggest re-express clusters: config-core lookup semantics,
eval report math/tables, faithfulness gates, twitter orchestration/retry.

## Explicit non-goals

- No keychain AES port: the persistent-profile design killed the Chrome-cookie path;
  if no caller needs it, the code dies unported (item 3).
- No bs4 port: text extraction from raw HTML, bounded (`FOLLOW_LIMIT` /
  `FETCH_TIMEOUT` / `MAX_PAGE_CHARS` preserved as behavior).
- `tools/*.py` gates stay Python permanently.
