# Testing Patterns and Findings

**Updated: 2026-09-13** — the Rust suite, its structural gates, the golden parity
tests that replaced the Python A/B, and the rules every test follows. The Python-era
version of this file (mock infrastructure for `lib/testing.py`, pytest autouse gates,
the `__pycache__` mutation trap) is in git history before this date; the rules it
stated are language-neutral and kept below.

---

## Overview

- `cargo test --manifest-path rust/Cargo.toml --all-features` — ~680 lib tests plus the
  integration suites under `rust/tests/`, no live model, no live browser, no network.
- Coverage floor **95% lines** (`gates/coverage_gate.sh`), a ratchet: it only moves up.
- `python3 -m pytest tools/tests -q` — the pytest for the SHELL tooling
  (`tools/gpu_lock.sh`). `tools/*.py` are dev gates, never the product.

**Run what the gate runs.** `tools/gate.sh --full` (= `make ci` = the pre-push hook) is
one step list declared once in `.gatesrc`: the house Rust gate (fmt, clippy `-D warnings`,
no `#[allow]`), `cargo audit`, the emoji gate, the 500-line cap, `cargo test`, the tools
pytest, and the coverage floor. A green `cargo test` alone proves less than it looks:
it misses coverage regressions and the lints.

## Structural gates (you cannot opt out by forgetting)

- **No real browser, no real cookies.** The collector's pure decisions live in
  `twitter/collect.rs`; the driver (`twitter/native.rs`) is exercised live, by a person,
  never by the suite. Cookie tests read fixture `cookies.sqlite` files.
- **Outputs stay in `tmp`.** Every writer takes its directory (or honours
  `TWITTER_OUTPUT_DIR` / `WEEKEND_OUTPUT_DIR`); the CLI dispatch tests redirect `HOME`
  to a fresh temp dir and assert the write landed under it and not under the real one.
- **Env-mutating tests are `#[serial_test::serial]`.** A test that sets
  `WEEKEND_OUTPUT_DIR` while another reads it is a flake, not a race worth debugging.
- **Data files the binary refuses to guess** (`conf/twitter.toml [fallback]`,
  `conf/eval_inputs.toml`, `conf/eval_vision.toml`) are shipped into the fake `HOME` by
  the dispatch tests, exactly as a checkout ships them.

## Golden parity tests

The Python implementation is gone; where it was the reference, its last verdicts are
frozen and the Rust side must reproduce them byte for byte:

| test | frozen from | asserts |
|---|---|---|
| `rust/tests/validator_parity.rs` | `tests/fixtures/validator_parity/expected_python_verdicts.json` | six taxes validators' `(score, reason)` per fixture answer |
| `rust/tests/weekend_parity.rs` | `tests/fixtures/weekend_parity/expected_python_payloads.json` | corpus cleaning, candidate lines, aggregator flags |
| `eval/validators/mixed_text_tests.rs` | verdicts computed on the shared prompts | mixed-signal + factual scorers |
| `eval/tasks_tests.rs` | the Python `TASKS` table | name, order, roles, `parse_json`, validator per row |
| `eval/prompts/mod.rs` tests | `conf/prompts.toml` | the eval prompt wraps the production instructions |
| `rust/tests/gpu_lock_shell_parity.rs` | `tools/gpu_lock.sh` | both lock halves read each other's owner file |

Each golden suite carries a calibration test that mutates an input and expects the
verdict to move, so a green run is evidence and not tautology. A change that moves a
golden is a change to reference behaviour: regenerate it deliberately and say so in
the diff.

## Rules for writing tests

### 0. Prove a new test can fail

A test you have only ever seen pass is not evidence. Break the behaviour it covers —
delete the guard, invert the condition, remove the conversion — confirm it goes red,
then restore. Do this before you trust any new assertion. Mutations verified to turn
the suite red this way include: disabling stagnation detection, disabling the runtime
budget, removing the millisecond-expiry conversion, removing the session-cookie check,
inverting the Following-tab branch, editing one word of `conf/prompts.toml`, and
renaming a grounding signal in a taxes fixture.

### 1. Every test must have a non-tautological assertion

A test that calls a function and asserts only that a mock was called would pass with
the bug in place. Verify a concrete outcome: the value, the file, the printed line.
Range checks (`0 <= score <= 100`) are a floor, never the sole assertion.

### 2. Test the real code, not a re-implementation

If the test re-derives the logic it is checking, it passes when both are wrong. Drive
the real binary (`tests/cli_dispatch_ztools.rs` runs `ztools` against a stub server)
or the real function with hand-built inputs.

### 3. Verify specific numbers

`assert_eq!(score, 100)`, not `assert!(score >= 0)`. When the exact value is hard to
predict, state why in the test and bound it as tightly as you can.

### 4. Use the real scorer/validator; mock only the LLM

A per-test stub HTTP server answering a canned body exercises the whole transport,
cleaning and scoring path. Mocking the validator measures the mock.

### 5. Discover expected values, do not guess them

Run the function once in isolation to learn the exact score, then pin it. The mixed
text and validator goldens were produced this way from the Python originals; a Rust
value copied back into its own test proves nothing.

## Patterns for hard-to-test code

- **Injection seams over process boundaries.** `LiveBrowserCollector.runner` is a
  function pointer so the passthrough logic is tested without a browser; the
  production wiring is one line and the tests use a capturing double.
- **Pure decisions out of the driver.** Scroll stop conditions, dedup and login checks
  are functions of observations (`twitter/collect.rs`); the browser loop only feeds
  them. The same split gives `weekend/report.rs` (parsers) vs `status.rs` (I/O).
- **Fixtures for filesystem shapes.** `cookies.sqlite`, plan directories with set
  mtimes, task snapshot dirs — built in a `tempfile::tempdir()` per test.
- **Data over literals.** Region lists, endpoint markers, fallback policy, eval inputs
  and vision fixtures are files under `conf/`; tests that need a variant write their
  own small TOML rather than patching a constant.

## Bugs found during test audits (kept because each is a class)

- A `since` window filter that stopped on the first old tweet instead of the first old
  PAGE dropped everything after a pinned tweet.
- A `--fetch-only` that printed "cached" and wrote nothing (class: message claims an
  effect the code does not perform). Fixed with a round-trip test through the loader.
- A stats value written for months that no consumer read (`clean` flag) — the
  timeout it should have gated inflated 138,000× on a thrashing box.
- Two stores for one fact (Python `wk` wrote `~/Documents`, the tab read
  `~/Documents/weekend_plans/`) so the status page called a fresh plan stale. One
  resolver now (`store::weekend_output_dir`).
- The word "unknown" rendered into plan tables as if it were data (class C4). One
  `fmt_missing` for every cell of both renderers, tested across seven spellings.
- Tweet-ID set equality as the collect-parity criterion: the same collector run twice
  minutes apart shares ~7 of 50 IDs (the feed is a per-load sample), so the instrument
  could not see parity. Replaced by shared-record agreement.

## A unit test must not read the live machine (2026-09-19)

Three tests asserted what THIS box was doing: real memory pressure through an
`Option` argument whose `None` meant "go and look"; the real
`/tmp/mac-osaurus-gpu.lock` for a "nothing there" case before the temp-dir
redirect; and both, twice, compared. All went red whenever a sweep ran. The rule
is a pure function over injected readings (`uncontended_verdict(lock_held,
pressure)`); the live function composes it; the test pins the rule. Grep
`#[test]` bodies for `memory_pressure()`, `foreign_holder()`, `/tmp/`, `sysctl`
and env reads: each is injected or a flake with a schedule.

## Loopback stubs answer either wire shape

`ztools/llm.rs` streams (`stream: true`) but reads a plain JSON completion too,
so every existing `{"choices":[{"message":{"content":…}}]}` stub keeps working
and a stall test is one stub that accepts and sends nothing (the verdict must
land at the stall budget, not the cap). Search stubs: every engine URL in a
config must be loopback (`bing_url`, `brave_url` too) — a default is the live
site, and a dead DDG stub falls through to it.
