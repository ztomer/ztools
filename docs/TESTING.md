# Testing Patterns and Findings

**Updated: 2026-10-05** — the Rust suite, its structural gates, the golden parity
tests that replaced the Python A/B, the `test_env` sandbox and the audit gates that
keep it honest, and the rules every test follows. The Python-era version of this file
(mock infrastructure for `lib/testing.py`, pytest autouse gates, the `__pycache__`
mutation trap) is in git history before 2026-09-13; the rules it stated are
language-neutral and kept below.

---

## Overview

- `cargo test --manifest-path rust/Cargo.toml --all-features` — 852 lib tests plus ~105
  across the integration suites under `rust/tests/`, no live model, no live browser, no
  network.
- Coverage floor **95% lines** (`gates/coverage_gate.sh`), a ratchet: it only moves up.
- `python3 -m pytest tools/tests -q` — the pytest for the SHELL tooling
  (`tools/gpu_lock.sh`). `tools/*.py` are dev gates, never the product.

**Run what the gate runs.** `tools/gate.sh --full` (= `make ci` = the pre-push hook) is
one step list declared once in `GOH_CI_STEPS` (`.gatesrc`): the house Rust gate (fmt,
clippy `-D warnings`, no `#[allow]`), `cargo audit`, the emoji gate, the 500-line cap,
`cargo test --all-features`, the tools pytest, `ruff check tools/`, the vendored crate's
own test suite, the coverage floor, and a secrets scan that names `vendor/` explicitly
(rather than inheriting the `GOH_EXCLUDE` that exempts it from the length/emoji checks).
A green `cargo test` alone proves less than it looks: it misses coverage regressions and
the lints.

## Structural gates (you cannot opt out by forgetting)

- **No real browser, no real cookies.** The collector's pure decisions live in
  `twitter/collect.rs`; the driver (`twitter/native.rs`) is exercised live, by a person,
  never by the suite. Cookie tests **synthesise** their Firefox `cookies.sqlite` into a
  per-test `tempfile::tempdir()` (`cookies_tests.rs::fixture_cookie_db` creates the
  `moz_cookies` table and inserts the rows) — there is no committed `.sqlite` fixture in
  the repo, and a test that read one would be reading the developer's real profile.
- **Outputs stay in `tmp`.** Every writer takes its directory (or honours
  `TWITTER_OUTPUT_DIR` / `WEEKEND_OUTPUT_DIR`), both of which `TestEnv` redirects.
- **Data files the binary refuses to guess** (`conf/twitter.toml [fallback]`,
  `conf/eval_inputs.toml`, `conf/eval_vision.toml`) are passed to the code **by path**,
  resolved at compile time from `env!("CARGO_MANIFEST_DIR")` — never through `$HOME`. A
  test that needs real shipped DATA asks for it by path; a test that needs a fake `~`
  takes the sandbox. There is no third answer, and the previous policy had one: it was
  "whichever test ran last won".
- **The three sandbox audit gates** (`rust/src/test_env/audit/`), which run on every
  `cargo test` and re-derive their expectations from the crate's own sources:
  1. `every_env_var_the_crate_reads_is_managed` — a variable the crate reads a path or a
     policy out of, and that the sandbox does not manage, fails the build. This is what
     makes "you cannot add a path-reading variable without `TestEnv` knowing" a property
     of the system rather than of whoever writes the next test.
  2. `every_hazard_test_constructs_a_test_env` — a test that builds a `ZtoolsConfig` or
     sets `record_signals: true` (both of whose defaults name `~/…`) must construct the
     guard **in its own body**. Seams do not count, deliberately: a guard reached through
     a helper is a guard the next test can call without.
  3. `no_test_names_a_fixed_directory_under_the_system_temp_dir` — `temp_dir().join(...)`
     with a fixed literal, which two `cargo test` runs on one Mac (or two agent sessions)
     write at once. `tempfile::tempdir()` is the fix; a pid suffix is the old workaround
     and is still flagged.

  All three used to carry an `OUTSIDE_THE_CONTRACT` allowlist keyed by file. **It is empty
  now** and kept as an empty list on purpose: the ratchet runs both ways — an entry whose
  file has been fixed fails the gate, and the gate prints the remaining debt per file so
  the list is a number somebody can watch fall rather than a permanent exemption.

## `test_env::TestEnv` — one sandbox for every variable this crate reads a path out of

`TestEnv::new()` takes a process-wide lock for the sandbox's whole lifetime, redirects 16
path variables into one fresh temp dir, clears the 12 policy knobs (28 managed in all),
and restores everything exactly on drop — including through a panic. It is reachable from
both compilation universes, so `rust/tests/**` uses it too.

**Why not `#[serial_test::serial]`.** It is one-sided: it excludes serial tests from each
other and from nothing else, while cargo runs every non-serial test in the binary
concurrently. A guarded test that reads a variable another guarded test just wrote is a
flake with a schedule. The guard takes the lock itself, so **a test is serialised by
constructing it, annotated or not**. `#[serial]` is still used where what is being
excluded is *this crate's own* serial set, not the environment.

**`$HOME` is a redirect like any other**, and that was the last thing standing between the
guard and the property it exists for. It was preserved — captured and restored, never
repointed — because `dirs::home_dir()` is process-global and the tests that resolved a
`~`-bearing default *through it, without taking the lock*, would break on a schedule when a
peer redirected it. That was never a reason to leave `$HOME` unmanaged; it was a reason to
bring those tests under the guard, which is what the allowlist tracked. Both hold now, and
`Point::Preserved` no longer exists: a variant with no member is a decision nobody can
audit. The regression that keeps it a redirect is
`restore_tests::every_managed_variable_is_sandboxed_and_restored`, which asserts all three
of the variable, the library that reads it (`dirs::home_dir()`) and the emptiness of the
directory it now names.

**A test's isolation is a coincidence of code order until the variable list is shared.**
Three instances of one defect turned up on 2026-10-04: a test that set `EVAL_SIGNALS_DIR`
but not `EVAL_OUTPUT_DIR` wrote `~/.config/ztools/outputs/gone-model/t1.txt` on every run;
four tests that passed `ZtoolsConfig::default()` were safe only because a branch elsewhere
happened not to read the cache; and a cookie test walked the developer's real Firefox
profile tree. Each had hand-rolled its own list of one or two variables, so the next
`forget` was one line away. The shared list is pre-paid — and one audit gate re-derives it
from the sources, so it cannot rot.

**Take the lock BEFORE you redirect.** `TestEnv::new()` used to build the sandbox and take
the lock after, which meant a thread waiting for the lock had already overwritten the
environment the holder was relying on: the holder's `~`-bearing defaults resolved into a
temp dir it did not own, and its restore put the *waiter's* values back. Nothing failed
loudly — one eval test simply started comparing one peer's sandbox against another's. `put`
documents that the lock is what makes its write sound, and a write taken before the lock is
the one case where that is untrue.

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
- **Fixtures for filesystem shapes.** A Firefox `cookies.sqlite`, plan directories with
  set mtimes, task snapshot dirs — built in a `tempfile::tempdir()` per test. Nothing
  under `tests/fixtures/` is a `.sqlite`: `git ls-files | rg '\.sqlite'` is empty, and
  `cookies_tests.rs::fixture_cookie_db` creates the `moz_cookies` table and inserts the
  rows at runtime. A committed cookie DB would be a copy of somebody's real profile.
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

## A fixture the repo does not own is a test that asserts nothing (2026-10-04)

`yoy_regression.rs` read `/tmp/yoy_out.txt` and `return`ed when the file was
absent — zero assertions, on every clean machine, under a name promising it
"must byte-match the Python validator's verdict". The answer had never been
committed and no CI step wrote it, so the guard it existed for (the SIGNED
traceable match, a real parity defect) was decorative.

The rule is about the SHAPE, not the path: a test whose input the repository
does not ship is a green no-op, because the only way to make it run is to
hand-place a file on the operator's disk. Three questions, in order:

1. Does the case belong in an existing committed fixture set? A golden set owns
   its inputs, so a case with a frozen expected value belongs there — one
   mechanism, covered by its completeness guard and its calibration test.
2. Is the expected value recoverable from the repo or git history? Provenance
   matters more than the test existing: here the yoy case turned out to be
   ALREADY in `validator_parity`, with a frozen Python verdict byte-identical
   to the one `yoy_regression.rs` pinned, so the file was a duplicate of a
   covered case as well as a no-op. Deleted, with the reasoning recorded in the
   golden's module doc.
3. Only then a new fixture dir — and say plainly where the expected value came
   from, because a golden whose expected value you invented is worse than no
   golden.

Before deleting a no-op test, break the production behaviour it named and watch
whatever replaces it go red. Here: re-`abs()`ing the reported deltas dropped
`traceable` from 4/4 to 0/4 in the golden, so nothing was lost.

## An assertion whose expected value is the fallback cannot see which branch ran (2026-10-04)

Written while fixing the above: a weather test served `{"error":true,...}` and
asserted the result equalled `fallback_forecast()`. That test stayed GREEN when
`parse_weather_json` was broken to return `Some(fallback_forecast())` for a
body with no `daily` block — the two states are the same string, so the
instrument was blind to exactly the branch it named. No black-box assertion on
the return value can separate them.

What works is giving the case an observable of its own:

- serve an input that is a DIFFERENT failure (a 200 whose `daily.time` is empty
  — the parse-fail path, reached after a successful round trip rather than
  instead of one), and
- have the stub RECORD its request line, then assert it was asked. That is also
  what turns "the parser works" into "the transport worked": a stub that
  discards its port and hand-feeds JSON to the parser covers half the function
  and advertises the whole of it.

Rule of thumb: if a break that changes behaviour leaves the test green, the
test has not measured the thing it is named after — fix the measurement before
concluding the behaviour is fine.

## Assert against what the stub RECORDED, not a copy you re-rendered (2026-10-05)

The carry-rule test — the one that keeps DATES/PRICE/AGES/LOCATION alive across the
weekend phase chain — rendered the template itself, with the same bindings production
used, and compared the result to the constant. **A constant compared with itself.** It
passed, and it kept passing when `("carry", CARRY_FIELDS)` was deleted from the
production binding list entirely: the test's own render still contained the rule, so the
one test guarding the binding could not see the binding go. The class-C2c failure it
exists to catch — every date column blank because the predecessor narrowed the payload —
would have shipped green. (The test's own doc comment is where that history lives.)

The fix is to drive the code and read what it actually sent
(`weekend_phases_tests.rs::the_carry_fields_rule_reaches_the_phase_that_binds_it`):
call `draft_activities` against a recording stub, pop the recorded request, parse the
`messages` array out of it, and assert the real prompt contains `CARRY_FIELDS`. That
test now goes red the moment the binding is dropped or mistyped.

One trap in doing it: **a recorded request is JSON, so it is not the prompt.** Every
newline in the body is a literal `\n` and every quote is an escape, so a substring
search for multi-line prompt text against the raw recorded body finds nothing and looks
like a failure of the thing you are testing. Parse the message out first.

This is the same move as the recorded request line in the section above, and it is the
general shape: **the copy of the input you make inside a test cannot see the code that
produced the real one.** Re-deriving an expected value from the same constants
production uses is `assert_eq!(CARRY_FIELDS, CARRY_FIELDS)` with a `render` in between.

## A structural assertion over a committed fixture cannot see the writer (2026-10-05)

Two golden tests asserted the *structure* of a committed document — the weekend plan's
heading text, its `| Dates |` column, the sentinel in the empty cells; the twitter
summary's paragraph break between the tweet count and the provenance block — by READING
the fixture. Both went green through exactly the break they were written to catch,
because the fixture did not move with the writer. They were auditing a FILE, and the
defect was in the PROGRAM that writes files. This is `differential-blindness` in a
golden's clothes: a committed baseline answers "did anything change", never "is the thing
still producing it".

Both now render live and assert on the render:
`format_golden_tests.rs::the_saved_plan_says_when_every_event_is` calls the real
`format_weekend_plan` through its own case builders, and
`summary_doc_tests.rs::the_provenance_block_is_its_own_paragraph` calls the real writer
for both the quiet and the degraded shape. That is the structural fix and it is visible
in the source; the historical claim — that each first draft sat green through the break
it was written to catch — is recorded in each test's own doc comment, which is where a
reader should look before re-deriving it by breaking a renderer.

The two kinds are not redundant and both are wanted. A byte-for-byte golden answers "did
the document change". A render-live structural assertion answers "does the writer still
produce the structure" — it is the only one that can fail while the fixture stays put.

One test here genuinely must read the fixture, and its blindness is the point rather than
a defect: `the_goldens_were_read_not_merely_recorded` audits whether the committed bytes
are the renderer's output for this file's inputs — a leftover column, a score that lost
its decimal, a word that reads as data. Its SUBJECT is the file, so reading it is
correct; what it therefore cannot tell you is whether the renderer still produces those
bytes. That claim is the render-live test's job, and it is why the pair must be read
together: **an audit of a fixture is not a test of its writer.**

## `ZTOOLS_UPDATE_GOLDENS=1` makes the sibling audit tests lie (2026-10-05)

`assert_golden` honours `ZTOOLS_UPDATE_GOLDENS=1` by *writing the fixture* and then
panicking, so blessing a change is always a deliberate second run. But
`the_goldens_were_read_not_merely_recorded` — the audit that checks the fixture really is
this file's renderer applied to this file's inputs — reads the same path from the same
test binary, where cargo runs tests concurrently. The audit and the rewrite are two
writers and one reader on one file, in one process.

The audit's verdict is a live function of the file's bytes, which is measurable. Same
binary, same test, one heading removed from `weekend_plan.md`:

```
$ cargo test --lib the_goldens_were_read_not_merely_recorded      # heading deleted
thread '...the_goldens_were_read_not_merely_recorded' panicked at
  src/ztools/weekend/../weekend/format_golden_tests.rs:299:9:
weekend_plan.md: fixed-activities heading
test result: FAILED. 1 passed; 1 failed

$ cp /tmp/goldenbak/weekend_plan.md tests/fixtures/user_documents/
$ cargo test --lib the_goldens_were_read_not_merely_recorded      # restored
test result: ok. 2 passed; 0 failed
```

So under the rewrite variable the audit may read the pre-rewrite bytes (pass, over a
fixture that is about to be replaced), the post-rewrite bytes (its findings are about a
change the run is still making), or a truncated file (spurious failure). Any of the three
is a verdict about a file that is moving, reported as though it were about the writer.

The rule is about the RUN, not the test: **one rewrite per run, and nothing else in that
binary may read the fixture while it is set.**

`ZTOOLS_UPDATE_GOLDENS` is in `TestEnv`'s `CLEARED` table
(`test_env/table.rs`), so a test that constructs the sandbox cannot see a stray export —
but the two golden tests deliberately construct **no** guard, because they are the tests
that must see the variable when a person sets it on purpose. So the exposure is exactly
as wide as the shell command: `ZTOOLS_UPDATE_GOLDENS=1 cargo test` hands the variable to
the whole binary, both golden tests and the audit alike, and a green audit line is then
an all-clear on bytes that are still moving. Two habits make it safe: run the rewrite as
its own invocation (`ZTOOLS_UPDATE_GOLDENS=1 cargo test --lib format_golden`), then
re-run the whole binary **without** the variable before believing anything it said.

## A test whose verdict depends on state another test writes (2026-10-08)

`eval::task_fingerprint` keeps a process-wide registry — the tasks this process
loaded, name to digest — because the history and signal writers reach those
stores by task NAME and neither signature can be widened from their side. That
registry is exactly the shape that makes a test's verdict a function of test
ORDER: `signals_tests::learning` recorded a sample under the task name `t`,
`report_tests` registered a different `t`, and the resulting reset made the
p95-blending case fail only when the two ran in the same binary.

The rule, which is the same rule as the golden-fixture one above over a
different shared object: **a test that reads or writes process-wide state must
either take names no other test uses, or reach the same decision through a
path that takes the state as an argument.** Both halves are now true here —
the aggregates have pure twins (`historical_stats`, `render_trends`,
`deltas_since`) that take the identity set explicitly, and the signal-learning
cases record through `record_task_signal` with a task they built themselves.
`record_signal` (the name path) is pinned once, in
`signals_tests::identity.rs`, against a name no other file uses.

## A box that is paging fails tests about a quiet box (2026-10-08)

`oversize::tests::an_uninjected_thrashing_verdict_reads_the_machine_and_still_answers`
and, through the same refusal, `eval_not_measured`'s full-suite case and all
three `drain_signal` cases, assert what happens when nothing is wrong: the
oversize gate must return an empty refusal, so the run proceeds to make
requests the stub is waiting for. On a machine that is already swapping
(measured 2026-10-08: swap 1.9GB, compressor 17.7GB) the gate refuses for a
real reason and those four go red with no code change involved.

Isolate before debugging: `git stash push -- rust/src/ && cargo test --no-fail-fast
--lib oversize` on clean HEAD is the same four failures, or it is your bug. The
environment-dependent half cannot be fixed from the test — it is the machine —
so the discipline is `regression-isolate`, then re-run when the box settles.
