# Roadmap — `ztools`

Forward-looking only. What shipped is in `CHANGELOG.md`; what was planned and is done
is in git history (house rule #13). This file is the open work, in the order it has to
happen, and nothing else.

## The contract this file keeps

Enforced by `tools/tests/test_roadmap.py` (gate step `python3 -m pytest tools/tests`),
so it holds whether or not anybody remembers it:

1. **Every item has an ID that is never reused** (`L1`, `G3`, …) and four fields:
   **Class** (the failure mode it is an instance of, because the class is what
   recurs), **Done when** (a command, test or gate that goes from red to green, never
   a judgement), **Blocked by** (an item ID, an owner decision, a cross-repo ask, or
   `nothing`), and **Why now** (what it hides while it stays open).
2. **A done item is deleted, not annotated.** "Closed 2026-…" paragraphs are how the
   previous version of this file grew to 575 lines of history around 20 lines of
   open work. The gate rejects them.
3. **Every repo path in backticks exists**, unless the item marks it `(new)`.
   Sixteen prompt rows once named a deleted tree for a month; a roadmap that names
   files is the same instrument and gets the same check.
4. **Every inbound reference resolves.** Docs that point here say `docs/ROADMAP.md <ID>`,
   and the gate fails on an ID this file does not define.

**How work is ranked.** The tools exist to run unattended and be trusted, so rank by
what a failure HIDES: a plan that says "quiet weekend" over a blocked search, a
summary steered by a tweet, a slot winner measured on a task that has since changed.
The same rule orders the machinery, and it puts the machinery first: a gate that
cannot fail hides more than a plan that reads oddly, because it is the thing that
would have noticed. Phases are SEQUENTIAL where a later phase's evidence depends on an
earlier one, and each phase says what it depends on.

## Where things stand — 2026-10-14

A dated summary, not checked by any gate; re-derive it before editing:
`gh run list --limit 3`, `git log --oneline origin/main..main`, `tools/gate.sh --full`.

- **The local gate is RED since goh v0.27.0** (`gates_of_heck` `51d31a5`, 2026-10-14):
  `tools/gate.sh --full` on `1076cde` failed 2 of 12 steps in 2 min 7 s -- the Rust gate's
  `no emptiness asserts` and the structural gate's `no early-exit pipe under pipefail`; with
  those two masked (scratch run, not committed) structural also refuses 13 undeclared
  stdin calls. Every other step was green at 09:00; RustSec published four `rustls`
  advisories the same day, so `cargo audit` is red now too (`A6`). Phase A is that list.
- **CI on `main` is still RED** (latest run 2026-10-12 14:43 UTC, `roadmap: rephase…`): every
  run since 2026-10-10 dies in `actions/setup-python` (`.python-version doesn't exist`). `A5`.
- Hooks reinstalled to goh 0.27.0's stock set (`1076cde`). Installed: `ztools 3.8.0`.
- The previous plan's `P2` (a release can be tagged over a red CI) is closed upstream:
  goh `5c5868e` makes the release kit refuse while CI on HEAD is red, pending or missing,
  and `tools/release.sh` execs that kit. Narrative of earlier phases:
  `git show 6a10475:docs/ROADMAP.md`.

## Phase A — wire in goh 0.27.0 (Phase 0)

Goal: `tools/gate.sh --full` green at HEAD, the commit pushed, and CI green on it. Depends on
nothing; everything else waits on it, because the pre-push hook refuses every push until it
lands. Size: S (one session). Exit: `tools/gate.sh --full` 12/12 green on HEAD, `git push`
accepted by the pre-push gate, `gh run list --branch main --limit 1` reports `success`.

### A1 — 13 child processes inherit stdin undeclared
- **Class:** a child that inherits a stdin nobody chose HANGS instead of failing (goh 3.4,
  `0c93f4d`: the check is on by default). Measured 2026-10-14 by
  goh's `subprocess-stdin`: 13 calls, 11 in `tools/tests/` (`test_gpu_lock_shell.py:62,91,98,271,295`,
  `test_install_platform_gate.py:73,248`, `test_ab_test_comparator.py:123`,
  `test_conftest_guard.py:50`, `test_rerun_truncated.py:198`, `test_sweep_models.py:162`) plus
  `tools/upgrade_tap.py:82` (deleted by `U1`) and `tools/vendor_lint_ratchet.py:124`.
- **Why now:** structural step `every child process names its stdin` is red at full scope, so
  pre-push refuses every push.
- **Done when:** each call names `stdin=subprocess.DEVNULL`, `input=` or `stdin=None` (none of
  these needs the caller's stdin; declare, do not `--seed`), and the step is green. Size: S.
- **Blocked by:** nothing.

### A2 — an early-exit `head` reads a pipe under pipefail
- **Class:** an early-exit consumer SIGPIPEs its producer and pipefail turns a hit into a
  failure, depending on load (goh 6.9, `496b6a5`). One site: `tools/osaurus_one.sh:179`,
  `FIRST="$(printf '%s\n' "$FOUND" | head -1)"`.
- **Why now:** structural step `no early-exit pipe under pipefail` is red; it is the first
  structural red, so it hides every step after it.
- **Done when:** the line reads `FIRST="$(head -n1 <<< "$FOUND")"` (the shape the finding
  prints) and the step is green with no `early_exit_pipe_seed.json` (new) written. Size: XS.
- **Blocked by:** nothing.

### A3 — two emptiness asserts in the eval tests
- **Class:** `assert!(!x.is_empty())` prints nothing about `x` when it fails (goh
  `empty-assert`, run by the house Rust gate). Sites: `rust/src/ztools/eval/smoke_tasks_tests.rs:235`
  and `:242`.
- **Why now:** the Rust gate's step `no emptiness asserts` is red (`[1/12]`).
- **Done when:** both read `assert_ne!(x.len(), 0, …)` with their messages kept, and step
  `[1/12]` is green. Size: XS.
- **Blocked by:** nothing.

### A4 — scanner floors are unset, so a scan that stops reading passes
- **Class:** a vacuous pass -- a scanner whose file selection broke reads nothing and reports
  clean (goh 4.3, `831d686`).
- **Why now:** advisory today, but it is the line that tells a blind gate from a clean one,
  and Phase A is the session that reads it.
- **Done when:** `.gatesrc` sets the twelve floors printed on 2026-10-14
  (`GOH_EMOJI_MIN_FILES=385 GOH_CONFLICT_MARKERS_MIN_FILES=439 GOH_FILE_LENGTH_MIN_FILES=285
  GOH_SECRETS_MIN_FILES=439 GOH_SHELL_LINT_MIN_FILES=8 GOH_EARLY_EXIT_PIPE_MIN_FILES=14
  GOH_DEAD_AFTER_EXEC_MIN_FILES=14 GOH_BARE_HOOK_INDEX_MIN_FILES=14 GOH_BARE_PYTHON_MIN_FILES=6
  GOH_CHECKOUT_CREDENTIALS_MIN_FILES=1 GOH_MD_LINKS_MIN_LINKS=4 GOH_UNREAPED_SPAWN_MIN_FILES=202`,
  re-read at the time, at or a little under each) plus `GOH_SUBPROCESS_STDIN_MIN_CALLS` at the
  post-`A1` count; the `[subject]` advisory line is gone; and one floor raised by one past its
  count turns `every scanner read its declared subject` red naming the key (then put back).
  Size: XS.
- **Blocked by:** `A1` (the stdin floor is counted after the declarations).

### A5 — CI checks out a file the repo does not track
- **Class:** CI reads a file only the developer's checkout has, so the local gate passes
  and the gate of record cannot start. `.github/workflows/ci.yml:98` says
  `python-version-file: .python-version`; `.gitignore:85` (`# pyenv`) ignores that file and
  it has never been tracked. Introduced by `22d8009` ("CI reads .python-version").
- **Why now:** CI has not reached the gate on any push since `83cf63b` (2026-10-10); goh
  `5c5868e` now refuses the next release while it stays red (goh BACKLOG Downstream, ztools).
  `--no-ci-check` exists and is not the fix.
- **Done when:** a pytest in `tools/tests/test_ci_files_tracked.py` (new) parses every
  `*-file:` / `hashFiles(...)` / `path:` input in `.github/workflows/ci.yml` and asserts each
  is `git ls-files`-tracked -- seen red on today's tree -- then `.python-version` is tracked
  (drop `.gitignore:85`) and the next CI run on `main` is `success`. Size: S.
- **Blocked by:** nothing.

### A6 — four rustls advisories published 2026-10-14
- **Class:** a dependency advisory published after the last green gate (RustSec, same day as
  goh 0.27.0): `rustls` 0.23.45 in `rust/Cargo.lock` carries RUSTSEC-2026-0346 (`Debug`
  impls expose TLS secrets), -0347 (buffered data to the outer server after ECH rejection),
  -0348 (Brotli certificate decompression allocates up to 1 GiB before authentication) and
  -0349 (an unauthenticated ServerHello downgrades later connections to classical key
  exchange). Fixed in 0.23.46, which `cargo update -p rustls` takes (dry run 2026-10-14).
- **Why now:** `cargo audit` (`[2/12]`) and the Rust gate's advisory step went red on the
  advisory DB fetched after the 09:00 gate run above; pre-push refuses on it.
- **Done when:** the lock holds `rustls` 0.23.46 or later in its own commit, and both audit
  steps are green with no ignore entry added. Size: XS.
- **Blocked by:** nothing.

## Phase P — the gate of record can fail

Goal: the two roadmap checks that currently pass over nothing can go red, and the docs say
what the step list runs. Depends on Phase A. Size: S. Exit: every new check in this phase
has been seen red once on the real tree.

### P3 — the roadmap's "has items" check passes over nothing
- **Class:** a gate made vacuous to accommodate an empty input.
  `test_the_roadmap_has_items_to_check` in `tools/tests/test_roadmap.py` asserts
  `len(items(roadmap)) >= 0`, which no input can fail; it was loosened in `6ad4a2b` when the
  file emptied. A broken `ITEM` regex would now report every rule satisfied.
- **Why now:** this file's whole contract rests on the parser seeing the items.
- **Done when:** the test compares `items()` against an independent count of
  `^### ` headings under `## Phase` sections (so an empty roadmap passes honestly and a
  parser that misses headings fails), and a fixture with a heading the parser cannot read
  is asserted red.
- **Blocked by:** nothing.

### P4 — an inbound roadmap reference the gate cannot see
- **Class:** a reference check that knows one word order and a subset of files.
  `rust-toolchain.toml` says "That mechanism is `X6` in `docs/ROADMAP.md`"; no `X6` has ever
  existed in this file (checked against `git log -p`), the `INBOUND` regex only matches
  `ROADMAP <ID>`, and root `*.toml` files are not in `inbound_sources()`.
- **Why now:** the comment also claims `gates/rust_gate.sh` prints no `rustc -V`; goh's
  gates/_proven.sh now keys on `rustc -V`, so the claim may be stale as well as its pointer.
- **Done when:** `inbound_sources()` covers root `*.toml`, the regex accepts
  `<ID>` … `docs/ROADMAP.md` within one sentence, a fixture of that form is asserted red,
  and the `rust-toolchain.toml` comment says what a red gate prints today (measured, quoted).
- **Blocked by:** nothing.

### P5 — `ruff format` is documented as a gate and nothing runs it
- **Class:** a doc that states a gate's contents without a check. `README.md` says
  "`ruff` is configured but not yet a gate" and `CLAUDE.md` says "Ruff is NOT in the gate's
  step list yet"; `.gatesrc` `GOH_CI_STEPS` does run `ruff check tools/`, and does NOT run
  `ruff format --check tools/`, which both docs tell you to run by hand.
- **Why now:** two docs disagree with the step list in opposite directions.
- **Done when:** `ruff format --check tools/` is a `GOH_CI_STEPS` step (seen red once on an
  unformatted fixture), and both docs say what the list runs — or quote it from `.gatesrc`
  instead of restating it.
- **Blocked by:** nothing.

## Phase D — the docs describe the Rust tool, not the retired Python one

Goal: no document a session reads first states something the code no longer does.
Depends on nothing, but do it after Phase P so the doc checks land on a green CI.
Size: S. Exit: the two checks below are green and each was seen red once.

### D1 — parity rows still say "deferred, recorded in ROADMAP"
- **Class:** a deferral recorded in two places, closed in one. `docs/PORT_PARITY.md` rows
  for `test_weekend_llm_1.py`, `test_weekend_llm_2.py`, `test_eval_report.py` and
  `test_signal_handling.py` say "B4 … Recorded in ROADMAP"; the plan has no such items
  because `9ce9677` (phase retries; timeout learning retired), `27624ff` (verbosity metrics)
  and `c83ac38` (Ctrl-C drain) closed them. The ledger's own count ("B4-deferred 2") is stale.
- **Why now:** a reader looking for those deferrals finds nothing and re-derives them.
- **Done when:** each row names the commit and the Rust test that covers it, the count line
  is recomputed from the table, and `grep -n "Recorded in ROADMAP" docs/PORT_PARITY.md`
  returns nothing.
- **Blocked by:** nothing.

### D2 — Python-era reference docs carry no banner
- **Class:** a historical document that does not announce it is historical.
  `docs/BUGS_twitter_llm_fallback.md`, `docs/LLM_GUIDANCE_AND_TEACHER_EVAL.md`,
  `docs/DEEP_MODEL_VS_GEMINI_EVALUATION.md`, `docs/eval_calibration_2026-07-11.md` and
  `docs/REPORT_WEAKNESS_CLASSES.md` cite `twitter/summarize.py`, `eval/validate.py`,
  `lib/osaurus_lib.py` and other files deleted at the Rust cutover (`ddb7d53`).
  `docs/REPORT_WEAKNESS_CLASSES.md` heads C16 "*(open)*" while its body says closed, and C11 is
  fixed in Rust (`rust/src/ztools/twitter/mod.rs` prints "fetched, processed").
- **Why now:** these are measurement records worth keeping, and they read as current.
- **Done when:** each carries a first-line banner ("Python-era record, superseded by …")
  or its file pointers resolve under `rust/`; a test over `docs/*.md` fails on a
  `*.py` product path in a doc without the banner (seen red before the banners land).
- **Blocked by:** nothing.

### D3 — `CLAUDE.md` points at gate lines that moved
- **Class:** a line-number citation with no check. `CLAUDE.md` cites `.gatesrc:26` for the
  coverage floor; `GOH_CI_STEPS` is at `.gatesrc:113` today.
- **Why now:** it is the first file a session reads.
- **Done when:** the citation names the variable, not a line (`GOH_CI_STEPS`), and a grep for
  `\.gatesrc:[0-9]` in `*.md` returns nothing.
- **Blocked by:** nothing.

## Phase U — house-layer adoption (advisory today, from goh's Downstream list)

Goal: every goh capability this repo still carries a private copy of, or has not wired, is
the house one. Depends on Phase A. Size: S. Exit: each item's check is a `GOH_CI_STEPS` step
or a deletion, and each was seen red once.

### U1 — a manual tap door that bypasses the tap's gate, and the repo's only status markers
- **Class:** two release paths, one of them ungated (goh 6.8, `bfac28b`): the release kit's
  tap step goes through the kit's tap bump, which runs the tap's own gate, but
  `tools/upgrade_tap.py` clones and pushes `ztomer/homebrew-tap` itself. Its docstring says
  nothing invokes it (re-checked 2026-10-14: no caller). The same file is all 3 of the repo's
  hand-rolled status markers (goh's `status-markers`, 284 files read: `tools/upgrade_tap.py:34`,
  `:38`, `:42`; goh 4.9, `51109d6`).
- **Why now:** it is a push to a shared repo that no gate sees, and it is the only thing
  keeping `status-markers` from adopting with no seed.
- **Done when:** `tools/upgrade_tap.py` and its `ARCHITECTURE.md` row are deleted, and
  `"$GOH_DIR/gates/goh.sh" status-markers` is a `GOH_CI_STEPS` step, green with no seed and
  seen red on a fixture marker. Size: XS.
- **Blocked by:** nothing.

### U2 — a vendored copy of the house GPU lock
- **Class:** a private copy of a house library drifts (goh `3103441`): `tools/gpu_lock.sh`
  duplicates `$GOH_DIR/lib/gpu_lock/gpu_lock.sh`, and its header cites a desktop-lock path
  that is gone.
- **Why now:** `rust/tests/gpu_lock_shell_parity.rs` judges the Rust client against this
  copy, not against the bash half every other repo sources.
- **Done when:** `tools/osaurus_one.sh` sources the house file, `tools/gpu_lock.sh` is
  deleted, and `rust/tests/gpu_lock_shell_parity.rs` plus `tools/tests/test_gpu_lock_shell.py`
  point at the house file (the parity test watched red against a deliberately skewed copy).
  Size: S.
- **Blocked by:** nothing.

### U3 — the vendored crate's tests are a plain `cargo test` with 26 emptiness asserts
- **Class:** a gate step that bypasses the house runner (goh Downstream, ztools):
  `GOH_CI_STEPS` runs `cargo test --manifest-path vendor/camoufox-rs/Cargo.toml --all-features`
  directly, and `vendor/camoufox-rs` carries 26 `assert!(…is_empty())` (counted 2026-10-14;
  `GOH_EXCLUDE='^vendor/'` keeps them out of the gate).
- **Why now:** low; the step runs and passes. Same owner decision as `H3`.
- **Done when:** the step runs through the house Rust layer (or the decision in `H3` drops
  the vendored crate), and the asserts read `assert_eq!(x.len(), 0)` / `assert_ne!`. Size: S.
- **Blocked by:** `H3`'s owner decision.

### U4 — the release script commits with `--no-verify`
- **Class:** a gate bypass written into a script (house rule: never `--no-verify`).
  `tools/release.sh:46` commits the version bump with `git commit … --no-verify`, so the
  commit that every release tags never ran the pre-commit gate.
- **Why now:** the kit gates the tree afterwards (`--gate "make ci"`), but the commit itself
  is the one place a staged-scope check (stdin, secrets) is exact.
- **Done when:** the flag is gone and a dry-run release on a clean tree still commits the
  bump through the hook. Size: XS.
- **Blocked by:** `A1`, `A2` (the hook must pass on the tree first).

## Phase F — owner-requested features

Goal: the owner's feature asks, each behind a test that can go red. Depends on Phase A (nothing
pushes before it). Size: S. Exit: each item's round-trip tests green and the command used once
from `routines`.

### F1 — `ztools weekend exclude <place>`, the seam the routines row button calls
- **Class:** a second writer of a user's config file. Owner, 2026-10-14: "weekend add an
  exclusion directly from the ui (button on the row)" -- the button lives in `routines`
  (its ROADMAP F1), and the only safe writer of the planner's exclusion list is this repo,
  so the counterpart is a CLI rather than `routines` editing TOML it does not own.
- **Why now:** `routines` F1 depends on it; without a command, the button would have to
  write `conf/weekend.toml` itself and inherit its traps.
- **Done when:** `ztools weekend exclude <place>` appends to the user-level `exclude_places`
  list (the file `weekend_exclusions_paths` names, `conf/weekend.toml` by default)
  idempotently, and `--undo` removes it; edits preserve comments and keep `exclude_places`
  ABOVE any table header (the C8 placement trap `conf/weekend.toml`'s own header explains:
  below a header TOML parses it as that table's key -- it once landed in the last
  `[[children]]` entry -- and the whole list was silently ignored, C8 in
  `docs/REPORT_WEAKNESS_CLASSES.md`). Round-trip tests on a TOML file with comments and a trailing table: add, add
  again (no duplicate), undo -- each re-parsed by the real config loader to prove the list is
  top-level, and one watched red with the entry appended at the end of the file. The place's
  identity is what the exclusion list matches on (today the place name, as
  `conf/weekend.toml` lists it), printed by the plan so `routines` sends that, never a row
  index. Size: S.
- **Blocked by:** nothing.

## Phase H — currency (house rule: newest release of every layer)

Goal: nothing in the stack is behind its newest release without a recorded reason.
Depends on `A5` (a CI that runs, to prove each move). Size: S. Exit: the commands named in
each item report nothing pending.

### H1 — lockfile refresh
- **Class:** dependency drift. `cargo update --dry-run --verbose` in `rust/`: 22
  compatible updates pending on 2026-10-14 (14 on 2026-10-12: `hyper` 1.12.0, `serde_json`
  1.0.154, `syn` 3.0.8, `toml` 1.1.8, …) and `brotli-decompressor` held at 6.0.1 (6.1.0
  available) by its parent.
- **Why now:** routine hygiene; a held-back transitive is how a duplicate major starts.
- **Done when:** `cargo update --verbose` shows no `Updating` line, any `Unchanged … available`
  line names its parent and reason here, `cargo tree -d` is empty, and the gate is green.
- **Blocked by:** `A5`.

### H2 — CI runner and actions ride moving tags
- **Class:** "latest means a named release, never a moving tag". `.github/workflows/ci.yml`
  runs on `macos-15` while the host is macOS 27, and uses `actions/checkout@v7`,
  `Swatinem/rust-cache@v2`, `taiki-e/install-action@v2`, `actions/setup-python@v7` —
  major tags the maintainer moves.
- **Why now:** a CI image two majors behind the host tests a platform the tool never runs
  on (it is macOS/arm64 by `install.sh`'s hard gate).
- **Done when:** the newest GitHub macOS arm64 image is named (measured from GitHub's runner
  list, not assumed), each action is pinned to a release SHA with its tag in a comment, and
  CI is green on it.
- **Blocked by:** `A5`.

### H3 — the vendored crate's clippy ratchet
- **Class:** suppressed findings held under a ceiling instead of fixed.
  `tools/vendor_lint_baseline.jsonc` holds 208 pedantic/nursery findings in
  `vendor/camoufox-rs` under shrink-only ceilings.
- **Why now:** low — the ratchet already stops growth. Parked until Phases P and D land.
- **Done when:** the baseline is empty and the step is replaced by `-D warnings`.
- **Blocked by:** owner decision on whether the vendored crate is worth the effort.

## Sources consolidated (2026-10-12)

Every agent store for this repo was read and given a disposition on 2026-10-12; nothing in
them is open beyond the items above. The table is `git show 47684e4:docs/ROADMAP.md`.
