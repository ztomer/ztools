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

## Where things stand — 2026-10-12

A dated summary, not checked by any gate; re-derive it before editing:
`gh run list --limit 3`, `git log --oneline origin/main..main`, `tools/gate.sh --full`.

- **CI on `main` is RED, and has been for four pushes** (runs from 2026-10-10 20:57 UTC on:
  `changelog: v3.6.0 stanza`, `chore: version 3.7.0`, `settings: …`, `docs: …`). Every one
  dies in `actions/setup-python` before the gate runs: `.python-version doesn't exist`. The
  last green run is `fix(eval): use float difference threshold…` (2026-10-10 01:10 UTC).
  Releases v3.6.0, v3.7.0 and v3.8.0 were tagged and published over that red. `P1`, `P2`.
- The local gate is green: `tools/gate.sh --full` passed 12/12 steps on HEAD `6a10475`
  (2026-10-12, 1 min 43 s). No CI run has reached the gate on any push since `83cf63b`
  (2026-10-10 01:10 UTC).
- Installed: `ztools 3.8.0` (Homebrew). HEAD is two commits past v3.8.0, docs/settings only.
- Phases W, M, T, L, G, R and S from the previous plan all landed; their narrative is in git
  (`git show 6a10475:docs/ROADMAP.md`) and in `CHANGELOG.md`. Product work has no open item;
  everything open below is the machinery that would have caught the red CI, and documents
  that still describe the retired Python implementation.

## Phase P — the gate of record is green and can fail

Goal: CI is green on `main`, releases cannot ship over a red CI, and the two roadmap checks
that currently pass over nothing can go red. Depends on nothing. Size: S (one session).
Exit: `gh run list --branch main --limit 1` reports `success` on a commit after `P1`, and
every new check in this phase has been seen red once on the real tree.

### P1 — CI checks out a file the repo does not track
- **Class:** CI reads a file only the developer's checkout has, so the local gate passes
  and the gate of record cannot start. `.github/workflows/ci.yml` says
  `python-version-file: .python-version`; `.gitignore:85` (`# pyenv`) ignores that file and
  it has never been tracked. Introduced by `22d8009` ("CI reads .python-version").
- **Why now:** CI has not reached the gate on any push since `83cf63b`; v3.6.0–v3.8.0 were
  proven only locally. Rule #17's "local gate identical to CI" is false while this stands.
- **Done when:** a pytest in `tools/tests/test_ci_files_tracked.py` (new) parses every
  `*-file:` / `hashFiles(...)` / `path:` input in `.github/workflows/ci.yml` and asserts each
  is `git ls-files`-tracked — seen red on today's tree — then `.python-version` is tracked
  (drop `.gitignore:85`) and the next CI run on `main` is `success`.
- **Blocked by:** nothing.

### P2 — a release can be tagged over a red CI
- **Class:** the release path consults the LOCAL gate only. `tools/release.sh` passes
  `--gate "make ci"` to the house release kit; nothing asks GitHub whether the gate of
  record passed on the commit being tagged, so v3.6.0–v3.8.0 shipped over `P1`.
- **Why now:** without it, the next environment-only CI break ships the same way.
- **Done when:** the release kit refuses to tag a commit whose CI run is not `success`
  (or names an explicit `--ci-skip <reason>`), proven by `tools/release.sh <ver> --dry-run`
  refusing on today's red `main`.
- **Blocked by:** cross-repo ask — the sequencing lives in `gates_of_heck`'s release kit, not
  here. If goh declines, a preflight in `tools/release.sh` carries it for this repo only.

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

## Phase H — currency (house rule: newest release of every layer)

Goal: nothing in the stack is behind its newest release without a recorded reason.
Depends on `P1` (a CI that runs, to prove each move). Size: S. Exit: the commands named in
each item report nothing pending.

### H1 — lockfile refresh
- **Class:** dependency drift. `cargo update --dry-run --verbose` in `rust/` (2026-10-12):
  14 compatible updates pending (`hyper` 1.12.0, `serde_json` 1.0.154, `syn` 3.0.8, `toml`
  1.1.8, …) and `brotli-decompressor` held at 6.0.1 (6.1.0 available) by its parent.
- **Why now:** routine hygiene; a held-back transitive is how a duplicate major starts.
- **Done when:** `cargo update --verbose` shows no `Updating` line, any `Unchanged … available`
  line names its parent and reason here, `cargo tree -d` is empty, and the gate is green.
- **Blocked by:** `P1`.

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
- **Blocked by:** `P1`.

### H3 — the vendored crate's clippy ratchet
- **Class:** suppressed findings held under a ceiling instead of fixed.
  `tools/vendor_lint_baseline.jsonc` holds 208 pedantic/nursery findings in
  `vendor/camoufox-rs` under shrink-only ceilings.
- **Why now:** low — the ratchet already stops growth. Parked until Phases P and D land.
- **Done when:** the baseline is empty and the step is replaced by `-D warnings`.
- **Blocked by:** owner decision on whether the vendored crate is worth the effort.

## Sources consolidated (2026-10-12)

- **This file's previous version** (`6a10475`): every item had landed; its narrative is in
  `git show 6a10475:docs/ROADMAP.md`. One claim in its 2026-10-10 sources list was wrong:
  "CI pinned to 3.14 is fixed (CI reads `.python-version`)" — that change is what turned CI
  red (`P1`).
- **docs/EVALUATION_WORKFLOW.md** (a "Gemini MUST inspect every eval run" directive plus
  four prompt actions): removed. Action 1 (context bounding) is pinned by
  `rust/src/ztools/eval/prompts/file_summary_pins_tests.rs`; Action 2 by
  `correct_weather_labels`; Action 3 is superseded (the planner carries LOCATION verbatim
  and the eval caps generic locations — a prompted city default would be the mandated-
  placeholder class C4); Action 4 by the `(@handle | timestamp)` rule in `conf/prompts.toml`
  and the DEGRADED banner. The per-run directive was disciplinary (rule #3) and linked a
  deleted Antigravity brain file. Recover: `git show 6a10475:docs/EVALUATION_WORKFLOW.md`.
- **Other `docs/*.md`**: reference and measurement records, not plans; kept (`D1`–`D3`).
- **`.opencode/opencode-loop/`**: empty; removed.
- **`.claude/`** in the repo: `.claude/settings.local.json` (one stale allow rule for the retired
  `model_eval` Python module) and an empty `worktrees/`; nothing to carry.
- **Claude Code CLI**: no transcripts; memory store kept (two feedback notes: "no invented
  constants", "macOS-only — hard-fail, don't verify Linux"). The latter is already
  structural (`install.sh` platform gate).
- **Claude desktop**: no ztools session. **Gemini CLI**: no ztools dir (only `scripts`'
  chats mention the path). **Antigravity**: no ztools conversation; the IDE conversation
  that mentions the path is a ZeroThunder one quoting old shell aliases (none survive in
  `~/.zshrc`) — left alone.
- **RUSTIFICATION_ROADMAP.md** and **ROADMAP-consolidated.md** (Projects root): no ztools
  item in either (grepped for the name, its tools and its modules).
- **git**: one branch (`main`), one worktree, no stash, clean tree.
