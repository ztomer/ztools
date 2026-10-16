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

- **Re-derived 2026-10-14: the gate is GREEN and Phase 0 is LANDED.** The list below used to say
  the gate had been red since goh v0.27.0; it is not. `gates/structural.sh --full` is green on
  every step Phase 0 was about — `no early-exit pipe under pipefail`, `every child process names
  its stdin`, `every scanner read its declared subject`, `gates refuse to pass over an empty
  tree`, `no bare python in a hook or gate`. `.python-version` is tracked (A5) and `rustls` is
  0.23.46 in `rust/Cargo.lock` (A6). Phase A is deleted from this file for that reason, per
  rule 2 above; its narrative is git history at the commit that closed each item.
- **CI on `main` was last RED** (every run since 2026-10-10 died in `actions/setup-python`).
  A5's tracking fix has landed, so the next run is the evidence — check `gh run list` rather
  than trusting this line.
- Hooks are goh 0.27.0's stock set. Installed: `ztools 3.8.0`.
- The previous plan's `P2` (a release can be tagged over a red CI) is closed upstream:
  goh `5c5868e` makes the release kit refuse while CI on HEAD is red, pending or missing,
  and `tools/release.sh` execs that kit. Narrative of earlier phases:
  `git show 6a10475:docs/ROADMAP.md`.

## Phase F — owner-requested features

Goal: the owner's feature asks, each behind a test that can go red. Depends on nothing (Phase 0
is landed, so pushes are no longer waiting on it). Size: S. Exit: each item's round-trip tests green and the command used once
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
Depends on nothing. Size: S. Exit: the commands named in
each item report nothing pending.

### H1 — lockfile refresh — one clause left, and no release can satisfy it
- **Class:** dependency drift. `cargo update --dry-run --verbose` in `rust/` had 21 compatible
  updates pending on 2026-10-14, and `brotli-decompressor` held at 6.0.1 (6.1.0 available).
- **Why now:** routine hygiene; a held-back transitive is how a duplicate major starts. Most of
  this row is done — all 21 updated, and `cargo update --dry-run --verbose` now reports ZERO
  package updates (the only `Updating` line left is the crates.io index fetch). It stays open
  for the last clause alone, which is the one that matters: a duplicate major is not hygiene,
  it is two copies of a library to keep in step.
- **Done when:** `cargo update --verbose` shows no package `Updating` line (met); every
  `Unchanged … available` line names its parent and reason here (met — see below); the gate is
  green (met). The remaining clause, `cargo tree -d` empty, is recorded as blocked below rather
  than claimed: ONE real duplicate remains, `miniz_oxide 0.8.9` (via `png 0.18.1`) beside
  `0.9.1` (via `flate2 1.1.10`). Note `libc v0.2.190` appears twice in that output but is ONE
  version reached by two paths (camoufox-rs and core-foundation), not a duplicate — reading
  the raw output as a count would have "fixed" nothing.
- **The held-back line, named:** `brotli-decompressor v6.0.1 (available: v6.1.0)`, held by its
  parent **`brotli v9.0.0`**, a direct dependency. 9.0.0 is the newest release and pins the 6.0
  line.
- **Blocked by:** an upstream `png` release. `png` 0.18.1 is the newest 0.18.x and
  `cargo update -p png` locks 0 packages, so the `miniz_oxide` 0.8 line cannot move until `png`
  adopts 0.9. Re-measure rather than assume: if a `png` has shipped, the refresh is one
  `cargo update` and this row goes.

### H2 — the CI runner image is two majors behind the host
- **Class:** "latest means a named release, never a moving tag". `.github/workflows/ci.yml`
  runs on `macos-15` while the host is macOS 27. The four actions are pinned
  (`tools/tests/test_ci_actions_pinned.py`); the runner image is what remains.
- **Why now:** a runner two majors behind the host judges a different machine -- it can pass
  what the host breaks and break what the host passes, and CI is the gate of record.
- **Done when:** the newest GitHub macOS arm64 image is named (measured from GitHub's runner
  list, not assumed), `ci.yml` runs on it, and CI is green on it.
- **Blocked by:** nothing.

## Sources consolidated (2026-10-12)

Every agent store for this repo was read and given a disposition on 2026-10-12; nothing in
them is open beyond the items above. The table is `git show 47684e4:docs/ROADMAP.md`.
