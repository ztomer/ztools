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

## Where things stand — 2026-11-07

A dated summary, not checked by any gate; re-derive it before editing:
`gh run list --limit 3`, `git log --oneline origin/main..main`, `ztools --version`, `tools/gate.sh --full`.

- **Releases:** https://github.com/ztomer/ztools/releases -- the tags are the record; a machine's
  own copy is read with `ztools --version`. v3.11.1 (`cb57715`): CI on
  `main` is green for that commit (run 42381838128). Since the previous rephase: v3.10.0 landed
  S1, the self-healing dependency table (`0c97219`: red first with the install step removed, 4 of
  8 fake-installer tests failed; real: `brew uninstall bat`, then `ztools deps` reinstalled bat
  0.26.1), so S1 and Phase S are deleted per rule 2; v3.11.0 moved Camoufox to 0.5.9 and the
  browser to 158.0.1-beta.41 (`b861af8`); v3.11.1 made the browser fetch non-interactive and
  streams its progress (`7c5911c`). The `miniz_oxide` duplicate H1 waited on is unified by the
  owner's OW.30 ruling: `rust/Cargo.toml` patches `png` to upstream `b2f10c6` (`359dc9b`, held by
  `tools/tests/test_miniz_oxide_single_version.py`); `cargo tree -d -e normal` prints nothing
  (re-measured 2026-11-07). What stays is removing that patch when `png` releases (H6).
- **The order of the open work, and who can move each item:**
  - agent-runnable now: H1 (the lockfile refresh: cargo reports 39 packages to lock in `rust/`
    and 5 in `vendor/camoufox-rs/Cargo.lock`), then H7 (the one major behind);
  - upstream-gated: H6 (a `png` release after 0.18.1 carrying `b2f10c6`).
- The narrative this section used to carry (the 2026-10-14 to 2026-10-24 gate, stdin floor, CI
  and B7 bullets, and the one-line degraded-banner ruling of 2026-10-21) is history:
  `git show cb57715:docs/ROADMAP.md`. Earlier phases: `git show 6a10475:docs/ROADMAP.md`.

## Standing rulings this file must not re-litigate

- **The degraded summary banner is ONE line** (owner, 2026-10-21: "way too much text to explain
  that raptor failed and we're using foundation. keep it all a single line."). Written by
  `rust/src/ztools/twitter/chain.rs`; the full text of every failure stays in the run record and
  the refused answer's file under `rejected/`.
- **Duplicated code across ztools, sys_updater and app_updates lives in ONE shared library**,
  `ztomer/house-lib` (owner, umbrella LIB.1, rulings OW.20/OW.22/OW.22b: "new private library
  for common tasks. keep GOH QA only."; made PUBLIC by OW.93 so the tap builds with no
  credential). gates_of_heck keeps QA only. ztools takes `selfheal`, the Camoufox rows and the
  platform gate (`install.sh` sources the library's shell form of it) from it; B1 landed in v3.12.0.
- **The `png` duplicate is closed by patching to upstream, not by a vendored fork** (owner,
  OW.30, `359dc9b`).

## Contracts other repos build on

### `ztools weekend exclude` — what the `routines` row button calls

The planner's exclusion list has ONE writer, this command (`rust/src/ztools/weekend/exclude.rs`);
`routines` (its own item F1) calls it rather than editing `conf/weekend.toml`. The contract,
pinned by `rust/tests/cli_weekend_exclude.rs` and `rust/src/ztools/weekend/exclude_tests.rs`:

- **Invocation:** `ztools weekend exclude "<place>"` adds; `ztools weekend exclude "<place>" --undo`
  removes. `<place>` is the row's IDENTITY: its bold name in the newest stored plan, exactly as the
  plan prints it (`| * 3.0/5 | **Kortright Centre for Conservation** (Vaughan) | ...` sends
  `Kortright Centre for Conservation`). Never a row index — rows reorder between runs. Matching is
  case-, whitespace- and quote-insensitive; the plan's spelling is what is stored.
- **Stdout:** exactly one JSON object, on every exit status:
  `{"schema":1,"action":"exclude"|"include","place":<as sent>,"result":"added"|"removed"|"unchanged"|"refused"|"failed","entry":<stored spelling>|null,"file":<path>|null,"entries":<list length>|null,"reason":<text>|null}`.
  `reason` is set exactly when `result` is `refused` or `failed`, and is written for a person.
- **Exit status:** `0` added, removed or unchanged (a repeat is a no-op, never an error); `3`
  refused — the place is not in the newest plan, is malformed, or the file's list sits under a
  table (C8), so retrying the same request cannot help; `1` failed — no config file, unreadable
  or invalid TOML, lock wait over 10 s, or the write failed; `2` is clap's own usage error.
- **Guarantees:** edits the file the planner reads (the first `weekend_exclusions_paths` entry
  with a top-level list), preserving comments and keeping the list above every table; atomic
  (temp file + rename, permissions kept, through a symlink to its target); serialised by an
  exclusive lock on the file, so concurrent presses lose no entry; a no-op leaves the file
  byte-identical. The plan store it validates against is `WEEKEND_OUTPUT_DIR`, the one the tab
  reads.

## Phase H — currency (house rule: newest release of every layer)

Goal: nothing in the stack is behind its newest release without a recorded reason.
Depends on nothing. Size: S. Exit: the commands named in each item report nothing pending.
Execution order: H1, H7, then H6 when upstream allows.

### H1 — lockfile refresh — 39 compatible updates pending in `rust/`, 5 in the vendored lockfile
- **Class:** dependency drift. Re-measured 2026-11-07: `cargo update --dry-run --verbose` in
  `rust/` reports "Locking 39 packages" (among them `aws-lc-rs` 1.18.2 -> 1.19.0, `cc` 1.9.0 ->
  1.11.1, `flate2` 1.1.10 -> 1.1.11, and `brotli` 9.0.0 -> 9.1.1 with `brotli-decompressor`
  6.0.1 -> 6.3.0, which lifts the hold the previous version of this row named), and the vendored
  `vendor/camoufox-rs/Cargo.lock` reports 5 (`jiff` 0.2.38 -> 0.2.39 and its siblings).
- **Why now:** routine hygiene; a held-back transitive is how a duplicate major starts, and the
  brotli hold that blocked part of this row is gone upstream.
- **Done when:** `cargo update --verbose` shows no package `Updating` line in `rust/` or in the
  vendored crate; every `Unchanged … available` line is either H7 or named here with its parent
  and reason; `cargo tree -d -e normal` still prints nothing; the gate is green.
- **Blocked by:** nothing.

### H7 — the one major behind: `rusqlite` 0.40.2 -> 0.41.1
- **Class:** dependency drift, the deliberate kind. It is the only `Unchanged … (available: …)`
  line `cargo update --dry-run --verbose` printed in `rust/` on 2026-11-07, and the house rule
  moves a major in its own commit, never silently and never "later".
- **Why now:** a major left behind is the next duplicate (`rust/deny.toml` already records a
  `hashbrown` pair reached through `rusqlite`).
- **Done when:** `rust/Cargo.toml` asks for the newest `rusqlite` major, the gate is green in its
  own commit, and `cargo update --verbose` prints no `Unchanged` line for it.
- **Blocked by:** H1 (refresh the compatible set first, so the major's diff is only the major).

### H6 — remove the `png` git patch when `png` releases `b2f10c6`
- **Class:** a git-pinned dependency standing in for a release. `rust/Cargo.toml` carries
  `[patch.crates-io] png = { git = …, rev = "b2f10c6a…" }` (OW.30, `359dc9b`) because the
  published `png` 0.18.1 still declares an unused `miniz_oxide 0.8`; upstream commit `b2f10c6`
  ("Remove import of miniz_oxide, unused since 0.17.12") removes it. Re-measured 2026-11-07:
  `cargo search png` still names 0.18.1 as the newest release.
- **Why now:** a git patch is outside `cargo audit`'s release metadata and outside the plain
  `cargo update` path; it should live only until upstream ships.
- **Done when:** the newest `png` release on crates.io contains `b2f10c6`, the `[patch]` stanza is
  deleted, `cargo tree -d -e normal` is still empty, and
  `tools/tests/test_miniz_oxide_single_version.py` is still green.
- **Blocked by:** upstream (the next `png` release after 0.18.1). Umbrella: "a png release
  (ztools H1)" in the umbrella roadmap (`../ROADMAP-consolidated.md`, section 2) names this row's predecessor.

## Sources consolidated (2026-10-12)

Every agent store for this repo was read and given a disposition on 2026-10-12; nothing in
them is open beyond the items above. The table is `git show 47684e4:docs/ROADMAP.md`.
