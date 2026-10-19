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

## Where things stand — 2026-10-16

A dated summary, not checked by any gate; re-derive it before editing:
`gh run list --limit 3`, `git log --oneline origin/main..main`, `tools/gate.sh --full`.

- **Re-derived 2026-10-14: the gate is GREEN and Phase 0 is LANDED.** The list below used to say
  the gate had been red since goh v0.27.0; it is not. `gates/structural.sh --full` is green on
  every step Phase 0 was about — `no early-exit pipe under pipefail`, `every child process names
  its stdin`, `every scanner read its declared subject`, `gates refuse to pass over an empty
  tree`, `no bare python in a hook or gate`. `.python-version` is tracked (A5) and `rustls` is
  0.23.46 in `rust/Cargo.lock` (A6). Phase A is deleted from this file for that reason, per
  rule 2 above; its narrative is git history at the commit that closed each item.
- **CI on `main` is GREEN**, run 38899800424 (2026-10-16) — the first green run since
  2026-10-10, and the `make ci` step inside it is the full 14-step gate. Two reds were behind
  it and the tree was innocent in both. (1) `Swatinem/rust-cache` caches `~/.cargo/bin`, so the
  manifest's `cargo install` met a binary a cache had restored with no record in
  `~/.cargo/.crates.toml`, and `cargo install` refuses exactly that (`binary ... already exists
  in destination`); the
  step runs under `set -euo pipefail`, so every gate step after it was skipped — for two days
  the gate of record ran nothing, and this file read the redness as the code's. goh's
  `required_tools.py` now installs per tool behind a presence guard, so a binary that is on
  PATH is a satisfied requirement whoever put it there. (2) The runner's `stable` clippy was
  1.98.0 where this host's is 1.99.0, and 1.98 fires `too_long_first_doc_paragraph` on two doc
  comments 1.99 accepts: the local 14/14 was green on a tree CI refused. Both are in git
  history; `git log --oneline -6` names them. Re-derive this bullet with `gh run list` rather
  than trusting it.
- **Umbrella B7 (b), `twitter-evening` failing, measured 2026-10-17:** not a model, endpoint
  or env fault -- raptor and `fm` both answered; the gate refused both answers over citation
  FORM (no `@`; time of day only; `;`-joined sources; unbracketed `per @h | ts`). The repair
  parses all four now (`docs/MODEL_QUIRKS.md`, CHANGELOG "B7 (b)"); no citation form seen
  in `rejected/` is left unparsed. The installed `ztools 3.8.0` predates it, so
  scheduled runs keep the old repair until the next release is installed.
- Hooks are goh 0.27.0's stock set. Installed: `ztools 3.8.0`.
- The previous plan's `P2` (a release can be tagged over a red CI) is closed upstream:
  goh `5c5868e` makes the release kit refuse while CI on HEAD is red, pending or missing,
  and `tools/release.sh` execs that kit. Narrative of earlier phases:
  `git show 6a10475:docs/ROADMAP.md`.

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

## Sources consolidated (2026-10-12)

Every agent store for this repo was read and given a disposition on 2026-10-12; nothing in
them is open beyond the items above. The table is `git show 47684e4:docs/ROADMAP.md`.
