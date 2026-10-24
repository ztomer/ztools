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

## Where things stand — 2026-10-21

A dated summary, not checked by any gate; re-derive it before editing:
`gh run list --limit 3`, `git log --oneline origin/main..main`, `tools/gate.sh --full`.

- **Re-derived 2026-10-14: the gate is GREEN and Phase 0 is LANDED.** The list below used to say
  the gate had been red since goh v0.27.0; it is not. `gates/structural.sh --full` is green on
  every step Phase 0 was about — `no early-exit pipe under pipefail`, `every child process names
  its stdin`, `every scanner read its declared subject`, `gates refuse to pass over an empty
  tree`, `no bare python in a hook or gate`. `.python-version` is tracked (A5) and `rustls` is
  0.23.46 in `rust/Cargo.lock` (A6). Phase A is deleted from this file for that reason, per
  rule 2 above; its narrative is git history at the commit that closed each item.
- **Re-measured 2026-10-20: the stdin floor is 18, against the 20 process calls a full scan
  reads.** It was 12 against 14, set on 2026-10-14 when the Python tools were fewer; at 12
  six calls could go unseen, and the floor is the only thing that says a scan has stopped
  reading its subject. Nothing is seeded: every one of the 20 calls declares its stdin, and
  the check is red the moment one of them does not.
- **The step is NAMED in `.gatesrc` (`GOH_SUBPROCESS_STDIN=on`), and it is proved red both
  ways (2026-10-20).** The house turned the check on for every repo on 2026-10-13; writing
  the key down says this repo chose it, in the file the next reader opens. Red-first, in a
  clean clone of the tree: with every declaration stripped (20 keywords out of 11 files) the
  check exits 1 and names the calls — "20 call(s) leave stdin undeclared in tracked source:
  tools/tests/test_gpu_lock_shell.py:106: subprocess.Popen inherits the caller's stdin; pass
  stdin= or input=" — and with the floor raised to 100 it exits 1 naming the key:
  "`GOH_SUBPROCESS_STDIN_MIN_CALLS` read 20 process call(s) and .gatesrc declares a floor of
  100". As committed it exits 0: "every one of 20 process call(s) declares its stdin".
- **CI on `main` is GREEN**, run 39699419247 (2026-10-21) — the run for `dc0ff71`, the commit
  the v3.9.2 tag names and the one the release kit required before it cut anything. The
  `make ci` step inside it is the full 14-step gate. Two reds were behind the previous run
  and the tree was innocent in both. (1)
  `Swatinem/rust-cache` caches `~/.cargo/bin`, so the
  manifest's `cargo install` met a binary a cache had restored with no record in
  `~/.cargo/.crates.toml`, and `cargo install` refuses exactly that (`binary ... already exists
  in destination`); the
  step runs under `set -euo pipefail`, so every gate step after it was skipped — for two days
  the gate of record ran nothing, and this file read the redness as the code's. goh's
  `required_tools.py` now installs per tool behind a presence guard, so a binary that is on
  PATH is a satisfied requirement whoever put it there. (2) The runner's `stable` clippy was
  1.98.0 where this host's is 1.99.0, and 1.98 fires `too_long_first_doc_paragraph` on two doc
  comments 1.99 accepts: the local 14/14 was green on a tree CI refused. Both are in git
  history; `git log --oneline -6` names them. A THIRD red was behind this run, and the tree
  WAS at fault: `tokio` 1.54.0, landed by the H1 refresh, was yanked upstream, and the
  advisory gate (`cargo audit`, `cargo deny`) went red on its own -- the lockfile moved to
  1.54.1 in `3b5caa4` before this push. Re-derive this bullet with `gh run list` rather
  than trusting it.
- **Umbrella rows "ztools CI red on a gitignored `.python-version`" and "tap bump outside the
  gate (6.8)" are CLOSED, re-derived 2026-10-24.** The first closed in `dbfa60c` (2026-10-14):
  `.python-version` is tracked and `tools/tests/test_ci_files_tracked.py` fails when any file a
  workflow reads is untracked; CI on `main` has been green since (latest run 39701639021 for
  `6bb2166`). The second closed in `3fb24d0`: `tools/release.sh` execs the house kit, whose tap
  step is the kit's tap_bump.sh, through the tap's own gate; no hand-rolled tap push remains in the tree
  (`grep homebrew-tap` finds only the kit's `--tap` argument). No release was cut.
- **Umbrella B7 (b), `twitter-evening` failing, measured 2026-10-17:** not a model, endpoint
  or env fault -- raptor and `fm` both answered; the gate refused both answers over citation
  FORM (no `@`; time of day only; `;`-joined sources; unbracketed `per @h | ts`). The repair
  parses all four now (`docs/MODEL_QUIRKS.md`, CHANGELOG "B7 (b)"); no citation form seen
  in `rejected/` is left unparsed. The repair shipped in v3.9.0 (2026-10-19) and the
  installed `ztools` is 3.9.2, so scheduled runs run the repair now.
- **The degraded summary banner is ONE line (owner ruling, 2026-10-21).** "way too much
  text to explain that raptor failed and we're using foundation. keep it all a single
  line." `rust/src/ztools/twitter/chain.rs` writes it: which model failed and why, which
  model answered instead, what was repaired before saving, and how far to trust the result
  -- one line, in the document and on the terminal, by the same function, so the two cannot
  drift apart. The FULL text of every failure is kept where a reader can open it: the run
  record (the CLI prints every reason under the notice) and the refused answer's file under
  `rejected/`, whose header carries the reason the gate gave. Shipped in v3.9.2.
- Hooks are goh 0.27.0's stock set. Installed: `ztools 3.9.2`.
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

## Phase S — self-healing runtime dependencies (owner ask O1)

Goal: a missing runtime dependency is installed, never a crash. Depends on nothing. Size: M.

### S1 — every runtime dependency is declared, probed, and installed on a miss
- **Owner's words (O1, umbrella §0b):** "ztools, app_updates, update_local should be self
  healing - if a dependency is not available it should be installed."
- **Class:** a dependency assumed rather than probed. The Camoufox browser was resolved from disk
  and a miss was a hard error telling the user to run `pip install camoufox && python3 -m camoufox
  fetch` by hand; `bat` was tried and silently skipped. One table (`rust/src/ztools/deps.rs` (new))
  declares every external binary, browser and Python module with its pinned version and install
  command; `rust/src/ztools/selfheal.rs` (new) is the shape (probe, install with a stated reason,
  re-probe, degrade with a reason) and is the same shape `app_updates` and `sys_updater` carry.
- **Done when:** the fake-installer tests in `rust/src/ztools/selfheal_tests.rs` (new) pass and
  were seen red first; `ztools deps` lists every row with its state; one real miss (`bat`,
  uninstalled with brew) is reinstalled by ztools itself.
- **Blocked by:** nothing.
- **Why now:** a fresh machine, or a cleared Camoufox cache, stops the timeline collector until a
  human reads an error and types an install command.

## Phase H — currency (house rule: newest release of every layer)

Goal: nothing in the stack is behind its newest release without a recorded reason.
Depends on nothing. Size: S. Exit: the commands named in
each item report nothing pending.

### H1 — lockfile refresh — refreshed 2026-10-21; the duplicate clause waits on the next `png` release
- **Class:** dependency drift. `cargo update --dry-run --verbose` in `rust/` had 21 compatible
  updates pending on 2026-10-14, and `brotli-decompressor` held at 6.0.1 (6.1.0 available).
- **Why now:** routine hygiene; a held-back transitive is how a duplicate major starts. Most of
  this row is done — all 21 updated, and that was the last batch that needed saying twice.
  Re-measured 2026-10-21: SEVEN compatible updates were pending in `rust/` (`cc` 1.8.0→1.9.0,
  `combine` 4.6.8→4.6.9, `crc32fast` 1.5.2→1.5.3, `find-msvc-tools` 0.1.14→0.1.15, `libc`
  0.2.190→0.2.191, `schannel` 0.1.29→0.1.30, `tokio` 1.53.3→1.54.0) and NINE more in the
  vendored crate's own tracked lockfile (`vendor/camoufox-rs/Cargo.lock` — the one CI's last
  step builds from, and the one the row's goal covers too: `defmt` 1.1.1→1.2.0, `defmt-macros`
  1.1.1→1.2.0, `defmt-parser` 1.0.0→1.1.0, `jiff` 0.2.37→0.2.38, `jiff-static` 0.2.37→0.2.38,
  `libc` 0.2.190→0.2.191, `serde_json` 1.0.151→1.0.154, `syn` 3.0.6→3.0.9, `unicode-ident`
  1.0.26→1.0.27). All sixteen are in. The clause that matters is still the one below: a
  duplicate major is not hygiene, it is two copies of a library to keep in step.
- **Done when:** `cargo update --verbose` shows no package `Updating` line (met 2026-10-21 —
  the only `Updating` line left is the crates.io index fetch, in `rust/` and in the vendored
  lockfile alike); every `Unchanged … available` line names its parent and reason here (met —
  see below); the gate is green (met). The remaining clause, `cargo tree -d` empty, is recorded
  as blocked below rather than claimed: ONE real duplicate remains, `miniz_oxide 0.8.9` (via
  `png 0.18.1`) beside `0.9.1` (via `flate2 1.1.10`), re-measured 2026-10-21 and unchanged by
  the refresh. The ONE in-tree route to it worth trying — swapping `flate2` to its `zlib-rs`
  backend, which does not use `miniz_oxide` at all — was tried on 2026-10-21 and MEASURED not to
  reach it; the measurement and its cause are below, so the next reader does not spend the round
  again. Note `libc v0.2.191` appears twice in that output but is ONE version reached by
  two kinds of edge: the first block is the normal-dependency tree (`camoufox-rs`,
  `core-foundation`, `dirs-sys`, …), the second the build-dependency tree (`cc`, `jobserver`).
  Same version, two places it is required — reading the raw output as a count of duplicates
  would have "fixed" nothing. The vendored lockfile has no duplicate at all
  (`cargo tree -d` there prints "nothing to print").
- **The held-back line, named:** `brotli-decompressor v6.0.1 (available: v6.1.1)`, held by its
  parent **`brotli v9.0.0`**, a direct dependency (`brotli = "9"` in `rust/Cargo.toml`).
  9.0.0 is the newest release of `brotli` and its requirement is `brotli-decompressor ~6.0`
  (with `default-features = false`), which EXCLUDES the 6.1 line — that is the whole reason,
  not "it pins the 6.0 line". It is the only `Unchanged … available` line `cargo update --verbose`
  prints in `rust/`; the vendored lockfile prints none. A refresh adds a held-back line whenever
  a parent stops accepting a newer child, so this list is re-derived, never inherited. Re-measured
  2026-10-21: the AVAILABLE version moved from 6.1.0 to 6.1.1; the hold itself is unchanged.
- **Blocked by:** the next `png` release — and by nothing in this repo's own manifests. Re-measured
  2026-10-21 with the backend swap applied and then reverted: BOTH copies of `miniz_oxide` sit
  inside `png 0.18.1`'s OWN dependency edge, so no change to THIS manifest removes either one.
    - `png 0.18.1` declares `miniz_oxide = { version = "0.8", features = ["simd"] }` — hard, not
      optional — and its SOURCE NEVER USES IT. The only occurrence outside its manifest and
      lockfile is one stale doc comment (`src/decoder/zlib.rs` in the registry copy), and png's
      own changelog says "Removed remaining uses of miniz_oxide for decoding". Upstream agrees:
      image-rs/image-png commit `b2f10c6a7e` ("Remove import of miniz_oxide, unused since
      0.17.12", merged 2026-09-23) deletes exactly that line from png's manifest. The published
      0.18.1 (2026-02-14) predates that commit, so the next `png` release carries the removal —
      and then the tree holds `miniz_oxide 0.9.1` once and `cargo tree -d` is empty with NO code
      change here.
    - `png 0.18.1` also declares `flate2 = "1.0.35"` WITH default features, and cargo unifies
      features across dependents: that request re-enables `flate2`'s `rust_backend` ->
      `miniz_oxide ^0.9` whatever this manifest asks for. Measured with the swap applied
      (`flate2 = { version = "1", default-features = false, features = ["zlib-rs"] }` and
      `png = { version = "0.18.1", features = ["zlib-rs"] }`), the resolved crate is
      `flate2 v1.1.10 FEATURES=any_impl,any_zlib,default,miniz_oxide,runtime_detection,rust_backend,zlib-rs`
      (`cargo tree -f "{p} FEATURES={f}" -e normal`) and `cargo tree -d` STILL prints
      `miniz_oxide v0.8.9` AND `miniz_oxide v0.9.1`. The swap changes which backend `flate2`
      COMPILES — `any_zlib` suppresses the `miniz_oxide` backend in `flate2`'s `src/ffi/mod.rs` —
      so it turns both copies into compiled-and-unused code and removes neither from the graph.
      The swap is otherwise SOUND — with it applied the whole suite passes (1089 lib tests, the
      PNG encode/decode pair among them), so what refuses it is the GRAPH, not the codec. It is
      also a real behaviour change for every `flate2` call in the tree (PNG encoding in the
      eval path, gzip and deflate in the twitter capture path), so it is not taken on its own: it
      buys no duplicate.
    - Re-derive rather than assume: `cargo update -p png --dry-run --verbose` prints
      `Locking 0 packages` today because 0.18.1 is still the newest release on crates.io (checked
      against the LIVE index, not a cached one: 0.17.16, 0.18.0-rc, 0.18.0, 0.18.1 — no 0.18.2, no
      0.19). When it ships, the refresh is one `cargo update` in `rust/` and this row goes.
    - The only in-tree route TODAY is carrying a fork of the codec: `png` vendored under
      `vendor/` with upstream's one-line manifest patch applied, which is what `vendor/patches/`
      is for. NOT taken, and left as an owner decision rather than a change: it makes this repo
      the owner of a codec's source and its advisories, to close one duplicate that upstream has
      already fixed and that `rust/deny.toml` records as living outside this repo ("none can be
      unified without patching `png`, `flate2`, `rusqlite` or a proc-macro crate upstream", which
      is why `multiple-versions` is `warn` there and not `deny`).

## Sources consolidated (2026-10-12)

Every agent store for this repo was read and given a disposition on 2026-10-12; nothing in
them is open beyond the items above. The table is `git show 47684e4:docs/ROADMAP.md`.
