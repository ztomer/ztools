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

## Where things stand — 2026-10-09

A dated summary of what the runs below already print. Nothing in this section is
checked by a gate, so re-derive it before editing it:
`git log --oneline origin/main..main`, `gh run list --limit 3`,
`tools/gate.sh --full`, and `"$GOH_DIR/gates/goh.sh" home-paths --exclude ^vendor/`.

- `origin/main` is at `18d87a9`; local `main` is at commit `3efd2af` holding the
  eight items completed today (`M1`, `M2`, `M3`, `M5`, `G2`, `G3`, `G4`, `R1`)
  along with swap/paging resilience across `oversize_tests`, `drain_signal`, and
  `eval_not_measured`.
- Prior CI runs (run 37854474001, run 37855122727) failed on workflow-level context,
  runner gate tool manifests, and tests reading home paths. All causes are resolved
  and covered by local tests (`tools/ci_tools.tsv` + `tools/tests/test_ci_tools.py`,
  `GOH_NO_HOME_PATHS=1` clean across 307 files).
- Landed today, each proven by a test made to fail first:
  **`M1`** (repo-relative file_summary checkout rows), **`M2`** (task identity
  fingerprinting and set-aside history aggregation), **`M3`** (production context-fit
  guard in `rust/src/ztools/llm.rs`, fallback skip down `model_fallback_chain`, and
  live Osaurus 8k window overflow probe documented in `docs/MODEL_QUIRKS.md`),
  **`M5`** (smoke prompts fit all documented windows), **`G2`** (307 files clean
  under `GOH_NO_HOME_PATHS=1`), **`G3`** (toolchain header cleaned), **`G4`** (CI gate
  tool manifest dynamic install + pinned tsv), **`R1`** (provenance ledger reading).
- Local verification on commit `3efd2af`:
  `tools/gate.sh --full` passes all 12 CI steps (coverage floor 95%, clippy, cargo audit,
  cargo deny, structural, pytest, ruff, camoufox tests, and secrets scan),
  `python3 -m pytest tools/tests -q` green (136 passed),
  `cargo test --manifest-path rust/Cargo.toml --all-features --lib` passes 893 lib tests.
- All temporary agent sessions (OpenCode and Claude worktrees/caches) are purged.

## Phase L — land what is built

Depends on nothing in itself. A measurement or gate built on an uncommitted tree
describes a state nobody can check out, so every phase after this depends on it; and a
green LOCAL gate says nothing about a checkout that lives somewhere else — CI is the
only instrument that can see that, which is why this phase leaves CI red until the
run itself is green.

### L2 — CI is the gate of record, and its runs are green
- **Class:** a gate that has not been run is indistinguishable from one that passes —
  and when it first DOES run, what it finds is this item's work, not a surprise.
- **Why now:** `18d87a9` landed the first fix on `main`, but GitHub Actions CI runs
  remained red awaiting the gate manifest (`G4`), home path elimination (`G2`), and
  relative checkout evaluation (`M1`). Committing and pushing the completed batch
  is required before any downstream measurement is sound.
- **Done when:** `gh run list --limit 3` shows a green push on `main` with no
  `--no-verify` anywhere, AND `tools/gate.sh --full` is green locally for the same
  commit. The first green run is evidence, not the fix; the fix is what it names.
- **Blocked by:** pushing this tree (the owner's call), then a runner that agrees.

## Phase M — what the eval records must mean what it says

Depends on `L`. The class for the whole phase: **a stored number that no longer
describes the thing its name says it describes.** `M1`, `M2`, and `M3` landed together
on 2026-10-08/09, so what is left is re-running the eval sweep against the tasks as
they now stand.

### M4 — Re-sweep, and re-derive `[best_models]`
- **Class:** a slot winner measured on a task that has since changed.
- **Why now:** `[best_models].think` (`qwen3.8-27b-jang_6d`) was chosen partly as "the
  only model above 60 on taxes_audit_readiness and file_summary_mixed together"; that
  half is unconfirmed. `foundation` is now NOT MEASURED on both file-summary tasks.
  `docs/MODEL_QUIRKS.md` points here for re-deriving its tables.
- **Done when:** a full-roster sweep under the final tasks is in the history with
  current fingerprints, `conf/config.toml`'s slot table and comments cite it, and the
  embedded-defaults drift test passes against the new slots.
- **Blocked by:** `L2`, idle GPU (runs serially through `tools/osaurus_one.sh`).

## Phase G — gates that still cannot fail

Depends on `L`. The class for the whole phase: **a check that reports green over
something it never inspected.** Both items left here are about the CHECKER rather
than the code: which checker ran (`G6`) and who owns the lint path (`G5`).

### G5 — Retire `tools/shell_lint_extra.sh` once the house shellcheck reaches it
- **Class:** a local fork of shared machinery. The hooks themselves converged on the
  stock ones on 2026-10-06 (`install.sh --force`, recorded in
  `.githooks/.goh-installed/`); this is what is left.
- **Why now:** the repo-local step exists only because the house shellcheck glob misses
  `.githooks/*` and `bin/*` (X5). Two lint paths for one class of file drift.
- **Done when:** X5 has shipped, the house structural gate goes red on a syntax error
  planted in `.githooks/pre-push` and in `bin/ab_test`, and the local step and script
  are deleted.
- **Blocked by:** `X5`.

### G6 — CI runs a NAMED release of `gates_of_heck`, not its `main`
- **Class:** a gate whose checker can change under it, so a verdict move cannot be
  attributed to the repo and `git bisect` of a gate change is impossible.
- **Why now:** `.gatesrc`'s hooks delegate to whatever `gates_of_heck` currently is,
  and CI clones its `main` at depth 1, so a push is checked by a different checker
  than the last push was. The standing house rule is that `main` is never a pin; a
  local checkout already sits 52 commits past v0.24.0 (`G2`'s counts, `L2`'s history
  and `X1`-`X3` all shipped inside that window unpinned).
- **Done when:** the workflow's clone in `.github/workflows/ci.yml` names a released
  tag, CI passes on it, and the local `GOH_DIR` reports the same tag.
- **Blocked by:** the owner's pick of the first pinned tag.

## Phase R — readings, not builds

Depends on nothing in this file; each is a measurement someone has to take. `R1`
was taken and recorded in `docs/MODEL_QUIRKS.md`; `R2` is the one that needs a
machine this desk does not have.

### R2 — A Linux pressure threshold from a Linux box
- **Class:** a reader that is right about what it reads and silent about what it
  cannot.
- **Why now:** `rust/src/ztools/eval/signals_platform.rs` reads swap and
  `MemAvailable` on Linux and deliberately not PSI, because a stall threshold that was
  not measured would be a guess.
- **Done when:** PSI `some`/`full` is measured under a quiet and a loaded sweep on a
  Linux host, and a threshold lands with that measurement cited.
- **Blocked by:** access to a Linux host.
