# Plan — the four open items

**This file is transient by construction.** House rule #13 allows exactly one
forward-looking file (`docs/BACKLOG.md`); per-feature plan files rot into a graveyard.
So this one has a death condition: when an item lands, its *plan* goes to git history
and only its *defect class* graduates into `BACKLOG.md`. When all four land, delete
this file. If it is still here with items ticked off, it has already started rotting.

Written 2026-08-19, immediately after a reboot, against `000b4b0`.

## Status, 2026-08-19

**Step 0 (cross-cutting instrumentation) and item 1's two structural gates are
DONE and green.** Nothing has been measured yet -- every change below is to the
instruments, taken deliberately before any number is recorded.

| | shipped | where |
|---|---|---|
| ✓ | Completeness derived by diffing asked-for against reported-back tasks | `eval/completeness.py` |
| ✓ | Truncated runs marked in history, refused by the averager, announced | `eval/report_history.py` |
| ✓ | `complete` carried into stats; mean prints `(partial)`; `N` column | `eval/report_core.py` |
| ✓ | Cross-model table rows are a UNION, not the first model's results | `eval/report_core.py` |
| ✓ | `GATE_TASKS` recorded WITH evidence; `ranking_mean` excludes them | `eval/discrimination.py` |
| ✓ | `disagreements()` re-derives the classification from each run's data | `eval/discrimination.py` |
| ✓ | Oversize warn-and-continue replaced by a refusal + override seam | `eval/cli_runtime.py`, `eval/cli.py` |
| ✓ | Sample-clean gate consults the GPU lock | `eval/samples.py` |

Gate at the time of writing: **2731 passed, 95.42% coverage** against a 95 floor
(2685 / 95.38% before this work). 20 mutants were run against the new guards;
19 were detected and the one that survived is described below.

**Four deviations from the plan as written**, each a deliberate improvement on it
rather than a shortcut:

1. **No separate `eval_quarantine.json`.** The plan called for one. A second store
   is a second thing consumers forget to read -- the parallel-pipeline drift class
   this repo already has a rule about. Truncated entries are instead MARKED in the
   one store and `load_historical_stats` refuses to average them, reporting an
   `excluded` count so the discrepancy stays visible. The individual task scores
   are real (the task that ran, ran); only the aggregate over them is not.
2. **Completeness is DERIVED, not threaded.** The plan said "thread completeness
   out of `run_eval`". There are two abandon paths (`watchdog.check_stall` and the
   consecutive-infra break) and a third is one bug away; a flag is the per-knob
   hook rule #12 warns about, and the path that forgets to set it is the one that
   ships. Diffing the task sets covers a future abandon path the day it is written.
3. **A second instance of the class was found and fixed.** `print_cross_model_comparison`
   took its ROW SET from `all_results[0]`, so a truncated first model silently
   deleted rows for every other model, and an empty first model blanked the whole
   table. Rule #6(b): grep for siblings, not just the instance.
4. **An existing test was part of the defect.** `test_model_too_big_for_memory`
   asserted `"70GB" in out`, which was true for both the warning it was written
   for and the refusal that replaced it -- it could see the number but not the
   consequence. Updated per rule #8 rather than left pinning the old behaviour.

**One mutant survived and found a real blind spot**, recorded because it is the
more useful half of the exercise: deleting the `continue` after the oversize
refusal -- print "Skipping" and then measure the model anyway -- passed every
assertion, because they only checked the message text. That is warn-and-continue
wearing a new word. The test now asserts `run_eval` was never called.

**A behaviour-change warning given earlier was WRONG and is retracted.** It said the
oversize gate would refuse `qwen3.8-27b-mxfp8` at 28.8GB. It does not: the gate reads
psutil's `available` (45.6GB on a settled machine), not the `Pages free` figure that
prediction was based on. See the correction in Preconditions, which also records a
real defect in the gate that the same investigation turned up.

**What is NOT done:** every measurement. Items 1, 2 and 3 all need the GPU and a
quiet machine, and item 4 is a standing practice rather than a task.

---

## Preconditions that apply to every item below

Measured at the time of writing, three minutes after boot:

| quantity | value | verdict |
|---|---|---|
| swap used | 0.00 MB | clean |
| compressor | 0.0 GB | clean |
| free | 21.9 GB | **MISREAD -- see correction** |
| active | 21.4 GB | boot settling |
| osaurus | pid 842, 9.5 GB resident, serving 1337 | a model was loaded |
| GPU lock | not held by anyone | free to take |

**CORRECTION, 50 minutes later.** The 21.9 GB above is `Pages free` from vm_stat, and
it is NOT available memory. macOS holds most reclaimable memory in `inactive` and
`speculative`, neither of which appears in `Pages free`:

    free 26.7 + inactive 16.9 + speculative 2.0 = 45.6 GB
    psutil available                            = 45.6 GB

The box was never short of memory. Re-measured on a settled machine: 7.2 GB used
total, load average 2.87, osaurus down to 2.52 GB resident having released the
model, largest non-osaurus process 0.62 GB.

Consequences, both of which were stated wrongly before this correction:

- **The oversize gate does NOT refuse `qwen3.8-27b-mxfp8`.** It reads psutil's
  `available` (45.6 GB), not `Pages free`, so 28.8 GB clears the 36.5 GB limit.
  Item 1 needs no `--allow-oversize`.
- **The "start from a known state" advice still stands**, but for a different
  reason: not because 9.5 GB of headroom is missing, but because a model already
  resident is a state you did not choose.

Load average WAS 48 at three minutes up. That is boot settling, not work -- wait for
it to fall below ~3 before the first timing, and re-check swap and compressor at that
moment rather than trusting any reading taken during boot.

**AND A DEFECT IN THE GATE THIS EXPOSED -- since FIXED.** `eval/samples.py`
deliberately gates sample cleanliness on PRESSURE (swap, compressor) rather than
headroom, and says why:

> After a sweep the page cache legitimately holds tens of GB of model weights and
> "available" drops to ~12GB on a perfectly healthy box, so a headroom threshold
> refuses to record on exactly the machine you most want readings from.

The oversize refusal first shipped in this session gated on HEADROOM -- the quantity
that docstring warns against. Clean file-backed pages holding model weights evict
instantly, but psutil does not count the `active` ones as available, so immediately
after a sweep the gate would refuse a model that runs fine.

FIXED. `eval/memory.py` now holds the machine-memory truth and BOTH gates read it,
so there is one definition of "thrashing" rather than two that drift:

- `pressure()` returns `(swap_gb, compressor_gb)` or None. Tri-state deliberately:
  None is "cannot tell" and is not the same as False.
- `is_thrashing()` is the unambiguous signal -- a machine already paying for memory
  it does not have.
- `reclaimable_available_gb()` is psutil's `available` PLUS active file-backed pages
  PLUS purgeable. That extra term is the whole point. The active-file-backed estimate
  over-subtracts on purpose, so it understates what is reclaimable and never
  overstates it: the safe direction for a gate whose failure mode is producing a
  wrong number.

`oversize_refusal` asks pressure FIRST and headroom second; `eval/cli.py` no longer
computes `(100 - mem_pct) / 100 * total` and hands it over, because a
percentage-of-total is wrong twice -- it ignores what the kernel would reclaim, and
it made a 64GB box with 7.2GB in use look nearly full. `samples.machine_is_uncontended`
delegates its pressure read to the same module.

Calibrated against the state that motivated it: with psutil reporting 4GB available
and 30GB of file-backed pages ACTIVE, the old gate saw 4GB and would refuse a 20GB
model; the new one sees ~32GB reclaimable and runs it. 11 mutants were run against
the new guards and all 11 were detected, including the two that reintroduce this
exact bug (headroom-only, and page-cache-not-counted).

One consequence worth knowing: the refusal now reads the real machine by default, so
a test asserting "70GB is refused" would pass on a 64GB laptop and fail on a 128GB
one. conftest's existing `deterministic_machine_contention` fixture was extended to
pin the memory readings, which is the rule it already enforced for pressure.

Every item runs under `tools/gpu_lock.sh` / `lib/gpu_lock.py`. One command at a time.

---

## Cross-cutting finding — the report layer cannot tell a truncated run from a complete one

This is not a fifth item; it is the shared root under items 1, 2 and 3, and fixing it
first makes all three cheaper. Stated separately so it is fixed once at the class level
rather than three times at the instance level (rule #6).

`eval/watchdog.py::check_stall` prints "these are NOT quality results" and `break`s the
task loop. Everything downstream is unaware:

- `eval/run.py:485` returns the partial `results` list with no marker.
- `eval/cli.py` appends it to `all_results` like any other.
- `eval/report_core.py:102 compute_score_stats` takes `statistics.mean(scores)` over
  whatever tasks happened to finish. It records `count`, and **nothing compares `count`
  to the number of tasks that were supposed to run.**
- `eval/report_history.py:32 save_historical_results` filters `is_test_model` and
  nothing else, so partial runs enter `~/.config/ztools/eval_history.json` permanently.
  `ornith-1.0-9b-mxfp8` has **55 entries** there; the 11/30 run is among them and is
  indistinguishable from the rest.
- `compute_task_winners` and `eval_results.json` inherit the same blindness.

The class: **a warning that exists only on stdout is not a gate.** The console said the
right thing and the JSON on disk recorded the wrong thing, and the JSON is what the next
session reads. This is the same shape as the truncated bonsai run reported at 62% when
its complete score was 79% — that incident was recorded as a lesson but never closed in
code.

Also confirmed absent while checking: **there is no task-weighting mechanism at all.**
`grep` for `TASK_WEIGHT`/`weights` over `eval/` and `lib/` finds nothing relevant. The
backlog's "gates are correctly weighted at zero for ranking" is a convention held in
someone's head; in code, `image_real` and `taxes_slip_qa` enter the mean at full weight
like every other task. Item 3 depends on this being real, not remembered.

**Fix, before items 1-3:**

1. Thread completeness out of `run_eval`: return `{"results": [...], "expected_tasks": N,
   "completed_tasks": M, "truncated_reason": str|None}` rather than a bare list.
2. `save_historical_results` REFUSES a truncated run — records it to a separate
   `eval_quarantine.json` with the reason, so nothing vanishes silently.
3. `compute_score_stats` carries `complete: bool`; `print_score_stats`,
   `compute_task_winners` and `_print_results` mark or exclude incomplete rows.
4. Introduce `RANKING_TASKS` / `GATE_TASKS` as data next to `TASKS` in
   `eval/tasks_core.py`, and make the mean a mean over ranking tasks only. Gates keep
   printing pass/fail and stop moving the number.
5. Tests: a synthetic 30-task run truncated at 11 must (a) not appear in history,
   (b) not win any task, (c) be labelled in the printed table. Prove each can fail by
   reverting the guard once (rule #2).

Done when: replaying `ornith`'s 11/30 shape through the harness produces no history
entry and no winner, and the existing complete runs are unaffected.

---

## Item 1 — `qwen3.8-27b-mxfp8` has no valid measurement

### Evidence

```
decode_tokens_per_sec_samples = [{"clean": false, "v": 0.1158}]
cold_start_seconds_samples    = [{"clean": false, "v": 74.2878}]
prefill_samples               = 1
disk_bytes                    = 28.8 GB
```

One sample per quantity, every one tagged unclean (taken at 18.07 GB compressor).
`samples.estimate_from` falls back to unclean history, so the displayed scalar is
`0.12 tok/s`; `samples.clean_estimate` correctly returns `None`, so the timeout path
already takes its documented floor rather than the ~138,000 s this reading once
produced. The guard works. The measurement is still missing.

### Plan

1. **Establish the baseline honestly.** `tools/osaurus_one.sh` to a known single server.
   Confirm swap/compressor at the moment of measurement, not from this document.
2. **Prove it answers before committing 30 tasks.**
   `python3 -m eval --model qwen3.8-27b-mxfp8 --task filename --quick`. If this cannot
   complete, the full run is guaranteed waste and item 1 becomes "is it measurable at
   all", answered NO with evidence.
3. **Full run under the lock**, with the item-0 completeness fix in place so a partial
   result cannot quietly become a score.
4. **Require n >= 2 clean samples** per quantity before treating the estimate as real —
   the rule already applied to prefill (`1165364`, `aba52b0`). One clean sample is an
   estimate of itself.

### The structural question this run must answer

`eval/cli.py:404` warns and proceeds:

```python
if model_mem_gb > avail_mem_gb * 0.8:
    console.print(f"{WARN} Model needs ~{model_mem_gb}GB, low memory - will be slower")
```

Warn-and-continue is the shape this repo keeps paying for (it is the same shape as the
arch `case` whose `*)` fallback ships an untested binary — see `target-platform-policy`).
The honest answer is probably a **hard refusal with an override seam**:

- Refuse when `model_mem_gb > avail_mem_gb * 0.8` unless `--allow-oversize` (or
  `EVAL_ALLOW_OVERSIZE=1`) is passed, and say the two numbers in the refusal.
- The seam must make **both directions testable on this machine** — the reject path and
  the accept path — exactly as the build-platform gate does.
- The test surface already exists: `references/tests/test_model_eval_main.py:488` patches
  `estimate_model_memory` to `70` and asserts the warning appears. That test becomes the
  refusal test, plus a sibling asserting the override lets it through. No 27 GB model
  needed to test either direction.
- `estimate_model_memory` (`eval/cli_runtime.py:141`) is already correct — it reads disk
  bytes, not the "27b" in the name — so the gate rests on a measured quantity.

**Decide the threshold from this run, not before it.** If 28.8 GB genuinely measures on
a 64 GB box with 21.9 GB free, `* 0.8` is too strict and the refusal would block real
work. If it thrashes, the refusal is the honest answer and the run produced the evidence
for it.

### The second structural gap — `machine_is_uncontended()` is blind to the GPU

`eval/samples.py:44` gates on swap (<= 8 GB) and compressor (<= 15 GB) only. A peer
agent session running its own eval leaves both quiet, so its contention is recorded as a
**clean** sample. This is documented in CLAUDE.md as a known limit and is unmitigated.

Proposed fix, cheap and exactly targeted at the recorded failure: **a sample is clean
only if this process holds the GPU lock.** `lib/gpu_lock.py` already exposes `holder()`
and `foreign_holder()`; `restart_server` already consults `foreign_holder()` for the same
reason (`eval/cli_runtime.py:264`), so the precedent and the plumbing exist.

Be honest about what this does and does not buy:
- ✓ catches the recorded failure — a peer session's eval contending for the GPU.
- ✗ does NOT catch non-eval Metal work (Blender, a game, a video encode). Nothing short
  of real GPU telemetry does, and `powermetrics` needs sudo.

So the docstring must state the residual blindness rather than implying the gate is now
complete. A gate that overstates its coverage is worse than one that admits its hole.

### Done when

n >= 2 clean samples for decode, cold start and prefill; the oversize gate is a
structural refusal with both directions tested; the sample-clean predicate consults the
lock; `MODEL_QUIRKS.md` records the real numbers **and** whether a 28.8 GB model is
measurable here at all.

### Risk

The single most expensive error of the previous session was recording numbers taken on a
contended box. If anything about the baseline looks off mid-run — load average, a second
osaurus, compressor climbing — **stop and discard**, do not finish the run and caveat it.

---

## Item 2 — `ornith-1.0-9b-mxfp8` is unmeasured and wedges the server

### Evidence

The 2026-08-19 run reached 11/30 with 6 timeouts. It left osaurus at 75% CPU with 0.7 GB
resident refusing completions. Historical mean 25% over 55 entries — and those 55 include
the partial run, so even the 25% is contaminated (see the cross-cutting finding).
Disk 10.1 GB, so this is not a memory problem.

### What is already fixed and only needs verifying

`osaurus stop` failing to kill it is **already addressed**: `restart_server`
(`eval/cli_runtime.py:235`) no longer hand-rolls `osascript` + `open -n`; it delegates to
`tools/osaurus_one.sh`, which enumerates every pid, escalates to SIGKILL, and polls the
port. Verify this during the run rather than planning new work for it.

### Plan

1. **Land the cross-cutting completeness fix first.** Without it, another partial run
   just adds more contaminated history entries.
2. **Purge the contaminated history.** The 11/30 run's entries must come out of
   `eval_history.json` — printed, not silently deleted. This is a data-integrity fix and
   is independent of whether the model is ever run again.
3. **One measured attempt, under the lock, watchdog armed.** `--quick` on a single task
   first, same as item 1: if it wedges on one task it will wedge on thirty.
4. **Then decide, and record the decision.** Two acceptable outcomes:
   - It completes 30/30 → it has a real score for the first time; rank it normally.
   - It wedges again → **remove it from the roster** rather than leaving it as a
     recurring trap. At a 25% mean it has no slot to win, and every future sweep pays
     the wedge cost to re-learn what is already known twice.
5. If removed, the removal needs a home: a `quarantine` list read by the sweep and the
   eval, with the reason and the date inline, so it is a recorded decision rather than a
   model that mysteriously stopped appearing.

### Done when

Either a complete 30/30 run exists, or the model is quarantined with its reason in
config and its contaminated history entries removed. **Not** "we tried again and it
wedged again" with nothing written down.

---

## Item 3 — the vision task is a gate, not a ranking

### Evidence

Measured over 8 models with complete 30/30 runs:

| task | distinct values / 8 | verdict |
|---|---|---|
| `taxes_yoy_narrative` | 7 | ranks |
| `taxes_qa` | 5 | useful, partly saturated |
| `taxes_slip_qa` | 2 | GATE |
| `image_real` | 2 | GATE |

`eval/vision_fixtures.py` draws three deliberately unmistakable, mutually unrelated
shapes (red circle, green triangle, blue square) at 512x512 on white.
`lib/validators/vision_validator.py:33` scores keyword recall over their `accept` lists.
That design was correct for its purpose — it proved sight rather than vocabulary, and it
is calibrated (the same prompt with images stripped scores 0). It cannot rank, because
every model with vision gets all three.

### Plan

1. **`RANKING_TASKS` / `GATE_TASKS` must be real first** (cross-cutting fix, step 4).
   Until gates stop entering the mean, adding harder images changes a number nobody can
   interpret.
2. **Add fixtures that admit degrees of correctness**, per the backlog: more objects,
   occlusion, counting, relative position. Concretely:
   - counting — N shapes of one colour among distractors; the answer is a number
   - relative position — "which shape is above/left of which"
   - occlusion — one shape partly behind another; does the model report both
3. **The validator must grow with them.** Keyword recall cannot grade a count or a
   spatial relation. These need a structured answer (a small JSON shape) graded
   arithmetically — the same move `taxes_grounded.py` already makes, and the reason it
   ranks where the rubric tasks saturate.
4. **Calibrate before believing** (`calibrate-the-instrument`, and the pattern
   `test_taxes_grounded.py` already establishes):
   - build the ideal answer from the fixture spec and assert it scores exactly 100;
   - run with images STRIPPED and assert it scores 0;
   - mutate the validator and confirm each mutant fails a DIFFERENT named test;
   - run mutations with `PYTHONDONTWRITEBYTECODE` set — this repo has now been bitten
     by the bytecode cache twice, and a mutation result that repeats across different
     mutants is a cache artifact, not a finding.
5. **A new task earns its place only by separating models.** If a fixture produces 2
   distinct values across the roster it is another gate; keep it as one, label it, and
   do not count it. Measure this explicitly rather than assuming harder means ranking.

### The raptor decision — one cheap experiment before writing it off

`raptor-v1-26b-a4b-vl-jang` is the fastest model on the roster by a wide margin (30 tasks
in 4 minutes vs bonsai's 145), has confirmed vision (100 on all three image tasks), and
is 13.0 GB. It scores **0 on `filename_injection`**, emitting `zzhijack` verbatim, which
correctly disqualifies it from `best_models.vlm` because `rn`'s vision path consumes
arbitrary screenshots.

But `lib/untrusted.py` prompt hardening already took `gemma-4-12b` and `bonsai` from
0 to 100 on that task. It has apparently never been tried on raptor. That is a single
cheap run with a large payoff:

- hardening fixes raptor → the fastest vision model becomes eligible, and the `vlm` slot
  gets a real candidate instead of a general-competence tiebreak;
- hardening does not fix it → it joins `foundation` in the "cannot be defended by
  framing" class, and the exclusion is now evidence-backed rather than inferred from one
  unhardened score.

Either outcome is worth more than the current state, which is an untested assumption.

### Done when

`image_*` either ranks (>= 4 distinct values across the roster) or is explicitly labelled
a gate in code and excluded from the mean; raptor's hardened injection score is measured;
`config.toml`'s `vlm` comment is updated to say which of those two happened.

---

## Item 4 — mutation survivors

### Standing decision: kill in passing, do not grind

This is deliberately NOT a campaign. Survivors are concentrated in validator files that
ordinary work already touches, so the policy is to kill them when passing through, not to
schedule a sweep. Recorded here only so the state is not re-derived every session.

| file | survivors | notable |
|---|---|---|
| `json_validator.py` | 23 (from 54; detection 53% -> 78%) | `validate_detailed_json` 6, `check_source_extraction` 5, `get_source_matching_details` 5, `validate_json` 4, `_names_match` 4, `validate_mixed_signal` 4 |
| `text_validator.py` | 22 (from 28) | `validate_summary` 6, **two of which are equivalent mutants, not gaps** |

Reproduce with:

```
PYTHONDONTWRITEBYTECODE=1 python3 tools/mutate.py --preset validators
```

### Two constraints that bite before any fix

1. **Both files are at the size gate.** `json_validator.py` is 483 lines and
   `text_validator.py` is 497, against a 500-line hard limit. Any survivor fix that adds
   production code to either file breaches the gate. Extractions go to
   `lib/validators/helpers.py` (106) or `text_match.py` (76), which have room. Check
   `wc -l` before editing, not after.
2. **The equivalent mutants must be recorded, not re-hunted.** `validate_summary`'s
   specificity tiers (`ratio >= 0.8`, `>= 0.5`) are absorbed by the misattribution cap:
   any ratio below 1.0 trips that cap and flattens the score to 45, so the half-credit
   tier cannot affect the result and no test can kill those two mutants. Write this into
   an exclusion list **with the reason inline**, so the next session does not spend an
   hour rediscovering it.

### The lesson these survivors already taught, worth not re-learning

- A fixture can satisfy a boundary under BOTH the original and the mutant. Killing
  `>= 2` vs `> 2` in `_names_match` needed a pair where no token is long enough for the
  fallback ("Blue Fox Cafe" / "Fox Blue Diner"), because the obvious fixture matched
  through the fallback either way.
- A clause inside an `or` cannot be tested by a fixture that satisfies the other clause.
- **Assert the value, not a range, whenever the range admits the mutant.** A test written
  while fixing this asserted `score < 100` where the right answer was 0 and the mutant
  produced 50 — satisfied by both, so it could not see the change it existed to catch.

### Done when

Nothing, on its own. This item never "completes"; it is a standing practice. The only
scheduled work is recording the two equivalent mutants so they stop being counted as
debt.

---

## Sequencing

| # | work | state |
|---|---|---|
| 0 | Cross-cutting: completeness, gate/ranking split | **DONE** — see Status above |
| 2 | Item 1 structural — oversize refusal + lock-aware sample gate | **DONE** — brought forward, see note below |
| 1 | Item 1 — measure `qwen3.8-27b-mxfp8` | needs the GPU and a quiet box |
| 3 | Item 2 — purge contaminated history, one ornith attempt, then decide | needs the GPU |
| 4 | Item 3 — harder fixtures + validator + raptor hardening probe | partly needs the GPU; fixtures and validator do not |
| — | Item 4 | in passing, forever. Only the equivalent-mutant record is scheduled |

Step 2 was brought forward ahead of step 1, against the original ordering. The
reason the plan gave for doing it after the run was to let the measurement decide
the THRESHOLD -- and that still holds, so the threshold was left at 0.8 exactly as
the warning had it. What did not need the measurement was the SHAPE: refusal
instead of warn-and-continue, with a seam that makes both directions testable. The
number still comes from the run.

The one ordering constraint that is not negotiable: **step 0 before any measurement.**
Every item here writes numbers to disk, and the layer that persists them currently cannot
tell a finished run from an abandoned one.
