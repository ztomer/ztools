# Python ↔ Rust parity ledger

Two implementations of the same four tools exist, and they have already drifted.

- **Python** — `references/**` in this repo, entry points `tw` / `wk` / `rn` / `ev`
  (only inside `.venv/bin`, so they need the venv active).
- **Rust** — `~/Projects/routines/src/ztools/*.rs`, subcommands `twitter-summarize`,
  `weekend-plan`, `image-renamer`, `model-eval`.

**The Rust port exists to escape the venv/uv dependency: one static binary, no Python
startup.** That is a real goal and it is why the port stays. This file is the price of
keeping it — the list of things that must be mirrored, so "we'll sync it later" is a
tracked debt instead of a memory.

## Which one actually runs (verified 2026-08-12)

| invocation | implementation | source |
|---|---|---|
| `twitter`, `weekend`, `rename_images`, `oeval` | **Rust** | `~/.zshrc:154-158` — shell *functions*, which beat PATH |
| the scheduled jobs | **Python** | `routines.toml:15` `uv run --frozen wk`; `routines-twitter.toml:8` `uv run --frozen tw` |
| `tw` / `wk` / `rn` / `ev` | Python | `.venv/bin/` only; not on PATH |

Interactive runs and scheduled runs therefore execute **different code**. Note also that
`$HOME/Projects/ztools/bin` is on PATH (`~/.zshrc:233`) carrying Python scripts named
`twitter`, `weekend`, `oeval`, `rename_images` — all shadowed by the shell functions and
dead since 16 Jul.

## Known divergences

### 1. The summarizer prompt is duplicated, not shared

`routines/src/ztools/twitter.rs:105` carries its own copy of the instruction text
("Use connecting phrases and narrative verbs to show how events relate"), mirroring
`references/eval/tasks_prompts.py: TWITTER_PROMPT`. Nothing derives one from the other.
Every prompt improvement therefore has to be made twice or it silently applies to half
the runs.

The Rust side reads exactly one thing from this repo — `~/Projects/ztools/conf/weekend.toml`
(`config_ztools.rs:73`). Prompts are not in shared config.

### 2. Model selection has already diverged, and the Rust default is not the measured best

| | twitter/summarize | source |
|---|---|---|
| Rust | `gemma-4-e4b-it-8bit` (hardcoded default) | `config_ztools.rs:76-81` |
| Python | `summarize = gemma-4-12b-it-mxfp8`, `json = foundation` | `conf/config.toml [best_models]` |

The 2026-08-12 sweep ranks `gemma-4-e4b-it-8bit` at 82.1 and `muse-glimmer-30b-jang_6m`
at 88.2 over 22 common tasks. So the interactive path runs a model the leaderboard does
not endorse, and editing `conf/config.toml` — the obvious place — does not change it.

### 3. The eval and its scorers are Python-only

`references/eval/**` plus `references/lib/validators/**` are what rank models and gate
quality. Rust has its own 194-line `model_eval.rs`. Any scorer fix here does not reach
the Rust binary's notion of which model is best.

## To mirror into Rust

Nothing below has been ported yet.

- [ ] **Fact-coverage scoring** (2026-08-12) — `validate_factual_coverage` matched key facts
  as exact substrings while the prompt orders paraphrase, so it scored verbatim copying.
  Now matches on identifying tokens with both sides through one tokenizer. See
  `references/lib/validators/text_match.py` and `docs/MODEL_QUIRKS.md`.
- [ ] **Model choice** — once the sweep completes, whatever `[best_models]` lands on has to be
  reflected in `config_ztools.rs` defaults, or better, read from shared config.
- [ ] **Greedy decoding in the eval** (temperature 0) — a leaderboard has to be reproducible.
- [ ] **Derived request timeouts** from measured cold-start / prefill / decode rates, instead of
  a flat ceiling that killed long generations and leaked server inference slots.

## How this gets resolved

By A/B test, later — not by argument. `bin/ab_test` already exists for exactly this: it
runs the native Rust build and the Python reference side by side and checks parity and
timing, and it was hardened once already after a version that printed "VERIFICATION OK"
on both branches of its own check.

Two things to fix in it before the results mean anything, given what this file records:
the two sides currently run **different models** (divergence 2), so an output comparison
would be measuring that rather than the implementations; and any quality verdict needs the
fixed fact-coverage scorer, since the old one scored verbatim copying. Decide the model on
both sides first, then A/B.

## The standing hazard

This is the "parallel reimplementation" failure mode: two pipelines that must agree, with
nothing forcing them to. The cheapest structural fix, if the static-binary goal allows it,
is to move the shared surface — prompts and model choice — into config both sides read,
leaving Rust to own only its transport and CLI. Until then, every entry above is a real
divergence a user can hit.
