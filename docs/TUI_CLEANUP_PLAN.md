# TUI Cleanup Plan: Dieter Rams / Susan Kare Edition

## Guiding Principles

1. **One prefix system** — `·` (step), `✗` (error), `!` (warning) everywhere. No `[PASS]`, `[WARN]`, `[FAIL]`, `[llm]`, `[cache]`, `[browser]`, `[!]`, `[ok]`, `[Wrn]`.
2. **No decorative borders** — no `====`, `────`, `━━━━`, `──`, `-` * 40 in terminal output (markdown/file output is fine).
3. **No emoji** in console output. `✗`/`·`/`!` are the only symbols. (Emoji rendering varies by terminal — unreliable.)
4. **One print system** — pick `print()` or `console.print()` per file, never both.
5. **Every character earns its place** — no debug/instructions printed on normal runs.
6. **Success is silent** — no `[OK]`, `[SUCCESS]`, or `✓` for normal operations. Summary line only at end.

## Phases

### Phase 1: Kill the loudest noise (immediate payoff)

| # | File | Line(s) | What | Fix |
|---|------|---------|------|-----|
| 1 | `model_eval.py` | 460–475 | 16-line debug instruction block printed every run | Move to `--help` text or guard behind `args.verbose` |

### Phase 2: Normalize prefix symbols (all tools)

Replace every prefix convention with the `·`/`✗`/`!` system:

| # | File | Pattern | Replace with |
|---|------|---------|--------------|
| 2 | `twit_summarize.py` | `[llm]` (12 lines) | `·` or drop (context is obvious) |
| 3 | `twit_browser.py` | `[cookies]`, `[browser]`, `[parse]` | Drop labels, keep message |
| 4 | `twit_output.py` | `[clean]`, `[!]` | Drop `[clean]`, `[!]` → `!` |
| 5 | `twitter_summarizer.py` | `[!]`, `[cache]`, `[ok]` | `!`, drop `[cache]`, drop `[ok]` |
| 6 | `lib/config_core.py` | `[ Wrn ]` | Drop label |
| 7 | `eval_run.py` | `[PASS]`, `[WARN]`, `[FAIL]` | `·`, `!`, `✗` |
| 8 | `model_eval.py` | `[PASS]`, `[WARN]`, `[FAIL]` | `·`, `!`, `✗` |
| 9 | `lib/quality_entry.py` | `──` section separators | Drop |
| 10 | `lib/quality_runner.py` | `✓`, `△` | `·`, `!` |

### Phase 3: Kill decorative borders

| # | File | Line(s) | What |
|---|------|---------|------|
| 11 | `eval_run.py` | 170–172 | `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━` banner |
| 12 | `twitter_summarizer.py` | 218, 220 | `"-" * 40` |
| 13 | `benchmark_quality.py` | 292–294, 357–359 | `====` borders |
| 14 | `benchmark_quality.py` | 299, 350 | `──` borders |

### Phase 4: Unify print/console.print

| # | File | Issue |
|---|------|-------|
| 15 | `twitter_summarizer.py` | Mixes `console.print` (line 186) with plain `print` (all others). Convert all to `print()` or all to `console.print()`. |
| 16 | `weekend_planner.py` | Mixes plain `print` (line 361 `[WARNING]`) with Rich. Pick one. |

### Phase 5: Remove emoji from console

| # | File | Line(s) | Emoji | Replace with |
|---|------|---------|-------|--------------|
| 17 | `eval_run.py` | 183 | `⚠️` | `!` |
| 18 | `model_eval.py` | 138, 162, 372, 375, 378 | `⚠️` | `!` |

### Phase 6 (bonus): Consistency polish

| # | File | What |
|---|------|------|
| 19 | `image_renamer.py` | Lines 112–134: 6 repeated `✗` error handlers — could DRY into a helper |
| 20 | `img_llm.py` | Lines 199, 216: error prints missing `✗` prefix (unlike `img_helpers.py`) |

## Verification

After each phase run:
```
python3 -m pytest tests/test_validators.py tests/test_config.py tests/test_content_processing.py tests/test_parse.py tests/test_parse2.py tests/test_weekend.py tests/test_twitter.py tests/test_taxes_validator.py -v
```

## Test Count Target

74 passed, 0 failed, no LLM-dependent tests included.
