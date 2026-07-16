# ZTools TUI & CLI Cleanup Plan

This document outlines the consolidated plan and roadmap for terminal/user interface cleanup across the `ztools` codebase, applying the design principles of terminal interface design experts (Dieter Rams / Susan Kare philosophies). 

---

## 🎨 Guiding Principles

1. **One Prefix System:** Use only `·` (step), `✗` (error), and `!` (warning) in console output. No arbitrary labels like `[PASS]`, `[FAIL]`, `[llm]`, `[cache]`, `[browser]`, `[Wrn]`, or `[ok]`.
2. **No Decorative Borders:** Do not print banners like `====`, `────`, `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`, or `"-" * 40` to stdout (markdown/file outputs are exempt).
3. **No Console Emojis:** Rely exclusively on `✗`/`·`/`!` symbols. Emoji rendering is terminal-dependent and visually noisy.
4. **Unified Printing System:** Pick either standard `print()` or `console.print()` on a per-file basis; do not mix them.
5. **Silent Success:** Operations should succeed silently or output a clean final summary (e.g. `· N processed, M skipped`). No redundant `[OK]`, `[SUCCESS]`, or `✓` lines for individual steps.

---

## 🏁 Completed Milestones

- [x] **Unified TUI Dashboard:** Centralized all four tools (`weekend`, `twitter`, `oeval`, `rename`) into a responsive Python `Textual` dashboard ([tui/app.py](file:///Users/ztomer/Projects/ztools/tui/app.py)).
- [x] **Active Task Scheduler:** Implemented the `⏱️ Task Scheduler` tab in the TUI view to schedule, monitor, and delete background automation tasks.
- [x] **Async Background Workers:** Added non-blocking async execution daemon loops (`scheduler_loop`) with dynamic status tags.
- [x] **Local Quality Gate Hook:** Built a pre-push hook ([.githooks/pre-push](file:///Users/ztomer/Projects/ztools/.githooks/pre-push)) enforcing Ruff checks and isolated tests (coverage >95%) matching the remote GitHub Actions runner exactly.

---

## 🛠️ Outstanding Cleanup Tasks

### Phase 1: Prefix Standardization
Normalize all logger and output statements across CLI scripts to use the `·`, `!`, `✗` prefixes.

| # | File | Legacy Format | Action |
|---|------|---------------|--------|
| 1.1 | `eval/run.py` & `eval/cli.py` | `[PASS]`, `[WARN]`, `[FAIL]` | Replace with `·`, `!`, `✗` |
| 1.2 | `twitter/summarize.py` | `[llm]` | Replace with `·` or drop |
| 1.3 | `twitter/browser.py` | `[cookies]`, `[browser]`, `[parse]` | Drop labels, keep raw message prefixed with `·` |
| 1.4 | `twitter/output.py` | `[clean]`, `[!]` | Replace `[!]` with `!`, drop `[clean]` |
| 1.5 | `twitter/cli.py` | `[!]`, `[cache]`, `[ok]` | `!`, drop `[cache]`, drop `[ok]` |
| 1.6 | `lib/config_core.py` | `[ Wrn ]` | Replace with `!` |
| 1.7 | `lib/quality_runner.py` | `✓`, `△` | Replace with `·`, `!` |

---

### Phase 2: Eliminate Console Decorative Borders & Banners
Remove visual clutter and character-printed lines from stdout runs.

*   [ ] **`eval/run.py` (Lines 170-172):** Remove `━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━` banner.
*   [ ] **`twitter/cli.py` (Lines 218, 220):** Remove `"-" * 40` separators.
*   [ ] **`eval/benchmark_quality.py`:** Remove `====` and `──` borders.
*   [ ] **`weekend/cli.py`:** Remove `=== Weekend Generator Started ===` and similar decorative header lines.

---

### Phase 3: Emoji Sanitization
Replace terminal-dependent emojis with high-fidelity unicode or standard prefixes.

*   [ ] **`eval/run.py` (Line 183):** Replace `⚠️` with `!`.
*   [ ] **`eval/cli.py`:** Replace `⚠️` with `!`.

---

### Phase 4: Output Styling & Wrap Formatting

#### 4.1 Twitter Wrap Alignment in TUI View
- **Issue:** Timeline outputs wrap mid-word and lack visual indentation.
- **Fix:** Align text body cleanly after the `@name` column and prevent mid-word wrapping.

#### 4.2 Weekend Planner Output Polish
- **Weather Dump:** Instead of dumping raw data on a single long line, format it as a readable sentence: `"Fri 27°C (rain), Sat 20°C (heavy rain), Sun 18°C (rain)"`.
- **Dangling Progress Spinners:** Replace the timing-heavy `⠏ ✓` spinners in `weekend/cli.py` with simple `·` prefixed lines that complete in-place.
- **Table Scores:** Remove the score column if it defaults to `N/A`, or implement actual scoring metrics.

#### 4.3 Image Renamer Output Compactness
- **Issue:** Printing long `old_name.png -> new_name.png` lines on every file creates cluttered terminal outputs.
- **Fix:** Print only the filename difference/renamed output concisely (e.g. showing just the changes).

---

### Phase 5: Print System Unification
Pick a single printing interface per file to avoid mixed terminal formats.

*   [ ] **`twitter/cli.py`:** Standardize on standard `print` or `console.print` exclusively (currently mixes both).
*   [ ] **`weekend/cli.py`:** Standardize console print format (removes inline mixed Rich markup).
