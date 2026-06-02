# Expert Reviews: TUI Quality

## Susan Kare (Iconography, Apple Macintosh)

> "Your symbol system is coherent — ·/✗/! reads like a unified icon family. That's good. But then you staple on Rich Progress spinners, === banners, and emoji. Imagine Helvetica next to Comic Sans. Pick one visual language."

**Verdict:** B+. Symbols right, ornamentation wrong.

### Fixes
- Kill Rich Progress in weekend_planner
- Kill `===` decorative borders
- Kill the `N/A` column — wasted pixels
- Weather: readable sentence, not raw data dump

---

## Dieter Rams (Industrial Design, Braun)

> "Less but better — your progress spinner shows 5 tasks with timing down to the second. That's not better, it's more. The N/A score column is dishonest: you're showing data you don't have. The weather line is not thorough — you just concatenated raw floats."

**Verdict:** C+. Functional but noisy. Every character must earn its place.

### Fixes
- Replace progress with 5 simple `·` lines that complete silently
- Kill the `N/A` column
- Format weather: `Fri 27°C (rain), Sat 20°C (heavy rain), Sun 18°C (rain)`
- Remove `===` header — adds nothing

---

## Linus Torvalds (Linux Kernel)

> "Taste matters. Renaming 42 files prints 42 lines of old_name -> new_name — that's noise. Show me the diff, not the full path twice. And Foundation as default model? Either it's good enough to be default or it isn't. Don't hide behind weak defaults."

**Verdict:** C. Too much noise, wrong model.

### Fixes
- Collapse renames: `· 42 renamed (_1 suffix stripped), 0 skipped, 0 errors`
- Show `✗` lines only for failures
- Change default model from foundation to something competent
- Rip out Progress infrastructure for 5 sequential calls

---

## Uncle Bob (Clean Architecture)

> "You're violating SRP all over the place. weekend_planner.py::main() fetches data, calls LLMs, and formats output. weekend_output.py exists but the main function still has console.print calls. Output formatting is a detail; your domain logic shouldn't know about console or Progress."

**Verdict:** D. Business logic knows about terminal rendering.

### Fixes
- Push all `console.print()` calls into `*_output.py` modules
- Main functions return data structures; presenters handle rendering
- `lib/tui.py` is good but not enough if callers scatter output throughout business logic

---

## Action Items (Ranked by Impact)

| # | Priority | Item | Tool | Expert |
|---|----------|------|------|--------|
| 1 | High | Default model: swap foundation to competent model | All tools | Torvalds |
| 2 | High | Replace Rich Progress spinners with `·` lines | weekend_planner | Kare, Rams |
| 3 | High | Kill `=== Weekend Generator Started ===` | weekend_planner | Kare, Rams |
| 4 | High | Collapse rename output (42 lines → 1 line + errors) | image_renamer | Torvalds |
| 5 | Medium | Move all console.print calls into *_output.py | weekend_planner | Uncle Bob |
| 6 | Medium | Format weather as readable sentence | weekend_planner | Rams |
| 7 | Medium | Kill `N/A` score column (or implement scoring) | weekend_planner | Rams |
| 8 | Low | Twitter timeline: consistent tabulation after @name | twitter_summarizer | Torvalds |
