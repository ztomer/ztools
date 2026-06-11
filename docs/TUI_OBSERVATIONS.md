# TUI Observations & Issues

## Cross-Cutting: Foundation Model Default

All tools default to `foundation`.

---

## Twitter (`twitter/cli.py`)

### Word Wrap in TUI View

The timeline output wraps mid-word. Need consistent tabulation/indentation after `@name` so the text body aligns cleanly across lines.

Current (wrapping breaks mid-word):
```
*   **@M5Stack** introduced the CardputerZero, a pocket-sized, Linux-powered device designed
for hacking, at 09:20.
```

Desired: text after `@name` column aligns, no mid-word breaks:
```
*   **@M5Stack**    introduced the CardputerZero, a pocket-sized, Linux-powered device
                    designed for hacking, at 09:20.
```

---

## Weekend (`weekend/cli.py`)

### Output

```
Using model: foundation
=== Weekend Generator Started ===
  Bounding Dates: June 05 to June 07, 2026
  Weather Forecast:
  Friday: 27.1°C, Precipitation (4.5mm)
  Saturday: 19.9°C, Precipitation (18.0mm)
  Sunday: 17.5°C, Precipitation (2.7mm)

✓ Fetched events             0:00:54
✓ Fetched venues             0:00:54
✓ Generated Fixed Activities 0:00:23
✓ Generated Transient Events 0:00:33
✓ Formatted output           0:00:12

                                                                             Weekend Plan: June 05 to June 07, 2026

Daily Forecast: Friday: 27.1°C, Precipitation (4.5mm) Saturday: 19.9°C, Precipitation (18.0mm) Sunday: 17.5°C, Precipitation (2.7mm)
```

### Issues

- **"=== Weekend Generator Started ==="** — decorative `===` borders are still present
- **Rich Progress spinners** — `⠏ ✓ Fetched events` uses dangling progress UI. The previous TUI cleanup removed Rich markup in some tools but this one still has it (this was flagged in Phase 4 but missed — uses Rich Progress with SpinnerColumn)
- **Weather line wraps poorly** — daily forecast on one long line
- **Table header uses `N/A` for scores** — all scores show `N/A`, which is the value prop
- **No summary line at end** — shows "Success!" and output path but no `· N renamed, M skipped` style summary (though this is a weekend plan, not a rename)
- **Still uses `===` decorative header** — was on the Phase 3 list but uses `[bold green]=== Weekend Generator Started ===` (Rich markup variant of decorative borders)

### Susan Kare / Dieter Rams Review

1. **Progress spinners** — The `⠏ ✓` Rich Progress pattern is visually heavy. 5 spinner tasks with timing is over-engineered for what amounts to "fetch, fetch, generate, generate, format." Replace with simple `·` prefixed lines that complete in-place.

2. **Header inconsistency** — Mixes `===` decorative header with otherwise clean output. Pick one style.

3. **Score column always `N/A`** — If scores are never populated, the column is noise. Either implement scoring or remove the column.

4. **Weather dump** — The forecast line dumps raw data. Format as a readable table or inline sentence: "Fri 27°C (rain), Sat 20°C (heavy rain), Sun 18°C (rain)."

---

## Rename (`rename/cli.py`)

### Output

```
· foundation
· 42 images
Renamed: 10_powerful_sentences_by_scott_adams_navigating_fa_1.jpeg -> 10_powerful_sentences_by_scott_adams_navigating_fa.jpeg
Renamed: 15_years_of_business_lessons_in_500_words_1_marryi_1.png -> 15_years_of_business_lessons_in_500_words_1_marryi.png
...
· No readable text, using VLM
...
· 42 renamed, 0 skipped, 0 errors
```

### Issues

- **Foundation model used** — all rename ops use `foundation` which is weak for VLM/rename tasks
- **`_1` suffix stripping** — every file has a `_1` before the extension that gets stripped (e.g. `_1.png` → `.png`). This might be correct dedup but looks aggressive — 42/42 files had `_1` stripped, suggesting it might be an artifact of the rename pattern rather than actual dedup
- **Summary line is good** — `· 42 renamed, 0 skipped, 0 errors` follows the TUI convention
- **VLM fallback** — `· No readable text, using VLM` correctly uses `STEP` prefix, good
- **No per-file error indicator** — if a file failed, would it show `✗`? The code has it but the output doesn't demonstrate it
- **Renamed line is long** — `old_name -> new_name` with 40+ char names creates very long lines. Consider `Renamed: ..._1.png -> .png` showing just the diff
