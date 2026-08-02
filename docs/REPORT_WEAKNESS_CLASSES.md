# Report weakness classes — `tw` and `wk`

**Stage 0 of G3** (`~/Projects/routines/AUTOMATION_PLAN.md` §3 G3, backlog item 9).
Catalogued 2026-08-02 against real, dated outputs. Nothing here is hypothetical: every
class is anchored to a quote from a shipped report and a root cause in the source.

Companion docs — build on these, do not duplicate:
`BUGS_twitter_llm_fallback.md` (the fallback cascade), `MODEL_QUIRKS.md` (per-model prompt
quirks), `TESTING.md` (test patterns), `eval_calibration_2026-07-11.md` (model sweep).

---

## Evidence corpus

| Report type | Sample | Dated |
| :--- | :--- | :--- |
| `wk` | `~/Documents/weekend_plan_July_31_to_August_02_2026.md` | 2026-07-31 |
| `tw` | `~/Documents/twitter_summaries/2026-07-29_1633_to_2026-07-31_1215.md` | 2026-07-31 |
| `tw` | `~/Documents/twitter_summaries/2026-07-27_0050_to_2026-07-29_1633.md` | 2026-07-29 |

Ground truth established on this machine 2026-08-02 (probe-first), not inferred:

- Osaurus at `localhost:1337` is **up**; `/v1/models` serves 9 models.
- `conf/twitter.toml:5` names `gemma-4-26b-a4b-it-4bit`; that model returns
  **HTTP 404** `"is not installed or registered with any provider"`. The key is also
  never read (see C13), so it is dead config rather than a live 404.
- `conf/phase_signals.json` keys `qwopus3.6-27b-v2-mlx-4bit` — also not installed.
- No `conf/models/*.toml` contains the `{exclusions}` placeholder (C8).
- 33 of 41 configured model prompts raise on `str.format()` (C1).

> Quoted rows below render the score glyph as ASCII `*`. The saved reports use
> U+2B50, which the repo emoji gate forbids; `weekend/output.py:41` already emits
> `*`, so the quotes match today's renderer.

---

## Correction to the plan's framing

AUTOMATION_PLAN.md §3 G3 states *"Every transient row's 'Duration / End Date' column is
empty"*. In the real 2026-07-31 sample the column is **not empty** — every row reads the
constant `2-3 hours`. That is a worse symptom, not a milder one: an empty cell is an honest
placeholder, a constant is a fabricated one. The plan also attributes the loss to "the DDGS
scrape/extraction drops the event's dates". The scrape is a contributing cause (C3), but the
**structural** cause is that the JSON schema the model is asked to fill has no date field at
all (C2) and the prompt orders the model to emit `"2-3 hours"` (C4).

---

## Class index

| ID | Class | Tool | Status (2026-08-02) |
| :--- | :--- | :--- | :--- |
| C1 | `SILENT-TEMPLATE-SUBSTITUTION-FAILURE` | `wk` | **FIXED** — shared renderer, raises |
| C2 | `DATE-DROPPED-AT-THE-LLM-BOUNDARY` | `tw` + `wk` | **C2b FIXED** (schema+renderer); **C2a OPEN** (`tw`) |
| C3 | `NO-RECENCY-FILTER` | `wk` | **PARTIAL** — dated rows filtered; undated are C7 |
| C4 | `MANDATED-PLACEHOLDER` | `wk` | **FIXED** — prompts no longer order constants |
| C5 | `UNVERIFIED-SEMANTIC-LABEL` | `wk` | **FIXED** — corrected in code, clear-cut cases |
| C6 | `PROVENANCE-LABEL-NOT-BACKED-BY-DATA` | `wk` | **FIXED** — heading renamed to Fit Score |
| C7 | `CLASSIFICATION-BY-QUERY-PROVENANCE` | `wk` | **OPEN** — now visible as a blank date |
| C8 | `UNENFORCED-USER-CONSTRAINT` | `wk` | **FIXED (3rd attempt)** — see C8b, the matcher class |
| C9 | `BACKEND-PROVENANCE-DISCARDED` | `tw` | **FIXED** — degraded runs are banner-marked |
| C10 | `UNSPECIFIED-OUTPUT-CONTRACT` | `tw` | **OPEN** |
| C11 | `COVERAGE-OVERSTATED` | `tw` | **LATENT — deliberately untouched** |
| C12 | `EVAL-DOES-NOT-EXERCISE-PRODUCTION` | both | **FIXED** — one shared renderer |
| C13 | `DECLARED-BUT-UNREAD-CONFIG` | `tw` | **OPEN** |
| C8b | `NAME-MATCHED-BY-CONTAINMENT` | `wk` | **FIXED** — token-subset matching |

C1, C2 and C12 are the load-bearing ones. C2 is the only class present in **both** tools,
which makes it the true class rather than two coincidental bugs. C12 is why none of the
others were caught: the evaluator does not execute the code that produces the reports.

---

## C1 · SILENT-TEMPLATE-SUBSTITUTION-FAILURE

**Invariant violated:** a prompt template either renders completely or fails loudly. It must
never reach a model half-rendered.

**Symptom.** The model is asked to find events for a weekend it was never told about. The
literal characters `{date_range}` are sent to the model.

**Root cause.** `conf/models/*.toml` prompts embed a JSON schema containing unescaped `{`.
`str.format()` therefore parses `{"transient_events"` as a replacement field and raises
`KeyError`. The call site swallows it:

`weekend/prompts.py:212-220`
```python
config_prompt = get_model_prompt(model, Task.WEEKEND_TRANSIENT)
if config_prompt:
    try:
        formatted = config_prompt.format(location=..., age_range=..., date_range=...)
    except (KeyError, IndexError, ValueError):
        formatted = config_prompt.replace("{}", f"{location} {age_range} {date_range}")
```
The `except` branch does a `"{}"` replace — but these templates use *named* fields, so
nothing is substituted. Twin site for the fixed branch: `weekend/prompts.py:157-168`.

**Probe (2026-08-02), the prompt actually sent to the model:**
```
Find 5-10 events for {date_range} in {location}. Kids ages {age_range}.
```

**Blast radius.** 33/41 configured prompts raise. For `weekend_transient` — where the caller
really does use keyword args — 6 of 7 model families deliver unsubstituted placeholders:
`foundation.toml:17`, `gemma.toml:12`, `laguna.toml:24`, `nemotron.toml:24`,
`qwen.toml:24`, `qwopus.toml:24`. Only `gemma_versions.toml:28` renders.
(`summarize` / `filename` also raise under keyword args, but their real call sites use
positional `{}` replacement — `twitter/summarize.py:181-182` — so they are **not** affected.
Stated explicitly so the class is not overclaimed.)

**Failing case:** `report_class_cases.py::C1_prompt_templates_render` — asserts every
`weekend_fixed` / `weekend_transient` prompt in `conf/models/*.toml` renders under the exact
keyword set the production call site passes, and that no `{name}` placeholder survives.

**Class-level fix, not instance:** escape braces at the boundary (or store the schema outside
the template), and delete the `except` that converts a template bug into silent corruption —
a failed render must raise. A gate over `conf/models/*.toml` keeps it dead.

---

## C2 · DATE-DROPPED-AT-THE-LLM-BOUNDARY

**Invariant violated:** when a full timestamp is in hand, the component that decides recency
must receive it. Do not truncate a datetime on the way into the model.

This is the same defect in both tools, which is what makes it a class.

### C2a · `tw`

`twitter/summarize.py:145`
```python
parts = [f"@{t['screen_name']} | {t['created_at'].strftime('%H:%M')}"]
```
`created_at` is a full timezone-aware UTC datetime (`twitter/browser_parse.py:59-60`) and
survives sorting and the cache round-trip. It is truncated to wall-clock time at the *only*
point where it could reach the LLM.

**Sample (2026-07-31, window `2026-07-29 16:33 → 2026-07-31 12:15`, i.e. ~2 days):**
> `- @jammles9 reported that Morocco is using its people to take over Ceuta and Melilla, with reports of looting and setting fires (@jammles9 | 23:21).`

`23:21` on which of the three days? Nothing in the file says. The date appears only in the
code-generated `**Period:**` header and the filename.

### C2b · `wk`

The transient JSON schema has no date field — only `duration` and `day`:

`weekend/prompts.py:70-73`
```
{"transient_events": [{"name": "str", "location": "str",
"target_ages": "str", "price": "str", "duration": "str",
"weather": "str", "day": "str"}]}
```
Dates *are* requested one phase earlier (`weekend/prompts.py:16`, "list its name, location,
dates, price...") and are then structurally discarded at the structuring phase. The renderer
then labels the duration field as if it were a date — `weekend/output.py:126`:
```python
duration = item.get("duration") or item.get("end_date") or ""
```
under the header `Duration / End Date` (`weekend/output.py:102-104`).
`weekend/llm.py:249` collapses `["day", "date", "dates", "event_date"]` into `day` without
parsing or validating any of them.

**Sample (2026-07-31):**
> `| * 4.8/5 | **Canada Day at Your Toronto Zoo** (Toronto Zoo) | 6-13 | $20-30 per child or free | 2-3 hours | Friday | outdoor |`

**Failing cases:** `report_class_cases.py::C2a_tw_timestamps_are_day_qualified` (every
timestamp in a multi-day `tw` report must be day-disambiguated) and
`::C2b_wk_transient_rows_carry_a_date` (every transient row must carry a parseable date, not
a duration).

**Class-level fix:** carry the datetime to the boundary in both tools — `%b %d %H:%M` in the
`tw` prefix, a required `start_date`/`end_date` in the `wk` schema — and add one shared
assertion that no renderer prints a date column sourced from a non-date field.

---

## C3 · NO-RECENCY-FILTER

**Invariant violated:** a report scoped to a window must filter its candidates against that
window, in code, not by asking the model nicely.

**Symptom.** A July 1 event in a July 31 – August 2 plan (quote above, C2b).

**Root cause — five sites, none of which filter:**

- `weekend/data.py:103` — `fetch_transient_events(dates_str, year, month_name)` accepts
  `dates_str` and **never uses it**. The caller passes the real Fri–Sun window
  (`weekend/cli.py:164`). This is the clearest place a filter was intended and is missing.
- `weekend/data.py:117-123` — queries interpolate only `{month_name} {year}`, so a
  "July 2026" query legitimately returns a July 1 event for a July 31 plan.
- `weekend/data.py:107`, `:159`, `:178` — `DDGS().text(q, max_results=...)` with no
  `timelimit=`. `ddgs` accepts and forwards it; the repo never passes it (zero hits).
- `weekend/cli.py:235-316` `_parse_transient` — filters on name-ish keys and rejects weather
  telemetry; performs **no date check** against `fri`/`sun`.
- `weekend/config.py:19` + `weekend/cli.py:161-168` — the `--use-cache` scrape cache has no
  TTL and is not keyed by weekend, so last month's scrape is served for this weekend.

The only date enforcement in the whole pipeline is textual pleading at
`weekend/prompts.py:37-38` and `:248-250` ("Filter these strictly! Ensure they match the
Dates provided!") — addressed to a model that, per C1, never received the dates.

**Failing case:** `report_class_cases.py::C3_no_row_predates_the_window` — every dated row
must fall inside `[friday, sunday]`.

---

## C4 · MANDATED-PLACEHOLDER

**Invariant violated (house rule "honest placeholders"):** a field that was not measured must
be visibly unmeasured. Never instruct a model to fabricate a constant that will be rendered
as if it were data.

**Symptom.** Identical price / duration / age on every row, presented as scraped fact.

**Root cause.** The prompt *orders* the constants. `weekend/prompts.py:75-79`:
```
MANDATORY default values:
- target_ages: "{age_range}"
- price: $20-30 per child or free
- duration: "2-3 hours"
- day: Friday/Saturday/Sunday
```
Fixed twin at `weekend/prompts.py:93-95` (`$18-35 per child or free`). Monolithic fallbacks
repeat both: `:178-180`, `:231-236`. Closed with `"Never leave any field empty."`
(`:86`, `:102`, `:185`) — which is precisely what converts "unknown" into a fabricated value.
Model configs restate it: `conf/models/qwen.toml:32-33`, and the same block in
`qwopus.toml`, `nemotron.toml`, `laguna.toml`, `foundation.toml`.

**Sample (2026-07-31)** — all eight fixed rows:
> `| * 3.9/5 | **Candyland Indoor Play Centre, Vaughan** (Vaughan) | 6-13 | $18-35 per child or free | indoor |`

The pipeline half-knows: `weekend/llm.py:362` treats `"2-3 hours"` as a *penalty sentinel*
(`if item["duration"].lower() not in ("", "2-3 hours")`) — the scorer recognises the value as
information-free while the renderer still prints it as fact.

**Failing case:** `report_class_cases.py::C4_no_column_is_constant_across_all_rows` — a
non-key column whose value is identical on every row is a fabricated default, not data.

**Class-level fix:** drop `MANDATORY default values` and `Never leave any field empty` from
every prompt; make the schema's optional fields genuinely optional; render a missing value as
the existing `—` sentinel (`weekend/output.py:13`).

---

## C5 · UNVERIFIED-SEMANTIC-LABEL

**Invariant violated:** a label that the code can check must be checked by the code.

**Symptom.** An indoor trampoline park labelled `outdoor`.

**Sample (2026-07-31), the top-ranked fixed row:**
> `| * 4.4/5 | **Sky Zone Trampoline Park, Toronto** (Toronto) | 6-13 | $18-35 per child or free | outdoor |`

**Root cause.** The value is free LLM choice (`weekend/prompts.py:81-84`), never recomputed.
The real Open-Meteo forecast (`weekend/data.py:69-100`) touches it only *after the fact* as a
ranking bonus (`weekend/llm.py:324-347`); nothing overwrites a wrong label.
`weekend/llm.py:364` again treats the canonical trio as low-information.

**The eval scorer for this is itself broken** — `lib/quality_weekend_scorers.py:8` puts the
two-letter string `"in"` in `INDOOR`, and `:96` matches by substring, so `"raining"`,
`"windy"` and `"fine"` all score as *indoor*. Calibrate the instrument before trusting it.

**Failing case:** `report_class_cases.py::C5_weather_label_matches_venue_kind` — a venue whose
name contains an unambiguous indoor marker (`indoor`, `trampoline park`, `museum`,
`play centre`) must not be labelled `outdoor`.

---

## C6 · PROVENANCE-LABEL-NOT-BACKED-BY-DATA

**Invariant violated:** a column's heading must name what the number actually is.

**Symptom.** Both tables are headed `(Ranked by Review Score)` and every row renders a
`N/5` star rating. There is no review score in the pipeline.

**Root cause.** `weekend/output.py:50` and `:97` append the heading; `weekend/output.py:39-41`
`_fmt_score` renders `item["score"]`, which is set by `weekend/llm.py:370-372` from
`_score_item` (`weekend/llm.py:280-367`) — an internal weather/age/completeness heuristic.
The real scraper, `weekend/data.py:170` `scrape_review_score`, is **never called** by the
pipeline; it is exported (`weekend/cli.py:47`, `weekend/__init__.py:34`) and carries five unit
tests (`tests/test_weekend_data.py:258-319`). Those tests are the trap: they make dead code
look maintained.

**Failing case:** `report_class_cases.py::C6_review_score_heading_requires_review_data` —
if the report claims "Review Score", `scrape_review_score` must appear on the live call path.

---

## C7 · CLASSIFICATION-BY-QUERY-PROVENANCE

**Invariant violated:** a row's category must be a property of the row, not of the query that
happened to surface it.

**Symptom.** Evergreen venue boilerplate ranked as a limited-time event.

**Sample (2026-07-31), in the *Transient / Limited-Time Events* table:**
> `| * 3.8/5 | **Discover family fun in Vaughan** (Various venues) | 6-13 | $0 | 2-3 hours | Saturday | both |`

That row is a tourism-page tagline. `Tiny Otters Indoor Playspace` on the row above is a
year-round venue; both are in the transient table.

**Root cause.** There is no classifier anywhere. The label is fixed by which DDGS query set
produced the text — `weekend/data.py:117-124` (transient) vs `:143-150` (fixed) — and carried
by two disjoint code paths (`weekend/llm.py:526-555` vs `:557-581`) to two tables
(`weekend/output.py:147`, `:172`). Nothing re-checks.

**Failing case:** `report_class_cases.py::C7_transient_rows_are_time_bounded` — a transient row
must have a date or end date; a row with none is evergreen and misfiled.

---

## C8 · UNENFORCED-USER-CONSTRAINT

**Invariant violated:** a constraint the user configured must be enforced in code. A prompt
mention is not enforcement — and here it is not even mentioned.

**Symptom.** Four of the fifteen rows in the 2026-07-31 plan are venues the user explicitly
excluded in `conf/weekend.toml`.

`conf/weekend.toml:23-40` excludes, among others, `Toronto Zoo`, `Sky Zone Toronto`,
`LEGOLAND Discovery Centre Toronto`. The report contains:
> `**Sky Zone Trampoline Park, Toronto**` · `**LEGOLAND Discovery Centre Toronto, Vaughan Mills**` · `**Canada Day at Your Toronto Zoo**` · `**Toronto Zoo - Free Community Drone Show**`

**Root cause — three independent layers. Any one alone breaks the feature.**

**Layer 1 (found 2026-08-02, now FIXED).** `conf/weekend.toml` declared
`exclude_places` *after* the `[[children]]` array-of-tables. TOML therefore parsed it as a
key of the **last child**, not as a top-level key, so `weekend/config.py:54`
`WEEKEND_CONFIG.get("exclude_places", [])` returned **`[]`**. The user's 16 exclusions were
invisible to the program, and the `.get` default swallowed the error silently:

```
>>> from weekend.config import EXCLUDE_PLACES
[]                      # before: 16 declared, 0 seen
['Canada's Wonderland', 'Ontario Science Centre', ...]   # after the key was moved
```
This is the same failure shape as C1 — a broad default converting a config error into silent
wrong behaviour. Fixed by moving the key above the first table, with a comment pinning the
ordering requirement. Layers 2 and 3 remain open.

**Layer 2.** `weekend/prompts.py:145` builds `exclusion_string` and it reaches a prompt
only through the `{exclusions}` placeholder at `:163`. **No `conf/models/*.toml` contains
`{exclusions}`** (verified 2026-08-02), and the built-in fallback prompt at
`weekend/prompts.py:170-186` never interpolates it either. The transient builder
(`weekend/prompts.py:204-239`) does not so much as compute it — which is why both Toronto Zoo
rows are in the transient table.

**Layer 3.** There is no post-parse filter at any later stage. Even a model that honoured the
instruction perfectly would not be *verified*.

Note `lib/quality_weekend_scorers.py:170-194` *does* score exclusions — but only against the
eval's own `WEEKEND_FIXED_REF` fixture, never against `conf/weekend.toml`. See C12.

**Failing case:** `report_class_cases.py::C8_no_excluded_place_appears` — no row's name or
location may match `conf/weekend.toml` `exclude_places`.

**Class-level fix:** enforce in code after parsing, not in the prompt. A user constraint that
is only ever a suggestion to a model is not a feature.

---

## C8b · NAME-MATCHED-BY-CONTAINMENT

**Invariant violated:** when matching a user-configured name against scraped
text, the config's wording is *not* a contiguous substring of the scraped
wording. Matching must tolerate reordering, interpolation and punctuation.

**Symptom.** An excluded venue shipped from a real run — twice, after C8 had been
declared fixed both times.

**Sample (real run, 2026-08-07, line 12 of the shipped plan):**
> `| * 1.9/5 | **Sky Zone Trampoline Park** (Vaughan/Toronto) | 5 and up |  | indoor |`

`conf/weekend.toml` excludes `Sky Zone Toronto`. The tokens are all present but
not contiguous, so `normalize_for_match(entry) in haystack` missed. The same run
*did* drop `LEGOLAND Discovery Centre Toronto`, and that single drop was
mistakenly taken as proof the class was closed.

**Root cause of the root cause — the escape hatch.** The first fix addressed only
the U+2019 instance (`Ripley's` vs `Ripley’s`) and then documented the remaining
gap as acceptable: *"Add the variant to `exclude_places` if one slips through."*
That comment converted a matcher defect into unbounded manual config maintenance
and is why the class survived a second review. A workaround written into a
docstring is not a fix; it is permission for the bug to stay.

**Fix.** `weekend/enforce.py:matches_exclusion` requires every *significant*
token of the entry to appear in the candidate, in any order:
`{sky, zone, toronto} ⊆ {sky, zone, trampoline, park, vaughan, toronto}`.
Still conservative — all tokens required — so `Toronto Zoo` does not match
`Toronto Islands`, `CN Tower` does not match `Tower of London`, and
`Little Canada` does not match `Canada Day at the Zoo`. Two tokenizer defects
surfaced while testing it: a possessive left a stray `s` token, and a
parenthetical was treated as required.

**The checker was wrong too — the fourth this session.** It passed the shipped
row because it used the same containment logic. Checker and enforcement now share
one function. Three sibling sites (holiday names, indoor markers, mandated price
literals) matched scraped text without punctuation folding and were fixed with
it; an en-dash `$20–30` would previously have escaped the C4 literal check.

**Failing case:** `test_report_class_fixes.py::test_C8_zero_excluded_venues_in_the_output_not_merely_one_drop`
— asserts **zero** excluded venues in the rendered output. "At least one drop
happened" is the assertion that let this through twice.

---

## C9 · BACKEND-PROVENANCE-DISCARDED

**Invariant violated (house rule "never let a degraded path be silent"):** the artifact must
record which backend produced it.

**Symptom.** A summary from the fourth-tier local fallback is byte-for-byte indistinguishable
from one produced by the primary 35B server model. Neither dated `tw` sample names a model;
the footer is script name and timestamp only.

**Root cause — provenance exists and is thrown away at each layer:**
- `lib/osaurus_lib.py:392` returns the served model: `"model": data.get("model", model)`.
- `twitter/summarize.py:223-239` reads only `result["content"]` and returns a bare `str`.
- `lib/llm/fallback.py:51` returns the raw result, no provenance wrapper.
- `twitter/output.py:79-85` `write_markdown` has no backend parameter at all; the footer at
  `twitter/output.py:105` is `*Generated by twitter_summarizer.py on {timestamp}*`.

**No degradation gate.** The only refusal is total failure —
`twitter/summarize.py:133` `critical = header_count == 0 and bullet_count == 0`. A summary
that trips `"Only 1 bullet points — may lack detail"` or `"Very short (60 chars)"`
(`:106-108`, `:228-238`) is printed as a console warning and **written to disk anyway**.
Every degradation signal is transient stdout chatter that dies with the terminal.

`BUGS_twitter_llm_fallback.md:186-188` calls the cascade "degrades gracefully"; in practice it
degrades *invisibly*, which is the defect.

**Failing case:** `report_class_cases.py::C9_report_names_its_backend` — the report must state
which model produced it, and mark itself when that was not the primary.

---

## C10 · UNSPECIFIED-OUTPUT-CONTRACT

**Invariant violated:** a format the reader depends on must be specified and enforced in code,
not left to sampling.

**Symptom.** Two consecutive runs of the same code, same prompt, same model, two formats.

**2026-07-29 sample:**
> `...showcasing a trained agent and renderer on 7,200 live worlds on one GPU at 07:22.`

**2026-07-31 sample:**
> `...with reports of looting and setting fires (@jammles9 | 23:21).`

**Root cause.** There is no bullet template anywhere in the codebase; everything below
`## Summary` is one opaque LLM string inserted at `twitter/output.py:102`. The prompt asks for
"when" without specifying a format (`conf/models/qwen.toml:43`). The model either echoes the
*input* prefix shape built at `twitter/summarize.py:153` (`[@handle | HH:MM]: `) or
paraphrases it. The eval goldens teach **both** styles — bracket form at
`lib/eval_data.py:171`, prose form at `:188` — so neither is scored as wrong.

Second-order: the handle is printed twice per bullet (`@jammles9 reported ... (@jammles9 |
23:21)`) because the prompt asks to "include who (@user mentions)" and the model also copies
the input prefix.

**Failing case:** `report_class_cases.py::C10_bullet_timestamp_format_is_uniform` — one
attribution format per report, and the same format across reports.

---

## C11 · COVERAGE-OVERSTATED  *(latent — evidence too thin to act on)*

**Invariant:** a stated count must be the count that was actually processed.

`twitter/output.py:98` prints `**Tweets:** {len(tweets)} from {unique_authors} accounts` —
the *fetched* count. The prompt builder silently truncates to a character budget:

`twitter/summarize.py:161-163`
```python
if used + len(line) + 1 > budget:
    break
```
Two problems: iteration is `reversed(tweets)` (`:144`) so overflow drops the **oldest**
tweets, and `break` (not `continue`) means one over-long tweet discards every older tweet
that would still have fit. The true count is returned (`:186`) and printed to the console
(`:215`) but never reaches the markdown. Separately, the bounded scroll can exit on a runtime
budget or stagnation (`twitter/browser.py:177-178`, `:193-203`) while
`twitter/output.py:91-92` still prints the full requested `**Period:**`.

**Honest status: not proven.** On the two samples (50 and 52 tweets) the budget almost
certainly was not hit, and no run log survives to confirm. The mechanism is recorded, not a
claim that the samples exhibit it. **Do not "fix" this without first reproducing it** — the
right next step is a probe that logs `len(tweets)` vs the returned `n` on a real run.

**Failing case:** `report_class_cases.py::C11_stated_count_matches_processed_count` — marked
`xfail(strict=False)` precisely because it cannot currently be triggered from a saved report;
it needs the count threaded into the artifact first.

---

## C12 · EVAL-DOES-NOT-EXERCISE-PRODUCTION

**Invariant violated (house rule "green tests are not a shipped feature"):** the evaluator
must run the code that produces the artifact.

This is the meta-class. It is why none of C1–C11 was caught.

- **Different substitution path.** `ev` renders prompts with `_safe_format_prompt`
  (`lib/config_tasks.py:38-58`), which uses `str.replace()` and therefore *succeeds* on the
  very templates that make production raise. Production uses `str.format()`
  (`weekend/prompts.py:159`, `:214`). **C1 is structurally invisible to `ev`.**
- **No reader for real outputs.** `ev` cannot score a saved report. The `tw` output directory
  appears exactly once in the codebase — `twitter/output.py:24`, where it is *written*. There
  is no flag, no code path, no reader. The plan's Stage 1 criterion ("`ev` on real `tw`/`wk`
  outputs") has no surface to run on today.
- **Fixtures cannot express the defects.** Every `--quality` case is a frozen `TestCase` in
  `lib/eval_data.py` (10 cases total). The `summarize` fixture is a **single-day** timeline
  (`lib/eval_data.py:171`), so C2a cannot fail; both C10 formats contain a `HH:MM` so the
  scorer (`lib/quality_scorers.py:347`) is satisfied either way.
- **No scorer exists** for stale dates, repeated column values, or evergreen-vs-transient.
  Completeness only tests `bool(val and str(val).strip())`
  (`lib/quality_weekend_scorers.py:70-77`) — a row filled entirely with the prompt's own
  mandated defaults (C4) scores **full marks**.
- **The weekend dimension weights sum to 0.95**, so a weekend composite can never exceed 95.
  Any "improvement to 100" claim on that task is arithmetically impossible.
- **The weather scorer is miscalibrated** — `"in"` as an indoor keyword (C5).
- `conf/phase_signals.json` and `conf/eval_signals.json` are **machine-written telemetry**
  (`eval/run.py:60-86`, `weekend/llm.py:64-73`), not test data — do not hand-edit them to add
  cases. `phase_signals.json` currently keys `qwopus3.6-27b-v2-mlx-4bit`, a model not
  installed on this machine, so its timeout override is inert.

**Failing case:** `report_class_cases.py::C12_eval_and_production_share_prompt_rendering` —
the eval's renderer and the production renderer must produce the same string for the same
template. They currently do not.

**Class-level fix:** one shared prompt-rendering function used by both, and an `ev` surface
that scores a real saved report file. Until then, an `ev` score is evidence about the
fixtures, not about `tw` or `wk`.

---

## C13 · DECLARED-BUT-UNREAD-CONFIG

**Invariant violated:** configuration that is declared must be read, or it silently lies about
what the tool does.

`conf/twitter.toml` declares six keys. Exactly one is read — `llm_url`, at
`twitter/output.py:26`. `model`, `max_scrolls`, `chrome_cookies_db`, `state_file` and
`output_dir` are not read from that file at all; the model actually used comes from
`conf/config.toml` `[best_models] summarize` via `twitter/cli.py:35`.

The declared `model = "gemma-4-26b-a4b-it-4bit"` is **not installed** (HTTP 404, probed
2026-08-02). Anyone reading the config would conclude `tw` runs a model it has never run.

The existing gate, `tools/check_config_debt.py`, detects hardcoded values in Python but has
no check for the inverse — a config key with no reader.

**Failing case:** `report_class_cases.py::C13_declared_config_keys_are_read` — every key in
`conf/twitter.toml` must have a reader, and every model named in config must exist on the
server.

---

## The Osaurus blocker — RESOLVED, and the Stage 0 diagnosis was wrong

Stage 0 reported that "every MLX model hangs; only the on-device `foundation`
model answers", and treated that as a property of the machine. **That was wrong,
and the correction matters more than the original finding.**

The real cause was a **wedged Osaurus process**. It had been up 25 hours with
117 MB RSS (no model resident) and 0% CPU while RAM was 85% free. After a clean
relaunch every configured model served normally:

| Model | Before (wedged) | After relaunch |
| :--- | :--- | :--- |
| `foundation` (on-device) | 1s | 1s |
| `gemma-4-e4b-it-8bit` | no response in 120s | **14s** |
| `qwen-agentworld-35b-a3b-mxfp8` (`summarize`) | no response in 300s | **13s** |
| `qwen3.6-35b-a3b-mxfp8-mtp` (`think`) | no response in 300s | **14s** |

Three things follow, and each is a defect in its own right:

1. **The health check could not see it.** `is_server_running()` asks
   `/v1/models` (`lib/osaurus_models.py:63-71`), which a wedged server answers
   instantly. `ev` printed `Server: OK` and then burned 600s per task. A
   readiness check that exercises a cheaper path than the work is not a readiness
   check. Fixed: `lib/osaurus_server.can_serve()` asks for a single token and
   returns a stated reason on failure.
2. **The auto-restart made it worse.** `restart_server()` quit Osaurus, failed to
   confirm the relaunch within its 20s window, and returned `False` — leaving
   **nothing listening on 1337**. `ensure_server()` then retries that up to three
   times. For an unattended routine this converts a recoverable wedge into a hard
   outage. **Still open** — `can_serve()` gives the honest signal but the restart
   path itself has not been reworked.
3. **This is C9's condition in the wild.** While wedged, a real `tw` run would
   have exhausted the server tier and silently landed on `foundation` — which is
   exactly why C9 was prioritised.

**`ev` baseline: still not taken.** The server can serve now, so it is finally
possible, but a `--quality` sweep is ~20 tasks x minutes/task and was not run in
the time available. No before/after number is claimed. What IS proven is stated
per class below, by real run or by test.

---

## Stage 1 outcome (2026-08-02)

**Proven by a REAL `wk` run** (weekend of 2026-08-07..09, `qwen3.6-35b-a3b-mxfp8-mtp`,
output at `~/Documents/weekend_plan_August_07_to_August_09_2026.md`):

| Class | Real-run result |
| :--- | :--- |
| C1 | model receives the real date range; run produced in-window events |
| C2b | `Dates` column holds ISO dates or an honest blank — never a duration |
| C3 | no stale or out-of-window row |
| C4 | no mandated literal; no fabricated constant column |
| C5 | no impossible weather label |
| C8 | **initially WRONG** — that run also shipped `Sky Zone Trampoline Park`. See C8b; re-verified after the matcher fix. |
| C6 | heading reads `Ranked by Fit Score (computed, not reviews)` |

**Test-proven only (no real run):** C9 and C12. C9's banner is exercised through
the real `write_markdown`, but a real `tw` run needs a live X session and was not
performed — so the degraded-path banner has not been observed in production.
C12 is a structural property (one shared renderer) and has no runtime artifact.

**Still open:** C2a (`tw` timestamps), C7, C10, C13. **C11 remains latent and deliberately untouched.**

**`restart_server` — FIXED and verified on the real app.** The port frees before
the process exits, so LaunchServices swallowed the relaunch and re-activated the
terminating instance. `_wait_until_down` now waits for the process (pgrep), the
launch is retried once (never another quit), and `ensure_server` gives a grace
period before it may quit again. Real run: `restart_server() -> True` in 2.2s
with the PID changing 93433 -> 93921 and the model serving afterwards. The first
attempt at this fix (wait for the port) was insufficient and a live run caught
it -- the same lesson as C8b.

### What the real run taught that the tests could not

C8 was declared fixed and test-proven, then a real run shipped an excluded venue
anyway: the config's ASCII `Ripley's` did not match the scraper's typographic
`Ripley’s`. **The checker missed it too**, because it normalised the same wrong
way — so it reported PASS on the very row the enforcement failed to drop. An
instrument that shares the bug it is measuring is worse than no instrument.

Two further checker bugs surfaced the same way, both punishing the pipeline for
telling the truth: a blank date cell was scored as C2b (it is C7), and a column
of honest blanks was scored as a fabricated constant. Neither could have been
found without running the real thing.

This is the strongest argument in this document for the `prove-before-claim`
rule: three of the checks written in Stage 0 were wrong, and every one of them
looked green until a real artifact went through.

---

## What Stage 1 should fix first

Ordered by (blast radius / effort), not by how bad the symptom looks:

1. **C1** — one-line-ish fix (escape the braces, delete the swallowing `except`), and it is
   the mechanical cause of the headline symptom. Nothing else in `wk` can be trusted while
   the model is receiving `{date_range}`.
2. **C12** — until the evaluator runs production's renderer and can read a real report, every
   subsequent fix is unverifiable. Build this second, before the content fixes, so the rest
   are measurable.
3. **C2** — the shared class; fix both tools with one assertion.
4. **C4 + C8** — both are "stop lying in the output": delete the mandated defaults, enforce
   the exclusion list in code.
5. **C9** — cheap (~5 lines) and makes the whole fallback question auditable after the fact.
6. **C3, C5, C6, C7, C10** — content-quality work that only becomes measurable after 2.
7. **C11** — probe before fixing. Do not act on the mechanism alone.
