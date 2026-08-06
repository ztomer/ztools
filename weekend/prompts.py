from lib.config import Task, get_model_prompt
from lib.prompt_render import POSITIONAL_SLOT, render_prompt
from lib.tui import debug_print
from weekend.config import AGE_RANGE, CITY, DATES_STR, EXCLUDE_PLACES, REGION


def _render_model_prompt(template, template_id, context, **fields):
    """Render a conf/models/*.toml prompt, or raise.

    The two template conventions (a single positional `{}` slot vs named
    `{placeholder}`s) are detected and dispatched EXPLICITLY. The predecessor
    guessed, and a broad `except` turned a wrong guess into a prompt that reached
    the model with `{date_range}` still in it -- see class C1.
    """
    return render_prompt(
        template,
        template_id=template_id,
        positional=context if POSITIONAL_SLOT in template else None,
        **fields,
    )

# Phase-specific template constants
PHASE_WEATHER_CONDENSE = """\
Given this weather forecast, summarize what to expect for the weekend in
1-2 sentences. Be specific about temperatures and conditions.

{weather_str}

Output only the summary, nothing else."""

# The phase chain used to NARROW at every step: extract asked for dates, draft
# asked only for "name, location, description", and refine only for
# "name + description". So the dates were discarded two phases before the schema
# that wanted them, and every date column rendered blank. Each phase now carries
# the same fields through verbatim -- see class C2c.
_CARRY_FIELDS = """Carry these fields through EXACTLY as they appear in the input,
never rewritten or dropped: DATES, PRICE, AGES, LOCATION. If the input does not
state one, write "unknown" for it -- never guess.

AGES means the age range the VENUE OR EVENT states for itself, copied from the
source. It is NOT the age range of the family this plan is for. Never fill it
with the family's ages -- if the source does not state an age range, "unknown"
is the correct answer."""

PHASE_EXTRACT_EVENTS = """\
Extract family-friendly event listings, near {location}, from the search
results below.

For each event output one line:
NAME | LOCATION | DATES | PRICE | AGES | short description

- DATES: copy the date text exactly as written, e.g. "Saturday, August 1 -
  Monday, August 3" or "23rd August" or "February 14-16". Include the year if
  the result shows one. Write "unknown" if the result gives no date.
- PRICE and AGES: copy verbatim if stated, else "unknown".
- WHERE: only include somewhere a family in {location} could drive to for a
  day out. The search engine returns results from all over the world -- a zoo in
  San Diego, a trampoline park in Dublin or Oswego is useless here however well
  it matches. If the result does not name a place in or near {location}, skip
  it. Do not "adapt" a foreign listing to the local area.
- Ignore ads and navigation text.
- DATES must be when the EVENT runs, not when the page was written or updated.
  If the only date on the page is a publication or "last updated" date, that is
  not an event date -- write "unknown".
- A result marked [THIS WEEKEND] already mentions a date inside the weekend
  being planned, so prefer those -- they are listed first. This is a hint about
  where to look, NOT a restriction: extract from the unmarked results too. Never
  invent an event to fill the list, and never move an event's dates to make it
  fit the weekend. Fewer real events beats more invented ones.

IS THIS AN ACTIVITY? Before listing anything, ask: "is this an actual thing a
family can go and DO at a specific time and place, or is it a page that LISTS
things?" A directory, guide, calendar, round-up, "what's on" page, "things to do
in X" article, blog archive or a venue's events INDEX is NOT an activity -- you
cannot attend a guide. Skip those results entirely, however well they match.
If the page names a SPECIFIC event, list that event, not the page.

Search results:
{raw_text}

One event per line, in the pipe-separated format above."""

PHASE_EXTRACT_VENUES = """\
Extract family-friendly venues from the search results below. For each venue,
list its name, location, price if available, and what it offers for kids.
Ignore irrelevant search results, ads, and navigation text.

Search results:
{raw_text}

List each relevant venue with key details, one per line."""

PHASE_DRAFT_TRANSIENT = """\
You are a family activity planner. Suggest 10 specific weekend activities for
families with kids ages {age_range} in {location}. Focus on time-limited events
happening specifically on {date_range}.

IS THIS AN ACTIVITY? Before listing anything, ask: "is this an actual thing a
family can go and DO at a specific time and place, or is it a page that LISTS
things?" A directory, guide, calendar, round-up, "what's on" page, "things to do
in X" article, blog archive or a venue's events INDEX is NOT an activity -- you
cannot attend a guide. Skip those results entirely, however well they match.
If the page names a SPECIFIC event, list that event, not the page.

The year is {year}. Every date you output must be in {year}. An event dated in
any other year does not belong in this plan -- drop it rather than re-dating it.

Weather: {weather_condensed}

Available events:
{cleaned_sources}

Output one line per suggestion in this EXACT format:
NAME | LOCATION | DATES | PRICE | AGES | short description

{carry}"""

PHASE_DRAFT_FIXED = """\
You are a family activity planner. Suggest 10 specific weekend activities for
families with kids ages {age_range} in {location}. Include a diverse mix of
year-round venues, outdoor seasonal places (parks, conservation areas, farms),
and indoor family spots. The year is {year}; the weekend is {date_range}.

Weather: {weather_condensed}
Prefer outdoor activities when clear/warm, and indoor venues when precipitation is expected.

Available venues:
{cleaned_sources}

Output one line per suggestion in this EXACT format:
NAME | LOCATION | DATES | PRICE | AGES | short description

{carry}"""

PHASE_REFINE = """\
Here are activity suggestions:

{draft_text}

Merge any near-duplicates, keep the best 8, remove low-quality or irrelevant
ones, and sort by overall appeal.

Output the refined list in the SAME pipe-separated format you received:
NAME | LOCATION | DATES | PRICE | AGES | short description

Carry DATES, PRICE, AGES and LOCATION through unchanged from the input. Merging
two entries keeps the more specific value, never "unknown" over a real one."""

# Class C4 (MANDATED-PLACEHOLDER) + C2b (DATE-DROPPED-AT-THE-LLM-BOUNDARY).
# This prompt used to ORDER the model to emit "$20-30 per child or free" and
# "2-3 hours" on every row and close with "Never leave any field empty" -- which
# is what turned "unknown" into a fabricated constant the report then rendered as
# fact. It also had no date field at all, so an event's real dates were
# structurally impossible to carry. Both are fixed here: unknown is now an
# explicit empty string, and start_date/end_date are first-class.
PHASE_STRUCTURE_TRANSIENT_SYSTEM = """\
Output JSON now. Use EXACT schema:
{{"transient_events": [{{"name": "str", "location": "str",
"target_ages": "str", "price": "str", "start_date": "str", "end_date": "str",
"duration": "str", "weather": "str", "day": "str"}}]}}

Rules for every field:
- Copy values from the source text. NEVER invent one.
- If the source does not state a value, output an empty string "" for it.
  An empty field is CORRECT and expected. Do not guess, do not use a typical
  or average value, and do not repeat a value from another row.
- start_date / end_date: ISO YYYY-MM-DD, from the DATES field of the input.
  The input dates are usually free text (e.g. "Saturday, August 1 - Monday,
  August 3") -- convert them to ISO. If the text gives no year, the year is
  {year}. If the input says "unknown", output "".
- target_ages: the age range the VENUE OR EVENT states for itself, copied
  from the input. NEVER the family's ages. If the input does not state one,
  output "".
- price: the actual price as written in the source, else "".

Weather: {weather_condensed}
Set weather from the activity type and the forecast above: "outdoor" for
outdoor activities (parks, zoo, sports), "indoor" for indoor venues (museums,
play centres, trampoline parks), "both" for flexible activities.

Output ONLY JSON."""

PHASE_STRUCTURE_FIXED_SYSTEM = """\
Output JSON now. Use EXACT schema:
{{"fixed_activities": [{{"name": "str", "location": "str",
"target_ages": "str", "price": "str", "weather": "str"}}]}}

Rules for every field:
- Copy values from the source text. NEVER invent one.
- If the source does not state a value, output an empty string "" for it.
  An empty field is CORRECT and expected. Do not guess a typical price or age
  range, and do not repeat a value from another row.

Weather: {weather_condensed}
Set weather from the activity type and the forecast above: "outdoor" for
outdoor activities (parks, zoo, sports), "indoor" for indoor venues (museums,
play centres, trampoline parks), "both" for flexible activities.

Output ONLY JSON."""


def build_weather_condense_prompt(weather_str):
    return PHASE_WEATHER_CONDENSE.format(weather_str=weather_str)


def build_source_extract_prompt(raw_text, source_type, location=None):
    template = PHASE_EXTRACT_EVENTS if source_type == "events" else PHASE_EXTRACT_VENUES
    location = location or f"{CITY}/{REGION}"
    if "{location}" in template:
        return template.format(raw_text=raw_text, location=location)
    return template.format(raw_text=raw_text)


def _default_year():
    from datetime import date

    return date.today().year


def build_draft_prompt(
    weather_condensed, cleaned_sources, source_type, location, age_range, date_range,
    year=None,
):
    template = PHASE_DRAFT_TRANSIENT if source_type == "transient" else PHASE_DRAFT_FIXED
    return template.format(
        weather_condensed=weather_condensed,
        cleaned_sources=cleaned_sources,
        location=location,
        age_range=age_range,
        date_range=date_range,
        year=year or _default_year(),
        carry=_CARRY_FIELDS,
    )


def build_refine_prompt(draft_text):
    return PHASE_REFINE.format(draft_text=draft_text)


def build_structure_system_prompt(source_type, age_range, weather_condensed="", year=None):
    template = (
        PHASE_STRUCTURE_TRANSIENT_SYSTEM
        if source_type == "transient"
        else PHASE_STRUCTURE_FIXED_SYSTEM
    )
    return template.format(
        age_range=age_range,
        weather_condensed=weather_condensed,
        year=year or _default_year(),
    )


def build_structure_user_prompt(draft_text):
    return f"Convert these activities to the schema:\n\n{draft_text}"


def build_fixed_system_prompt(
    model: str = None, location: str = None, age_range: str = None, venues_str: str = ""
):
    exclusion_string = ", ".join(EXCLUDE_PLACES)

    location = location or f"{CITY}/{REGION}"
    age_range = age_range or AGE_RANGE

    config_prompt = get_model_prompt(model, Task.WEEKEND_FIXED) if model else ""

    debug_print(
        f"[DEBUG] build_fixed_system_prompt: model={model}, "
        f"location={location}, age_range={age_range}",
        flush=True,
    )
    if config_prompt:
        # The positional slot in these templates reads "...from this list", so it
        # wants the scraped venues. The old except-path filled it with
        # "<location> <age_range>" instead -- a second bug the swallow hid.
        formatted = _render_model_prompt(
            config_prompt,
            f"{model}:{Task.WEEKEND_FIXED.value}",
            venues_str,
            location=location,
            age_range=age_range,
            date_range=DATES_STR,
            exclusions=exclusion_string,
        )
        debug_print(f"[DEBUG] prompt after render (first 200): {formatted[:200]}", flush=True)
        return formatted

    return f"""\
    Output JSON now. Use EXACT schema:
    {{"fixed_activities": [{{"name": "str", "location": "str",
    "target_ages": "str", "price": "str", "weather": "str"}}]}}

    Extract 10 popular {location} venues for families with kids ages {age_range}.
    Include location (city only), target_ages, price in CAD, weather.

    Copy values from the source text; NEVER invent one. If the source does not
    state a value, output "" for it -- an empty field is correct. Do not guess a
    typical price or age range, and do not repeat a value from another row.

    Set weather from activity type: "outdoor" for outdoor activities, "indoor"
    for indoor venues (museums, play centres, trampoline parks), "both" for
    flexible.
    """


def build_fixed_user_prompt(dates_str, weather_str, venues_str):
    return f"""\
    Current Context for the upcoming weekend:
    Dates: {dates_str}
    {weather_str}

    Potential Venues and Current Exhibits:
    {venues_str}

    Execute the task based on the system instructions and the provided context
    to find 10 year-round fixed activities, prioritizing current exhibits or
    highly-rated venues from the context. Output ONLY JSON.
    """


def build_transient_system_prompt(
    model: str = None,
    location: str = None,
    age_range: str = None,
    date_range: str = None,
    events_str: str = "",
):
    location = location or f"{CITY}/{REGION}"
    age_range = age_range or AGE_RANGE
    date_range = date_range or DATES_STR

    config_prompt = get_model_prompt(model, Task.WEEKEND_TRANSIENT) if model else ""
    if config_prompt:
        return _render_model_prompt(
            config_prompt,
            f"{model}:{Task.WEEKEND_TRANSIENT.value}",
            events_str,
            location=location,
            age_range=age_range,
            date_range=date_range,
        )

    return f"""\
    Output JSON now. Use EXACT schema:
    {{"transient_events": [{{"name": "str", "location": "str",
    "target_ages": "str", "price": "str", "start_date": "str", "end_date": "str",
    "duration": "str", "weather": "str", "day": "str"}}]}}

    Extract {location} family events for {date_range}.
    ONLY include an event if the source text shows it happens within
    {date_range}. Drop anything outside that window.

    Copy values from the source text; NEVER invent one. If the source does not
    state a value, output "" for it -- an empty field is correct. Do not guess a
    typical price, duration or age range, and do not repeat a value from another
    row. start_date/end_date are ISO YYYY-MM-DD taken from the source only.

    Set weather from activity type: "outdoor" for outdoor activities, "indoor"
    for indoor venues (museums, play centres, trampoline parks), "both" for
    flexible.
    """


def build_transient_user_prompt(dates_str, weather_str, events_str):
    return f"""\
    Current Context for the upcoming weekend:
    Dates: {dates_str}
    {weather_str}

    High-Signal Transient Events (Filter these strictly! Ensure they match the
    Dates provided!):
    {events_str}

    Execute the task based on the system instructions and the provided context.
    Output ONLY JSON.
    """
