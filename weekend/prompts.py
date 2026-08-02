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

PHASE_EXTRACT_EVENTS = """\
Extract family-friendly event listings from the search results below. For each
event, list its name, location, dates, price if available, and description.
Ignore irrelevant search results, ads, and navigation text.

Search results:
{raw_text}

List each relevant event with key details, one per line."""

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

Weather: {weather_condensed}

Available events:
{cleaned_sources}

List specific activity suggestions, one per line. Include name, location, and
brief description."""

PHASE_DRAFT_FIXED = """\
You are a family activity planner. Suggest 10 specific weekend activities for
families with kids ages {age_range} in {location}. Focus on year-round venues
and fixed-location activities.

Weather: {weather_condensed}

Available venues:
{cleaned_sources}

List specific activity suggestions, one per line. Include name, location, and
brief description."""

PHASE_REFINE = """\
Here are activity suggestions:

{draft_text}

Merge any near-duplicates, keep the best 8, remove low-quality or irrelevant
ones, and sort by overall appeal. Output the refined list, one per line with
name + description."""

PHASE_STRUCTURE_TRANSIENT_SYSTEM = """\
Output JSON now. Use EXACT schema:
{{"transient_events": [{{"name": "str", "location": "str",
"target_ages": "str", "price": "str", "duration": "str",
"weather": "str", "day": "str"}}]}}

MANDATORY default values:
- target_ages: "{age_range}"
- price: $20-30 per child or free
- duration: "2-3 hours"
- day: Friday/Saturday/Sunday

Weather: {weather_condensed}
Set weather based on the activity type and forecast above: "outdoor" for
outdoor activities (parks, zoo, sports) in nice weather, "indoor" for indoor
venues (museums, play centres), "both" for flexible activities.

Never leave any field empty. Output ONLY JSON."""

PHASE_STRUCTURE_FIXED_SYSTEM = """\
Output JSON now. Use EXACT schema:
{{"fixed_activities": [{{"name": "str", "location": "str",
"target_ages": "str", "price": "str", "weather": "str"}}]}}

MANDATORY default values:
- target_ages: "{age_range}"
- price: $18-35 per child or free

Weather: {weather_condensed}
Set weather based on the activity type and forecast above: "outdoor" for
outdoor activities (parks, zoo, sports) in nice weather, "indoor" for indoor
venues (museums, play centres), "both" for flexible activities.

Never leave any field empty. Output ONLY JSON."""


def build_weather_condense_prompt(weather_str):
    return PHASE_WEATHER_CONDENSE.format(weather_str=weather_str)


def build_source_extract_prompt(raw_text, source_type):
    template = PHASE_EXTRACT_EVENTS if source_type == "events" else PHASE_EXTRACT_VENUES
    return template.format(raw_text=raw_text)


def build_draft_prompt(
    weather_condensed, cleaned_sources, source_type, location, age_range, date_range
):
    template = PHASE_DRAFT_TRANSIENT if source_type == "transient" else PHASE_DRAFT_FIXED
    return template.format(
        weather_condensed=weather_condensed,
        cleaned_sources=cleaned_sources,
        location=location,
        age_range=age_range,
        date_range=date_range,
    )


def build_refine_prompt(draft_text):
    return PHASE_REFINE.format(draft_text=draft_text)


def build_structure_system_prompt(source_type, age_range, weather_condensed=""):
    template = (
        PHASE_STRUCTURE_TRANSIENT_SYSTEM
        if source_type == "transient"
        else PHASE_STRUCTURE_FIXED_SYSTEM
    )
    return template.format(age_range=age_range, weather_condensed=weather_condensed)


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

    MANDATORY default values:
    - target_ages: "{age_range}"
    - price: $18-35 per child or free

    Set weather based on activity type: "outdoor" for outdoor activities,
    "indoor" for indoor venues, "both" for flexible.

    Never leave any field empty.
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
    "target_ages": "str", "price": "str", "duration": "str",
    "weather": "str", "day": "str"}}]}}

    Extract {location} family events for {date_range}.

    MANDATORY default values:
    - target_ages: "{age_range}"
    - price: $20-30 per child or free
    - duration: "2-3 hours"
    - day: Friday/Saturday/Sunday

    Set weather based on activity type: "outdoor" for outdoor activities,
    "indoor" for indoor venues, "both" for flexible.
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
