from weekend_config import EXCLUDE_PLACES, CITY, REGION, AGE_RANGE, DATES_STR
from lib.config import get_model_prompt, Task


def build_fixed_system_prompt(model: str = None, location: str = None, age_range: str = None):
    exclusion_string = ", ".join(EXCLUDE_PLACES)

    location = location or f"{CITY}/{REGION}"
    age_range = age_range or AGE_RANGE

    config_prompt = get_model_prompt(model, Task.WEEKEND_FIXED) if model else ""

    from weekend_planner import debug_print
    debug_print(f"[DEBUG] build_fixed_system_prompt: model={model}, location={location}, age_range={age_range}", flush=True)
    if config_prompt:
        try:
            formatted = config_prompt.format(
                location=location,
                age_range=age_range,
                date_range=DATES_STR,
                exclusions=exclusion_string,
            )
        except (KeyError, IndexError):
            formatted = config_prompt.replace("{}", f"{location} {age_range}")
        debug_print(f"[DEBUG] prompt after format (first 200): {formatted[:200]}", flush=True)
        return formatted

    return f"""
    Output JSON now. Use EXACT schema: {{"fixed_activities": [{{"name": "str", "location": "str", "target_ages": "str", "price": "str", "weather": "str"}}]}}

    Extract 10 popular {location} venues for families with kids ages {age_range}.
    Include location (city only), target_ages, price in CAD, weather.

    MANDATORY default values:
    - target_ages: "{age_range}"
    - price: $18-35 per child or free
    - weather: "indoor"

    Never leave any field empty.
    """


def build_fixed_user_prompt(dates_str, weather_str, venues_str):
    return f"""
    Current Context for the upcoming weekend:
    Dates: {dates_str}
    {weather_str}

    Potential Venues and Current Exhibits:
    {venues_str}

    Execute the task based on the system instructions and the provided context to find 10 year-round fixed activities, prioritizing current exhibits or highly-rated venues from the context. Output ONLY JSON.
    """


def build_transient_system_prompt(model: str = None, location: str = None, age_range: str = None, date_range: str = None):
    location = location or f"{CITY}/{REGION}"
    age_range = age_range or AGE_RANGE
    date_range = date_range or DATES_STR

    config_prompt = get_model_prompt(model, Task.WEEKEND_TRANSIENT) if model else ""
    if config_prompt:
        try:
            formatted = config_prompt.format(
                location=location,
                age_range=age_range,
                date_range=date_range,
            )
        except (KeyError, IndexError):
            formatted = config_prompt.replace("{}", f"{location} {age_range} {date_range}")
        return formatted

    return f"""
    Output JSON now. Use EXACT schema: {{"transient_events": [{{"name": "str", "location": "str", "target_ages": "str", "price": "str", "duration": "str", "weather": "str", "day": "str"}}]}}

    Extract {location} family events for {date_range}.

    MANDATORY default values:
    - target_ages: "{age_range}"
    - price: $20-30 per child or free
    - duration: "2-3 hours"
    - weather: "indoor"
    - day: Friday/Saturday/Sunday
    """


def build_transient_user_prompt(dates_str, weather_str, events_str):
    return f"""
    Current Context for the upcoming weekend:
    Dates: {dates_str}
    {weather_str}

    High-Signal Transient Events (Filter these strictly! Ensure they match the Dates provided!):
    {events_str}

    Execute the task based on the system instructions and the provided context. Output ONLY JSON.
    """
