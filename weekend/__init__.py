"""Weekend planner."""

from weekend.config import (
    DEBUG_EVENTS_FILE, DEBUG_VENUES_FILE,
    load_events_cache, save_events_cache,
    load_venues_cache, save_venues_cache,
    load_weekend_config,
    WEEKEND_CONFIG, EXCLUDE_PLACES, CHILDREN, CHILDREN_STR, CITY, REGION, AGE_RANGE, DATES_STR,
    MODEL_CONFIG, MODEL_NAME, OSAURUS_BASE_URL, OSAURUS_APP,
    is_server_running_ours, restart_osaurus, ensure_server,
)
from weekend.data import (
    get_weekend_date_objects, get_weekend_dates_string,
    fetch_weather, fetch_transient_events, fetch_fixed_venues, scrape_review_score,
)
from weekend.prompts import (
    build_fixed_system_prompt, build_fixed_user_prompt,
    build_transient_system_prompt, build_transient_user_prompt,
)
from weekend.llm import (
    get_llm_json, normalize_llm_items, fetch_scores_for_items,
)
from weekend.output import (
    build_markdown_tables, print_to_cli, print_header, print_step, print_info, print_warning, print_summary,
)
from weekend.cli import parse_args, main
