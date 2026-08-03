import json
import os
from pathlib import Path

from lib.config import Task
from lib.llm.fallback import call_with_fallback
from lib.mlx_lib import (
    call_mlx,
    find_text_mlx_model,
    process_mlx_content,
)
from lib.osaurus_lib import (
    _extract_json_only,
    apply_model_quirks,
    call_llm_api,
    get_best_model,
    panic_dump,
    strip_thinking,
)
from lib.tui import STEP, WARN, debug_print
from weekend.config import OSAURUS_BASE_URL, ensure_server
from weekend.phases import (  # noqa: F401  (re-export shim)
    draft_activities,
    extract_sources,
    refine_draft,
    structure_to_json,
)
from weekend.scoring import _score_item, fetch_scores_for_items  # noqa: F401  (re-export shim)

# LLM API defaults
LLM_TEMPERATURE = float(os.environ.get("WEEKEND_LLM_TEMPERATURE", "0.1"))
LLM_API_TIMEOUT = int(os.environ.get("WEEKEND_LLM_TIMEOUT", "1800"))
LLM_MAX_RETRIES = int(os.environ.get("WEEKEND_LLM_MAX_RETRIES", "5"))

# Phase pipeline timeouts (seconds) — set high for slow models (qwopus).
# phase_signals.json learns actual per-model latencies and tightens on reruns.
PHASE_TIMEOUT_WEATHER = int(os.environ.get("WEEKEND_PHASE_TIMEOUT", "900"))
PHASE_TIMEOUT_EXTRACT = int(os.environ.get("WEEKEND_PHASE_TIMEOUT", "900"))
PHASE_TIMEOUT_DRAFT = int(os.environ.get("WEEKEND_PHASE_TIMEOUT", "900"))
PHASE_TIMEOUT_REFINE = int(os.environ.get("WEEKEND_PHASE_TIMEOUT", "900"))
PHASE_TIMEOUT_STRUCTURE = int(os.environ.get("WEEKEND_PHASE_TIMEOUT", "900"))
PHASE_MAX_RETRIES = int(os.environ.get("WEEKEND_PHASE_MAX_RETRIES", "3"))
# Tracked file — see the EVAL_SIGNALS_DIR note in eval/run.py. tests/conftest.py
# redirects this to a tmp dir so `pytest` never dirties the working tree.
PHASE_SIGNALS_DIR = Path(
    os.environ.get("PHASE_SIGNALS_DIR", str(Path(__file__).parent.parent / "conf"))
)
PHASE_SIGNALS_PATH = PHASE_SIGNALS_DIR / "phase_signals.json"

# Timing, scoring, and source constants (Mitchell Hashimoto & John Carmack design)
_mlx_fallback_str = os.environ.get("WEEKEND_MLX_FALLBACKS", "qwen,llama,phi")
DEFAULT_MLX_FALLBACKS = [m.strip() for m in _mlx_fallback_str.split(",") if m.strip()]
POPULATED_FIELDS_BASE_SCORE = 3.0
OVERLAP_SCORE_HIGH = 3.0
OVERLAP_SCORE_LOW = 1.5
WEATHER_MATCH_BONUS = 2.0
WEATHER_PARTIAL_BONUS = 1.0
PRICE_MATCH_BONUS = 0.5
LOCATION_MATCH_BONUS = 0.5
SCORE_DIVISOR = 2.0
SCORE_MAX_LIMIT = 5.0
SOURCE_TYPE_TRANSIENT = "transient"
SOURCE_TYPE_FIXED = "fixed"
SOURCE_TYPE_EVENTS = "events"
SOURCE_TYPE_VENUES = "venues"
WEATHER_PREVIEW_LIMIT = 200
BATCH_GROWTH_STREAK_LIMIT = 3


def _load_phase_signals():
    try:
        if PHASE_SIGNALS_PATH.exists():
            return json.loads(PHASE_SIGNALS_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_phase_signals(signals):
    PHASE_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PHASE_SIGNALS_PATH.write_text(json.dumps(signals, indent=2, sort_keys=True))


def _call_llm(
    system_prompt,
    user_prompt,
    timeout,
    max_retries=PHASE_MAX_RETRIES,
    parse_json=False,
    phase_key=None,
    use_foundation=False,
):
    target_model = get_best_model(Task.JSON)
    last_content = None
    _attempt = [0]

    signals = _load_phase_signals() if phase_key else {}
    model_signals = signals.setdefault(target_model, {}) if phase_key else {}
    phase_signals = model_signals.get(phase_key, {}) if phase_key else {}
    base_timeout = phase_signals.get("timeout", timeout) if phase_key else timeout
    current_timeout = base_timeout

    def call_fn(model_name):
        nonlocal last_content, current_timeout
        _attempt[0] += 1
        current_timeout = base_timeout
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        messages = apply_model_quirks(messages, model_name)

        debug_print(
            f"· _call_llm: phase={phase_key}, attempt={_attempt[0]}, "
            f"base={base_timeout}, timeout={current_timeout}",
            flush=True,
        )
        debug_print(
            f"· system={len(messages[0]['content'])} chars, "
            f"user={len(messages[1]['content'])} chars",
            flush=True,
        )

        result = call_llm_api(
            OSAURUS_BASE_URL.rstrip("/"),
            model_name,
            messages,
            temperature=LLM_TEMPERATURE,
            timeout=current_timeout,
            parse_json=parse_json,
        )

        if result and "content" in result:
            raw_content = result["content"]
            last_content = raw_content
            cleaned = strip_thinking(raw_content)
            if parse_json:
                try:
                    json_str = _extract_json_only(cleaned)
                    if json_str is not None:
                        debug_print(f"· JSON extracted, length={len(json_str)}", flush=True)
                        return json.loads(json_str)
                    debug_print("· _extract_json_only returned None", flush=True)
                except Exception as e:
                    debug_print(f"· JSON parse error: {e}", flush=True)
                return None
            return cleaned
        else:
            err = result.get("error", "no content") if isinstance(result, dict) else str(result)
            print(f"{WARN} Osaurus API error: {err[:100]}")
            debug_print(f"· No content in result: {result}", flush=True)
        return None

    def mlx_fn():
        mlx_model = find_text_mlx_model(DEFAULT_MLX_FALLBACKS)
        if mlx_model:
            print(f"{STEP} MLX fallback: {mlx_model.name}")
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                messages = apply_model_quirks(messages, getattr(mlx_model, "name", str(mlx_model)))
                mlx_sys = next(
                    (m["content"] for m in messages if m["role"] == "system"), system_prompt
                )
                mlx_usr = next((m["content"] for m in messages if m["role"] == "user"), user_prompt)
                raw = call_mlx(mlx_model, f"System: {mlx_sys}\n\nUser: {mlx_usr}")
                if raw:
                    cleaned = process_mlx_content(raw)
                    if parse_json:
                        json_str = _extract_json_only(cleaned)
                        if json_str is not None:
                            return json.loads(json_str)
                        raise ValueError("No valid JSON in MLX response")
                    return cleaned
            except Exception as e:
                print(f"{WARN} MLX failed: {e}")
        return None

    def foundation_fn():
        from lib.foundation_lib import call_foundation, foundation_available

        if not foundation_available():
            return None
        print(f"{STEP} On-device Foundation Models fallback")
        try:
            raw = call_foundation(system_prompt, user_prompt, parse_json=parse_json)
        except Exception as e:
            print(f"{WARN} Foundation failed: {e}")
            return None
        if raw is None:
            return None
        if parse_json:
            json_str = _extract_json_only(raw)
            if json_str is not None:
                return json.loads(json_str)
            print(f"{WARN} Foundation returned no valid JSON")
            return None
        return raw

    # Forced Foundation mode: skip the server entirely and use the on-device
    # model as the primary path.
    if use_foundation:
        foundation_result = foundation_fn()
        if foundation_result is not None:
            return foundation_result
        print(f"{WARN} --use-foundation requested but Foundation Models unavailable")

    result = call_with_fallback(
        [target_model],
        call_fn,
        restart_fn=ensure_server,
        mlx_fn=mlx_fn,
        foundation_fn=foundation_fn,
        max_server_retries=max_retries - 1,
        label="Osaurus",
    )

    if result is not None and phase_key and _attempt[0] > 1:
        model_signals[phase_key] = {"timeout": current_timeout}
        _save_phase_signals(signals)

    if parse_json and result is None and last_content is not None:
        debug_print("· All retries failed, dumping content")
        panic_dump(last_content)

    return result


def get_llm_json(system_prompt, user_prompt, max_retries=LLM_MAX_RETRIES, use_foundation=False):
    result = _call_llm(
        system_prompt,
        user_prompt,
        timeout=LLM_API_TIMEOUT,
        max_retries=max_retries,
        parse_json=True,
        use_foundation=use_foundation,
    )
    if result is None:
        print(f"{WARN} Failed to parse JSON, returning empty result")
    return result


def normalize_llm_items(items, field_mapping=None):
    if not items:
        return items

    normalized = []
    NAME_KEYS = ["name", "activity", "activity_name", "title", "event", "event_name", "description"]
    LOC_KEYS = ["location", "address", "venue", "place"]
    AGE_KEYS = ["target_ages", "age_group", "ages", "age_range"]
    PRICE_KEYS = ["price", "cost", "pricing", "fee"]
    WEATHER_KEYS = ["weather", "setting", "type", "indoor_outdoor"]
    DAY_KEYS = ["day", "date", "dates", "event_date"]
    DUR_KEYS = ["duration", "end_date", "time"]

    for item in items:
        if isinstance(item, str):
            normalized.append({"name": item})
        elif isinstance(item, dict):
            if field_mapping:
                for model_field, standard_field in field_mapping.items():
                    if model_field in item and standard_field not in item:
                        item[standard_field] = item[model_field]

            for keys, std in [
                (NAME_KEYS, "name"),
                (LOC_KEYS, "location"),
                (AGE_KEYS, "target_ages"),
                (PRICE_KEYS, "price"),
                (WEATHER_KEYS, "weather"),
                (DAY_KEYS, "day"),
                (DUR_KEYS, "duration"),
            ]:
                if std not in item:
                    for k in keys:
                        if k in item:
                            item[std] = item[k]
                            break

            normalized.append(item)
    return normalized


def condense_weather(weather_str):
    from weekend.prompts import build_weather_condense_prompt

    result = _call_llm(
        "",
        build_weather_condense_prompt(weather_str),
        timeout=PHASE_TIMEOUT_WEATHER,
        phase_key="condense_weather",
    )
    return result if result else weather_str[:WEATHER_PREVIEW_LIMIT]


EXTRACT_SIGNALS_PATH = Path(__file__).parent.parent / "conf" / "extract_signals.json"
DEFAULT_BATCH_SIZE = 3
MAX_BATCH_SIZE = 5



def generate_weekend_plan(
    model,
    weather_str,
    events_str,
    venues_str,
    dates_str,
    location,
    age_range,
    date_range,
    use_foundation=False,
    plan_year=None,
):
    from weekend.output import print_warning

    # The model dated four events 2025 in a 2026 plan; only the C3 window filter
    # caught them. The year is now stated explicitly at every phase that emits a
    # date rather than being inferred from the range string.
    if plan_year is None:
        from datetime import date as _d

        plan_year = _d.today().year

    condensed_weather = condense_weather(weather_str)

    cleaned_events = extract_sources(
        events_str, SOURCE_TYPE_EVENTS, model_name=model, use_foundation=use_foundation
    )
    draft_transient = draft_activities(
        condensed_weather or weather_str[:WEATHER_PREVIEW_LIMIT],
        cleaned_events,
        SOURCE_TYPE_TRANSIENT,
        location,
        age_range,
        date_range,
        use_foundation=use_foundation,
        year=plan_year,
    )
    json_transient = {}
    if draft_transient:
        refined_transient = refine_draft(draft_transient)
        json_transient = (
            structure_to_json(
                refined_transient, SOURCE_TYPE_TRANSIENT, age_range, condensed_weather,
                year=plan_year,
            )
            or {}
        )
    else:
        print_warning("Transient draft failed, using monolithic fallback")
        from weekend.prompts import build_transient_system_prompt, build_transient_user_prompt

        sys_t = build_transient_system_prompt(
            model,
            location=location,
            age_range=age_range,
            date_range=date_range,
            events_str=events_str,
        )
        usr_t = build_transient_user_prompt(dates_str, weather_str, events_str)
        json_transient = get_llm_json(sys_t, usr_t, use_foundation=use_foundation) or {}

    cleaned_venues = extract_sources(
        venues_str, SOURCE_TYPE_VENUES, model_name=model, use_foundation=use_foundation
    )
    draft_fixed = draft_activities(
        condensed_weather or weather_str[:WEATHER_PREVIEW_LIMIT],
        cleaned_venues,
        SOURCE_TYPE_FIXED,
        location,
        age_range,
        date_range,
        use_foundation=use_foundation,
        year=plan_year,
    )
    json_fixed = {}
    if draft_fixed:
        refined_fixed = refine_draft(draft_fixed)
        json_fixed = (
            structure_to_json(
                refined_fixed, SOURCE_TYPE_FIXED, age_range, condensed_weather,
                year=plan_year,
            )
            or {}
        )
    else:
        print_warning("Fixed draft failed, using monolithic fallback")
        from weekend.prompts import build_fixed_system_prompt, build_fixed_user_prompt

        sys_f = build_fixed_system_prompt(
            model, location=location, age_range=age_range, venues_str=venues_str
        )
        usr_f = build_fixed_user_prompt(dates_str, weather_str, venues_str)
        json_fixed = get_llm_json(sys_f, usr_f, use_foundation=use_foundation) or {}

    return json_transient, json_fixed
