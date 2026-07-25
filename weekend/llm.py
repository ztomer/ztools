import json
import os
import re
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


def _score_item(item, weather_str="", age_range=""):
    CLOUD_KEYWORDS = [
        "cloudy",
        "overcast",
        "rain",
        "snow",
        "storm",
        "wet",
        "cold (<10",
        "drizzle",
        "showers",
        "thunder",
    ]
    SUN_KEYWORDS = [
        "sunny",
        "clear",
        "warm",
        "sun",
        "hot",
        "outdoor",
        "outside",
        "fair",
        "dry",
        "pleasant",
    ]

    score = 0.0

    fields = ["name", "location", "price", "target_ages", "weather", "day", "duration"]
    populated = sum(1 for f in fields if item.get(f))
    score += (populated / len(fields)) * POPULATED_FIELDS_BASE_SCORE

    if age_range and item.get("target_ages"):
        age_nums = sorted(set(int(n) for n in re.findall(r"\d+", str(age_range))))
        target_nums = sorted(set(int(n) for n in re.findall(r"\d+", str(item["target_ages"]))))
        if age_nums and target_nums:
            overlap = max(
                0, min(age_nums[-1], target_nums[-1]) - max(age_nums[0], target_nums[0]) + 1
            )
            if overlap >= 2:
                score += OVERLAP_SCORE_HIGH
            elif overlap == 1:
                score += OVERLAP_SCORE_LOW

    if item.get("weather"):
        w = item["weather"].lower()
        is_cloudy = any(k in w for k in CLOUD_KEYWORDS)
        is_sunny = any(k in w for k in SUN_KEYWORDS)
        is_outdoor = "outdoor" in w
        is_indoor = "indoor" in w and not is_outdoor
        forecast_cloudy = (
            any(k in weather_str.lower() for k in CLOUD_KEYWORDS) if weather_str else False
        )
        forecast_sunny = (
            any(k in weather_str.lower() for k in SUN_KEYWORDS) if weather_str else False
        )
        if is_indoor:
            score += WEATHER_PARTIAL_BONUS
        elif is_outdoor and forecast_sunny:
            score += WEATHER_MATCH_BONUS
        elif is_outdoor and forecast_cloudy:
            pass
        elif is_cloudy and forecast_cloudy:
            score += WEATHER_MATCH_BONUS
        elif is_sunny and forecast_sunny:
            score += WEATHER_MATCH_BONUS
        elif is_sunny or is_cloudy:
            score += WEATHER_PARTIAL_BONUS

    if item.get("price") and item["price"].lower() not in ("", "free", "n/a", "tbd"):
        score += PRICE_MATCH_BONUS
    if item.get("location") and len(item["location"]) > 5:
        score += LOCATION_MATCH_BONUS

    if item.get("name"):
        name = item["name"]
        if len(name) > 20:
            score += 0.3
        if any(c.isdigit() for c in name):
            score += 0.2
        if len(name.split()) >= 3:
            score += 0.3
    if item.get("duration") and item["duration"].lower() not in ("", "2-3 hours"):
        score += 0.3
    if item.get("weather") and item["weather"].lower() not in ("", "indoor", "outdoor", "both"):
        score += 0.2

    return min(round(score / SCORE_DIVISOR, 1), SCORE_MAX_LIMIT)


def fetch_scores_for_items(items, weather_str="", age_range=""):
    for item in items:
        item["score"] = _score_item(item, weather_str=weather_str, age_range=age_range)


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


def _load_extract_signals():
    try:
        if EXTRACT_SIGNALS_PATH.exists():
            return json.loads(EXTRACT_SIGNALS_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_extract_signals(signals):
    EXTRACT_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXTRACT_SIGNALS_PATH.write_text(json.dumps(signals, indent=2, sort_keys=True))


def _extract_group_prompt(lines, source_type):
    from weekend.prompts import build_source_extract_prompt

    return build_source_extract_prompt("\n".join(lines), source_type)


def extract_sources(raw_text, source_type, model_name="default", use_foundation=False):
    signals = _load_extract_signals()
    per_type = signals.setdefault(model_name or "default", {}).setdefault(source_type, {})
    batch_size = per_type.get("batch_size", DEFAULT_BATCH_SIZE)
    phase_key = f"extract_sources/{source_type}"

    lines = [line for line in raw_text.split("\n") if line.startswith("- ")]
    if not lines:
        return raw_text

    results = []
    streak = 0
    i = 0
    while i < len(lines):
        chunk = lines[i : i + batch_size]
        prompt = _extract_group_prompt(chunk, source_type)
        result = _call_llm(
            "",
            prompt,
            timeout=PHASE_TIMEOUT_EXTRACT,
            phase_key=phase_key,
            use_foundation=use_foundation,
        )

        if result:
            results.append(result)
            streak += 1
            i += batch_size
            if streak >= BATCH_GROWTH_STREAK_LIMIT and batch_size < MAX_BATCH_SIZE:
                batch_size += 1
                per_type["batch_size"] = batch_size
                _save_extract_signals(signals)
        else:
            per_type["timeouts"] = per_type.get("timeouts", 0) + 1
            batch_size = max(batch_size // 2, 1)
            per_type["batch_size"] = batch_size
            _save_extract_signals(signals)
            streak = 0
            if batch_size == 1:
                results.append(chunk[0])
                i += 1
                batch_size = per_type.get("batch_size", DEFAULT_BATCH_SIZE)

    if not results:
        return raw_text
    return "\n".join(results)


def draft_activities(
    weather_condensed,
    cleaned_sources,
    source_type,
    location,
    age_range,
    date_range,
    use_foundation=False,
):
    from weekend.prompts import build_draft_prompt

    prompt = build_draft_prompt(
        weather_condensed, cleaned_sources, source_type, location, age_range, date_range
    )
    return _call_llm(
        "",
        prompt,
        timeout=PHASE_TIMEOUT_DRAFT,
        phase_key=f"draft_activities/{source_type}",
        use_foundation=use_foundation,
    )


def refine_draft(draft_text, use_foundation=False):
    from weekend.prompts import build_refine_prompt

    result = _call_llm(
        "",
        build_refine_prompt(draft_text),
        timeout=PHASE_TIMEOUT_REFINE,
        phase_key="refine_draft",
        use_foundation=use_foundation,
    )
    return result if result else draft_text


def structure_to_json(text, source_type, age_range, weather_condensed="", use_foundation=False):
    from weekend.prompts import build_structure_system_prompt, build_structure_user_prompt

    sys_prompt = build_structure_system_prompt(source_type, age_range, weather_condensed)
    usr_prompt = build_structure_user_prompt(text)
    return _call_llm(
        sys_prompt,
        usr_prompt,
        timeout=PHASE_TIMEOUT_STRUCTURE,
        parse_json=True,
        phase_key=f"structure_to_json/{source_type}",
        use_foundation=use_foundation,
    )


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
):
    from weekend.output import print_warning

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
    )
    json_transient = {}
    if draft_transient:
        refined_transient = refine_draft(draft_transient)
        json_transient = (
            structure_to_json(
                refined_transient, SOURCE_TYPE_TRANSIENT, age_range, condensed_weather
            )
            or {}
        )
    else:
        print_warning("Transient draft failed, using monolithic fallback")
        from weekend.prompts import build_transient_system_prompt, build_transient_user_prompt

        sys_t = build_transient_system_prompt(
            model, location=location, age_range=age_range, date_range=date_range
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
    )
    json_fixed = {}
    if draft_fixed:
        refined_fixed = refine_draft(draft_fixed)
        json_fixed = (
            structure_to_json(refined_fixed, SOURCE_TYPE_FIXED, age_range, condensed_weather) or {}
        )
    else:
        print_warning("Fixed draft failed, using monolithic fallback")
        from weekend.prompts import build_fixed_system_prompt, build_fixed_user_prompt

        sys_f = build_fixed_system_prompt(model, location=location, age_range=age_range)
        usr_f = build_fixed_user_prompt(dates_str, weather_str, venues_str)
        json_fixed = get_llm_json(sys_f, usr_f, use_foundation=use_foundation) or {}

    return json_transient, json_fixed
