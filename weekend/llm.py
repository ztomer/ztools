import json
import re
from pathlib import Path

from weekend.config import OSAURUS_BASE_URL, ensure_server

from lib.osaurus_lib import (
    get_best_model,
    call_llm_api,
    strip_thinking,
    panic_dump,
    apply_model_quirks,
    _extract_json_only,
)
from lib.llm.fallback import call_with_fallback
from lib.config import Task
from lib.tui import debug_print, WARN

# LLM API defaults
LLM_TEMPERATURE = 0.1
LLM_API_TIMEOUT = 1800
LLM_MAX_RETRIES = 5

# Phase pipeline timeouts (seconds)
PHASE_TIMEOUT_WEATHER = 60
PHASE_TIMEOUT_EXTRACT = 300
PHASE_TIMEOUT_DRAFT = 300
PHASE_TIMEOUT_REFINE = 120
PHASE_TIMEOUT_STRUCTURE = 120
PHASE_MAX_RETRIES = 3
from lib.mlx_lib import (
    find_text_mlx_model,
    call_mlx,
    process_mlx_content,
)


def _call_llm(system_prompt, user_prompt, timeout, max_retries=PHASE_MAX_RETRIES, parse_json=False):
    target_model = get_best_model(Task.JSON)
    last_content = None

    def call_fn(model_name):
        nonlocal last_content
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        messages = apply_model_quirks(messages, model_name)

        debug_print(f"[llm] _call_llm: {len(messages)} messages, timeout={timeout}, parse_json={parse_json}", flush=True)
        debug_print(f"[llm] system={len(messages[0]['content'])} chars, user={len(messages[1]['content'])} chars", flush=True)

        result = call_llm_api(
            OSAURUS_BASE_URL.rstrip("/"),
            model_name,
            messages,
            temperature=LLM_TEMPERATURE,
            timeout=timeout,
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
                        debug_print(f"[llm] JSON extracted, length={len(json_str)}", flush=True)
                        return json.loads(json_str)
                    debug_print("[llm] WARNING: _extract_json_only returned None", flush=True)
                except Exception as e:
                    debug_print(f"[llm] JSON parse error: {e}", flush=True)
                return None
            return cleaned
        else:
            err = result.get("error", "no content") if isinstance(result, dict) else str(result)
            print(f"{WARN} Osaurus API error: {err[:100]}")
            debug_print(f"[llm] No content in result: {result}", flush=True)
        return None

    def mlx_fn():
        mlx_model = find_text_mlx_model(["qwen", "llama", "phi"])
        if mlx_model:
            print(f"[llm] MLX fallback: {mlx_model.name}")
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                messages = apply_model_quirks(messages, getattr(mlx_model, "name", str(mlx_model)))
                mlx_sys = next((m["content"] for m in messages if m["role"] == "system"), system_prompt)
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
                print(f"[llm] MLX failed: {e}")
        return None

    result = call_with_fallback(
        [target_model], call_fn,
        restart_fn=ensure_server, mlx_fn=mlx_fn,
        max_server_retries=max_retries - 1, label="Osaurus",
    )

    if parse_json and result is None and last_content is not None:
        debug_print("[llm] All retries failed, dumping content")
        panic_dump(last_content)

    return result


def get_llm_json(system_prompt, user_prompt, max_retries=LLM_MAX_RETRIES):
    result = _call_llm(system_prompt, user_prompt, timeout=LLM_API_TIMEOUT, max_retries=max_retries, parse_json=True)
    if result is None:
        print("[llm] WARNING: Failed to parse JSON, returning empty result")
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

            for keys, std in [(NAME_KEYS, "name"), (LOC_KEYS, "location"),
                          (AGE_KEYS, "target_ages"), (PRICE_KEYS, "price"),
                          (WEATHER_KEYS, "weather"), (DAY_KEYS, "day"), (DUR_KEYS, "duration")]:
                if std not in item:
                    for k in keys:
                        if k in item:
                            item[std] = item[k]
                            break

            normalized.append(item)
    return normalized


def _score_item(item, weather_str="", age_range=""):
    CLOUD_KEYWORDS = ["cloudy", "overcast", "indoor", "rain", "snow", "storm", "wet", "cold (<10"]
    SUN_KEYWORDS = ["sunny", "clear", "warm", "sun", "hot", "outdoor", "outside"]

    score = 0.0

    fields = ["name", "location", "price", "target_ages", "weather", "day", "duration"]
    populated = sum(1 for f in fields if item.get(f))
    score += (populated / len(fields)) * 3.0

    if age_range and item.get("target_ages"):
        age_nums = sorted(set(int(n) for n in re.findall(r'\d+', str(age_range))))
        target_nums = sorted(set(int(n) for n in re.findall(r'\d+', str(item["target_ages"]))))
        if age_nums and target_nums:
            overlap = len(set(range(age_nums[0], age_nums[-1] + 1)) & set(range(target_nums[0], target_nums[-1] + 1)))
            if overlap >= 2:
                score += 3.0
            elif overlap == 1:
                score += 1.5

    if item.get("weather"):
        w = item["weather"].lower()
        is_cloudy = any(k in w for k in CLOUD_KEYWORDS)
        is_sunny = any(k in w for k in SUN_KEYWORDS)
        forecast_cloudy = any(k in weather_str.lower() for k in CLOUD_KEYWORDS) if weather_str else False
        forecast_sunny = any(k in weather_str.lower() for k in SUN_KEYWORDS) if weather_str else False
        if is_cloudy and forecast_cloudy:
            score += 2.0
        elif is_sunny and forecast_sunny:
            score += 2.0
        elif is_cloudy or is_sunny:
            score += 1.0

    if item.get("price") and item["price"].lower() not in ("", "free", "n/a", "tbd"):
        score += 0.5
    if item.get("location") and len(item["location"]) > 5:
        score += 0.5

    return min(round(score / 2.0, 1), 5.0)


def fetch_scores_for_items(items, weather_str="", age_range=""):
    for item in items:
        item["score"] = _score_item(item, weather_str=weather_str, age_range=age_range)


def condense_weather(weather_str):
    from weekend.prompts import build_weather_condense_prompt
    result = _call_llm("", build_weather_condense_prompt(weather_str), timeout=PHASE_TIMEOUT_WEATHER)
    return result if result else weather_str[:200]


EXTRACT_SIGNALS_PATH = Path(__file__).parent.parent / "conf" / "extract_signals.json"
DEFAULT_BATCH_SIZE = 5
MAX_BATCH_SIZE = 10


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


def extract_sources(raw_text, source_type, model_name="default"):
    signals = _load_extract_signals()
    per_type = signals.setdefault(model_name or "default", {}).setdefault(source_type, {})
    batch_size = per_type.get("batch_size", DEFAULT_BATCH_SIZE)

    lines = [l for l in raw_text.split("\n") if l.startswith("- ")]
    if not lines:
        return raw_text

    results = []
    streak = 0
    i = 0
    while i < len(lines):
        chunk = lines[i:i + batch_size]
        prompt = _extract_group_prompt(chunk, source_type)
        result = _call_llm("", prompt, timeout=PHASE_TIMEOUT_EXTRACT)

        if result:
            results.append(result)
            streak += 1
            i += batch_size
            if streak >= 3 and batch_size < MAX_BATCH_SIZE:
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


def draft_activities(weather_condensed, cleaned_sources, source_type, location, age_range, date_range):
    from weekend.prompts import build_draft_prompt
    prompt = build_draft_prompt(weather_condensed, cleaned_sources, source_type, location, age_range, date_range)
    return _call_llm("", prompt, timeout=PHASE_TIMEOUT_DRAFT)


def refine_draft(draft_text):
    from weekend.prompts import build_refine_prompt
    result = _call_llm("", build_refine_prompt(draft_text), timeout=PHASE_TIMEOUT_REFINE)
    return result if result else draft_text


def structure_to_json(text, source_type, age_range):
    from weekend.prompts import build_structure_system_prompt, build_structure_user_prompt
    sys_prompt = build_structure_system_prompt(source_type, age_range)
    usr_prompt = build_structure_user_prompt(text)
    return _call_llm(sys_prompt, usr_prompt, timeout=PHASE_TIMEOUT_STRUCTURE, parse_json=True)


def generate_weekend_plan(model, weather_str, events_str, venues_str, dates_str, location, age_range, date_range):
    from weekend.output import print_step, print_warning

    condensed_weather = condense_weather(weather_str)

    cleaned_events = extract_sources(events_str, "events", model_name=model)
    draft_transient = draft_activities(condensed_weather or weather_str[:200], cleaned_events, "transient", location, age_range, date_range)
    json_transient = {}
    if draft_transient:
        refined_transient = refine_draft(draft_transient)
        json_transient = structure_to_json(refined_transient, "transient", age_range) or {}
    else:
        print_warning("Transient draft failed, using monolithic fallback")
        from weekend.prompts import build_transient_system_prompt, build_transient_user_prompt
        sys_t = build_transient_system_prompt(model, location=location, age_range=age_range, date_range=date_range)
        usr_t = build_transient_user_prompt(dates_str, weather_str, events_str)
        json_transient = get_llm_json(sys_t, usr_t) or {}

    cleaned_venues = extract_sources(venues_str, "venues", model_name=model)
    draft_fixed = draft_activities(condensed_weather or weather_str[:200], cleaned_venues, "fixed", location, age_range, date_range)
    json_fixed = {}
    if draft_fixed:
        refined_fixed = refine_draft(draft_fixed)
        json_fixed = structure_to_json(refined_fixed, "fixed", age_range) or {}
    else:
        print_warning("Fixed draft failed, using monolithic fallback")
        from weekend.prompts import build_fixed_system_prompt, build_fixed_user_prompt
        sys_f = build_fixed_system_prompt(model, location=location, age_range=age_range)
        usr_f = build_fixed_user_prompt(dates_str, weather_str, venues_str)
        json_fixed = get_llm_json(sys_f, usr_f) or {}

    return json_transient, json_fixed
