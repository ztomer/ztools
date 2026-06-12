import json
import re

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
from lib.mlx_lib import (
    find_text_mlx_model,
    call_mlx,
    process_mlx_content,
)


def get_llm_json(system_prompt, user_prompt, max_retries=LLM_MAX_RETRIES):
    target_model = get_best_model(Task.JSON)
    last_content = None

    def call_fn(model_name):
        nonlocal last_content
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        messages = apply_model_quirks(messages, model_name)

        debug_print(f"[llm] Calling API with {len(messages)} messages, system={len(messages[0]['content'])}, user={len(messages[1]['content'])}", flush=True)

        result = call_llm_api(
            OSAURUS_BASE_URL.rstrip("/"),
            model_name,
            messages,
            temperature=LLM_TEMPERATURE,
            timeout=LLM_API_TIMEOUT,
            parse_json=True,
        )

        debug_print(f"[llm] API response keys: {result.keys() if isinstance(result, dict) else type(result)}", flush=True)
        debug_print(f"[llm] API response preview: {str(result)[:200]}", flush=True)

        if result and "content" in result:
            try:
                raw_content = result["content"]
                last_content = raw_content
                debug_print(f"[llm] Raw content length: {len(raw_content)}", flush=True)
                debug_print(f"[llm] Raw content preview: {raw_content[:300]}", flush=True)

                cleaned = strip_thinking(raw_content)
                debug_print(f"[llm] After strip_thinking length: {len(cleaned)}", flush=True)
                debug_print(f"[llm] After strip_thinking preview: {cleaned[:300]}", flush=True)

                json_str = _extract_json_only(cleaned)
                if json_str is not None:
                    debug_print(f"[llm] JSON extracted successfully, length: {len(json_str)}", flush=True)
                    return json.loads(json_str)
                else:
                    debug_print(f"[llm] WARNING: _extract_json_only returned None", flush=True)
                    raise ValueError("No valid JSON found")
            except Exception as e:
                debug_print(f"[llm] JSON parse error: {e}", flush=True)
        else:
            err = result.get("error", "no content") if isinstance(result, dict) else str(result)
            print(f"{WARN} Osaurus API error: {err[:100]}")
            debug_print(f"[llm] WARNING: No content in result: {result}", flush=True)

        return None

    def mlx_fn():
        mlx_model = find_text_mlx_model(["qwen", "llama", "phi"])
        if mlx_model:
            print(f"[llm] Falling back to MLX: {mlx_model.name}")
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                messages = apply_model_quirks(messages, getattr(mlx_model, "name", str(mlx_model)))
                mlx_sys = next((m["content"] for m in messages if m["role"] == "system"), system_prompt)
                mlx_usr = next((m["content"] for m in messages if m["role"] == "user"), user_prompt)

                raw = call_mlx(
                    mlx_model, f"System: {mlx_sys}\n\nUser: {mlx_usr}"
                )
                if raw:
                    cleaned = process_mlx_content(raw)
                    json_str = _extract_json_only(cleaned)
                    if json_str is not None:
                        return json.loads(json_str)
                    else:
                        raise ValueError("No valid JSON found")
            except Exception as e:
                print(f"[llm] MLX failed: {e}")
        return None

    result = call_with_fallback(
        [target_model],
        call_fn,
        restart_fn=ensure_server,
        mlx_fn=mlx_fn,
        max_server_retries=max_retries - 1,
        label="Osaurus",
    )

    if result is None and last_content is not None:
        debug_print("[llm] All retries failed, dumping content")
        panic_dump(last_content)

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
