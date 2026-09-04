"""Reshape whatever the model returned into the row dicts the report needs.

Split out of weekend/cli.py, which crossed the 500-line limit. These two are
pure shape-normalisation over untrusted LLM output — no I/O, no config — so
they belong next to the report rather than inside the command entry point.
"""

from lib.config import get_model_top_keys
from lib.tui import debug_print

from weekend.llm import normalize_llm_items

# Shape heuristics for untrusted model output: how many entries a bare list must
# carry before it is read as a list of items rather than as noise.
MIN_FIXED_LIST_LEN = 1
EMPTY_LIST_LIMIT = 0
MIN_TRANSIENT_LIST_LEN = 2
FALLBACK_LIST_LEN_HIGH = 3
FALLBACK_LIST_LEN_LOW = 2


def _parse_fixed(json_fixed, actual_model, field_mapping):
    fixed_keys = get_model_top_keys(actual_model).get(
        "fixed",
        [
            "fixed_activities",
            "year_round_fixed_activities",
            "venues",
            "places",
            "activities",
            "items",
        ],
    )
    name_keys = ["name"] + [k for k, v in field_mapping.items() if v == "name"]

    if isinstance(json_fixed, list) and len(json_fixed) >= MIN_FIXED_LIST_LEN:
        valid_items = [
            i for i in json_fixed if isinstance(i, dict) and any(i.get(nk) for nk in name_keys)
        ]
        if valid_items:
            return normalize_llm_items(valid_items, field_mapping=field_mapping)

    if isinstance(json_fixed, dict):
        for key in fixed_keys:
            if (
                json_fixed.get(key)
                and isinstance(json_fixed.get(key), list)
                and len(json_fixed.get(key)) > EMPTY_LIST_LIMIT
            ):
                raw = json_fixed[key]
                valid_items = [
                    i for i in raw if isinstance(i, dict) and any(i.get(nk) for nk in name_keys)
                ]
                if valid_items:
                    debug_print(
                        f"[DEBUG] Found valid in key '{key}': {len(valid_items)} items", flush=True
                    )
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

        if any(json_fixed.get(nk) for nk in name_keys):
            debug_print("[DEBUG] Single object, wrapping in list", flush=True)
            return normalize_llm_items([json_fixed], field_mapping=field_mapping)

        for k, v in json_fixed.items():
            if isinstance(v, list) and len(v) >= MIN_FIXED_LIST_LEN:
                valid_items = [i for i in v if isinstance(i, dict) and i.get("name")]
                if valid_items:
                    debug_print(f"[DEBUG] Fallback key '{k}': {len(valid_items)} items", flush=True)
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

    return []


def _parse_transient(json_transient, actual_model, field_mapping):
    name_keys = ["name"] + [k for k, v in field_mapping.items() if v == "name"]
    all_name_keys = name_keys + ["description", "title", "event", "summary", "activity_name"]

    if isinstance(json_transient, list) and len(json_transient) >= MIN_TRANSIENT_LIST_LEN:
        filtered = [
            i
            for i in json_transient
            if isinstance(i, dict)
            and not any(k in i for k in ["temperature", "condition", "precipitation"])
        ]
        if not filtered:
            return []

        def _normalize_with_fallback(items):
            result = []
            for item in items:
                new_item = dict(item)
                if not new_item.get("name"):
                    for alt in ["description", "activity_name", "title"]:
                        if new_item.get(alt):
                            new_item["name"] = new_item.pop(alt)
                            break
                result.append(new_item)
            return result

        valid_items = [
            i for i in filtered if isinstance(i, dict) and any(i.get(nk) for nk in all_name_keys)
        ]
        if valid_items:
            result = _normalize_with_fallback(valid_items)
            return normalize_llm_items(result, field_mapping=field_mapping)

        alt_items = [
            i
            for i in filtered
            if isinstance(i, dict)
            and any(
                i.get(ak) for ak in ["description", "title", "event", "summary", "activity_name"]
            )
        ]
        if alt_items:
            result = _normalize_with_fallback(alt_items)
            return normalize_llm_items(result, field_mapping=field_mapping)

    if isinstance(json_transient, dict):
        transient_keys = get_model_top_keys(actual_model).get(
            "transient", ["transient_events", "events", "activities", "recommendations"]
        )

        for key in transient_keys:
            if json_transient.get(key) and isinstance(json_transient.get(key), list):
                raw = json_transient[key]
                valid_items = [i for i in raw if isinstance(i, dict) and i.get("name")]
                if valid_items:
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

        forecast = json_transient.get("weekend_forecast")
        if forecast and isinstance(forecast, dict):
            all_events = []
            for day_data in forecast.values():
                if isinstance(day_data, dict) and isinstance(day_data.get("events"), list):
                    all_events.extend(day_data["events"])
            if all_events:
                valid_items = [i for i in all_events if isinstance(i, dict) and i.get("name")]
                if valid_items:
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

        if json_transient.get("name"):
            return normalize_llm_items([json_transient], field_mapping=field_mapping)

        for k, v in json_transient.items():
            if isinstance(v, list) and len(v) >= FALLBACK_LIST_LEN_HIGH:
                valid_items = [i for i in v if isinstance(i, dict) and i.get("name")]
                if valid_items:
                    return normalize_llm_items(valid_items, field_mapping=field_mapping)

        for k, v in json_transient.items():
            if isinstance(v, list) and len(v) >= FALLBACK_LIST_LEN_LOW:
                return normalize_llm_items(v, field_mapping=field_mapping)

    return []
