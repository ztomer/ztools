#!/usr/bin/env python3
import re
import json
from typing import Any, Optional, List, Dict, Union

from .content_processing import clean_model_output, extract_content_from_code_blocks, remove_inline_thinking
from .config_getters import get_model_config

clean_output = clean_model_output


def _extract_json_only(content: str) -> Optional[str]:
    if not content:
        return None
    code_block = extract_content_from_code_blocks(content)
    if code_block:
        content = code_block
        content = clean_model_output(content)
    else:
        content = clean_model_output(content)
        content = remove_inline_thinking(content)
    content = content.strip()

    def find_json(start_char, end_char):
        starts = [i for i, c in enumerate(content) if c == start_char]
        ends = [i for i, c in enumerate(content) if c == end_char]
        ends.reverse()
        for s in starts[:15]:
            for e in ends[:15]:
                if e > s:
                    candidate = content[s:e+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        continue
        return None

    res = find_json('[', ']')
    if res:
        return res
    res = find_json('{', '}')
    if res:
        return res
    return None


def extract_json(content: str, model: str = None) -> Union[Dict[str, Any], List[Any], None]:
    if content:
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        content = re.sub(r'\|[:\-]+\|', '', content)
    json_str = _extract_json_only(content)
    if json_str:
        try:
            data = json.loads(json_str)
            data = normalize_keys(data, model)
            data = fix_json_years(data)
            return filter_json_items(data)
        except json.JSONDecodeError:
            pass
    data = _extract_plain_list(content)
    if data:
        data = normalize_keys(data, model)
        data = fix_json_years(data)
        return filter_json_items(data)
    data = normalize_text_output(content)
    if data:
        data = normalize_keys(data, model)
        return filter_json_items(data)
    return None


def _extract_plain_list(content: str) -> Optional[List[Dict[str, Any]]]:
    if not content:
        return None
    items = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+[\.\)]\s+(.+)$|^- (.+)$", line)
        if match:
            name = (match.group(1) or match.group(2)).strip()
            if name and not name.startswith("#"):
                items.append({"name": name})
        elif line and not line.startswith("#") and len(line) > 1:
            items.append({"name": line})
    return items if items else None


TOP_LEVEL_KEYS = {
    "fixed_activities": "fixed_activities",
    "activities": "fixed_activities",
    "year_round_activities": "fixed_activities",
    "indoor_play_places": "fixed_activities",
    "play_places": "fixed_activities",
    "venues": "fixed_activities",
    "transient_events": "transient_events",
    "events": "transient_events",
    "limited_time_events": "transient_events",
}

KEY_NORMALIZATIONS = {
    "event": "name",
    "title": "activity",
    "place": "name",
    "venue": "location",
    "address": "location",
    "where": "location",
    "date": "day",
    "when": "day",
    "time": "duration",
    "age_group": "target_ages",
    "ages": "target_ages",
    "audience": "target_ages",
    "who": "target_ages",
    "cost": "price",
    "pricing": "price",
    "type": "weather",
    "setting": "weather",
    "indoor_outdoor": "weather",
}


def normalize_keys(data: Any, model: str = None) -> Any:
    if not data:
        return data
    model_mappings = {}
    if model:
        config = get_model_config(model)
        model_mappings = config.get("key_mappings", {})
    all_mappings = {**KEY_NORMALIZATIONS, **model_mappings}
    if isinstance(data, dict):
        for old_key, new_key in TOP_LEVEL_KEYS.items():
            if old_key in data:
                arr = data[old_key]
                if isinstance(arr, list):
                    return {new_key: [normalize_keys(item, model) for item in arr]}
                elif isinstance(arr, dict):
                    for k, v in arr.items():
                        if isinstance(v, list):
                            return {new_key: [normalize_keys(item, model) for item in v]}
                return {new_key: arr}
        result = {}
        for key, value in data.items():
            new_key = all_mappings.get(key, key)
            result[new_key] = normalize_keys(value, model)
        if len(result) == 1:
            only_key = list(result.keys())[0]
            only_value = result[only_key]
            if isinstance(only_value, str) and only_value:
                result["name"] = only_value
                if only_key != "name":
                    del result[only_key]
        return result
    if isinstance(data, list):
        fixed = []
        for item in data:
            if isinstance(item, dict) and len(item) == 1:
                key = list(item.keys())[0]
                value = item[key]
                if isinstance(value, str) and value:
                    item["name"] = value
            fixed.append(item)
        merged = merge_flat_dicts(fixed)
        return [normalize_keys(item, model) for item in merged]
    return data


def merge_flat_dicts(items: List[Any]) -> List[Dict[str, Any]]:
    if not items:
        return items
    return items


def filter_json_items(items: List[Any]) -> List[Any]:
    if not items:
        return items
    filtered = []
    for item in items:
        if isinstance(item, dict):
            keys = [str(k) for k in item.keys()]
            if any(k.strip().startswith("|") for k in keys):
                continue
            vals = [str(v) for v in item.values() if v]
            text = " ".join(vals).lower()
            if text.startswith("based on") or text.startswith("note:"):
                continue
            if "temperature" in text and "conditions" in text:
                continue
            if all(v in ("", "-", ":", None, " | ", "|", "---") for v in item.values()):
                continue
            filtered.append(item)
        elif isinstance(item, str):
            text = item.strip().lower()
            if text.startswith("|") or text.startswith(":") or text.startswith("based on"):
                continue
            if text in ("---", "----"):
                continue
            if text:
                filtered.append(item)
        else:
            filtered.append(item)
    return filtered


def fix_json_years(items: List[Any]) -> List[Any]:
    if not items:
        return items
    fixed = []
    for item in items:
        if isinstance(item, dict):
            fixed_item = {}
            for k, v in item.items():
                if isinstance(v, str):
                    v = re.sub(r'\b2626\b', '2026', v)
                    v = re.sub(r'\b26(?!\d)', '2026', v)
                fixed_item[k] = v
            fixed.append(fixed_item)
        else:
            fixed.append(item)
    return fixed


def normalize_text_output(text_output: str) -> List[Dict[str, Any]]:
    if not text_output:
        return []
    items = []
    for line in text_output.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^\d+[\.\)]\s*([^\-:?]+)(?:\s*[-:]\s*(.+))?$", line)
        if not match:
            match = re.match(r"^-\s*([^\-:?]+)(?:\s*[-:]\s*(.+))?$", line)
        if match:
            name = (match.group(1) or "").strip()
            details = (match.group(2) or "").strip() if match.lastindex and match.group(2) else ""
            if name and not name.startswith("*"):
                item = {"name": name}
                parts = [p.strip() for p in re.split(r"[-:,]", details) if p.strip()] if details else []
                if len(parts) > 0 and parts[0]:
                    item["location"] = parts[0]
                if len(parts) > 1 and parts[1]:
                    item["target_ages"] = parts[1]
                if len(parts) > 2 and parts[2]:
                    item["price"] = parts[2]
                if len(parts) > 3 and parts[3]:
                    item["weather"] = parts[3]
                items.append(item)
    return items
