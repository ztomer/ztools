import re
import json
from typing import List, Tuple, Any, Dict

from lib.validators.constants import (
    MAX_SCORE, JSON_ITEMS_WEIGHT, JSON_SCHEMA_WEIGHT,
    JSON_COMPLETENESS_WEIGHT, JSON_QUALITY_WEIGHT,
)

STOPWORDS = {
    "the", "and", "for", "with", "this", "that", "from", "are", "was",
    "has", "have", "but", "not", "you", "all", "can", "her", "his",
    "had", "they", "been", "will", "would", "could", "what", "when",
    "where", "who", "which", "why", "how",
}

JSON_STRUCTURE_WEIGHT = 20
JSON_COUNT_GOOD = 25
JSON_COUNT_OK = 15
JSON_VALIDITY_WEIGHT = 30
JSON_VALIDITY_THRESHOLD = 0.7
JSON_SOURCE_WEIGHT = 25

DETAILED_STRUCTURE_WEIGHT = 15
DETAILED_COUNT_GOOD = 15
DETAILED_COUNT_OK = 10
DETAILED_QUALITY_WEIGHT = 40
DETAILED_SOURCE_WEIGHT = 30
DETAILED_VALIDITY_THRESHOLD = 0.8
DETAIL_THRESHOLD_HIGH = 0.8
DETAIL_THRESHOLD_MID = 0.5
DETAIL_PARTIAL_HIGH = DETAILED_QUALITY_WEIGHT * 8 // 10
DETAIL_PARTIAL_MID = DETAILED_QUALITY_WEIGHT * 5 // 10
DETAIL_PARTIAL_LOW = DETAILED_QUALITY_WEIGHT * 2 // 10

MIN_ITEMS_GOOD = 10
MIN_ITEMS_OK = 5

DETAIL_REQUIRED_FIELDS = 3

NAME_FIELDS = ["name", "event", "title", "activity", "place"]
DETAIL_FIELDS = [
    "name", "event", "title", "activity", "place",
    "location", "venue", "address", "where",
    "day", "date", "when", "time", "duration",
    "target_ages", "age_group", "ages", "audience", "who",
    "price", "cost", "pricing",
    "weather", "type", "indoor_outdoor", "setting",
]


def extract_list_from_dict(data: Any) -> List[Any]:
    if isinstance(data, dict):
        keys = [
            "activities", "items", "results", "data",
            "fixed_activities", "transient_events",
            "events", "places", "venues", "recommendations",
        ]
        for key in keys:
            if key in data and isinstance(data[key], list):
                return data[key]
        for value in data.values():
            if isinstance(value, dict):
                found = extract_list_from_dict(value)
                if found:
                    return found
        best = []
        for value in data.values():
            if isinstance(value, list) and len(value) > len(best):
                best = value
        return best
    return data if isinstance(data, list) else []


def is_valid_list_item(item: Any) -> bool:
    if isinstance(item, str):
        return bool(item.strip())
    elif isinstance(item, dict):
        return any(item.get(field) for field in ["name", "activity"])
    return False


def has_item_details(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    name_fields = ["name", "event", "title", "activity", "place"]
    has_name = any(item.get(f) for f in name_fields)
    if not has_name:
        return False
    for field in DETAIL_FIELDS:
        if field not in name_fields and item.get(field):
            return True
    return False


def check_source_extraction(items: List[Any], source_text: str) -> float:
    if not items or not source_text:
        return 0.0
    source_lower = source_text.lower()
    source_terms = {
        t for t in source_lower.split()
        if len(t) >= 3 and t not in STOPWORDS
    }
    if not source_terms:
        return 0.0
    matches = 0
    for item in items:
        if isinstance(item, dict):
            item_text = " ".join(
                str(v).lower() for v in item.values() if v
            )
        elif isinstance(item, str):
            item_text = item.lower()
        else:
            item_text = str(item).lower()
        if not item_text:
            continue
        item_terms = {
            t for t in item_text.split()
            if len(t) >= 3 and t not in STOPWORDS
        }
        common = item_terms & source_terms
        if len(common) >= 2:
            matches += 1
    return matches / len(items) if items else 0.0


def get_source_matching_details(items: List[Dict], source_text: str) -> Dict:
    if not items or not source_text:
        return {"matched": [], "unmatched": [], "ratio": 0.0, "source_preview": ""}
    source_lower = source_text.lower()
    source_terms = {
        t for t in source_lower.split()
        if len(t) >= 3 and t not in STOPWORDS
    }
    if not source_terms:
        return {"matched": [], "unmatched": [], "ratio": 0.0, "source_preview": source_text[:50]}
    matched = []
    unmatched = []
    for item in items:
        if isinstance(item, dict):
            item_text = " ".join(str(v).lower() for v in item.values() if v)
            item_name = item.get("name", item.get("event", item.get("title", "")))
        elif isinstance(item, str):
            item_text = item.lower()
            item_name = item
        else:
            item_text = str(item).lower()
            item_name = str(item)
        if not item_text:
            continue
        item_terms = {
            t for t in item_text.split()
            if len(t) >= 3 and t not in STOPWORDS
        }
        common = item_terms & source_terms
        if len(common) >= 2:
            matched.append(item_name or "unnamed")
        else:
            unmatched.append(item_name or "unnamed")
    return {
        "matched": matched,
        "unmatched": unmatched,
        "ratio": len(matched) / len(items) if items else 0.0,
        "source_preview": source_text[:50],
    }


def validate_json(data: Any, source_text: str = "") -> Tuple[int, str]:
    if not data:
        return 0, "empty response"
    items = extract_list_from_dict(data) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    if not items:
        return 0, "no items found"
    score = 0
    failures = []
    score += JSON_STRUCTURE_WEIGHT
    if len(items) >= MIN_ITEMS_GOOD:
        score += JSON_COUNT_GOOD
    elif len(items) >= MIN_ITEMS_OK:
        score += JSON_COUNT_OK
    else:
        failures.append(
            f"only {len(items)} items (need {MIN_ITEMS_GOOD}+)"
        )
    valid_items = sum(1 for item in items if is_valid_list_item(item))
    if valid_items == len(items):
        score += JSON_VALIDITY_WEIGHT
    elif valid_items >= len(items) * JSON_VALIDITY_THRESHOLD:
        score += JSON_COUNT_OK
        failures.append(f"only {valid_items}/{len(items)} items are valid")
    else:
        failures.append(f"only {valid_items}/{len(items)} items are valid")
    if source_text and items:
        source_ratio = check_source_extraction(items, source_text)
        if source_ratio >= 0.8:
            score += JSON_SOURCE_WEIGHT
        elif source_ratio >= 0.5:
            score += JSON_SOURCE_WEIGHT // 2
        elif source_ratio > 0:
            score += JSON_SOURCE_WEIGHT // 4
        else:
            failures.append("not from input (hallucinated)")
    return min(MAX_SCORE, score), "; ".join(failures)


def has_required_fields(item: Dict, required: List[str]) -> bool:
    return all(item.get(field) for field in required)


REQUIRED_DETAILED_FIELDS = ["name", "location"]


def validate_detailed_json(data: Any, source_text: str = "") -> Tuple[int, str]:
    if not data:
        return 0, "empty response"
    items = extract_list_from_dict(data) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    if not items:
        return 0, "no items found"
    score = 0
    failures = []
    score += DETAILED_STRUCTURE_WEIGHT
    if len(items) >= MIN_ITEMS_GOOD:
        score += DETAILED_COUNT_GOOD
    elif len(items) >= MIN_ITEMS_OK:
        score += DETAILED_COUNT_OK
    else:
        failures.append(
            f"only {len(items)} items (need {MIN_ITEMS_GOOD}+)"
        )
    valid_with_details = 0
    for item in items:
        if isinstance(item, dict) and has_item_details(item):
            valid_with_details += 1
    all_have_details = valid_with_details == len(items)
    most_have_details = valid_with_details >= len(items) * 0.8
    if all_have_details:
        score += DETAILED_QUALITY_WEIGHT
    elif most_have_details:
        score += DETAIL_PARTIAL_HIGH
    elif valid_with_details == 0:
        failures.append("no items with details")
    else:
        failures.append(
            f"only {valid_with_details}/{len(items)} have details"
        )
    names = [
        item.get("name", "")
        if isinstance(item, dict) else str(item)
        for item in items
    ]
    unique_names = set(n for n in names if n)
    if len(unique_names) < len(names):
        duplicate_ratio = (len(names) - len(unique_names)) / len(names)
        if duplicate_ratio > 0.1:
            score -= int(duplicate_ratio * 20)
            failures.append(
                f"duplicates ({int(duplicate_ratio*100)}%)"
            )
    else:
        score += JSON_QUALITY_WEIGHT
    source_ratio = 0.0
    if source_text and items:
        source_ratio = check_source_extraction(items, source_text)
        if source_ratio >= 0.8:
            score += DETAILED_SOURCE_WEIGHT
        elif source_ratio >= 0.5:
            score += DETAILED_SOURCE_WEIGHT // 2
        elif source_ratio > 0:
            score += DETAILED_SOURCE_WEIGHT // 4
        else:
            failures.append("not from input (hallucinated)")
    return min(MAX_SCORE, score), "; ".join(failures[:DETAIL_REQUIRED_FIELDS])
