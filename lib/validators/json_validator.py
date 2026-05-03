# JSON Validation functions

import re
import json
from typing import List, Tuple, Any, Dict

from lib.validators.constants import (
    MAX_SCORE, JSON_ITEMS_WEIGHT, JSON_SCHEMA_WEIGHT,
    JSON_COMPLETENESS_WEIGHT, JSON_QUALITY_WEIGHT,
)


def extract_list_from_dict(data: Dict) -> List[Dict]:
    """Extract list from potentially nested dict."""
    if not isinstance(data, dict):
        return data if isinstance(data, list) else []
    
    # Check for common list keys
    for key in ["items", "activities", "events", "venues", "results", "data", "places"]:
        if key in data and isinstance(data[key], list):
            return data[key]
    
    # Return the dict wrapped in list
    return [data]


def is_valid_list_item(item: Any) -> bool:
    """Check if item is a non-empty dict."""
    return isinstance(item, dict) and bool(item)


def check_source_extraction(items: List[Dict], source_text: str) -> float:
    """Check how many items match source text.
    
    Returns ratio of items that have fields matching source.
    """
    if not source_text or not items:
        return 0.0
    
    source_lower = source_text.lower()
    matched = 0
    
    for item in items:
        # Check if any field value is in source
        for value in item.values():
            if isinstance(value, str) and value.lower() in source_lower:
                matched += 1
                break
    
    return matched / len(items) if items else 0.0


def get_source_matching_details(items: List[Dict], source_text: str) -> Dict:
    """Get detailed matching info for debugging."""
    if not source_text or not items:
        return {"matched": [], "unmatched": [], "ratio": 0.0}
    
    source_lower = source_text.lower()
    matched = []
    unmatched = []
    
    for item in items:
        name = item.get("name", "")
        item_matched = False
        
        for value in item.values():
            if isinstance(value, str) and value.lower() in source_lower:
                item_matched = True
                break
        
        if item_matched:
            matched.append(name)
        else:
            unmatched.append(name or str(item)[:30])
    
    return {
        "matched": matched,
        "unmatched": unmatched,
        "ratio": len(matched) / len(items) if items else 0.0,
    }


def has_item_details(item: dict) -> bool:
    """Check if item has sufficient details beyond name only."""
    if not isinstance(item, dict):
        return False
    # Has any field beyond just name
    return len(item.keys()) > 1


MIN_ITEMS_GOOD = 5
MIN_ITEMS_OK = 3
JSON_STRUCTURE_WEIGHT = 20
JSON_COUNT_GOOD = 20
JSON_COUNT_OK = 10
JSON_VALIDITY_WEIGHT = 20
JSON_VALIDITY_THRESHOLD = 0.7
JSON_SOURCE_WEIGHT = 20


def validate_json(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Score simple JSON lists based on validity and structure."""
    if not data:
        return 0, "empty response"
    
    if isinstance(data, dict):
        data = extract_list_from_dict(data)
    
    items = data if isinstance(data, list) else []
    if not items:
        return 0, "no items found"
    
    score = 0
    failures = []
    
    # Valid JSON structure
    score += JSON_STRUCTURE_WEIGHT
    
    # Proper count of items
    if len(items) >= MIN_ITEMS_GOOD:
        score += JSON_COUNT_GOOD
    elif len(items) >= MIN_ITEMS_OK:
        score += JSON_COUNT_OK
    else:
        failures.append(f"only {len(items)} items (need {MIN_ITEMS_GOOD}+)")
    
    # Each item should be valid
    valid_items = sum(1 for item in items if is_valid_list_item(item))
    
    if valid_items == len(items):
        score += JSON_VALIDITY_WEIGHT
    elif valid_items >= len(items) * JSON_VALIDITY_THRESHOLD:
        score += JSON_COUNT_OK
        failures.append(f"only {valid_items}/{len(items)} items are valid")
    else:
        failures.append(f"only {valid_items}/{len(items)} items are valid")
    
    # Check if items use input data
    if source_text and items:
        source_ratio = check_source_extraction(items, source_text)
        if source_ratio >= 0.8:
            score += JSON_SOURCE_WEIGHT
        elif source_ratio >= 0.5:
            score += JSON_SOURCE_WEIGHT // 2
        elif source_ratio > 0:
            score += JSON_SOURCE_WEIGHT // 4
        else:
            failures.append("not from input")
    
    return min(MAX_SCORE, score), "; ".join(failures)


def has_required_fields(item: Dict, required: List[str]) -> bool:
    """Check if item has all required fields."""
    return all(item.get(field) for field in required)


REQUIRED_DETAILED_FIELDS = ["name", "location"]


def validate_detailed_json(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Score objects with details based on quality, not count.
    
    Strict checks:
    - Items must have full details
    - Source matching is critical
    """
    if not data:
        return 0, "empty response"
    
    if isinstance(data, dict):
        data = extract_list_from_dict(data)
    
    items = data if isinstance(data, list) else []
    if not items:
        return 0, "no items found"
    
    score = 0
    failures = []
    
    # Valid JSON structure
    score += JSON_SCHEMA_WEIGHT
    
    # Check each item has required fields
    valid_items = sum(1 for item in items if has_required_fields(item, REQUIRED_DETAILED_FIELDS))
    if valid_items == len(items):
        score += JSON_COMPLETENESS_WEIGHT
    elif valid_items > 0:
        score += JSON_COMPLETENESS_WEIGHT // 2
        failures.append(f"{valid_items}/{len(items)} have name + location")
    else:
        failures.append("no items with name + location")
    
    # Check for duplicates
    names = [item.get("name", "") for item in items]
    unique_names = set(n for n in names if n)
    if len(unique_names) < len(names):
        failures.append(f"{len(names) - len(unique_names)} duplicates")
    else:
        score += JSON_QUALITY_WEIGHT
    
    # KEY: Source matching
    if source_text and items:
        source_ratio = check_source_extraction(items, source_text)
        if source_ratio >= 0.9:
            score += JSON_SOURCE_WEIGHT
        elif source_ratio >= 0.5:
            score += JSON_SOURCE_WEIGHT // 2
            failures.append(f"only {source_ratio*100:.0f}% from source")
        elif source_ratio > 0:
            score += JSON_SOURCE_WEIGHT // 4
            failures.append(f"mostly extracted ({source_ratio*100:.0f}%)")
        else:
            failures.append("not extracted from source")
    
    return min(MAX_SCORE, score), "; ".join(failures)


JSON_SOURCE_WEIGHT = 20