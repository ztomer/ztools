import json
import re
from typing import Any, List, Tuple

from lib.quality_models import Score, ScoreCard, TestCase, _str, _lower

INDOOR = {"indoor", "inside", "in", "cover", "shelter"}
OUTDOOR = {"outdoor", "outside", "park", "open-air", "nature", "trail"}
RAIN = {"rain", "precipitation", "storm", "shower", "drizzle", "wet"}
CLEAR = {"clear", "sunny", "sun", "fair", "dry", "fine"}


def _extract_items(output: str) -> Tuple[List[dict], List[str]]:
    """Extract list of item dicts from JSON output. Returns (items, failures)."""
    if not output.strip():
        return [], ["empty output"]
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return [], ["invalid JSON"]
    if isinstance(data, dict):
        vals = [v for v in data.values() if isinstance(v, list)]
        data = vals[0] if vals else []
    if not isinstance(data, list):
        return [], ["not a list"]
    items = [i for i in data if isinstance(i, dict)]
    if not items:
        return [], ["no dict items in list"]
    return items, []


def _parse_reference(case: TestCase) -> dict:
    """Parse reference metadata from test case. Returns dict with defaults."""
    try:
        ref = json.loads(case.reference) if case.reference else {}
    except (json.JSONDecodeError, TypeError):
        ref = {}
    ref.setdefault("age_range", [6, 13])
    ref.setdefault("weather", {})
    ref.setdefault("exclude", [])
    ref.setdefault("source_item_names", [])
    ref.setdefault("expected_count", [5, 10])
    return ref


def _has_field(item: dict, field: str) -> bool:
    val = item.get(field)
    return bool(val and _str(val).strip())


def _age_overlap(ta: str, ref_min: int, ref_max: int) -> float:
    nums = sorted(set(int(n) for n in re.findall(r'\d+', ta)))
    if not nums:
        return 0.0
    item_min, item_max = nums[0], nums[-1]
    overlap = max(0, min(item_max, ref_max) - max(item_min, ref_min) + 1)
    item_span = item_max - item_min + 1
    ref_span = ref_max - ref_min + 1
    if item_span == 0:
        return 0.0
    return overlap / max(item_span, ref_span)


def _score_weekend_completeness(output: str, case: TestCase) -> Score:
    items, fails = _extract_items(output)
    if fails:
        return Score("Completeness", 0, 0.25, fails)
    REQUIRED = ["name", "location", "target_ages", "price", "weather"]
    complete = sum(1 for item in items if all(_has_field(item, f) for f in REQUIRED))
    ratio = complete / len(items) if items else 0
    score = ratio * 100
    failures = []
    if ratio < 0.8:
        failures.append(f"{complete}/{len(items)} items have all required fields")
    return Score("Completeness", score, 0.25, failures)


def _score_weekend_weather_match(output: str, case: TestCase) -> Score:
    items, fails = _extract_items(output)
    if fails:
        return Score("Weather", 0, 0.20, fails)
    ref = _parse_reference(case)
    weather_map = ref.get("weather", {})
    if not weather_map:
        return Score("Weather", 100, 0.20)
    total = 0.0
    count = 0
    for item in items:
        item_weather = _lower(item.get("weather", ""))
        item_day = _lower(item.get("day", ""))
        if not item_weather:
            continue
        has_indoor = any(k in item_weather for k in INDOOR)
        has_outdoor = any(k in item_weather for k in OUTDOOR)
        cond = weather_map.get(item_day, "") if item_day else ""
        is_rainy = any(k in _lower(cond) for k in RAIN)
        is_clear = any(k in _lower(cond) for k in CLEAR)
        if is_rainy and has_indoor:
            total += 1.0
        elif is_clear and has_outdoor:
            total += 1.0
        elif has_indoor or has_outdoor:
            total += 0.5
        else:
            total += 0.25
        count += 1
    if count == 0:
        return Score("Weather", 50, 0.20, failures=["no weather-labeled items"])
    score = (total / count) * 100
    failures = []
    if score < 60:
        failures.append("many items mismatch weather conditions")
    return Score("Weather", score, 0.20, failures)


def _score_weekend_age_match(output: str, case: TestCase) -> Score:
    items, fails = _extract_items(output)
    if fails:
        return Score("Age", 0, 0.20, fails)
    ref = _parse_reference(case)
    age_min, age_max = ref.get("age_range", [6, 13])
    scores = []
    for item in items:
        ta = _str(item.get("target_ages", ""))
        if not ta:
            scores.append(0.5)
        else:
            overlap = _age_overlap(ta, age_min, age_max)
            scores.append(overlap)
    avg = sum(scores) / len(scores) if scores else 0
    score = avg * 100
    failures = []
    if avg < 0.5:
        failures.append("most items have poor age range match")
    return Score("Age", score, 0.20, failures)


def _score_weekend_source_grounding(output: str, case: TestCase) -> Score:
    items, fails = _extract_items(output)
    if fails:
        return Score("Source", 0, 0.20, fails)
    source = _lower(case.input_text)
    source_names = _parse_reference(case).get("source_item_names", [])
    scored = 0
    for item in items:
        name = _lower(item.get("name", ""))
        if not name:
            continue
        if source_names:
            if any(sn in name for sn in source_names):
                scored += 1
        else:
            tokens = set(re.findall(r'[a-z]+', name))
            if tokens & set(re.findall(r'[a-z]+', source)):
                scored += 1
    ratio = scored / len(items) if items else 0
    score = ratio * 100
    failures = []
    if ratio < 0.6:
        failures.append(f"only {scored}/{len(items)} items grounded in source")
    return Score("Source", score, 0.20, failures)


def _score_weekend_exclusions(output: str, case: TestCase) -> Score:
    items, fails = _extract_items(output)
    if fails:
        return Score("Exclusions", 0, 0.10, fails)
    ref = _parse_reference(case)
    exclude = [_lower(e) for e in ref.get("exclude", [])]
    if not exclude:
        return Score("Exclusions", 100, 0.10)
    total = len(items)
    excluded = 0
    for item in items:
        name = _lower(item.get("name", ""))
        loc = _lower(item.get("location", ""))
        if any(e in name or name in e for e in exclude):
            excluded += 1
        elif any(e in loc or loc in e for e in exclude):
            excluded += 1
    clean = total - excluded
    ratio = clean / total if total else 1.0
    score = ratio * 100
    failures = []
    if excluded > 0:
        failures.append(f"{excluded}/{total} items match excluded places")
    return Score("Exclusions", score, 0.10, failures)


TASK_SCORERS_WEEKEND = {
    "weekend_transient": [
        _score_weekend_completeness,
        _score_weekend_weather_match,
        _score_weekend_age_match,
        _score_weekend_source_grounding,
        _score_weekend_exclusions,
    ],
    "weekend_fixed": [
        _score_weekend_completeness,
        _score_weekend_weather_match,
        _score_weekend_age_match,
        _score_weekend_source_grounding,
        _score_weekend_exclusions,
    ],
}
