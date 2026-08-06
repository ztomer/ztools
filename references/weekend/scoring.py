"""Item scoring for weekend plan rows -- pure logic, no LLM and no I/O.

Extracted from weekend/llm.py to keep that module under the repo's 500-line cap.
weekend/llm.py re-exports both names, so existing imports keep working (the same
shim pattern as lib/config.py and lib/osaurus_lib.py).

NOTE (class C6, docs/REPORT_WEAKNESS_CLASSES.md): the number `_score_item`
returns is an internal heuristic over field completeness, age overlap and
weather fit. It is NOT a review score, even though weekend/output.py currently
heads its tables "Ranked by Review Score".
"""

import re

POPULATED_FIELDS_BASE_SCORE = 3.0
OVERLAP_SCORE_HIGH = 3.0
OVERLAP_SCORE_LOW = 1.5
WEATHER_MATCH_BONUS = 2.0
WEATHER_PARTIAL_BONUS = 1.0
PRICE_MATCH_BONUS = 0.5
LOCATION_MATCH_BONUS = 0.5
SCORE_DIVISOR = 2.0
SCORE_MAX_LIMIT = 5.0


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
