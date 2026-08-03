import datetime
import os
import re
import sys
import time

import requests
from ddgs import DDGS

from lib.tui import STEP, WARN
from weekend.config import CITY, LATITUDE, LONGITUDE, REGION, TIMEZONE

# HTTP request defaults
WEATHER_API_TIMEOUT = int(os.environ.get("WEEKEND_WEATHER_TIMEOUT", "10"))

# Web search retries and limits
FETCH_RETRIES = int(os.environ.get("WEEKEND_FETCH_RETRIES", "3"))
DDGS_MAX_RESULTS = int(os.environ.get("WEEKEND_DDGS_MAX_RESULTS", "8"))
REVIEW_MAX_RESULTS = int(os.environ.get("WEEKEND_REVIEW_MAX_RESULTS", "5"))
MAX_BODY_LENGTH = int(os.environ.get("WEEKEND_MAX_BODY_LENGTH", "300"))

# Rate limiting
RATE_LIMIT_SLEEP = float(os.environ.get("WEEKEND_RATE_LIMIT_SLEEP", "0.5"))

# Date, weather, and scraper constants (Mitchell Hashimoto & John Carmack design)
FRIDAY_WEEKDAY_INDEX = 4
DAYS_IN_WEEK = 7
SUNDAY_DELTA_DAYS = 2

WEATHER_API_URL = os.environ.get(
    "WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast"
)
DAILY_METEO_VARS = os.environ.get("WEEKEND_METEO_VARS", "temperature_2m_max,precipitation_sum")
PRECIPITATION_THRESHOLD = float(os.environ.get("WEEKEND_PRECIP_THRESHOLD", "0.5"))
FORECAST_HEADER = os.environ.get("WEEKEND_FORECAST_HEADER", "Daily Forecast:\n")

SCRAPE_ATTEMPTS = int(os.environ.get("WEEKEND_SCRAPE_ATTEMPTS", "3"))
DEFAULT_REVIEW_SCORE = float(os.environ.get("WEEKEND_DEFAULT_REVIEW_SCORE", "0.0"))

# Pre-compiled regular expressions (John Carmack optimization)
RATING_OUT_OF_FIVE_RE = re.compile(r"([0-4]\.\d)\s*/?\s*5", re.IGNORECASE)
RATING_LABEL_RE = re.compile(r"rating[:\s]*([0-4]\.\d)", re.IGNORECASE)


def _clean_search_results(results, default_label="Item", max_body=MAX_BODY_LENGTH):
    """Dedupe scrape results and drop the ones with no in-region evidence.

    Class C16, scoped at SOURCE. Probed on the real scraper: a query with no
    good local matches ("Vaughan community centres kids <month> <year>") is
    answered with whatever matched the YEAR -- a mass-spectrometry conference,
    an Astana film festival, a Bulgarian concert. There was no relevance floor,
    and every one of those reached the model as a candidate activity.

    Filtering here is safe because these are RAW results, not curated rows:
    dropping noise costs nothing, and `has_region_evidence` errs towards keeping
    anything ambiguous so the model still gets the final say (see
    `weekend/prompts.py`). A curated row that lacks region evidence is a
    judgement call and is never dropped by code.
    """
    from weekend.region import has_region_evidence

    seen = set()
    cleaned = []
    dropped = 0
    for r in results:
        title = r.get("title", "").strip()
        norm = title.lower().rstrip(".,!?:; ") if title else ""
        if not norm or norm in seen:
            continue
        seen.add(norm)
        body = (r.get("body") or "")[:max_body].strip()
        if not has_region_evidence(f"{title} {body}"):
            dropped += 1
            continue
        cleaned.append(f"- {title or default_label}: {body}")
    if dropped:
        print(f"{STEP} Dropped {dropped} out-of-region search result(s)")
    return "\n".join(cleaned)


def get_weekend_date_objects():
    today = datetime.date.today()
    friday = today + datetime.timedelta((FRIDAY_WEEKDAY_INDEX - today.weekday()) % DAYS_IN_WEEK)
    sunday = friday + datetime.timedelta(days=SUNDAY_DELTA_DAYS)
    return friday, sunday


def get_weekend_dates_string(friday, sunday):
    return f"{friday.strftime('%B %d')} to {sunday.strftime('%B %d, %Y')}"


def fetch_weather(friday, sunday):
    try:
        url = WEATHER_API_URL
        params = {
            "latitude": LATITUDE,
            "longitude": LONGITUDE,
            "daily": DAILY_METEO_VARS,
            "timezone": TIMEZONE,
            "start_date": friday.strftime("%Y-%m-%d"),
            "end_date": sunday.strftime("%Y-%m-%d"),
        }
        with requests.Session() as s:
            resp = s.get(url, params=params, timeout=WEATHER_API_TIMEOUT).json()
        daily = resp.get("daily", {})

        dates = daily.get("time", [])
        precip_array = daily.get("precipitation_sum", [])
        temp_array = daily.get("temperature_2m_max", [])

        forecasts = []
        for i in range(len(dates)):
            date_str = dates[i]
            precip = precip_array[i] if i < len(precip_array) else 0
            temp = temp_array[i] if i < len(temp_array) else 0
            condition = "Precipitation" if precip > PRECIPITATION_THRESHOLD else "Clear"
            day_name = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
            forecasts.append(f"{day_name}: {temp:.1f}°C, {condition} ({precip}mm)")

        return FORECAST_HEADER + "\n".join(forecasts)
    except Exception as e:
        print(f"{WARN} Weather fetch failed: {e}", file=sys.stderr)
        return "Forecast: Precipitation expected (fallback due to error)."


def fetch_transient_events(dates_str, year, month_name):
    def safe_search(q, retries=FETCH_RETRIES):
        for attempt in range(retries):
            try:
                results = list(DDGS().text(q, max_results=DDGS_MAX_RESULTS))
                return results
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(2**attempt)
                else:
                    break
        return []

    try:
        # NOTE: no province anchor here on purpose. Probed against the real
        # scraper: adding "Ontario" to EVENT queries degraded them (it surfaced
        # ontario.ca forest-fire bulletins and tennis fixtures), while the same
        # anchor clearly improved the VENUE queries below. Evidence, not symmetry.
        _transient_template_str = os.environ.get(
            "WEEKEND_TRANSIENT_QUERIES",
            "{REGION} family events {month_name} {year}||"
            "{REGION} Zoo special events {month_name} {year}||"
            "kids activities {CITY} {month_name} {year}||"
            "{REGION} museum family programs {month_name} {year}||"
            "{CITY} community centre kids programs {month_name}",
        )
        queries = [
            q.format(REGION=REGION, CITY=CITY, year=year, month_name=month_name)
            for q in _transient_template_str.split("||")
        ]

        all_results = []
        for q in queries:
            results = safe_search(q)
            all_results.extend(results)

        corpus = _clean_search_results(all_results, "Event", max_body=MAX_BODY_LENGTH)

        # Class C14 follow-up: web search for "events" returns almost nothing but
        # directory pages. The model correctly refuses to list one as an activity,
        # which left the plan nearly empty -- so read the events OUT of them
        # instead of discarding the whole harvest.
        from weekend.followup import follow_aggregators

        followed = follow_aggregators(all_results)
        return f"{corpus}\n{followed}" if followed else corpus
    except Exception as e:
        print(f"{WARN} Transient event fetch failed: {e}", file=sys.stderr)
        return "Error fetching transient events."


def fetch_fixed_venues(year, month_name):
    try:
        _fixed_template_str = os.environ.get(
            "WEEKEND_FIXED_VENUE_QUERIES",
            "indoor play centre {CITY} {REGION} Ontario prices||"
            "trampoline park {REGION} Ontario kids||"
            "children museum {REGION} Ontario||"
            "family arcade {CITY} Ontario||"
            "playplace {CITY} Ontario indoor kids prices",
        )
        queries = [
            q.format(CITY=CITY, REGION=REGION, year=year)
            for q in _fixed_template_str.split("||")
        ]

        all_results = []
        for q in queries:
            try:
                results = list(DDGS().text(q, max_results=DDGS_MAX_RESULTS))
                all_results.extend(results)
            except Exception as e:
                print(f"{WARN} Query failed: {q[:30]}... - {e}")

        return _clean_search_results(all_results, "Venue/Exhibit", max_body=MAX_BODY_LENGTH)
    except Exception as e:
        print(f"{WARN} Fixed venue fetch failed: {e}", file=sys.stderr)
        return "Error fetching fixed venues."


def scrape_review_score(place_name):
    for attempt in range(SCRAPE_ATTEMPTS):
        try:
            time.sleep(RATE_LIMIT_SLEEP)
            _review_template = os.environ.get(
                "WEEKEND_REVIEW_QUERY", '"{place_name}" rating review 5 stars'
            )
            query = _review_template.format(place_name=place_name)
            results = list(DDGS().text(query, max_results=REVIEW_MAX_RESULTS))
            combined = " ".join([r.get("title", "") + " " + r.get("body", "") for r in results])

            match = RATING_OUT_OF_FIVE_RE.search(combined)
            if match:
                return float(match.group(1))
            match2 = RATING_LABEL_RE.search(combined)
            if match2:
                return float(match2.group(1))
            break
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2**attempt)
            else:
                break
    return DEFAULT_REVIEW_SCORE
