import datetime
import sys
import time
import re
import requests
from ddgs import DDGS


def get_weekend_date_objects():
    today = datetime.date.today()
    friday = today + datetime.timedelta((4 - today.weekday()) % 7)
    sunday = friday + datetime.timedelta(days=2)
    return friday, sunday


def get_weekend_dates_string(friday, sunday):
    return f"{friday.strftime('%B %d')} to {sunday.strftime('%B %d, %Y')}"


def fetch_weather(friday, sunday):
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 43.8361,
            "longitude": -79.5083,
            "daily": "temperature_2m_max,precipitation_sum",
            "timezone": "America/Toronto",
            "start_date": friday.strftime("%Y-%m-%d"),
            "end_date": sunday.strftime("%Y-%m-%d"),
        }
        resp = requests.get(url, params=params, timeout=10).json()
        daily = resp.get("daily", {})

        dates = daily.get("time", [])
        precip_array = daily.get("precipitation_sum", [])
        temp_array = daily.get("temperature_2m_max", [])

        forecasts = []
        for i in range(len(dates)):
            date_str = dates[i]
            precip = precip_array[i] if i < len(precip_array) else 0
            temp = temp_array[i] if i < len(temp_array) else 0
            condition = "Precipitation" if precip > 0.5 else "Clear"
            day_name = datetime.datetime.strptime(
                date_str, "%Y-%m-%d").strftime("%A")
            forecasts.append(
                f"{day_name}: {temp:.1f}°C, {condition} ({precip}mm)")

        return "Daily Forecast:\n" + "\n".join(forecasts)
    except Exception as e:
        print(f"[ERROR] Weather fetch failed: {e}", file=sys.stderr)
        return "Forecast: Precipitation expected (fallback due to error)."


def fetch_transient_events(dates_str, year, month_name):
    def safe_search(q, retries=3):
        for attempt in range(retries):
            try:
                results = list(DDGS().text(q, max_results=8))
                return results
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(2 ** attempt)
                else:
                    break
        return []

    try:
        queries = [
            "Ontario Science Centre family workshops April 2026",
            "Toronto Zoo special events April 2026",
            "LEGOLAND Discovery Centre Toronto April May 2026",
            "Royal Ontario Museum ROM family programs April 2026",
            "Vaughan community centres kids April 2026",
        ]

        all_results = []
        for q in queries:
            results = safe_search(q)
            all_results.extend(results)

        seen = set()
        unique_results = []
        for r in all_results:
            title = r.get("title", "")
            if title and title not in seen:
                seen.add(title)
                unique_results.append(r)

        text_output = "\n".join(
            [
                f"- {r.get('title', 'Event')}: {r.get('body', '')}"
                for r in unique_results
            ]
        )
        return text_output
    except Exception as e:
        print(f"[ERROR] Transient event fetch failed: {e}", file=sys.stderr)
        return "Error fetching transient events."


def fetch_fixed_venues(year, month_name):
    try:
        queries = [
            "indoor play centre Toronto Vaughan 2026 prices",
            "trampoline park Toronto kids 2026",
            "children museum Toronto 2026",
            "family arcade Vaughan 2026",
            "playplace Vaughan indoor kids 2026 prices",
        ]

        all_results = []
        for q in queries:
            try:
                results = list(DDGS().text(q, max_results=8))
                all_results.extend(results)
            except Exception as e:
                print(f"[WARN] Query failed: {q[:30]}... - {e}")

        seen = set()
        unique_results = []
        for r in all_results:
            title = r.get("title", "")
            if title and title not in seen:
                seen.add(title)
                unique_results.append(r)

        text_output = "\n".join(
            [
                f"- {r.get('title', 'Venue/Exhibit')}: {r.get('body', '')}"
                for r in unique_results
            ]
        )
        return text_output
    except Exception as e:
        print(f"[ERROR] Fixed venue fetch failed: {e}", file=sys.stderr)
        return "Error fetching fixed venues."


def scrape_review_score(place_name):
    for attempt in range(3):
        try:
            time.sleep(0.5)
            query = f'"{place_name}" rating review 5 stars'
            results = list(DDGS().text(query, max_results=5))
            combined = " ".join([r.get("title", "") + " " + r.get("body", "") for r in results])

            match = re.search(r"([0-4]\.\d)\s*/?\s*5", combined, re.IGNORECASE)
            if match:
                return float(match.group(1))
            match2 = re.search(r"rating[:\s]*([0-4]\.\d)", combined, re.IGNORECASE)
            if match2:
                return float(match2.group(1))
            break
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(2 ** attempt)
            else:
                break
    return 0.0
