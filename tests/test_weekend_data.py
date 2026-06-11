"""Tests for weekend.data module."""
import datetime
from unittest.mock import patch, MagicMock
import pytest


class TestGetWeekendDateObjects:
    def test_returns_friday_and_sunday(self):
        from weekend.data import get_weekend_date_objects
        friday, sunday = get_weekend_date_objects()
        assert friday.weekday() == 4  # Friday
        assert sunday.weekday() == 6  # Sunday
        assert (sunday - friday).days == 2


class TestGetWeekendDatesString:
    def test_format(self):
        from weekend.data import get_weekend_dates_string
        friday = datetime.date(2026, 4, 10)
        sunday = datetime.date(2026, 4, 12)
        result = get_weekend_dates_string(friday, sunday)
        assert "April 10" in result
        assert "April 12, 2026" in result


class TestFetchWeather:
    def test_successful_forecast(self):
        from weekend.data import fetch_weather
        friday = datetime.date(2026, 4, 10)
        sunday = datetime.date(2026, 4, 12)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "daily": {
                "time": ["2026-04-10", "2026-04-11", "2026-04-12"],
                "temperature_2m_max": [15.5, 18.2, 12.0],
                "precipitation_sum": [0.0, 2.5, 0.3],
            }
        }
        with patch("weekend.data.requests.get", return_value=mock_response) as mock_get:
            result = fetch_weather(friday, sunday)
        assert "Daily Forecast" in result
        assert "Friday" in result
        assert "Saturday" in result
        assert "Sunday" in result
        assert "Clear" in result
        assert "Precipitation" in result

    def test_fallback_on_exception(self, capsys):
        from weekend.data import fetch_weather
        friday = datetime.date(2026, 4, 10)
        sunday = datetime.date(2026, 4, 12)
        with patch("weekend.data.requests.get", side_effect=Exception("network")):
            result = fetch_weather(friday, sunday)
        assert "fallback" in result.lower() or "Precipitation" in result
        out = capsys.readouterr()
        assert "Weather fetch failed" in out.err

    def test_empty_daily(self):
        from weekend.data import fetch_weather
        friday = datetime.date(2026, 4, 10)
        sunday = datetime.date(2026, 4, 12)
        mock_response = MagicMock()
        mock_response.json.return_value = {"daily": {}}
        with patch("weekend.data.requests.get", return_value=mock_response):
            result = fetch_weather(friday, sunday)
        assert "Daily Forecast" in result

    def test_short_precipitation_array(self):
        from weekend.data import fetch_weather
        friday = datetime.date(2026, 4, 10)
        sunday = datetime.date(2026, 4, 12)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "daily": {
                "time": ["2026-04-10", "2026-04-11", "2026-04-12"],
                "temperature_2m_max": [15.0],  # shorter
                "precipitation_sum": [],  # empty
            }
        }
        with patch("weekend.data.requests.get", return_value=mock_response):
            result = fetch_weather(friday, sunday)
        assert "Daily Forecast" in result


class TestFetchTransientEvents:
    def test_successful(self):
        from weekend.data import fetch_transient_events
        mock_results = [
            {"title": "Festival A", "body": "Description A"},
            {"title": "Festival B", "body": "Description B"},
        ]
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text.return_value = mock_results
        with patch("weekend.data.DDGS", fake_ddgs):
            result = fetch_transient_events("April 10-12", 2026, "April")
        assert "Festival A" in result
        assert "Festival B" in result
        assert "Description A" in result

    def test_dedup(self):
        from weekend.data import fetch_transient_events
        mock_results = [
            {"title": "Same", "body": "Desc 1"},
            {"title": "Same", "body": "Desc 2"},  # duplicate title
        ]
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text.return_value = mock_results
        with patch("weekend.data.DDGS", fake_ddgs):
            result = fetch_transient_events("April 10-12", 2026, "April")
        # Should only contain one occurrence of "Same"
        assert result.count("Same") == 1

    def test_outer_exception(self, capsys):
        from weekend.data import fetch_transient_events
        # Make r.get fail by passing a non-dict
        fake_ddgs = MagicMock()
        class Failing:
            def get(self, key, default=None):
                raise Exception("simulated get failure")
        fake_ddgs.return_value.text.return_value = [Failing()]
        with patch("weekend.data.DDGS", fake_ddgs):
            result = fetch_transient_events("April 10-12", 2026, "April")
        assert "Error" in result
        out = capsys.readouterr()
        assert "Transient event fetch failed" in out.err

    def test_safe_search_429_retry(self):
        from weekend.data import fetch_transient_events
        call_count = [0]
        def mock_text(q, max_results=8):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("429 rate limit")
            return [{"title": "After retry", "body": "x"}]
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text = mock_text
        with patch("weekend.data.DDGS", fake_ddgs), \
             patch("weekend.data.time.sleep"):
            result = fetch_transient_events("April 10-12", 2026, "April")
        # Retried and got results
        assert "After retry" in result

    def test_safe_search_429_all_fail(self):
        from weekend.data import fetch_transient_events
        def mock_text(q, max_results=8):
            raise Exception("429 rate limit")
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text = mock_text
        with patch("weekend.data.DDGS", fake_ddgs), \
             patch("weekend.data.time.sleep"):
            result = fetch_transient_events("April 10-12", 2026, "April")
        # All retries failed, returns empty
        assert result == ""

    def test_safe_search_non_429(self):
        from weekend.data import fetch_transient_events
        def mock_text(q, max_results=8):
            raise Exception("Connection refused")  # not 429, breaks out
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text = mock_text
        with patch("weekend.data.DDGS", fake_ddgs):
            result = fetch_transient_events("April 10-12", 2026, "April")
        # Breaks out, no retry
        assert result == ""


class TestFetchFixedVenues:
    def test_successful(self):
        from weekend.data import fetch_fixed_venues
        mock_results = [
            {"title": "Venue A", "body": "Desc A"},
        ]
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text.return_value = mock_results
        with patch("weekend.data.DDGS", fake_ddgs):
            result = fetch_fixed_venues(2026, "April")
        assert "Venue A" in result

    def test_dedup(self):
        from weekend.data import fetch_fixed_venues
        mock_results = [
            {"title": "Same", "body": "1"},
            {"title": "Same", "body": "2"},
        ]
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text.return_value = mock_results
        with patch("weekend.data.DDGS", fake_ddgs):
            result = fetch_fixed_venues(2026, "April")
        assert result.count("Same") == 1

    def test_query_failure_continues(self, capsys):
        from weekend.data import fetch_fixed_venues
        call_count = [0]
        def mock_text(q, max_results=8):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("query fail")
            return [{"title": f"Q{call_count[0]}", "body": "x"}]
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text = mock_text
        with patch("weekend.data.DDGS", fake_ddgs):
            result = fetch_fixed_venues(2026, "April")
        # Some queries succeeded despite one failing
        assert "Q1" in result
        assert "Q3" in result
        out = capsys.readouterr()
        assert "Query failed" in out.out

    def test_outer_exception(self, capsys):
        from weekend.data import fetch_fixed_venues
        # Make the body of the outer try fail
        fake_ddgs = MagicMock()
        class Failing:
            def get(self, key, default=None):
                raise Exception("simulated get failure")
        fake_ddgs.return_value.text.return_value = [Failing()]
        with patch("weekend.data.DDGS", fake_ddgs):
            result = fetch_fixed_venues(2026, "April")
        assert "Error" in result
        out = capsys.readouterr()
        assert "Fixed venue fetch failed" in out.err


class TestScrapeReviewScore:
    def test_match_slash_format(self):
        from weekend.data import scrape_review_score
        mock_results = [
            {"title": "Review", "body": "Great place 4.5 / 5 stars"},
        ]
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text.return_value = mock_results
        with patch("weekend.data.DDGS", fake_ddgs), \
             patch("weekend.data.time.sleep"):
            result = scrape_review_score("Place")
        assert result == 4.5

    def test_match_rating_format(self):
        from weekend.data import scrape_review_score
        mock_results = [
            {"title": "Review", "body": "rating: 4.2 out of 5"},
        ]
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text.return_value = mock_results
        with patch("weekend.data.DDGS", fake_ddgs), \
             patch("weekend.data.time.sleep"):
            result = scrape_review_score("Place")
        assert result == 4.2

    def test_no_match_returns_zero(self):
        from weekend.data import scrape_review_score
        mock_results = [
            {"title": "No score", "body": "no rating here"},
        ]
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text.return_value = mock_results
        with patch("weekend.data.DDGS", fake_ddgs), \
             patch("weekend.data.time.sleep"):
            result = scrape_review_score("Place")
        assert result == 0.0

    def test_429_retry(self):
        from weekend.data import scrape_review_score
        call_count = [0]
        def mock_text(q, max_results=5):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("429 rate limit")
            return [{"title": "Review", "body": "4.8 / 5"}]
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text = mock_text
        with patch("weekend.data.DDGS", fake_ddgs), \
             patch("weekend.data.time.sleep"):
            result = scrape_review_score("Place")
        assert result == 4.8

    def test_non_429_breaks(self):
        from weekend.data import scrape_review_score
        def mock_text(q, max_results=5):
            raise Exception("connection error")
        fake_ddgs = MagicMock()
        fake_ddgs.return_value.text = mock_text
        with patch("weekend.data.DDGS", fake_ddgs), \
             patch("weekend.data.time.sleep"):
            result = scrape_review_score("Place")
        assert result == 0.0
