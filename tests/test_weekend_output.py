"""Tests for weekend_output module."""
import pytest
from unittest.mock import patch, MagicMock


class TestPrintHelpers:
    def test_print_header(self, capsys):
        from weekend_output import print_header
        from rich.console import Console
        from io import StringIO
        fake_console = Console(file=StringIO(), force_terminal=True, width=200, force_interactive=True)
        with patch("weekend_output.console", fake_console):
            print_header("Label", "Value")
        out = fake_console.file.getvalue()
        assert "Label" in out
        assert "Value" in out

    def test_print_step(self):
        from weekend_output import print_step
        from rich.console import Console
        from io import StringIO
        fake_console = Console(file=StringIO(), force_terminal=True, width=200, force_interactive=True)
        with patch("weekend_output.console", fake_console):
            print_step("Hello step")
        out = fake_console.file.getvalue()
        assert "Hello step" in out

    def test_print_info(self):
        from weekend_output import print_info
        from rich.console import Console
        from io import StringIO
        fake_console = Console(file=StringIO(), force_terminal=True, width=200, force_interactive=True)
        with patch("weekend_output.console", fake_console):
            print_info("Label", "Value")
        out = fake_console.file.getvalue()
        assert "Label" in out
        assert "Value" in out

    def test_print_warning(self):
        from weekend_output import print_warning
        from rich.console import Console
        from io import StringIO
        fake_console = Console(file=StringIO(), force_terminal=True, width=200, force_interactive=True)
        with patch("weekend_output.console", fake_console):
            print_warning("Be careful")
        out = fake_console.file.getvalue()
        assert "Be careful" in out

    def test_print_summary(self):
        from weekend_output import print_summary
        from rich.console import Console
        from io import StringIO
        fake_console = Console(file=StringIO(), force_terminal=True, width=200, force_interactive=True)
        with patch("weekend_output.console", fake_console):
            print_summary("✓", 5, 3, "/tmp/out.md", 120.0)
        out = fake_console.file.getvalue()
        assert "Weekend plan" in out
        # Numbers get ANSI codes from rich
        import re
        # Strip ANSI codes
        clean = re.sub(r'\x1b\[[0-9;]*m', '', out)
        assert "5 fixed" in clean
        assert "3 transient" in clean
        assert "minutes" in clean


class TestFmtScore:
    def test_positive_score(self):
        from weekend_output import _fmt_score
        assert _fmt_score({"score": 4.5}) == "⭐ 4.5/5"

    def test_zero_score(self):
        from weekend_output import _fmt_score
        assert _fmt_score({"score": 0}) == ""

    def test_no_score_key(self):
        from weekend_output import _fmt_score
        assert _fmt_score({}) == ""


class TestFmtMissing:
    def test_present(self):
        from weekend_output import _fmt_missing
        assert _fmt_missing("value") == "value"

    def test_empty(self):
        from weekend_output import _fmt_missing
        assert _fmt_missing("") == "—"

    def test_none(self):
        from weekend_output import _fmt_missing
        assert _fmt_missing(None) == "—"


class TestBuildFixedTable:
    def test_empty(self):
        from weekend_output import _build_fixed_table
        assert _build_fixed_table([]) == ""

    def test_with_scores(self):
        from weekend_output import _build_fixed_table
        items = [{"name": "Activity A", "location": "Toronto", "target_ages": "5-12",
                  "price": "Free", "weather": "outdoor", "score": 4.5}]
        result = _build_fixed_table(items)
        assert "Fixed / Year-Round" in result
        assert "Ranked by Review Score" in result
        assert "Activity A" in result
        assert "4.5" in result

    def test_without_scores(self):
        from weekend_output import _build_fixed_table
        items = [{"name": "Activity A", "location": "Toronto"}]
        result = _build_fixed_table(items)
        assert "Activity A" in result
        assert "Ranked" not in result

    def test_alternative_keys(self):
        from weekend_output import _build_fixed_table
        items = [{"title": "My Title", "address": "123 Main", "age_group": "3-7",
                  "cost": "$5", "weather_appropriateness": "indoor", "activity_name": "Alt"}]
        result = _build_fixed_table(items)
        # title has priority over activity_name when name is missing
        assert "My Title" in result
        assert "(123 Main)" in result
        assert "indoor" in result

    def test_unknown_name(self):
        from weekend_output import _build_fixed_table
        items = [{"name": "", "location": "x"}]
        result = _build_fixed_table(items)
        assert "Unknown" in result

    def test_strips_bold(self):
        from weekend_output import _build_fixed_table
        items = [{"name": "**Bold Name**"}]
        result = _build_fixed_table(items)
        # Bold should be stripped from the name
        assert "Bold Name" in result


class TestBuildTransientTable:
    def test_empty(self):
        from weekend_output import _build_transient_table
        assert _build_transient_table([]) == ""

    def test_with_scores(self):
        from weekend_output import _build_transient_table
        items = [{"name": "Event A", "location": "Toronto", "score": 4.2,
                  "target_ages": "All", "price": "Free", "duration": "2hrs",
                  "day": "Saturday", "weather": "outdoor"}]
        result = _build_transient_table(items)
        assert "Transient" in result
        assert "Event A" in result
        assert "4.2" in result

    def test_without_scores(self):
        from weekend_output import _build_transient_table
        items = [{"name": "Event A"}]
        result = _build_transient_table(items)
        assert "Event A" in result

    def test_alternative_keys(self):
        from weekend_output import _build_transient_table
        items = [{"event": "Festival", "address": "Vaughan", "age_group": "8-14",
                  "cost": "$10", "end_date": "April 15", "dates": "Sat-Sun",
                  "weather_appropriateness": "indoor"}]
        result = _build_transient_table(items)
        assert "Festival" in result

    def test_missing_day(self):
        from weekend_output import _build_transient_table
        items = [{"name": "Event", "day": ""}]
        result = _build_transient_table(items)
        # Should show em-dash for missing day
        assert "—" in result

    def test_unknown_name(self):
        from weekend_output import _build_transient_table
        items = [{"name": "", "title": ""}]
        result = _build_transient_table(items)
        assert "Unknown" in result


class TestBuildMarkdownTables:
    def test_full_build(self):
        from weekend_output import build_markdown_tables
        # Mock fetch_scores to add score keys
        def add_score(items, **kwargs):
            for item in items:
                item.setdefault("score", 4.0)
        with patch("weekend_output.fetch_scores_for_items", side_effect=add_score):
            result = build_markdown_tables(
                dates_str="April 10-12, 2026",
                weather_str="Sunny",
                structured_data={"transient_events": [
                    {"name": "Event A", "location": "Toronto"},
                ]},
                fixed_activities=[
                    {"name": "Fixed A", "location": "Toronto"},
                ],
            )
        assert "Weekend Plan" in result
        assert "April 10-12, 2026" in result
        assert "Sunny" in result
        assert "Fixed A" in result
        assert "Event A" in result

    def test_structured_data_list(self):
        from weekend_output import build_markdown_tables
        with patch("weekend_output.fetch_scores_for_items", return_value=None):
            result = build_markdown_tables(
                dates_str="April",
                weather_str="",
                structured_data=[{"name": "E1"}],
                fixed_activities=[],
            )
        assert "E1" in result

    def test_structured_data_no_transient(self):
        from weekend_output import build_markdown_tables
        with patch("weekend_output.fetch_scores_for_items", return_value=None):
            result = build_markdown_tables(
                dates_str="April",
                weather_str="",
                structured_data={},
                fixed_activities=[],
            )
        assert "Weekend Plan" in result

    def test_grouping_by_name(self):
        from weekend_output import build_markdown_tables
        with patch("weekend_output.fetch_scores_for_items", return_value=None):
            # Same name, different days
            result = build_markdown_tables(
                dates_str="April",
                weather_str="",
                structured_data={
                    "transient_events": [
                        {"name": "Same", "day": "Sat"},
                        {"name": "Same", "day": "Sun"},
                    ]
                },
                fixed_activities=[],
            )
        # Should be grouped, days combined
        assert result.count("Same") == 1  # grouped

    def test_alternative_structured_keys(self):
        from weekend_output import build_markdown_tables
        with patch("weekend_output.fetch_scores_for_items", return_value=None):
            result = build_markdown_tables(
                dates_str="April",
                weather_str="",
                structured_data={"events": [{"name": "E1"}]},
                fixed_activities=[],
            )
        assert "E1" in result

    def test_activities_structured_key(self):
        from weekend_output import build_markdown_tables
        with patch("weekend_output.fetch_scores_for_items", return_value=None):
            result = build_markdown_tables(
                dates_str="April",
                weather_str="",
                structured_data={"activities": [{"name": "A1"}]},
                fixed_activities=[],
            )
        assert "A1" in result


class TestPrintToCli:
    def test_prints_markdown(self):
        from weekend_output import print_to_cli
        from rich.console import Console
        from io import StringIO
        fake_console = Console(file=StringIO(), force_terminal=True, width=200, force_interactive=True)
        with patch("weekend_output.console", fake_console):
            print_to_cli("# Hello\n\nWorld")
        out = fake_console.file.getvalue()
        assert "Hello" in out
        assert "World" in out
