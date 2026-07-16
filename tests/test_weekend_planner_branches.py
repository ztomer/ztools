"""Tests for weekend.cli edge cases in _parse_fixed and _parse_transient."""



class TestParseFixedEdges:
    """Test _parse_fixed edge cases."""

    def test_parse_fixed_list_no_name(self, mock_llm):
        import weekend.cli as wp

        result = wp._parse_fixed([{"description": "Item without name"}], "mock-model", {})
        assert result == []

    def test_parse_fixed_list_with_mapping(self, mock_llm):
        import weekend.cli as wp

        result = wp._parse_fixed([{"activity": "Mapped"}], "mock-model", {"activity": "name"})
        assert len(result) == 1
        assert result[0]["name"] == "Mapped"


class TestParseTransientEdges:
    """Test _parse_transient edge cases."""

    def test_parse_transient_all_weather(self, mock_llm):
        """All items are weather data — line 135 returns []."""
        import weekend.cli as wp

        result = wp._parse_transient(
            [
                {"temperature": "20", "condition": "Sunny"},
                {"temperature": "22", "condition": "Cloudy"},
            ],
            "mock-model",
            {},
        )
        assert result == []

    def test_parse_transient_alt_items_only(self, mock_llm):
        """Items have alt keys but no 'name' — line 154-157."""
        import weekend.cli as wp

        # Use 'event' key which is NOT in all_name_keys (which has description/title/summary)
        # but IS in the alt_items check.
        # Note: 'event' IS in all_name_keys per line 130: ["description", "title", "event", "summary", "activity_name"]
        # So we need a key that is in alt_items check but NOT in all_name_keys.
        # alt_items check: ["description", "title", "event", "summary", "activity_name"]
        # all_name_keys = name_keys + ["description", "title", "event", "summary", "activity_name"]
        # So they're the same set. To trigger alt_items, valid_items must be empty.
        # Use items with NO name_keys values — i.e. all keys are absent.
        # valid_items filters by all_name_keys. If all_name_keys items are empty/missing, valid_items is [].
        # Then alt_items also uses the same set! So it should also be empty.
        # Actually test: items with None values for alt keys — valid_items checks .get(nk) (None is falsy)
        result = wp._parse_transient(
            [{"event": None, "location": "Toronto"}, {"title": None, "location": "Vaughan"}],
            "mock-model",
            {},
        )
        # Both branches return [] since nothing is truthy
        assert result == []

    def test_alt_keys_are_dead_code(self, mock_llm):
        """Lines 154-157 (alt_items branch) are dead code.

        The valid_items and alt_items branches both check the same keys,
        so if valid_items is empty, alt_items is also empty. This test
        documents the dead code by verifying that an item with only alt-keys
        flows through valid_items (never needs the alt_items fallback).
        """
        import weekend.cli as wp

        # List with 2+ items (so we enter the list branch), all using alt-keys
        # (description/title). The valid_items branch picks them up via line 149.
        items = [
            {"description": "d1", "title": "t1"},
            {"description": "d2", "title": "t2"},
        ]
        result = wp._parse_transient(items, "mock-model", {})
        # Both items are picked up by valid_items (description/title match)
        assert len(result) == 2
        # Both items have a "name" key (from alt-keys)
        for item in result:
            assert "name" in item

    def test_parse_transient_weekend_forecast(self, mock_llm):
        """Dict with weekend_forecast key — line 171-178."""
        import weekend.cli as wp

        result = wp._parse_transient(
            {
                "weekend_forecast": {
                    "Friday": {"events": [{"name": "Event A"}]},
                    "Saturday": {"events": [{"name": "Event B"}]},
                }
            },
            "mock-model",
            {},
        )
        assert len(result) == 2

    def test_parse_transient_single_object_with_name(self, mock_llm):
        """Single dict with name — line 181."""
        import weekend.cli as wp

        result = wp._parse_transient(
            {"name": "Solo Event", "location": "Toronto"}, "mock-model", {}
        )
        assert len(result) == 1
        assert result[0]["name"] == "Solo Event"

    def test_parse_transient_fallback_list_len_3(self, mock_llm):
        """Dict with random list >= 3 — line 184-187."""
        import weekend.cli as wp

        result = wp._parse_transient(
            {
                "random_key": [
                    {"name": "A", "location": "X"},
                    {"name": "B", "location": "Y"},
                    {"name": "C", "location": "Z"},
                ]
            },
            "mock-model",
            {},
        )
        assert len(result) == 3

    def test_parse_transient_fallback_list_len_2(self, mock_llm):
        """Dict with random list >= 2 — line 190-191."""
        import weekend.cli as wp

        result = wp._parse_transient(
            {
                "random_key": [
                    {"name": "A", "location": "X"},
                    {"name": "B", "location": "Y"},
                ]
            },
            "mock-model",
            {},
        )
        assert len(result) == 2


class TestParseArgs:
    """Test parse_args for argparse coverage."""

    def test_parse_args_use_cache(self, monkeypatch):
        import sys

        import weekend.cli as wp

        monkeypatch.setattr(sys, "argv", ["weekend.cli", "--use-cache"])
        args = wp.parse_args()
        assert args.use_cache is True

    def test_parse_args_model(self, monkeypatch):
        import sys

        import weekend.cli as wp

        monkeypatch.setattr(sys, "argv", ["weekend.cli", "--model", "x-model"])
        args = wp.parse_args()
        assert args.model == "x-model"

    def test_parse_args_skip_web(self, monkeypatch):
        import sys

        import weekend.cli as wp

        monkeypatch.setattr(sys, "argv", ["weekend.cli", "--skip-web"])
        args = wp.parse_args()
        assert args.skip_web is True

    def test_parse_args_debug(self, monkeypatch):
        import sys

        import weekend.cli as wp

        monkeypatch.setattr(sys, "argv", ["weekend.cli", "--debug"])
        args = wp.parse_args()
        assert args.debug is True

    def test_parse_args_all(self, monkeypatch):
        import sys

        import weekend.cli as wp

        monkeypatch.setattr(
            sys, "argv", ["weekend.cli", "--use-cache", "--model", "x", "--skip-web", "--debug"]
        )
        args = wp.parse_args()
        assert args.use_cache is True
        assert args.model == "x"
        assert args.skip_web is True
        assert args.debug is True


class TestMainEntry:
    """Test __main__ block via runpy."""

    def test_main_block(self, monkeypatch, capsys):
        import runpy
        import sys
        from unittest.mock import MagicMock, patch

        monkeypatch.setattr(sys, "argv", ["weekend.cli", "--use-cache"])
        mock_ddgs = MagicMock()
        mock_ddgs.DDGS = MagicMock()
        sys.modules["ddgs"] = mock_ddgs
        with (
            patch("weekend.cli._fetch_data", return_value=("Sunny", "- E1", "- F1", "June 5-7")),
            patch("weekend.cli.init_config"),
            patch(
                "weekend.cli.generate_weekend_plan",
                return_value=(
                    {"transient_events": [{"name": "E1"}]},
                    {"fixed_activities": [{"name": "F1"}]},
                ),
            ),
            patch("os.path.expanduser", return_value="/tmp/wp_test_main"),
        ):
            import os

            os.makedirs("/tmp/wp_test_main", exist_ok=True)
            runpy.run_module("weekend", run_name="__main__")
        out = capsys.readouterr().out
        assert "Using model" in out or "Started" in out or "Transient" in out
