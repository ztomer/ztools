import pytest
from weekend_planner import (
    build_markdown_tables,
    normalize_llm_items,
    print_to_cli,
    print_summary,
    print_header,
)


class TestBuildMarkdownTables:
    def test_empty_dates_and_weather(self):
        result = build_markdown_tables("", "", {}, [])
        assert result is not None
        assert "# Weekend Plan" in result

    def test_with_data(self):
        dates = "Sat May 4 - Sun May 5"
        weather = "18C / Sunny"
        data = {
            "items": [
                {"name": "Park", "time": "10am", "cost": "free"},
            ]
        }
        activities = [
            {"name": "Zoo", "location": "Toronto", "price": "$20"},
        ]
        result = build_markdown_tables(dates, weather, data, activities)
        # Activities list is rendered with name/location/price columns
        assert "Zoo" in result
        assert "Toronto" in result
        assert "Sat May 4" in result
        assert "18C / Sunny" in result
        # And the markdown header is correct
        assert result.startswith("# Weekend Plan: Sat May 4 - Sun May 5")
        # Activity row contains name in bold, location in parens, price as $20
        assert "**Zoo** (Toronto)" in result
        assert "$20" in result

    def test_empty_activities(self):
        dates = "Sat May 4"
        weather = "Sunny"
        result = build_markdown_tables(dates, weather, {}, [])
        assert result is not None

    def test_weather_table_structure(self):
        result = build_markdown_tables("Sat May 4", "22C", {}, [])
        # Both dates and weather are echoed in the header section
        assert "Sat May 4" in result
        assert "22C" in result


class TestNormalizeLlmItems:
    def test_empty_list(self):
        assert normalize_llm_items([]) == []

    def test_none_input(self):
        result = normalize_llm_items(None)
        assert result is None

    def test_strings_to_dicts(self):
        result = normalize_llm_items(["item1", "item2"])
        assert len(result) == 2
        assert result[0]["name"] == "item1"
        assert result[1]["name"] == "item2"

    def test_dicts_preserved(self):
        items = [{"name": "test", "location": "here"}]
        result = normalize_llm_items(items)
        assert result == items

    def test_field_mapping(self):
        items = [{"title": "Park", "loc": "Toronto"}]
        mapping = {"title": "name", "loc": "location"}
        result = normalize_llm_items(items, mapping)
        assert len(result) == 1
        assert result[0]["name"] == "Park"
        assert result[0]["location"] == "Toronto"

    def test_field_mapping_partial(self):
        items = [{"title": "Park", "location": "Toronto"}]
        mapping = {"title": "name"}
        result = normalize_llm_items(items, mapping)
        assert result[0]["name"] == "Park"
        assert result[0]["location"] == "Toronto"


class TestPrintFunctions:
    def test_print_header_output(self, capsys):
        print_header("Test", "value")
        captured = capsys.readouterr()
        assert "Test" in captured.out
        assert "value" in captured.out

    def test_print_to_cli_none(self):
        with pytest.raises((TypeError, AttributeError)):
            print_to_cli(None)

    def test_print_to_cli_empty(self, capsys):
        print_to_cli("")
        captured = capsys.readouterr()
        # Empty string → no output (or only whitespace)
        assert captured.out.strip() == ""

    def test_print_to_cli_content(self, capsys):
        print_to_cli("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_print_summary_output(self, capsys):
        print_summary("OK", 3, 5, "/tmp/test.md", 1.5)
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestGetFunctions:
    def test_get_model_field_mapping_exists(self):
        from weekend_planner import get_model_field_mapping
        result = get_model_field_mapping("qwen3.6")
        assert isinstance(result, dict)

    def test_get_model_top_keys_exists(self):
        from weekend_planner import get_model_top_keys
        result = get_model_top_keys("qwen3.6")
        assert result is not None
        assert isinstance(result, dict) or isinstance(result, list)
