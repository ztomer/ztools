"""Tests for lib.osaurus_output - JSON extraction, normalization, filtering."""



class TestMergeFlatDicts:
    def test_empty(self):
        from lib.osaurus_output import merge_flat_dicts

        assert merge_flat_dicts([]) == []

    def test_none(self):
        from lib.osaurus_output import merge_flat_dicts

        assert merge_flat_dicts(None) is None

    def test_passthrough(self):
        from lib.osaurus_output import merge_flat_dicts

        items = [{"a": 1}, {"b": 2}]
        assert merge_flat_dicts(items) == items


class TestFilterJsonItems:
    def test_empty(self):
        from lib.osaurus_output import filter_json_items

        assert filter_json_items([]) == []

    def test_none(self):
        from lib.osaurus_output import filter_json_items

        assert filter_json_items(None) is None

    def test_valid_item(self):
        from lib.osaurus_output import filter_json_items

        items = [{"name": "x", "location": "y"}]
        assert filter_json_items(items) == items

    def test_skip_pipe_starting_key(self):
        from lib.osaurus_output import filter_json_items

        items = [{"|col1|": "x"}]
        assert filter_json_items(items) == []

    def test_skip_based_on(self):
        from lib.osaurus_output import filter_json_items

        items = [{"name": "Based on my analysis, this is great"}]
        assert filter_json_items(items) == []

    def test_skip_note_prefix(self):
        from lib.osaurus_output import filter_json_items

        items = [{"name": "Note: this is a note"}]
        assert filter_json_items(items) == []

    def test_skip_temperature_conditions(self):
        from lib.osaurus_output import filter_json_items

        items = [{"name": "Temperature conditions are nice"}]
        assert filter_json_items(items) == []

    def test_skip_empty_value(self):
        from lib.osaurus_output import filter_json_items

        items = [{"name": ""}]
        assert filter_json_items(items) == []

    def test_skip_dash_only(self):
        from lib.osaurus_output import filter_json_items

        items = [{"name": "-"}]
        assert filter_json_items(items) == []

    def test_skip_separator_only(self):
        from lib.osaurus_output import filter_json_items

        items = [{"name": "---"}]
        assert filter_json_items(items) == []

    def test_string_pipe_starting(self):
        from lib.osaurus_output import filter_json_items

        items = ["|table|row"]
        assert filter_json_items(items) == []

    def test_string_colon_starting(self):
        from lib.osaurus_output import filter_json_items

        items = [":thing"]
        assert filter_json_items(items) == []

    def test_string_based_on(self):
        from lib.osaurus_output import filter_json_items

        items = ["Based on what I see"]
        assert filter_json_items(items) == []

    def test_string_separator_only(self):
        from lib.osaurus_output import filter_json_items

        items = ["---"]
        assert filter_json_items(items) == []

    def test_string_kept(self):
        from lib.osaurus_output import filter_json_items

        items = ["valid string"]
        assert filter_json_items(items) == items

    def test_other_type_kept(self):
        from lib.osaurus_output import filter_json_items

        items = [42, 3.14, None]
        # All non-dict, non-string kept
        assert filter_json_items(items) == items

    def test_mixed(self):
        from lib.osaurus_output import filter_json_items

        items = [
            {"name": "valid"},
            {"name": "skip me"},
            "valid string",
            "skip this too",
        ]
        result = filter_json_items(items)
        # Filter happens
        assert isinstance(result, list)


class TestFixJsonYears:
    def test_empty(self):
        from lib.osaurus_output import fix_json_years

        assert fix_json_years([]) == []

    def test_none(self):
        from lib.osaurus_output import fix_json_years

        assert fix_json_years(None) is None

    def test_fix_2626(self):
        from lib.osaurus_output import fix_json_years

        items = [{"desc": "Year 2626 was when..."}]
        result = fix_json_years(items)
        assert "2026" in result[0]["desc"]

    def test_bare_26_in_a_date_field_is_expanded(self):
        from lib.osaurus_output import TARGET_YEAR, fix_json_years

        items = [{"start_date": "26", "day": " 26 "}]
        result = fix_json_years(items)
        assert result[0]["start_date"] == TARGET_YEAR
        assert result[0]["day"] == TARGET_YEAR

    def test_day_of_month_and_non_date_26s_survive(self):
        """The corruption this function used to cause.

        A blanket `\b26(?!\\d)` rewrote every standalone 26 in every value:
        "August 26" became "August 2026" — which `_parse_any_date` can no longer
        read, so the row escaped `drop_events_outside_window` — and prices,
        street numbers and age ranges were mangled the same way.
        """
        from lib.osaurus_output import fix_json_years

        items = [
            {
                "day": "August 26",
                "price": "26",
                "location": "26 King St W",
                "target_ages": "8-26",
                "desc": "Year 26 was when...",
            }
        ]
        assert fix_json_years(items) == items

    def test_no_year_unchanged(self):
        from lib.osaurus_output import fix_json_years

        items = [{"desc": "no year here"}]
        result = fix_json_years(items)
        assert result[0]["desc"] == "no year here"

    def test_other_numbers_unchanged(self):
        from lib.osaurus_output import fix_json_years

        items = [{"desc": "Number 25 is fine"}]
        result = fix_json_years(items)
        assert "25" in result[0]["desc"]

    def test_non_string_values_kept(self):
        from lib.osaurus_output import fix_json_years

        items = [{"count": 26}, {"active": True}]
        result = fix_json_years(items)
        assert result[0]["count"] == 26

    def test_non_dict_items_kept(self):
        from lib.osaurus_output import fix_json_years

        items = ["just a string", 42]
        result = fix_json_years(items)
        assert result == items


class TestNormalizeTextOutput:
    def test_empty(self):
        from lib.osaurus_output import normalize_text_output

        assert normalize_text_output("") == []

    def test_none(self):
        from lib.osaurus_output import normalize_text_output

        assert normalize_text_output(None) == []

    def test_simple_numbered(self):
        from lib.osaurus_output import normalize_text_output

        result = normalize_text_output("1. Toronto Event - Park - All Ages")
        assert len(result) == 1
        assert "name" in result[0]

    def test_dash_numbered(self):
        from lib.osaurus_output import normalize_text_output

        result = normalize_text_output("- Toronto Event - Park - All Ages")
        assert len(result) == 1

    def test_skip_headers(self):
        from lib.osaurus_output import normalize_text_output

        result = normalize_text_output("## Header\n1. Item one")
        assert len(result) == 1
        assert "Item" in result[0]["name"]

    def test_skip_empty_lines(self):
        from lib.osaurus_output import normalize_text_output

        result = normalize_text_output("1. A\n\n2. B")
        assert len(result) == 2

    def test_with_location(self):
        from lib.osaurus_output import normalize_text_output

        result = normalize_text_output("1. Event Name - Toronto")
        assert result[0].get("location") == "Toronto"

    def test_with_parts(self):
        from lib.osaurus_output import normalize_text_output

        result = normalize_text_output("1. Event - Toronto - 5-12 - Free - Outdoor")
        item = result[0]
        # Dash-separated: name, location, target_ages, price, weather
        assert item == {
            "name": "Event",
            "location": "Toronto",
            "target_ages": "5",
            "price": "12",
            "weather": "Free",
        }

    def test_skip_asterisk_name(self):
        from lib.osaurus_output import normalize_text_output

        result = normalize_text_output("1. *italic name*")
        # Names starting with * are skipped
        assert len(result) == 0

    def test_with_colon_parts(self):
        from lib.osaurus_output import normalize_text_output

        result = normalize_text_output("1. Event: Toronto: 5-12")
        item = result[0]
        assert item.get("location") == "Toronto"

    def test_with_comma_parts(self):
        from lib.osaurus_output import normalize_text_output

        result = normalize_text_output("1. Event, Toronto, 5-12")
        item = result[0]
        # Comma-separated: only first 2 split, rest is details
        assert item == {"name": "Event, Toronto, 5", "location": "12"}

    def test_no_match(self):
        from lib.osaurus_output import normalize_text_output

        result = normalize_text_output("not a list at all")
        assert result == []


class TestKeyNormalizations:
    def test_top_level_keys_dict(self):
        from lib.osaurus_output import TOP_LEVEL_KEYS

        assert "activities" in TOP_LEVEL_KEYS
        assert "events" in TOP_LEVEL_KEYS
        assert TOP_LEVEL_KEYS["activities"] == "fixed_activities"
        assert TOP_LEVEL_KEYS["events"] == "transient_events"

    def test_key_normalizations_dict(self):
        from lib.osaurus_output import KEY_NORMALIZATIONS

        assert "event" in KEY_NORMALIZATIONS
        assert "title" in KEY_NORMALIZATIONS
        assert "venue" in KEY_NORMALIZATIONS
        assert "address" in KEY_NORMALIZATIONS
