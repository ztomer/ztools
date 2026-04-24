import pytest
from lib.osaurus_lib import extract_json, normalize_keys, filter_json_items, fix_json_years


def test_extract_json_valid():
    content = '''[{"name": "Event", "location": "Park"}]'''
    result = extract_json(content)
    assert result is not None
    assert len(result) > 0


def test_extract_json_with_markdown():
    content = '''**[{"name": "Event", "location": "Park"}]**'''
    result = extract_json(content)
    assert result is not None


def test_normalize_keys():
    data = [{"event": "Name", "venue": "Location"}]
    result = normalize_keys(data, "qwen")
    # Should normalize event->name, venue->location
    assert any("name" in str(r) for r in result)


def test_filter_json_items():
    items = [
        {"name": "Valid Event"},
        {"name": "Based on the data"},
        {"location": "Park"},
    ]
    filtered = filter_json_items(items)
    # Should filter out "Based on..."
    assert len(filtered) < len(items)


def test_fix_json_years():
    items = [{"name": "Event 2626"}, {"date": "April 20-22, 2626"}]
    fixed = fix_json_years(items)
    assert any("2026" in str(v) for item in fixed for v in item.values())

