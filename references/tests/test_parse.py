from lib.osaurus_lib import extract_json, filter_json_items, fix_json_years, normalize_keys


def test_extract_json_valid():
    content = """[{"name": "Event", "location": "Park"}]"""
    result = extract_json(content)
    # Valid JSON array of 1 item, passed through unchanged
    assert result == [{"name": "Event", "location": "Park"}]


def test_extract_json_with_markdown():
    content = """**[{"name": "Event", "location": "Park"}]**"""
    result = extract_json(content)
    # Bold markdown wrapper stripped, JSON extracted
    assert result == [{"name": "Event", "location": "Park"}]


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
