import pytest
from lib.validators_lib import (
    validate_json,
    validate_detailed_json,
    validate_summary,
    validate_filename,
    is_valid_filename_char,
    has_filename_format,
    count_content_lines,
)

def test_validate_json_valid_data():
    data = ["Item 1", "Item 2", "Item 3"]
    score, msg = validate_json(data)
    assert score == 100
    assert msg == ""

def test_validate_json_invalid_structure():
    data = "Not a list"
    score, msg = validate_json(data)
    assert score == 0
    assert "no items found" in msg

def test_validate_detailed_json_valid_data():
    data = [
        {"name": "Place 1", "location": "Loc 1", "weather": "Clear"},
        {"name": "Place 2", "location": "Loc 2", "price": "Free"},
        {"name": "Place 3", "weather": "Rainy", "description": "Good place"}
    ]
    score, msg = validate_detailed_json(data)
    assert score == 100
    assert msg == ""

def test_validate_detailed_json_missing_details():
    data = [
        {"name": "Place 1"},
        {"name": "Place 2"},
        {"name": "Place 3"}
    ]
    score, msg = validate_detailed_json(data)
    # They have a name but no details.
    # Score should be >0 but <100. Actually, structure=20, items=30, valid=0. Total 50.
    assert score < 100
    assert "no items with details" in msg

def test_validate_summary_valid():
    text = "## Summary\\n- Fact 1\\n- Fact 2\\n- Fact 3\\nAnother line here with more words to reach 100 chars. This is a very long text indeed to test the summary validator."
    score, msg = validate_summary(text)
    assert score > 0

def test_validate_filename_valid():
    filename = "this_is_a_valid_filename"
    score, msg = validate_filename(filename)
    assert score == 100
    assert msg == ""

def test_validate_filename_invalid():
    filename = "this is an invalid filename!"
    score, msg = validate_filename(filename)
    assert score < 100
    assert "invalid chars" in msg

def test_is_valid_filename_char():
    assert is_valid_filename_char('a')
    assert is_valid_filename_char('1')
    assert is_valid_filename_char('_')
    assert is_valid_filename_char('-')
    assert not is_valid_filename_char('!')
    assert not is_valid_filename_char(' ')

def test_has_filename_format():
    assert has_filename_format("file.txt")
    assert has_filename_format("file_name")
    assert not has_filename_format("file")
