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
    data = ["Toronto Event", "Vaughan Festival", "Markham Fair", "Richmond Hill Celebration", "Scarborough Day", "North York Festival", "Etobicoke Event", "East York Fair", "York Celebration", "Downtown Festival"]
    source_text = "Toronto Event and Vaughan Festival, Markham Fair plus Richmond Hill Celebration, Scarborough Day with North York Festival, Etobicoke Event also East York Fair and York Celebration, plus Downtown Festival"
    score, msg = validate_json(data, source_text)
    assert score == 100
    assert msg == ""

def test_validate_json_invalid_structure():
    data = "Not a list"
    score, msg = validate_json(data)
    assert score == 0
    assert "no items found" in msg

def test_validate_detailed_json_valid_data():
    data = [
        {"name": "Place 1", "location": "Loc 1"},
        {"name": "Place 2", "location": "Loc 2"},
        {"name": "Place 3", "location": "Loc 3"},
        {"name": "Place 4", "location": "Loc 4"},
        {"name": "Place 5", "location": "Loc 5"},
        {"name": "Place 6", "location": "Loc 6"},
        {"name": "Place 7", "location": "Loc 7"},
        {"name": "Place 8", "location": "Loc 8"},
        {"name": "Place 9", "location": "Loc 9"},
        {"name": "Place 10", "location": "Loc 10"},
    ]
    source_text = "Place 1 at Loc 1, Place 2 at Loc 2, Place 3 at Loc 3, Place 4 at Loc 4, Place 5 at Loc 5, Place 6 at Loc 6, Place 7 at Loc 7, Place 8 at Loc 8, Place 9 at Loc 9, Place 10 at Loc 10"
    score, msg = validate_detailed_json(data, source_text)
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


def test_check_source_extraction():
    from lib.validators_lib import check_source_extraction
    
    items = [
        {"name": "Spring Festival at Downsview Park"},
        {"name": "Indoor Coding Workshop", "location": "Downsview"},
    ]
    source = "Spring Festival at Downsview Park and Indoor Coding Workshop"
    ratio = check_source_extraction(items, source)
    assert ratio >= 0.5


def test_get_source_matching_details():
    from lib.validators_lib import get_source_matching_details
    
    items = [{"name": "Valid Event"}]
    source = "Valid Event happens"
    details = get_source_matching_details(items, source)
    assert "ratio" in details
    assert len(details["matched"]) >= 1


def test_has_item_details():
    from lib.validators_lib import has_item_details
    
    # Should pass: has name + detail
    assert has_item_details({"name": "Event", "location": "Park"})
    # Should fail: no name field (name added in normalize_keys, not validator)
