"""Tests for lib.validators.helpers."""
import pytest


class TestHasTextHeaders:
    def test_h2(self):
        from lib.validators.helpers import has_text_headers
        assert has_text_headers("## Title") is True

    def test_h3(self):
        from lib.validators.helpers import has_text_headers
        assert has_text_headers("### Subtitle") is True

    def test_h1_not_matched(self):
        from lib.validators.helpers import has_text_headers
        assert has_text_headers("# Top") is False

    def test_no_header(self):
        from lib.validators.helpers import has_text_headers
        assert has_text_headers("just text") is False

    def test_header_in_middle(self):
        from lib.validators.helpers import has_text_headers
        assert has_text_headers("intro\n## Section\nbody") is True

    def test_empty(self):
        from lib.validators.helpers import has_text_headers
        assert has_text_headers("") is False


class TestCountContentLines:
    def test_basic(self):
        from lib.validators.helpers import count_content_lines
        assert count_content_lines("a\nb\nc") == 3

    def test_excludes_headers(self):
        from lib.validators.helpers import count_content_lines
        assert count_content_lines("a\n## Header\nb") == 2

    def test_excludes_empty(self):
        from lib.validators.helpers import count_content_lines
        assert count_content_lines("a\n\n\nb") == 2

    def test_empty(self):
        from lib.validators.helpers import count_content_lines
        assert count_content_lines("") == 0

    def test_none(self):
        from lib.validators.helpers import count_content_lines
        assert count_content_lines(None) == 0

    def test_only_headers(self):
        from lib.validators.helpers import count_content_lines
        assert count_content_lines("## a\n### b") == 0


class TestIsValidFilenameChar:
    def test_letter(self):
        from lib.validators.helpers import is_valid_filename_char
        assert is_valid_filename_char("a") is True

    def test_digit(self):
        from lib.validators.helpers import is_valid_filename_char
        assert is_valid_filename_char("5") is True

    def test_underscore(self):
        from lib.validators.helpers import is_valid_filename_char
        assert is_valid_filename_char("_") is True

    def test_dash(self):
        from lib.validators.helpers import is_valid_filename_char
        assert is_valid_filename_char("-") is True

    def test_dot(self):
        from lib.validators.helpers import is_valid_filename_char
        assert is_valid_filename_char(".") is True

    def test_space(self):
        from lib.validators.helpers import is_valid_filename_char
        assert is_valid_filename_char(" ") is False

    def test_slash(self):
        from lib.validators.helpers import is_valid_filename_char
        assert is_valid_filename_char("/") is False

    def test_special(self):
        from lib.validators.helpers import is_valid_filename_char
        assert is_valid_filename_char("@") is False


class TestHasFilenameFormat:
    def test_underscore(self):
        from lib.validators.helpers import has_filename_format
        assert has_filename_format("my_file") is True

    def test_dash(self):
        from lib.validators.helpers import has_filename_format
        assert has_filename_format("my-file") is True

    def test_dot(self):
        from lib.validators.helpers import has_filename_format
        assert has_filename_format("file.txt") is True

    def test_no_separator(self):
        from lib.validators.helpers import has_filename_format
        assert has_filename_format("myfile") is False

    def test_empty(self):
        from lib.validators.helpers import has_filename_format
        assert has_filename_format("") is False


class TestExtractBestFilenameCandidate:
    def test_simple(self):
        from lib.validators.helpers import _extract_best_filename_candidate
        assert _extract_best_filename_candidate("my_filename") == "my_filename"

    def test_multiline(self):
        from lib.validators.helpers import _extract_best_filename_candidate
        # First line must be skipped (too short or invalid)
        result = _extract_best_filename_candidate("## think\nmy_filename\nmore")
        # "## think" is skipped because of # prefix
        assert "my_filename" in result

    def test_skips_code_block(self):
        from lib.validators.helpers import _extract_best_filename_candidate
        result = _extract_best_filename_candidate("```\n```\nmy_filename")
        # Code block lines are skipped
        assert "my_filename" in result

    def test_skips_header(self):
        from lib.validators.helpers import _extract_best_filename_candidate
        result = _extract_best_filename_candidate("## Header\nmy_filename")
        assert "my_filename" in result

    def test_short_line_skipped(self):
        from lib.validators.helpers import _extract_best_filename_candidate
        result = _extract_best_filename_candidate("ab\nabcdef")
        # "ab" is too short (< 3), so it's skipped
        # "abcdef" is taken
        assert "abcdef" in result

    def test_long_line_skipped(self):
        from lib.validators.helpers import _extract_best_filename_candidate
        result = _extract_best_filename_candidate("a" * 70 + "\nmy_file")
        # First line is too long, skipped
        assert "my_file" in result

    def test_empty(self):
        from lib.validators.helpers import _extract_best_filename_candidate
        assert _extract_best_filename_candidate("") == ""

    def test_whitespace_only(self):
        from lib.validators.helpers import _extract_best_filename_candidate
        assert _extract_best_filename_candidate("   \n  \n") == ""

    def test_no_valid_candidate(self):
        from lib.validators.helpers import _extract_best_filename_candidate
        # Lines too short, no good candidate
        result = _extract_best_filename_candidate("a\nb\nc")
        # Falls back to first 50 chars of stripped text
        assert "a" in result


class TestStripBacktickValue:
    def test_simple(self):
        from lib.validators.helpers import strip_backtick_value
        assert strip_backtick_value("hello") == "hello"

    def test_single_backticks(self):
        from lib.validators.helpers import strip_backtick_value
        assert strip_backtick_value("`hello`") == "hello"

    def test_code_block(self):
        from lib.validators.helpers import strip_backtick_value
        assert strip_backtick_value("```hello```") == "hello"

    def test_empty(self):
        from lib.validators.helpers import strip_backtick_value
        assert strip_backtick_value("") == ""

    def test_none(self):
        from lib.validators.helpers import strip_backtick_value
        assert strip_backtick_value(None) == ""

    def test_int(self):
        from lib.validators.helpers import strip_backtick_value
        assert strip_backtick_value(42) == "42"

    def test_whitespace(self):
        from lib.validators.helpers import strip_backtick_value
        assert strip_backtick_value("  `hello`  ") == "hello"

    def test_only_open_code_block(self):
        from lib.validators.helpers import strip_backtick_value
        result = strip_backtick_value("```hello")
        # Strips leading ``` (3 chars) but keeps the text after it
        assert result == "hello"


class TestNormalizeWhitespace:
    def test_basic(self):
        from lib.validators.helpers import normalize_whitespace
        assert normalize_whitespace("hello  world") == "hello world"

    def test_tabs(self):
        from lib.validators.helpers import normalize_whitespace
        assert normalize_whitespace("hello\tworld") == "hello world"

    def test_newlines(self):
        from lib.validators.helpers import normalize_whitespace
        assert normalize_whitespace("hello\nworld") == "hello world"

    def test_strip(self):
        from lib.validators.helpers import normalize_whitespace
        assert normalize_whitespace("  hello  ") == "hello"

    def test_empty(self):
        from lib.validators.helpers import normalize_whitespace
        assert normalize_whitespace("") == ""

    def test_none(self):
        from lib.validators.helpers import normalize_whitespace
        assert normalize_whitespace(None) == ""


class TestExtractJsonList:
    def test_valid_list(self):
        from lib.validators.helpers import extract_json_list
        result = extract_json_list('[{"a": 1}, {"b": 2}]')
        assert result == [{"a": 1}, {"b": 2}]

    def test_in_text(self):
        from lib.validators.helpers import extract_json_list
        result = extract_json_list("before text [{\"x\": 1}] after")
        assert result == [{"x": 1}]

    def test_no_list(self):
        from lib.validators.helpers import extract_json_list
        assert extract_json_list("just text") == []

    def test_empty(self):
        from lib.validators.helpers import extract_json_list
        assert extract_json_list("") == []

    def test_none(self):
        from lib.validators.helpers import extract_json_list
        assert extract_json_list(None) == []

    def test_invalid_json(self):
        from lib.validators.helpers import extract_json_list
        assert extract_json_list("[{invalid}]") == []


class TestHasItemDetails:
    def test_dict_with_one_key(self):
        from lib.validators.helpers import has_item_details
        assert has_item_details({"name": "x"}) is False

    def test_dict_with_two_keys(self):
        from lib.validators.helpers import has_item_details
        assert has_item_details({"name": "x", "age": 1}) is True

    def test_empty_dict(self):
        from lib.validators.helpers import has_item_details
        assert has_item_details({}) is False

    def test_none(self):
        from lib.validators.helpers import has_item_details
        assert has_item_details(None) is False

    def test_string(self):
        from lib.validators.helpers import has_item_details
        assert has_item_details("not a dict") is False

    def test_list(self):
        from lib.validators.helpers import has_item_details
        assert has_item_details([1, 2]) is False
