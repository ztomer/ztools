import pytest
from unittest.mock import patch, MagicMock

from img_helpers import (
    clean_filename,
    extract_first_line,
    extract_full_text,
    is_meaningful_text,
    is_non_human_readable,
    _strip_instruction_prefix,
)


class TestCleanFilename:
    def test_basic_cleaning(self):
        assert clean_filename("Hello World!!!") == "hello_world"

    def test_special_chars_removed(self):
        result = clean_filename("photo@#$%^&*.png")
        assert result == "photopng"
        assert "unnamed" not in result

    def test_whitespace_to_underscore(self):
        assert clean_filename("my cool photo") == "my_cool_photo"

    def test_dashes_to_underscore(self):
        assert clean_filename("my-cool-photo") == "my_cool_photo"

    def test_lowercase(self):
        assert clean_filename("HELLO WORLD") == "hello_world"

    def test_truncation(self):
        result = clean_filename("a" * 100, max_length=20)
        assert len(result) <= 20
        assert result == result.rstrip("_")

    def test_empty_returns_unnamed(self):
        assert clean_filename("") == "unnamed"

    def test_only_special_chars(self):
        assert clean_filename("!@#$%") == "unnamed"

    def test_trailing_underscore_stripped(self):
        assert clean_filename("hello_") == "hello"


class TestIsMeaningfulText:
    def test_empty_text(self):
        assert is_meaningful_text("") is False

    def test_whitespace_only(self):
        assert is_meaningful_text("   ") is False

    def test_single_long_alphanum(self):
        assert is_meaningful_text("ABCD1234") is False

    def test_all_uppercase_no_spaces(self):
        assert is_meaningful_text("HELLOWORLD") is False

    def test_normal_sentence(self):
        assert is_meaningful_text("Hello world") is True

    def test_two_short_words_below_min(self):
        assert is_meaningful_text("hi there", min_word_count=3) is False

    def test_multiple_short_words(self):
        assert is_meaningful_text("a quick brown fox") is True

    def test_single_word_with_spaces(self):
        assert is_meaningful_text("meaningful text", min_word_count=2) is True


class TestIsNonHumanReadable:
    def test_empty_text(self):
        assert is_non_human_readable("") is True

    def test_short_text(self):
        assert is_non_human_readable("ab") is True

    def test_hf_code(self):
        assert is_non_human_readable("HFabc123456") is True

    def test_hh_code(self):
        assert is_non_human_readable("HH123456789") is True

    def test_handle(self):
        assert is_non_human_readable("@username") is True

    def test_short_uppercase(self):
        assert is_non_human_readable("ABC") is True

    def test_code_with_digits(self):
        assert is_non_human_readable("ABC123") is True

    def test_readable_sentence(self):
        assert is_non_human_readable("Hello world") is False

    def test_handle_with_underscore(self):
        assert is_non_human_readable("@user_name") is False


class TestStripInstructionPrefix:
    def test_here_is_a(self):
        assert _strip_instruction_prefix("Here is a filename: my_file") == "my_file"

    def test_heres_the(self):
        assert _strip_instruction_prefix("Here's the filename: output") == "output"

    def test_suggested(self):
        assert _strip_instruction_prefix("suggested name: my_photo") == "my_photo"

    def test_renamed_to(self):
        assert _strip_instruction_prefix("renamed to: new_name") == "new_name"

    def test_filename_is(self):
        assert _strip_instruction_prefix("Filename: result") == "result"

    def test_no_prefix(self):
        assert _strip_instruction_prefix("my_file") == "my_file"

    def test_case_insensitive(self):
        assert _strip_instruction_prefix("SUGGESTED FILENAME: cool") == "cool"

    def test_whitespace_handling(self):
        assert _strip_instruction_prefix("  Here is a file:  test  ") == "test"


class TestExtractFirstLine:
    def test_returns_first_line(self):
        mock_image = MagicMock()
        with patch("img_helpers.Image.open", return_value=mock_image), \
             patch("img_helpers.pytesseract.image_to_string", return_value="First line\nSecond line\nThird"):
            result = extract_first_line("test.png")
        assert result == "First line"

    def test_returns_none_for_empty(self):
        mock_image = MagicMock()
        with patch("img_helpers.Image.open", return_value=mock_image), \
             patch("img_helpers.pytesseract.image_to_string", return_value=""):
            result = extract_first_line("test.png")
        assert result is None

    def test_skips_blank_lines(self):
        mock_image = MagicMock()
        with patch("img_helpers.Image.open", return_value=mock_image), \
             patch("img_helpers.pytesseract.image_to_string", return_value="\n\nFirst real line"):
            result = extract_first_line("test.png")
        assert result == "First real line"

    def test_exception_returns_none(self, capsys):
        with patch("img_helpers.Image.open", side_effect=Exception("bad file")):
            result = extract_first_line(MagicMock(name="broken.png"))
        assert result is None
        out = capsys.readouterr()
        assert "broken.png" in out.out or "broken.png" in out.err


class TestExtractFullText:
    def test_returns_full_text(self):
        mock_image = MagicMock()
        with patch("img_helpers.Image.open", return_value=mock_image), \
             patch("img_helpers.pytesseract.image_to_string", return_value="Line 1\nLine 2"):
            result = extract_full_text("test.png")
        assert result == "Line 1\nLine 2"

    def test_returns_none_for_empty(self):
        mock_image = MagicMock()
        with patch("img_helpers.Image.open", return_value=mock_image), \
             patch("img_helpers.pytesseract.image_to_string", return_value=""):
            result = extract_full_text("test.png")
        assert result is None

    def test_exception_returns_none(self, capsys):
        with patch("img_helpers.Image.open", side_effect=Exception("bad file")):
            result = extract_full_text(MagicMock(name="broken.png"))
        assert result is None
        out = capsys.readouterr()
        assert "broken.png" in out.out or "broken.png" in out.err
