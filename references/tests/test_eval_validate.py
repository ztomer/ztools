
from eval.validate import safe_content, validate_file_summary


class TestSafeContent:
    def test_none_returns_empty(self):
        assert safe_content({"content": None}) == ""

    def test_dict_content_stringified(self):
        result = safe_content({"content": {"key": "val"}})
        assert isinstance(result, str)
        assert "key" in result

    def test_normal_string(self):
        assert safe_content({"content": "hello"}) == "hello"

    def test_missing_key(self):
        assert safe_content({}) == ""

    def test_empty_string(self):
        assert safe_content({"content": ""}) == ""


class TestValidateFileSummary:
    def test_empty_data(self):
        score, msg = validate_file_summary(None)
        assert score == 0
        assert "empty" in msg

    def test_empty_list(self):
        score, msg = validate_file_summary([])
        assert score == 0
        assert "empty" in msg

    def test_few_files(self):
        data = [{"path": "a.py", "desc": "does stuff"}]
        score, msg = validate_file_summary(data)
        assert "only" in msg

    def test_all_detailed_scores_100(self):
        data = [
            {"path": "a.py", "desc": "parses config files and loads settings"},
            {"path": "b.py", "desc": "validates JSON output format"},
            {"path": "c.py", "desc": "fetches data from external API"},
            {"path": "d.py", "desc": "handles error processing logic"},
        ]
        score, msg = validate_file_summary(data)
        assert score == 100
        assert msg == ""

    def test_no_content_details(self):
        data = [
            {"path": "a.py", "desc": "some file"},
            {"path": "b.py", "desc": "another file"},
        ]
        score, msg = validate_file_summary(data)
        # 0 detailed out of 2 → 0; only 2 files (<3) → 25
        assert score == 25
        assert "no content details" in msg

    def test_dict_input_with_parsed_json(self):
        data = {"main.py": "parses input data", "utils.py": "validates output"}
        score, msg = validate_file_summary(data)
        # Dict becomes [{main.py: parses input data}, {utils.py: validates output}] → 2 items
        # Both descs have content verbs ("parses", "validates") → 2/2 = 100% detailed
        # 2/2 >= 0.8 → score 85
        assert score == 85
        assert msg == ""

    def test_dict_no_content_detail(self):
        data = {"main.py": "a python script", "utils.py": "helper functions"}
        score, msg = validate_file_summary(data)
        assert "no content details" in msg

    def test_string_with_text_headers(self):
        content = (
            "## Main module\nhandles configuration and api calls\n## Utils\nvalidation helpers"
        )
        score, msg = validate_file_summary(content)
        # String is not JSON-parseable as a list/dict of items → score 20, "no headers"
        assert score == 20
        assert "no headers" in msg

    def test_string_too_short(self):
        score, msg = validate_file_summary("hello")
        assert "no headers" in msg

    def test_partially_detailed(self):
        data = [
            {"path": "a.py", "desc": "parses config files"},
            {"path": "b.py", "desc": "some random script"},
            {"path": "c.py", "desc": "another generic description"},
            {"path": "d.py", "desc": "yet another file"},
        ]
        score, msg = validate_file_summary(data)
        # 4 files <5 files, 1/4 detailed = 25% → 0
        assert score == 50

    def test_non_dict_items_skipped(self):
        """Lines 47, 51: non-dict items and items without path/desc are skipped."""
        # 4 valid items (all detailed) + 3 invalid (skipped)
        # num_files = 7 (all items), 4 detailed / 7 = 57% → score 85
        data = [
            "string item",
            42,
            None,
            {"path": "a.py", "desc": "parses config files"},
            {"path": "b.py", "desc": "validates JSON output format"},
            {"path": "c.py", "desc": "fetches data from external API"},
            {"path": "d.py", "desc": "handles error processing logic"},
        ]
        score, msg = validate_file_summary(data)
        assert score == 85

    def test_no_path_no_desc_skipped(self):
        """Items missing path or desc are skipped (line 51)."""
        # 4 valid detailed, 2 invalid → num_files=6, 4/6 = 67% → 85
        data = [
            {"path": "a.py", "desc": "parses config files"},
            {"path": "b.py"},  # no desc
            {"desc": "no path here"},  # no path
            {"path": "c.py", "desc": "validates JSON output format"},
            {"path": "d.py", "desc": "fetches data from external API"},
            {"path": "e.py", "desc": "handles error processing logic"},
        ]
        score, msg = validate_file_summary(data)
        assert score == 85

    def test_no_items_returns_zero(self):
        """Line 59: items list has only non-dict entries, so no content details."""
        data = [
            "string",
            42,
            None,
        ]
        score, msg = validate_file_summary(data)
        assert score == 25
        assert "no content details" in msg

    def test_medium_detail_count(self):
        """Line 63: detailed_count is 50-80% of total."""
        # 4 items, 2 detailed = 50%
        data = [
            {"path": "a.py", "desc": "parses config files"},
            {"path": "b.py", "desc": "validates JSON output format"},
            {"path": "c.py", "desc": "some random script"},
            {"path": "d.py", "desc": "another file"},
        ]
        score, msg = validate_file_summary(data)
        assert score == 85

    def test_two_detailed(self):
        """Line 65: detailed_count is exactly 2."""
        # 5 items, 2 detailed
        data = [
            {"path": "a.py", "desc": "parses config files"},
            {"path": "b.py", "desc": "validates JSON output format"},
            {"path": "c.py", "desc": "some random script"},
            {"path": "d.py", "desc": "another file"},
            {"path": "e.py", "desc": "yet another file"},
        ]
        score, msg = validate_file_summary(data)
        assert score == 70

    def test_one_detailed(self):
        """Score 50 for 1 detailed item."""
        data = [
            {"path": "a.py", "desc": "parses config files"},
            {"path": "b.py", "desc": "some random script"},
            {"path": "c.py", "desc": "another file"},
            {"path": "d.py", "desc": "yet another file"},
        ]
        score, msg = validate_file_summary(data)
        assert score == 50

    def test_long_string_with_headers(self):
        """Line 91: string input that's long enough."""
        content = (
            "## Header\nThis is a long string that should pass the 200 character threshold for additional scoring in the file summary validation function."
            * 3
        )
        score, msg = validate_file_summary(content)
        # String isn't valid JSON, has text headers (+20) and length >= 200 (+20) = 40
        assert score == 40
        assert msg == ""

    def test_dict_input_medium_detail(self):
        """Lines 116, 118, 120: dict path with various detailed counts."""
        # 5 items, 3 detailed = 60% → 70
        data = {
            "a.py": "parses config files",
            "b.py": "validates JSON output format",
            "c.py": "fetches data from API",
            "d.py": "some random script",
            "e.py": "another file",
        }
        score, msg = validate_file_summary(data)
        assert score == 70

    def test_dict_input_two_detailed(self):
        """Line 118: exactly 2 detailed items in dict."""
        # 5 items, 2 detailed = 40% → 55
        data = {
            "a.py": "parses config files",
            "b.py": "validates JSON output format",
            "c.py": "some random script",
            "d.py": "another file",
            "e.py": "yet another file",
        }
        score, msg = validate_file_summary(data)
        assert score == 55

    def test_dict_input_one_detailed(self):
        """Line 120: exactly 1 detailed item in dict."""
        data = {
            "a.py": "parses config files",
            "b.py": "some random script",
            "c.py": "another file",
        }
        score, msg = validate_file_summary(data)
        assert score == 40

    def test_dict_input_empty_filepath_skipped(self):
        """Line 106: items with empty filepath or summary are skipped in dict path."""
        data = {
            "": "parses config files",  # empty filepath
            "b.py": "",  # empty summary
            "c.py": "validates JSON output format",
            "d.py": "fetches data from external API",
            "e.py": "handles error processing logic",
        }
        score, msg = validate_file_summary(data)
        # num_files = 5 (all keys), detailed = 3/5 = 60% → 70
        assert score == 70
        assert msg == ""

    def test_no_items_returns_zero_non_dict(self):
        """Items list with only non-dict entries: num_files is len(items) which is > 0,
        but detailed_count is 0, so falls through to else branch (score 25).
        Line 59 is unreachable since empty list is caught earlier."""
        data = ["string", 42, None]
        score, msg = validate_file_summary(data)
        # num_files = 3, < 4, so "only 3 files" added
        # detailed_count = 0, so "no content details" added
        assert "only 3 files" in msg
        assert "no content details" in msg
        assert score == 25


class TestFileSummaryRejectsFilenameInference:
    """The task exists to catch filename inference; the checks must be real.

    The docstring promised "no filename-only summaries" and "no generic patterns
    like 'a python script'", but the implementation was a single keyword scan
    whose verb list included `config`, `model`, `api` and `client` — so "a
    python config" counted as a detailed description.
    """

    DETAILED = [
        {"path": "lib/config_loader.py", "desc": "Parses TOML config and validates keys"},
        {"path": "lib/api_client.py", "desc": "Sends chat requests and handles retries"},
        {"path": "lib/report.py", "desc": "Renders scorecards into markdown and writes files"},
        {"path": "lib/extract.py", "desc": "Extracts JSON from noisy output, strips thinking"},
    ]

    def test_real_descriptions_still_score_full(self):
        score, msg = validate_file_summary(self.DETAILED)
        assert score == 100
        assert msg == ""

    def test_filename_echoes_do_not_count_as_detail(self):
        data = [
            {"path": "lib/config_loader.py", "desc": "config loader"},
            {"path": "lib/api_client.py", "desc": "api client"},
            {"path": "lib/report.py", "desc": "Renders scorecards into markdown and writes files"},
            {"path": "lib/extract.py", "desc": "Extracts JSON from noisy output"},
        ]
        score, msg = validate_file_summary(data)
        assert score < 100
        assert "filename-only" in msg

    def test_generic_descriptions_do_not_count_as_detail(self):
        data = [
            {"path": "lib/one.py", "desc": "a python script"},
            {"path": "lib/two.py", "desc": "a config file"},
            {"path": "lib/three.py", "desc": "Renders scorecards into markdown"},
            {"path": "lib/four.py", "desc": "Extracts JSON from noisy model output"},
        ]
        score, msg = validate_file_summary(data)
        assert score < 100
        assert "generic description" in msg
