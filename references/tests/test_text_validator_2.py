"""Tests for lib.validators.text_validator."""


class TestValidateFileSummary:
    def test_empty(self):
        from lib.validators.text_validator import validate_file_summary

        score, msg = validate_file_summary(None)
        assert score == 0
        assert "empty" in msg

    def test_empty_list(self):
        from lib.validators.text_validator import validate_file_summary

        score, msg = validate_file_summary([])
        assert score == 0
        # Empty list is falsy
        assert "empty" in msg

    def test_dict_input(self):
        from lib.validators.text_validator import validate_file_summary

        score, msg = validate_file_summary({"path": "x.py", "desc": "python file"})
        # Wrapped in list — 1 item → "only 1 items" failure
        assert "only 1 items" in msg
        # 30 (count 0 + paths 25 + quality 0) but not full 30 since count check fails
        # Actually: 30 (count 5/100 × 60) + 25 (path 50% × 50) + 0 (no unique descs) = 55
        assert score < 75

    def test_string_input(self):
        from lib.validators.text_validator import validate_file_summary

        score, _ = validate_file_summary("not a list")
        # Not a dict or list, treated as empty list
        assert score == 0

    def test_too_few_items(self):
        from lib.validators.text_validator import validate_file_summary

        items = [{"path": "x.py", "desc": "a python file"}]
        score, msg = validate_file_summary(items)
        assert "only 1 items" in msg

    def test_good_count(self):
        from lib.validators.text_validator import validate_file_summary

        items = [
            {"path": f"file{i}.py", "desc": f"description for file {i} which is good"}
            for i in range(5)
        ]
        score, _ = validate_file_summary(items)
        # 30 (count) + 30 (paths) + 40 (quality) = 100
        assert score == 100

    def test_unrealistic_paths(self):
        from lib.validators.text_validator import validate_file_summary

        items = [{"path": "filename", "desc": "specific unique description here"} for _ in range(5)]
        score, msg = validate_file_summary(items)
        # No . or / in paths
        assert "unrealistic paths" in msg

    def test_partial_paths(self):
        from lib.validators.text_validator import validate_file_summary

        items = [
            {"path": "x.py", "desc": "specific desc 1"},
            {"path": "y.py", "desc": "specific desc 2"},
            {"path": "filename", "desc": "specific desc 3"},
            {"path": "another", "desc": "specific desc 4"},
            {"path": "another2", "desc": "specific desc 5"},
        ]
        # 2/5 = 40% real paths (boundary: exactly 0.4 → partial credit, no failure msg)
        score, msg = validate_file_summary(items)
        # Partial credit: 30 (count) + 30 (path 60% × 50) + 30 (quality) = 90
        assert score == 90
        # 40% is at the boundary — partial credit, no unrealistic failure
        assert "unrealistic" not in msg

        # Now test < 40% to verify the failure message
        items2 = [
            {"path": "x.py", "desc": "specific desc 1"},
            {"path": "filename", "desc": "specific desc 2"},
            {"path": "another", "desc": "specific desc 3"},
            {"path": "another2", "desc": "specific desc 4"},
            {"path": "another3", "desc": "specific desc 5"},
        ]
        score2, msg2 = validate_file_summary(items2)
        assert "unrealistic" in msg2

    def test_generic_descriptions(self):
        from lib.validators.text_validator import validate_file_summary

        items = [{"path": "x.py", "desc": "personal document"} for _ in range(5)]
        score, msg = validate_file_summary(items)
        # Generic descriptions - quality score is 15 (unique but generic)
        # No specific "generic" failure msg
        assert score == 75  # 30 + 30 + 15

    def test_one_specific(self):
        from lib.validators.text_validator import validate_file_summary

        items = [
            {"path": "x.py", "desc": "personal document"},
            {"path": "y.py", "desc": "specific unique description for the file"},
            {"path": "z.py", "desc": "another generic item"},
            {"path": "a.py", "desc": "another generic item"},
            {"path": "b.py", "desc": "another generic item"},
        ]
        score, _ = validate_file_summary(items)
        # 1 specific + 3 unique generic → 100
        assert score == 100

    def test_no_descriptions(self):
        from lib.validators.text_validator import validate_file_summary

        items = [
            {"path": "x.py"},  # No desc field
            {"path": "y.py"},
            {"path": "z.py"},
            {"path": "a.py"},
            {"path": "b.py"},
        ]
        score, _ = validate_file_summary(items)
        # unique_descs is empty → 60 (30 + 30 + 0)
        assert score == 60

    def test_description_field_alias(self):
        from lib.validators.text_validator import validate_file_summary

        items = [
            {"path": "x.py", "description": "specific description here"},
            {"path": "y.py", "description": "another specific one here"},
            {"path": "z.py", "description": "third specific here"},
            {"path": "a.py", "description": "fourth specific here"},
            {"path": "b.py", "description": "fifth specific here"},
        ]
        score, _ = validate_file_summary(items)
        # Uses "description" as fallback
        assert score > 50


class TestSummaryMatchesItsPrompt:
    """The validator must reward what the summarize prompt actually orders.

    The prompt says: start with a `## Executive Summary` paragraph, and end every
    bullet with `(@handle | Mon DD HH:MM)`. Counting only `user N` tokens and
    requiring synthesis prose ABOVE the first header made a prompt-perfect
    summary score 85 with "no user mentions" while `@user 1..3` padding scored
    100 — the validator rewarded violating the prompt.
    """

    CONFORMANT = (
        "## Executive Summary\n"
        "The week's discussion centered on funding rounds and model releases, with\n"
        "several threads converging on inference cost. Participants reported new\n"
        "benchmarks and confirmed pricing changes across providers.\n\n"
        "## Funding\n"
        "- Series B closed at $40M, led by an existing investor "
        "(@TechCrunch | Mar 15 08:00)\n"
        "- Follow-on announced for infrastructure spend (@benedictevans | Mar 15 09:30)\n\n"
        "## Models\n"
        "- New model release confirmed with lower latency (@simonw | Mar 16 11:05)\n"
        "- Community shared early evaluation numbers (@karpathy | Mar 16 14:20)\n"
    )

    def test_prompt_conformant_summary_reaches_ok(self):
        from lib.validators.text_validator import validate_summary

        score, failures = validate_summary(self.CONFORMANT)
        assert score >= 90, f"{score}: {failures}"
        assert "no user mentions" not in failures

    def test_user_n_padding_does_not_outscore_real_handles(self):
        from lib.validators.text_validator import validate_summary

        padded = self.CONFORMANT + "\n- @user 1 responded, @user 2 asked, @user 3 confirmed\n"
        assert validate_summary(padded)[0] <= validate_summary(self.CONFORMANT)[0]

    def test_real_handles_are_counted_distinctly(self):
        from lib.validators.text_validator import count_distinct_users

        assert count_distinct_users("(@TechCrunch | Mar 15) and (@simonw | Mar 16)") == 2
        # Repeating one handle is not extra coverage.
        assert count_distinct_users("@simonw @simonw @simonw") == 1
        # The legacy synthetic-timeline form still counts.
        assert count_distinct_users("user 1 asked, user 2 replied") == 2
        # Email and domain tokens are not people.
        assert count_distinct_users("contact support@example.com or sales@example.com") == 0
