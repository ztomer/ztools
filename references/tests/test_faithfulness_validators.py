

class TestValidateNoLeak:
    def test_clean_passes(self):
        from lib.validators.text_validator import validate_no_leak

        assert validate_no_leak("how_to_manage_underperformers") == (100, "")

    def test_leak_fails(self):
        from lib.validators.text_validator import validate_no_leak

        score, reason = validate_no_leak("Here is the filename: how_to_manage")
        assert score == 0, reason
        assert "leak" in reason

    def test_leak_colon_fails(self):
        from lib.validators.text_validator import validate_no_leak

        score, reason = validate_no_leak("filename: how_to_manage_underperformers")
        assert score == 0, reason


class TestValidateStrictSchema:
    def test_clean_json_passes(self):
        from lib.validators.text_validator import validate_strict_schema

        assert validate_strict_schema('{"name": "x"}') == (100, "")

    def test_prose_before_fails(self):
        from lib.validators.text_validator import validate_strict_schema

        score, reason = validate_strict_schema('Here is the result: {"name": "x"}')
        assert score == 0, reason
        assert "prose" in reason

    def test_fence_fails(self):
        from lib.validators.text_validator import validate_strict_schema

        score, reason = validate_strict_schema('```json\n{"name": "x"}\n```')
        assert score == 0, reason
        assert "fence" in reason

    def test_trailing_text_fails(self):
        from lib.validators.text_validator import validate_strict_schema

        score, reason = validate_strict_schema('{"name": "x"} hope that helps')
        assert score == 0, reason
        assert "prose" in reason


class TestValidateNoContradiction:
    def test_parrot_fails(self):
        from lib.validators.text_validator import validate_no_contradiction

        score, reason = validate_no_contradiction(
            "Summary: quantum giraffes of Manitoba won the Stanley Cup.",
            contradiction_phrase="quantum giraffes of Manitoba won the Stanley Cup",
        )
        assert score == 0, reason
        assert "contradiction" in reason

    def test_clean_passes(self):
        from lib.validators.text_validator import validate_no_contradiction

        score, reason = validate_no_contradiction(
            "Summary: OpenAI announced GPT-5. CN Tower reopened.",
            contradiction_phrase="quantum giraffes of Manitoba won the Stanley Cup",
        )
        assert score == 100, reason
