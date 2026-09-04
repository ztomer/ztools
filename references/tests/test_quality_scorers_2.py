"""Tests for lib.quality_scorers - all scoring functions."""

import pytest
from lib.quality_models import TestCase


@pytest.fixture
def mock_llm():
    from lib.testing import MockLLM

    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


def make_case(input_text="", reference="", description="test", task="filename"):
    return TestCase(
        task=task,
        input_text=input_text,
        reference=reference,
        description=description,
    )


class TestSummarizeStructure:
    def test_empty(self, mock_llm):
        from lib.quality_scorers import _score_summarize_structure

        s = _score_summarize_structure("", make_case())
        assert s.score == 0

    def test_no_structure(self, mock_llm):
        from lib.quality_scorers import _score_summarize_structure

        s = _score_summarize_structure("Just a wall of text without any structure.", make_case())
        assert s.score == 20
        assert any("header" in f or "bullet" in f for f in s.failures)

    def test_headers_only(self, mock_llm):
        from lib.quality_scorers import _score_summarize_structure

        out = "## Section One\nSome content\n## Section Two\nMore content\n" * 3
        s = _score_summarize_structure(out, make_case())
        assert s.score == 70

    def test_bullets_only(self, mock_llm):
        from lib.quality_scorers import _score_summarize_structure

        out = "- Item one\n- Item two\n- Item three\n" * 5
        s = _score_summarize_structure(out, make_case())
        assert s.score == 50

    def test_full_structure(self, mock_llm):
        from lib.quality_scorers import _score_summarize_structure

        out = "## Title\n- Bullet\n- Bullet\n## Section\n- More\n" * 5
        s = _score_summarize_structure(out, make_case())
        assert s.score == 100

    def test_template(self, mock_llm):
        from lib.quality_scorers import _score_summarize_structure

        out = "## Title\n**Who: user 1\n**What: did thing\n**When: now\n**Where: here\n" * 3
        s = _score_summarize_structure(out, make_case())
        assert s.score <= 60
        assert any("template" in f for f in s.failures)

    def test_too_short(self, mock_llm):
        from lib.quality_scorers import _score_summarize_structure

        s = _score_summarize_structure("## Short\nbody", make_case())
        # 13 chars < 150 → 0 length, 50 header
        assert s.score == 50
        assert any("short" in f for f in s.failures)

    def test_too_long(self, mock_llm):
        from lib.quality_scorers import _score_summarize_structure

        out = "## Title\n- Bullet\n" + ("filler " * 500)
        s = _score_summarize_structure(out, make_case())
        # 3000+ chars > 350 (good), header (20), bullet (30), -long penalty
        assert s.score == 80
        assert any("long" in f for f in s.failures)


class TestSummarizeSpecificity:
    def test_empty(self, mock_llm):
        from lib.quality_scorers import _score_summarize_specificity

        s = _score_summarize_specificity("", make_case())
        assert s.score == 0

    def test_no_input_events(self, mock_llm):
        from lib.quality_scorers import _score_summarize_specificity

        s = _score_summarize_specificity("any output with user 1", make_case("plain text"))
        # No events in input -> ts_score = 0
        assert s.score >= 0

    def test_no_user_mentions(self, mock_llm):
        from lib.quality_scorers import _score_summarize_specificity

        s = _score_summarize_specificity(
            "Long output without any user mention.", make_case("user 1 said thing")
        )
        assert any("no user" in f for f in s.failures)

    def test_full_specificity(self, mock_llm):
        from lib.quality_scorers import _score_summarize_specificity

        out = "At 10:00 user 1 said hi. At 11:00 user 2 responded. At 12:00 user 3 confirmed."
        inp = "10:00 user 1 said hi\n11:00 user 2 said back\n12:00 user 3 confirmed"
        s = _score_summarize_specificity(out, make_case(inp))
        assert s.score == 100

    def test_missing_timestamps(self, mock_llm):
        from lib.quality_scorers import _score_summarize_specificity

        out = "user 1 said hi. user 2 said back. user 3 confirmed." * 5
        inp = "10:00 user 1 said hi\n11:00 user 2 said back\n12:00 user 3 confirmed"
        s = _score_summarize_specificity(out, make_case(inp))
        assert any("timestamp" in f for f in s.failures)


class TestFileCompleteness:
    def test_empty(self, mock_llm):
        from lib.quality_scorers import _score_file_completeness

        s = _score_file_completeness("", make_case(reference="[]"))
        assert s.score == 0
        assert "empty" in s.failures

    def test_invalid_json(self, mock_llm):
        from lib.quality_scorers import _score_file_completeness

        s = _score_file_completeness("not json", make_case(reference="[]"))
        assert s.score == 0
        assert any("invalid" in f for f in s.failures)

    def test_not_list(self, mock_llm):
        from lib.quality_scorers import _score_file_completeness

        s = _score_file_completeness('{"a": 1}', make_case(reference="[]"))
        assert s.score == 0
        assert "not a list" in s.failures

    def test_full_coverage(self, mock_llm):
        from lib.quality_scorers import _score_file_completeness

        ref = '[{"path": "a.txt", "desc": "x"}, {"path": "b.txt", "desc": "y"}]'
        out = '[{"path": "a.txt"}, {"path": "b.txt"}]'
        s = _score_file_completeness(out, make_case(reference=ref))
        assert s.score == 100

    def test_missing_files(self, mock_llm):
        from lib.quality_scorers import _score_file_completeness

        ref = '[{"path": "a.txt", "desc": "x"}, {"path": "b.txt", "desc": "y"}]'
        out = '[{"path": "a.txt"}]'
        s = _score_file_completeness(out, make_case(reference=ref))
        assert s.score == 50
        assert any("missing" in f for f in s.failures)

    def test_no_ref_paths(self, mock_llm):
        from lib.quality_scorers import _score_file_completeness

        # exp_paths empty -> ratio = 0 (per `if exp_paths else 0`)
        s = _score_file_completeness("[]", make_case(reference="[]"))
        assert s.score == 0


class TestFileAccuracy:
    def test_empty(self, mock_llm):
        from lib.quality_scorers import _score_file_accuracy

        s = _score_file_accuracy("", make_case(reference="[]"))
        assert s.score == 0

    def test_invalid_json(self, mock_llm):
        from lib.quality_scorers import _score_file_accuracy

        s = _score_file_accuracy("not json", make_case(reference="[]"))
        assert s.score == 0
        assert any("invalid" in f for f in s.failures)

    def test_not_list(self, mock_llm):
        from lib.quality_scorers import _score_file_accuracy

        s = _score_file_accuracy('{"a": 1}', make_case(reference="[]"))
        assert "not a list" in s.failures

    def test_no_items_scored(self, mock_llm):
        from lib.quality_scorers import _score_file_accuracy

        ref = '[{"path": "a.txt", "desc": "x"}]'
        out = "[]"
        s = _score_file_accuracy(out, make_case(reference=ref))
        assert "no items scored" in s.failures

    def test_path_not_found(self, mock_llm):
        from lib.quality_scorers import _score_file_accuracy

        ref = '[{"path": "a.txt", "desc": "x"}]'
        out = '[{"path": "b.txt", "desc": "y"}]'
        s = _score_file_accuracy(out, make_case(reference=ref))
        # No items scored, all failed
        assert s.score == 0
        assert any("no items scored" in f for f in s.failures)

    def test_no_description(self, mock_llm):
        from lib.quality_scorers import _score_file_accuracy

        ref = '[{"path": "a.txt", "desc": "x"}]'
        out = '[{"path": "a.txt"}]'
        s = _score_file_accuracy(out, make_case(reference=ref))
        # Empty desc -> continue without incrementing count, no items scored
        assert s.score == 0
        assert any("no items scored" in f for f in s.failures)

    def test_desc_mismatch(self, mock_llm):
        from lib.quality_scorers import _score_file_accuracy

        ref = '[{"path": "a.txt", "desc": "important database connection module"}]'
        out = '[{"path": "a.txt", "desc": "totally unrelated stuff"}]'
        s = _score_file_accuracy(out, make_case(reference=ref))
        assert any("mismatch" in f for f in s.failures)

    def test_desc_match(self, mock_llm):
        from lib.quality_scorers import _score_file_accuracy

        ref = '[{"path": "a.txt", "desc": "important database connection module"}]'
        out = '[{"path": "a.txt", "desc": "important database connection stuff"}]'
        s = _score_file_accuracy(out, make_case(reference=ref))
        assert s.score >= 70

    def test_desc_substring_match(self, mock_llm):
        from lib.quality_scorers import _score_file_accuracy

        # ref_desc has tokens "important" "database"
        # out_desc has "importantdatabase" -> substring match
        ref = '[{"path": "a.txt", "desc": "database important"}]'
        out = '[{"path": "a.txt", "desc": "importantdatabase stuff"}]'
        s = _score_file_accuracy(out, make_case(reference=ref))
        # Should get high score due to substring match
        assert s.score >= 50

    def test_empty_ref_desc(self, mock_llm):
        from lib.quality_scorers import _score_file_accuracy

        ref = '[{"path": "a.txt", "desc": ""}]'
        out = '[{"path": "a.txt", "desc": "anything"}]'
        s = _score_file_accuracy(out, make_case(reference=ref))
        # ref_tokens is empty -> continue -> count stays 0 -> no items scored
        assert "no items scored" in s.failures


class TestFileFormat:
    def test_empty(self, mock_llm):
        from lib.quality_scorers import _score_file_format

        s = _score_file_format("", make_case())
        assert s.score == 0

    def test_invalid_json(self, mock_llm):
        from lib.quality_scorers import _score_file_format

        s = _score_file_format("not json", make_case())
        assert s.score == 0
        assert any("invalid" in f for f in s.failures)

    def test_not_list(self, mock_llm):
        from lib.quality_scorers import _score_file_format

        s = _score_file_format('{"a": 1}', make_case())
        assert "not a list" in s.failures

    def test_empty_array(self, mock_llm):
        from lib.quality_scorers import _score_file_format

        s = _score_file_format("[]", make_case())
        assert s.score == 30
        assert "empty array" in s.failures

    def test_full_valid(self, mock_llm):
        from lib.quality_scorers import _score_file_format

        out = '[{"path": "a.txt", "desc": "x"}]'
        s = _score_file_format(out, make_case())
        assert s.score == 100

    def test_partial_valid(self, mock_llm):
        from lib.quality_scorers import _score_file_format

        out = '[{"path": "a.txt", "desc": "x"}, {"path": "b.txt"}]'
        s = _score_file_format(out, make_case())
        assert s.score == 50
        assert any("schema" in f for f in s.failures)


class TestTaskScorers:
    def test_task_scorers_dict(self, mock_llm):
        from lib.quality_scorers import TASK_SCORERS

        assert "filename" in TASK_SCORERS
        assert "summarize" in TASK_SCORERS
        assert "file_summary" in TASK_SCORERS
        assert len(TASK_SCORERS["filename"]) == 3
        assert len(TASK_SCORERS["summarize"]) == 4
        assert len(TASK_SCORERS["file_summary"]) == 3


class TestScoreOutput:
    def test_empty_output(self, mock_llm):
        from lib.quality_scorers import score_output

        sc = score_output("", "filename", make_case())
        assert sc.dimensions == []

    def test_generic_filename(self, mock_llm):
        from lib.quality_scorers import score_output

        sc = score_output("filename.txt", "filename", make_case())
        assert len(sc.dimensions) == 3
        assert all(d.score == 0 for d in sc.dimensions)

    def test_filename_task(self, mock_llm):
        from lib.quality_scorers import score_output

        sc = score_output("hello_world.txt", "filename", make_case("hello world"))
        assert len(sc.dimensions) == 3

    def test_summarize_task(self, mock_llm):
        from lib.quality_scorers import score_output

        sc = score_output("Long output text. " * 20, "summarize", make_case("input"))
        assert len(sc.dimensions) == 4

    def test_file_summary_task(self, mock_llm):
        from lib.quality_scorers import score_output

        sc = score_output(
            '[{"path": "a.txt", "desc": "x"}]',
            "file_summary",
            make_case(reference='[{"path": "a.txt", "desc": "x"}]'),
        )
        assert len(sc.dimensions) == 3

    def test_unknown_task(self, mock_llm):
        from lib.quality_scorers import score_output

        sc = score_output("anything", "unknown_task", make_case())
        assert sc.dimensions == []
