"""Tests for lib.quality_scorers - all scoring functions."""
import json
import pytest
from unittest.mock import MagicMock
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


class TestFilenameRelevance:
    def test_empty(self, mock_llm):
        from lib.quality_scorers import _score_filename_relevance
        s = _score_filename_relevance("", make_case("hello world", "hello"))
        assert s.score == 0
        assert "empty" in s.failures

    def test_full_match(self, mock_llm):
        from lib.quality_scorers import _score_filename_relevance
        s = _score_filename_relevance("hello world", make_case("hello world there", "hello"))
        assert s.score == 100

    def test_partial_match_60(self, mock_llm):
        from lib.quality_scorers import _score_filename_relevance
        s = _score_filename_relevance("hello there", make_case("hello there friend", "hello"))
        assert s.score == 100  # 100% tokens covered

    def test_low_coverage(self, mock_llm):
        from lib.quality_scorers import _score_filename_relevance
        s = _score_filename_relevance("xyz", make_case("hello world there", "hello"))
        assert s.score == 25
        assert any("covered" in f for f in s.failures)

    def test_mid_coverage(self, mock_llm):
        from lib.quality_scorers import _score_filename_relevance
        s = _score_filename_relevance("hello abc def", make_case("hello abc xyz def qrs", "hello"))
        # 3/4 = 0.75 -> 100
        assert s.score == 100

    def test_substring_match(self, mock_llm):
        from lib.quality_scorers import _score_filename_relevance
        s = _score_filename_relevance("prefix_hello", make_case("hello world", "hello"))
        # "hello" is substring of "prefix_hello"
        assert s.score >= 75

    def test_substring_match_inner_break(self, mock_llm):
        """Test the inner substring match path: inp token is substring of out token."""
        from lib.quality_scorers import _score_filename_relevance
        # "world" is not in {prefix, hello}, but is substring of "helloworld" - no wait
        # Better: output has "helloworld" as a single token, input has "hello" and "world"
        # "hello" is direct substring of "helloworld" → matches.add
        # Actually re.findall splits on non-alphanumeric, so "helloworld" stays as one token
        # "hello" is in "helloworld" (substring) → True branch
        s = _score_filename_relevance("helloworld_there", make_case("hello there", "hello"))
        # hello is substring of helloworld
        # there is direct match
        assert s.score >= 75

    def test_mid_score_50(self, mock_llm):
        from lib.quality_scorers import _score_filename_relevance
        # 2/5 = 0.4 -> 75. 1/5 = 0.2 -> 50
        s = _score_filename_relevance("hello", make_case("hello world there friend", "world"))
        # coverage 0.25 (between 0.2 and 0.4) -> score = 50
        assert s.score == 50

    def test_no_input_tokens(self, mock_llm):
        from lib.quality_scorers import _score_filename_relevance
        s = _score_filename_relevance("anything", make_case("the a an", "the"))
        # All stopwords removed, no tokens
        # coverage = 0, so score falls to 25 branch
        assert s.score == 25
        assert any("covered" in f for f in s.failures)

    def test_missing_key_concepts(self, mock_llm):
        from lib.quality_scorers import _score_filename_relevance
        # Output has hello (high coverage), reference has world+there+friend+abc+def
        # but output only has hello -> ref_matches is empty -> cap doesn't apply
        # Need to have at least 1 ref token present, but < 40% of ref tokens
        s = _score_filename_relevance("hello world",
                                       make_case("hello world there friend",
                                                 "world there friend abc def ghi jkl"))
        # 2 of 6 ref tokens present, < 0.4 -> cap at 60
        assert s.score <= 60
        assert any("missing" in f for f in s.failures)


class TestFilenameFormat:
    def test_empty(self, mock_llm):
        from lib.quality_scorers import _score_filename_format
        s = _score_filename_format("", make_case())
        assert s.score == 0
        assert "empty" in s.failures

    def test_generic(self, mock_llm):
        from lib.quality_scorers import _score_filename_format
        s = _score_filename_format("filename.txt", make_case())
        assert s.score == 0
        assert any("generic" in f for f in s.failures)

    def test_question(self, mock_llm):
        from lib.quality_scorers import _score_filename_format
        s = _score_filename_format("what is this?", make_case())
        assert s.score <= 50
        assert any("question" in f for f in s.failures)

    def test_please(self, mock_llm):
        from lib.quality_scorers import _score_filename_format
        s = _score_filename_format("please_name.txt", make_case())
        assert s.score <= 50

    def test_spaces(self, mock_llm):
        from lib.quality_scorers import _score_filename_format
        s = _score_filename_format("hello world.txt", make_case())
        assert s.score <= 60
        assert any("space" in f for f in s.failures)

    def test_invalid_chars(self, mock_llm):
        from lib.quality_scorers import _score_filename_format
        s = _score_filename_format("hello@world.txt", make_case())
        assert s.score <= 80
        assert any("invalid" in f for f in s.failures)

    def test_too_long(self, mock_llm):
        from lib.quality_scorers import _score_filename_format
        s = _score_filename_format("a" * 70 + ".txt", make_case())
        assert s.score <= 80

    def test_uppercase(self, mock_llm):
        from lib.quality_scorers import _score_filename_format
        s = _score_filename_format("Hello.txt", make_case())
        assert s.score <= 90

    def test_no_separators(self, mock_llm):
        from lib.quality_scorers import _score_filename_format
        s = _score_filename_format("helloworld", make_case())
        assert s.score <= 90

    def test_perfect(self, mock_llm):
        from lib.quality_scorers import _score_filename_format
        s = _score_filename_format("hello_world.txt", make_case())
        assert s.score == 100


class TestFilenameConciseness:
    def test_empty(self, mock_llm):
        from lib.quality_scorers import _score_filename_conciseness
        s = _score_filename_conciseness("", make_case())
        assert s.score == 0
        assert "empty" in s.failures

    def test_question(self, mock_llm):
        from lib.quality_scorers import _score_filename_conciseness
        s = _score_filename_conciseness("what?", make_case())
        assert s.score == 0

    def test_space(self, mock_llm):
        from lib.quality_scorers import _score_filename_conciseness
        s = _score_filename_conciseness("hello world", make_case())
        assert s.score == 10
        assert any("space" in f for f in s.failures)

    def test_too_short(self, mock_llm):
        from lib.quality_scorers import _score_filename_conciseness
        s = _score_filename_conciseness("abc", make_case())
        assert s.score == 50
        assert any("short" in f for f in s.failures)

    def test_very_short(self, mock_llm):
        from lib.quality_scorers import _score_filename_conciseness
        s = _score_filename_conciseness("hello", make_case())
        assert s.score == 75

    def test_optimal(self, mock_llm):
        from lib.quality_scorers import _score_filename_conciseness
        s = _score_filename_conciseness("hello-world.txt", make_case())
        assert s.score == 100

    def test_slightly_long(self, mock_llm):
        from lib.quality_scorers import _score_filename_conciseness
        s = _score_filename_conciseness("a" * 50 + ".txt", make_case())
        assert s.score == 80
        assert any("long" in f for f in s.failures)

    def test_too_long(self, mock_llm):
        from lib.quality_scorers import _score_filename_conciseness
        s = _score_filename_conciseness("a" * 70 + ".txt", make_case())
        assert s.score == 40
        assert any("long" in f for f in s.failures)

    def test_filler_words(self, mock_llm):
        from lib.quality_scorers import _score_filename_conciseness
        s = _score_filename_conciseness("the-hello-world.txt", make_case())
        assert s.score <= 100
        # 'the' is a filler word
        assert any("filler" in f for f in s.failures) or s.score < 100


class TestSummarizeCompleteness:
    def test_empty(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness
        s = _score_summarize_completeness("", make_case("user 1: hello"))
        assert s.score == 0
        assert any("empty" in f for f in s.failures)

    def test_too_short(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness
        s = _score_summarize_completeness("too short", make_case("user 1: hello"))
        assert s.score == 0

    def test_no_users_in_input(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness
        s = _score_summarize_completeness("Long enough text to pass minimum check." * 5,
                                          make_case("plain text input"))
        # user_ratio defaults to 1, no users constraint
        assert s.score >= 0

    def test_no_events_in_input(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness
        s = _score_summarize_completeness("Long enough text." * 10,
                                          make_case("plain text"))
        assert s.score >= 0

    def test_full_coverage(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness
        text = "Long enough text " * 10
        out = "user 1 said something at 10:00 about launch and beta and dns migration"
        s = _score_summarize_completeness(out, make_case(text))
        assert s.score >= 90

    def test_low_users(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness
        text = "user 1, user 2, user 3 said things"
        s = _score_summarize_completeness("only one person talked about launch",
                                          make_case(text))
        assert any("users" in f for f in s.failures)

    def test_low_events(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness
        text = "10:00 user 1 said things\n11:00 user 2 said things\n12:00 user 3 said things\n" * 5
        s = _score_summarize_completeness("very long output without timestamps " * 5,
                                          make_case(text))
        assert any("events" in f for f in s.failures)

    def test_low_topics(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness
        text = "discussion about launch access beta and migration " * 5
        s = _score_summarize_completeness("very long output without any keywords " * 5,
                                          make_case(text))
        assert any("topics" in f for f in s.failures)


class TestSummarizeSynthesis:
    def test_empty(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis
        s = _score_summarize_synthesis("", make_case())
        assert s.score == 0
        assert "empty" in s.failures

    def test_no_synthesis(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis
        s = _score_summarize_synthesis("Just some text without synthesis.",
                                       make_case())
        assert s.score < 40

    def test_with_summary(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis
        out = "Overall, the conversation was about migration. user 1 asked things and user 2 responded with thanks."
        s = _score_summarize_synthesis(out, make_case())
        assert s.score >= 40

    def test_with_tldr(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis
        out = "TL;DR this is a short summary of the thread. user 1 asked user 2 and user 2 confirmed."
        s = _score_summarize_synthesis(out, make_case())
        assert s.score >= 40

    def test_narrative_verbs(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis
        out = "The user asked, the team responded, they confirmed. " * 3
        s = _score_summarize_synthesis(out, make_case())
        # Multiple narrative verbs -> higher narrative_score
        assert "no narrative" not in " ".join(s.failures)

    def test_user_action(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis
        out = "user 1 asked a question\nuser 2 confirmed the answer\nuser 3 thanked everyone"
        s = _score_summarize_synthesis(out, make_case())
        # User actions present
        assert "no relationship" not in " ".join(s.failures)

    def test_no_top_level(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis
        out = "## Section\nContent here"  # starts with header
        s = _score_summarize_synthesis(out, make_case())
        # top_level becomes empty
        assert s.score >= 0

    def test_with_top_level_pre_header(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis
        # Text before a header (after a newline) - top_level has content
        out = "Overall summary of the conversation.\n\n## Section\nDetails here. user 1 asked user 2 confirmed."
        s = _score_summarize_synthesis(out, make_case())
        # header_match captures the \n## ... pattern
        # top_level is the part before that header
        assert s.score >= 40


class TestSummarizeStructure:
    def test_empty(self, mock_llm):
        from lib.quality_scorers import _score_summarize_structure
        s = _score_summarize_structure("", make_case())
        assert s.score == 0

    def test_no_structure(self, mock_llm):
        from lib.quality_scorers import _score_summarize_structure
        s = _score_summarize_structure("Just a wall of text without any structure.",
                                       make_case())
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
        assert s.score <= 50

    def test_too_long(self, mock_llm):
        from lib.quality_scorers import _score_summarize_structure
        out = "## Title\n- Bullet\n" + ("filler " * 500)
        s = _score_summarize_structure(out, make_case())
        assert s.score <= 80


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
        s = _score_summarize_specificity("Long output without any user mention.",
                                          make_case("user 1 said thing"))
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
        s = _score_file_completeness('[]', make_case(reference="[]"))
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
        out = '[]'
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
        sc = score_output('[{"path": "a.txt", "desc": "x"}]', "file_summary",
                          make_case(reference='[{"path": "a.txt", "desc": "x"}]'))
        assert len(sc.dimensions) == 3

    def test_unknown_task(self, mock_llm):
        from lib.quality_scorers import score_output
        sc = score_output("anything", "unknown_task", make_case())
        assert sc.dimensions == []
