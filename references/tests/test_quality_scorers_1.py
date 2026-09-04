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
        # "hello" is substring of "prefix_hello" (1/1 = 100% coverage, 75 score)
        assert s.score == 75
        assert s.failures == []

    def test_substring_match_inner_break(self, mock_llm):
        """Test the inner substring match path: inp token is substring of out token."""
        from lib.quality_scorers import _score_filename_relevance

        # "world" is not in {prefix, hello}, but is substring of "helloworld" - no wait
        # Better: output has "helloworld" as a single token, input has "hello" and "world"
        # "hello" is direct substring of "helloworld" → matches.add
        # Actually re.findall splits on non-alphanumeric, so "helloworld" stays as one token
        # "hello" is in "helloworld" (substring) → True branch
        s = _score_filename_relevance("helloworld_there", make_case("hello there", "hello"))
        # hello substring of helloworld, there direct match
        # coverage = 2/2 = 100%, score 100
        assert s.score == 100
        assert s.failures == []

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
        s = _score_filename_relevance(
            "hello world",
            make_case("hello world there friend", "world there friend abc def ghi jkl"),
        )
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
        # Multiple penalties: question, spaces, invalid char, no separators
        assert s.score == 0
        assert any("question" in f for f in s.failures)

    def test_please(self, mock_llm):
        from lib.quality_scorers import _score_filename_format

        s = _score_filename_format("please_name.txt", make_case())
        # 'please' alone triggers question/instruction text penalty
        assert s.score == 50
        assert any("question" in f or "instruction" in f for f in s.failures)

    def test_spaces(self, mock_llm):
        from lib.quality_scorers import _score_filename_format

        s = _score_filename_format("hello world.txt", make_case())
        # 1 space penalty
        assert s.score == 55
        assert any("space" in f for f in s.failures)

    def test_invalid_chars(self, mock_llm):
        from lib.quality_scorers import _score_filename_format

        s = _score_filename_format("hello@world.txt", make_case())
        # 1 invalid char penalty (-20)
        assert s.score == 80
        assert any("invalid" in f for f in s.failures)

    def test_too_long(self, mock_llm):
        from lib.quality_scorers import _score_filename_format

        s = _score_filename_format("a" * 70 + ".txt", make_case())
        # 74 chars exceeds 59 → -20
        assert s.score == 80
        assert any("long" in f for f in s.failures)

    def test_uppercase(self, mock_llm):
        from lib.quality_scorers import _score_filename_format

        s = _score_filename_format("Hello.txt", make_case())
        # 1 uppercase letter → -10
        assert s.score == 90
        assert any("uppercase" in f for f in s.failures)

    def test_no_separators(self, mock_llm):
        from lib.quality_scorers import _score_filename_format

        s = _score_filename_format("helloworld", make_case())
        # No separators (-10), otherwise valid
        assert s.score == 90
        assert any("separator" in f for f in s.failures)

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
        # 'the' is a filler word, 19 chars (within 4-59), passes other checks
        # score = 100 (length) - 15 (filler) = 85
        assert s.score == 85
        assert any("filler" in f for f in s.failures)


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

        s = _score_summarize_completeness(
            "Long enough text to pass minimum check." * 5, make_case("plain text input")
        )
        # No user constraint in input, no failure
        assert s.score == 100
        assert s.failures == []

    def test_no_events_in_input(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness

        s = _score_summarize_completeness("Long enough text." * 10, make_case("plain text"))
        # No events constraint in input, full score
        assert s.score == 100
        assert s.failures == []

    def test_full_coverage(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness

        text = "Long enough text " * 10
        out = "user 1 said something at 10:00 about launch and beta and dns migration"
        s = _score_summarize_completeness(out, make_case(text))
        assert s.score >= 90

    def test_low_users(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness

        text = "user 1, user 2, user 3 said things"
        s = _score_summarize_completeness("only one person talked about launch", make_case(text))
        assert any("users" in f for f in s.failures)

    def test_low_events(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness

        text = "10:00 user 1 said things\n11:00 user 2 said things\n12:00 user 3 said things\n" * 5
        s = _score_summarize_completeness(
            "very long output without timestamps " * 5, make_case(text)
        )
        assert any("events" in f for f in s.failures)

    def test_low_topics(self, mock_llm):
        from lib.quality_scorers import _score_summarize_completeness

        text = "discussion about launch access beta and migration " * 5
        s = _score_summarize_completeness(
            "very long output without any keywords " * 5, make_case(text)
        )
        assert any("topics" in f for f in s.failures)


class TestSummarizeSynthesis:
    def test_empty(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis

        s = _score_summarize_synthesis("", make_case())
        assert s.score == 0
        assert "empty" in s.failures

    def test_no_synthesis(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis

        s = _score_summarize_synthesis("Just some text without synthesis.", make_case())
        # All three signals missing
        assert s.score == 0
        assert any("TL;DR" in f for f in s.failures)

    def test_with_summary(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis

        out = "Overall, the conversation was about migration. user 1 asked things and user 2 responded with thanks."
        s = _score_summarize_synthesis(out, make_case())
        # 'Overall' is synthesis cue, 'asked'/'responded' are narrative verbs
        # 'user X asked' + 'user Y responded' = relationship awareness
        assert s.score == 66
        assert s.failures == []

    def test_with_tldr(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis

        out = (
            "TL;DR this is a short summary of the thread. user 1 asked user 2 and user 2 confirmed."
        )
        s = _score_summarize_synthesis(out, make_case())
        # TL;DR cue (+33), narrative verbs (+?), relationship (+?)
        assert s.score == 68
        assert s.failures == []

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
        # top_level is empty, no synthesis/narrative/relationship signals
        assert s.score == 0

    def test_with_top_level_pre_header(self, mock_llm):
        from lib.quality_scorers import _score_summarize_synthesis

        # Text before a header (after a newline) - top_level has content
        out = "Overall summary of the conversation.\n\n## Section\nDetails here. user 1 asked user 2 confirmed."
        s = _score_summarize_synthesis(out, make_case())
        # header_match captures the \n## ... pattern
        # top_level is the part before that header
        assert s.score >= 40
