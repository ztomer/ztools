import json

import pytest

from eval.benchmark_quality import score_filename, score_summarize, score_file_summary


FILENAME_CASES = [
    {"input": "", "expected_keywords": [], "human_score_expectation": 0, "description": "x"},
    {"input": "x", "expected_keywords": ["login"], "human_score_expectation": 0, "description": "x"},
]


class TestScoreFilename:
    def test_empty_output(self):
        score, failures = score_filename("", FILENAME_CASES[0])
        assert score == 0
        assert "empty" in failures

    def test_generic_name(self):
        score, failures = score_filename("screenshot.png", FILENAME_CASES[0])
        assert score == 0
        assert any("generic" in f for f in failures)

    def test_generic_unnamed(self):
        score, failures = score_filename("unnamed", FILENAME_CASES[0])
        assert score == 0
        assert any("generic" in f for f in failures)

    def test_keyword_match_all(self):
        case = {
            "input": "", "expected_keywords": ["login", "error", "invalid"],
            "human_score_expectation": 100, "description": "x",
        }
        score, failures = score_filename("login_error_invalid_creds.png", case)
        # All 3 keywords present, valid format
        assert score == 100
        assert len(failures) == 0

    def test_keyword_match_partial(self):
        case = {
            "input": "", "expected_keywords": ["login", "error", "invalid", "credential"],
            "human_score_expectation": 100, "description": "x",
        }
        score, failures = score_filename("login_error.png", case)
        # 2/4 keywords matched, but combined with format score = 100
        # (keyword coverage is partial, but everything else is good)
        assert score == 100
        assert failures == []

    def test_keyword_no_match(self):
        case = {
            "input": "", "expected_keywords": ["summer", "festival", "park"],
            "human_score_expectation": 0, "description": "x",
        }
        score, failures = score_filename("random_text.txt", case)
        assert score == 50
        assert any("no keywords matched" in f for f in failures)

    def test_keyword_ratio_30(self):
        """ratio >= 0.3 — line 134 kw_score = 30."""
        case = {
            "input": "", "expected_keywords": ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"],
            "human_score_expectation": 0, "description": "x",
        }
        # 3/10 = 0.3 (alpha, beta, gamma all in name)
        score, failures = score_filename("alpha_beta_gamma.png", case)
        # kw_score=30, fmt_score=50, total=80
        assert score == 80

    def test_keyword_ratio_low(self):
        """ratio < 0.3 with some matches — line 136 kw_score = 15."""
        # Use keywords with no substring overlap with the others or with "alpha"
        case = {
            "input": "", "expected_keywords": ["alpha", "lemon", "tiger", "ocean", "river", "mountain", "desert", "forest", "cloud", "storm"],
            "human_score_expectation": 0, "description": "x",
        }
        # 1/10 = 0.1 (alpha only, "lemon" is not in "alpha_lemon"...)
        # Wait: "lemon" IS in "alpha_lemon.png"!
        # Use output that only contains 1 keyword
        score, failures = score_filename("alpha.png", case)
        # 1/10 = 0.1, matches > 0, kw_score=15, fmt_score=50, total=65
        assert score == 65

    def test_too_long(self):
        case = {
            "input": "", "expected_keywords": ["test"],
            "human_score_expectation": 0, "description": "x",
        }
        long_name = "a" * 70 + ".txt"
        score, failures = score_filename(long_name, case)
        assert any("too long" in f for f in failures)

    def test_has_spaces(self):
        case = {
            "input": "", "expected_keywords": ["login"],
            "human_score_expectation": 0, "description": "x",
        }
        score, failures = score_filename("login error.png", case)
        assert any("has spaces" in f for f in failures)

    def test_not_lowercase_penalty(self):
        case = {
            "input": "", "expected_keywords": ["login"],
            "human_score_expectation": 0, "description": "x",
        }
        score, failures = score_filename("LoginError.png", case)
        assert any("not lowercase" in f for f in failures)

    def test_question_text_penalty(self):
        case = {
            "input": "", "expected_keywords": ["login"],
            "human_score_expectation": 0, "description": "x",
        }
        score, failures = score_filename("what is this login error.png?", case)
        assert any("invalid format" in f for f in failures) or any("question" in f for f in failures)

    def test_invalid_chars_penalty(self):
        case = {
            "input": "", "expected_keywords": ["test"],
            "human_score_expectation": 0, "description": "x",
        }
        score, failures = score_filename("test@file#.png", case)
        assert any("invalid" in f.lower() for f in failures)


class TestScoreSummarize:
    SUMMARY_50 = "Short."
    SUMMARY_150 = "A" * 150
    SUMMARY_300 = "A" * 300
    SUMMARY_500 = "A" * 500

    def test_empty_output(self):
        score, failures = score_summarize("", {"expected_users": [], "expected_topics": []})
        assert score == 0
        assert any("empty" in f or "short" in f for f in failures)

    def test_too_short(self):
        score, failures = score_summarize(self.SUMMARY_50, {"expected_users": [], "expected_topics": []})
        assert score == 0

    def test_minimal_length(self):
        case = {"expected_users": ["@user1"], "expected_topics": ["launch"]}
        score, failures = score_summarize("Some text about the launch here. " * 5, case)
        assert score > 0

    def test_user_mentions_scoring(self):
        case = {"expected_users": ["@user1", "@user2", "@user3", "@user4"], "expected_topics": ["launch"]}
        text = (
            "## Summary\n"
            "Product launch this week.\n"
            "- User1 posted about it\n"
            "- User2 replied with feedback\n"
            "- User3 asked about access\n"
            "- User4 shared their experience\n"
            "- @User5 provided a review\n"
        )
        score, failures = score_summarize(text, case)
        # Users: 5/4 = full, topic launch in text, structure good
        # user_score=37, topic_score=20, structure=10 = 67
        assert score == 67
        assert "topics" in failures[0]

    def test_topic_coverage(self):
        case = {"expected_users": ["@user1"], "expected_topics": ["launch", "access", "beta", "feedback"]}
        text = (
            "## Launch Announcement\n"
            "The product launch was announced this week.\n"
            "Early access is now available for beta testers.\n"
            "We received great feedback from the community.\n"
        )
        score, failures = score_summarize(text, case)
        # Users: 0/1, topics: 4/4, structure: header
        # topic_score=20, structure=10, partial user = 55
        assert score == 55
        assert any("users" in f for f in failures)

    def test_structure_with_headers_and_bullets(self):
        case = {"expected_users": ["@user1"], "expected_topics": ["launch"]}
        text = "## Summary\n- Item 1\n- Item 2\n" * 20
        score, failures = score_summarize(text, case)
        # 0 users + 0 topics (case-incompatible markers), 20 header + 30 bullet
        assert score == 50
        assert any("users" in f for f in failures)

    def test_structure_bullets_only(self):
        case = {"expected_users": ["@user1"], "expected_topics": ["launch"]}
        text = "- Item 1\n- Item 2\n" * 20
        score, failures = score_summarize(text, case)
        # 3 lines per topic recognition + 30 bullet only
        assert score == 33
        assert any("users" in f for f in failures)

    def test_structure_no_headers_no_bullets_long(self):
        """len >= 200, no headers, no bullets — line 192 struct_score = 10."""
        case = {"expected_users": ["@user1"], "expected_topics": ["launch"]}
        text = "A " * 150  # ~300 chars, no headers, no bullets
        score, failures = score_summarize(text, case)
        # 300 chars length (10), no headers, no bullets, no markers
        assert score == 28
        assert any("users" in f for f in failures)

    def test_length_depth_scoring(self):
        case = {"expected_users": ["@user1", "@user2"], "expected_topics": ["launch", "access"]}
        long_text = "## Summary\n- User1: announced launch\n- User2: asked about access\n" * 30
        score, failures = score_summarize(long_text, case)
        assert score > 0

    def test_no_user_mentions_triggers_failure(self):
        case = {"expected_users": ["@user1", "@user2", "@user3"], "expected_topics": ["launch"]}
        text = "## Summary\nGeneral discussion about the launch.\n" * 10
        score, failures = score_summarize(text, case)
        assert any("users" in f for f in failures)


class TestScoreFileSummary:
    def test_empty_output(self):
        score, failures = score_file_summary("", {"expected_paths": ["eval_lib.py"]})
        assert score == 0
        assert "empty" in failures

    def test_invalid_json(self):
        score, failures = score_file_summary("not json", {"expected_paths": ["eval_lib.py"]})
        assert score == 0
        assert "invalid JSON" in failures

    def test_not_a_list(self):
        score, failures = score_file_summary('{"path": "test.py"}', {"expected_paths": ["test.py"]})
        assert score == 0
        assert "not a list" in failures

    def test_all_paths_matched(self):
        case = {"expected_paths": ["eval_lib.py", "validators.py", "config.py", "osaurus_lib.py"]}
        output = json.dumps([
            {"path": "eval_lib.py", "desc": "model evaluation functions"},
            {"path": "validators.py", "desc": "validation logic for JSON output"},
            {"path": "config.py", "desc": "configuration management"},
            {"path": "osaurus_lib.py", "desc": "LLM API client library"},
        ])
        score, failures = score_file_summary(output, case)
        # 4/4 paths match (40), all 4 detailed (40), real paths (20)
        assert score == 100
        assert len(failures) == 0

    def test_partial_path_match_penalty(self):
        case = {"expected_paths": ["eval_lib.py", "validators.py", "config.py", "osaurus_lib.py"]}
        output = json.dumps([
            {"path": "eval_lib.py", "desc": "model evaluation functions"},
            {"path": "unknown.py", "desc": "mystery script"},
        ])
        score, failures = score_file_summary(output, case)
        assert score < 80

    def test_path_ratio_50(self):
        """ratio >= 0.5 — line 235 path_score = 30."""
        case = {"expected_paths": ["eval_lib.py", "validators.py", "config.py", "osaurus_lib.py"]}
        output = json.dumps([
            {"path": "eval_lib.py", "desc": "model evaluation functions"},
            {"path": "validators.py", "desc": "validation logic for output"},
            {"path": "unknown.py", "desc": "mystery script"},
        ])
        # 2/4 = 0.5, path_score=30, desc_score=45 (3 meaningful), total=75
        score, failures = score_file_summary(output, case)
        assert score == 75

    def test_generic_descriptions_penalty(self):
        case = {"expected_paths": ["eval_lib.py"]}
        output = json.dumps([
            {"path": "eval_lib.py", "desc": "pe"},
            {"path": "validators.py", "desc": "system file"},
            {"path": "config.py", "desc": "configuration file"},
            {"path": "osaurus_lib.py", "desc": "personal document"},
        ])
        score, failures = score_file_summary(output, case)
        assert any("meaningful" in f for f in failures)

    def test_no_paths_matched_penalty(self):
        case = {"expected_paths": ["eval_lib.py", "validators.py", "config.py", "osaurus_lib.py"]}
        output = json.dumps([
            {"path": "unrelated.py", "desc": "some other module"},
        ])
        score, failures = score_file_summary(output, case)
        assert score < 50
        assert any("paths" in f for f in failures)

    def test_meaningful_descriptions_bonus(self):
        case = {"expected_paths": ["eval_lib.py", "validators.py"]}
        output = json.dumps([
            {"path": "eval_lib.py", "desc": "model evaluation functions and utilities"},
            {"path": "validators.py", "desc": "JSON and text validation logic"},
        ])
        score, failures = score_file_summary(output, case)
        # 2/2 paths match, 2/2 have meaningful descs (long, has verbs/nouns)
        # path_score=40 + desc_score=40 = 80
        assert score == 80
        assert failures == []
