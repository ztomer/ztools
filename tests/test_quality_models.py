"""Tests for lib.quality_models."""
import pytest


class TestScore:
    def test_score_creation(self):
        from lib.quality_models import Score
        s = Score(name="x", score=80.0, weight=1.0)
        assert s.name == "x"
        assert s.score == 80.0
        assert s.weight == 1.0
        assert s.failures == []

    def test_score_with_failures(self):
        from lib.quality_models import Score
        s = Score(name="x", score=50.0, weight=1.0, failures=["bad"])
        assert s.failures == ["bad"]

    def test_score_weighted(self):
        from lib.quality_models import Score
        s = Score(name="x", score=80.0, weight=0.5)
        assert s.weighted == 40.0


class TestScoreCard:
    def test_scorecard_creation(self):
        from lib.quality_models import ScoreCard
        sc = ScoreCard(model="m", task="json", case_id="c1", dimensions=[], output="o")
        assert sc.model == "m"
        assert sc.task == "json"
        assert sc.case_id == "c1"
        assert sc.dimensions == []
        assert sc.output == "o"
        assert sc.elapsed == 0.0

    def test_composite_no_dimensions(self):
        from lib.quality_models import ScoreCard
        sc = ScoreCard(model="m", task="t", case_id="c", dimensions=[], output="o")
        assert sc.composite == 0.0

    def test_composite_with_dimensions(self):
        from lib.quality_models import ScoreCard, Score
        sc = ScoreCard(
            model="m", task="t", case_id="c",
            dimensions=[
                Score(name="a", score=80.0, weight=1.0),
                Score(name="b", score=60.0, weight=0.5),
            ],
            output="o",
        )
        # 80*1.0 + 60*0.5 = 110
        assert sc.composite == 110.0

    def test_total_weight(self):
        from lib.quality_models import ScoreCard, Score
        sc = ScoreCard(
            model="m", task="t", case_id="c",
            dimensions=[
                Score(name="a", score=80.0, weight=1.0),
                Score(name="b", score=60.0, weight=0.5),
            ],
            output="o",
        )
        assert sc.total_weight == 1.5

    def test_total_weight_no_dimensions(self):
        from lib.quality_models import ScoreCard
        sc = ScoreCard(model="m", task="t", case_id="c", dimensions=[], output="o")
        assert sc.total_weight == 0

    def test_report_no_failures(self):
        from lib.quality_models import ScoreCard, Score
        sc = ScoreCard(
            model="m", task="json", case_id="c1",
            dimensions=[Score(name="validity", score=80.0, weight=1.0)],
            output="o",
            elapsed=1.5,
        )
        result = sc.report()
        assert "json" in result
        assert "validity" in result
        assert "80.0%" in result
        assert "1.5s" in result

    def test_report_with_failures(self):
        from lib.quality_models import ScoreCard, Score
        sc = ScoreCard(
            model="m", task="json", case_id="c1",
            dimensions=[Score(name="validity", score=50.0, weight=1.0, failures=["bad"])],
            output="o",
            elapsed=1.5,
        )
        result = sc.report()
        assert "FAIL" in result
        assert "bad" in result


class TestTestCase:
    def test_testcase_creation(self):
        from lib.quality_models import TestCase
        tc = TestCase(task="json", input_text="x", reference="y", description="z")
        assert tc.task == "json"
        assert tc.input_text == "x"
        assert tc.reference == "y"
        assert tc.description == "z"


class TestStrHelpers:
    def test_str_with_value(self):
        from lib.quality_models import _str
        assert _str("hello") == "hello"
        assert _str(42) == "42"

    def test_str_with_none(self):
        from lib.quality_models import _str
        assert _str(None) == ""

    def test_lower(self):
        from lib.quality_models import _lower
        assert _lower("HELLO") == "hello"

    def test_lower_none(self):
        from lib.quality_models import _lower
        assert _lower(None) == ""

    def test_lower_int(self):
        from lib.quality_models import _lower
        assert _lower(42) == "42"
