"""Tests for lib.quality_entry regression mode."""

from unittest.mock import patch


def _make_scorecard(model, task, case_id, dims, elapsed=1.0):
    from lib.quality_models import Score, ScoreCard

    scores = [Score(name, score, weight) for name, score, weight in dims]
    return ScoreCard(model, task, case_id, scores, "", elapsed)


class TestRegressionOnly:
    def test_reconstructs_scorecard_with_correct_weights(self):
        """Baseline scores must be reconstructed with scorers' weights, not 0.0."""
        baseline = {
            "model-a::filename::test_case_1": {
                "composite": 85.0,
                "dimensions": {"Relevance": 90.0, "Format": 80.0, "Conciseness": 85.0},
                "elapsed": 1.5,
            }
        }
        import lib.quality_entry as qe

        with (
            patch("lib.quality_entry.load_baseline", return_value=baseline),
            patch("lib.quality_entry.compare_to_baseline") as mock_cmp,
            patch("sys.argv", ["quality_entry", "--regression-only"]),
        ):
            mock_cmp.return_value = []
            qe.main()

        scorecards = mock_cmp.call_args[0][0]
        assert len(scorecards) == 1
        sc = scorecards[0]
        for d in sc.dimensions:
            assert d.weight > 0.0, f"Dimension {d.name} has zero weight"
        expected = 90 * 0.40 + 80 * 0.35 + 85 * 0.25
        assert sc.composite == expected, f"Composite should be {expected}, got {sc.composite}"

    def test_missing_baseline_shows_message(self):
        import lib.quality_entry as qe

        with (
            patch("lib.quality_entry.load_baseline", return_value={}),
            patch.object(qe, "print") as mock_print,
            patch("sys.argv", ["quality_entry", "--regression-only"]),
        ):
            qe.main()
            output = "".join(c[0][0] for c in mock_print.call_args_list)
            assert "No baseline found" in output

    def test_weights_match_scorer_definitions(self):
        """Dimension weights should match those defined in quality_scorers."""
        from lib.quality_models import TestCase
        from lib.quality_scorers import TASK_SCORERS

        expected = {}
        for task, scorers in TASK_SCORERS.items():
            for scorer_fn in scorers:
                dummy = scorer_fn("", TestCase(task, "", "", ""))
                expected.setdefault(task, {})[dummy.name] = dummy.weight

        assert "filename" in expected
        assert expected["filename"]["Relevance"] == 0.40
        assert expected["filename"]["Format"] == 0.35
        assert expected["filename"]["Conciseness"] == 0.25

    def test_compare_to_baseline_reports_regression(self):
        """When composite differs from baseline, a warning should be emitted."""
        prev = _make_scorecard(
            "m",
            "filename",
            "c1",
            [
                ("Relevance", 90, 0.40),
                ("Format", 80, 0.35),
                ("Conciseness", 85, 0.25),
            ],
        )
        curr = _make_scorecard(
            "m",
            "filename",
            "c1",
            [
                ("Relevance", 50, 0.40),
                ("Format", 80, 0.35),
                ("Conciseness", 85, 0.25),
            ],
        )

        from lib.quality_report import compare_to_baseline

        with patch("lib.quality_report.load_baseline") as mock_load:
            mock_load.return_value = {
                "m::filename::c1": {
                    "composite": prev.composite,
                    "dimensions": {d.name: d.score for d in prev.dimensions},
                    "elapsed": prev.elapsed,
                }
            }
            warnings = compare_to_baseline([curr])
        assert any("Relevance" in w for w in warnings), (
            f"Expected Relevance regression warning, got: {warnings}"
        )
