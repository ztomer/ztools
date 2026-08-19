"""A task that every model passes must not move the ranking number.

`image_real` and `taxes_slip_qa` each produced 2 distinct values over 8 complete
runs. Before this, "gates are weighted at zero for ranking" was a convention held
in someone's head -- nothing in eval/ mentioned weights, so both gates entered
every mean at full weight.

Each test here was proven able to fail by reverting its guard first (rule #2).
"""

from eval.discrimination import (
    GATE_TASKS,
    classify,
    disagreements,
    distinct_values,
    is_gate,
    ranking_mean,
    ranking_tasks,
    scores_by_task,
)
from eval.report_core import compute_score_stats, print_score_stats


def _record(model, scores, complete=True):
    """scores: {task: score}"""
    return {
        "model": model,
        "results": [{"task": t, "quality_score": s} for t, s in scores.items()],
        "completeness": {"complete": complete},
    }


class TestTheRecordedGates:
    def test_the_two_measured_gates_are_recorded_with_evidence(self):
        assert is_gate("image_real")
        assert is_gate("taxes_slip_qa")
        for task, evidence in GATE_TASKS.items():
            assert "distinct values" in evidence, f"{task} recorded without evidence"

    def test_a_ranking_task_is_not_a_gate(self):
        assert not is_gate("taxes_yoy_narrative")
        assert ranking_tasks(["taxes_yoy_narrative", "image_real"]) == ["taxes_yoy_narrative"]


class TestRankingMean:
    def test_a_gate_does_not_move_the_ranking_mean(self):
        """The whole point: a task everyone passes cannot order anyone."""
        results = [
            {"task": "taxes_yoy_narrative", "quality_score": 40},
            {"task": "image_real", "quality_score": 100},
        ]
        assert ranking_mean(results) == 40
        # The plain mean would have been 70 -- a 30-point lift from a task that
        # every model with vision also scores 100 on.

    def test_a_run_of_only_gates_falls_back_rather_than_reporting_zero(self):
        """`ev --task image_real` must not report a passing model as a failure."""
        assert ranking_mean([{"task": "image_real", "quality_score": 100}]) == 100

    def test_no_results_is_zero_not_a_crash(self):
        assert ranking_mean([]) == 0.0
        assert ranking_mean(None) == 0.0

    def test_stats_carry_both_numbers(self):
        stats = compute_score_stats(
            [_record("m", {"taxes_yoy_narrative": 40, "image_real": 100})]
        )
        assert stats["m"]["ranking_mean"] == 40
        assert stats["m"]["mean"] == 70
        assert stats["m"]["gate_tasks"] == 1

    def test_the_ranking_mean_decides_the_printed_order(self, capsys):
        """A model that wins only on gates must not sort above one that ranks."""
        stats = compute_score_stats(
            [
                _record("gate_winner", {"taxes_yoy_narrative": 10, "image_real": 100}),
                _record("real_winner", {"taxes_yoy_narrative": 50, "image_real": 0}),
            ]
        )
        # By plain mean gate_winner (55) beats real_winner (25).
        assert stats["gate_winner"]["mean"] > stats["real_winner"]["mean"]
        print_score_stats(stats)
        out = capsys.readouterr().out
        assert out.index("real_winner") < out.index("gate_winner")

    def test_a_stats_dict_without_the_field_still_prints(self, capsys):
        """Hand-built stats predate this field; falling back beats printing 0."""
        print_score_stats(
            {"m": {"mean": 85.0, "median": 85.0, "stdev": 5.0, "min": 80, "max": 90}}
        )
        assert "85.0" in capsys.readouterr().out


class TestDerivingTheClassificationFromData:
    def _eight_models(self, per_task):
        return [
            _record(f"m{i}", {task: scores[i] for task, scores in per_task.items()})
            for i in range(8)
        ]

    def test_the_measured_shape_reproduces(self):
        """7-of-8-at-100 is a gate; a wide spread ranks."""
        data = self._eight_models(
            {
                "taxes_slip_qa": [100, 100, 100, 100, 100, 100, 100, 50],
                "taxes_yoy_narrative": [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )
        verdicts = classify(data)
        assert verdicts["taxes_slip_qa"] == "gate"
        assert verdicts["taxes_yoy_narrative"] == "ranks"

    def test_too_few_models_is_unknown_not_gate(self):
        """A narrow spread over 2 models is a sample-size artefact, not a finding."""
        data = [_record("a", {"t": 100}), _record("b", {"t": 100})]
        assert classify(data)["t"] == "unknown"

    def test_a_truncated_run_does_not_narrow_the_spread(self):
        """Otherwise an incomplete run could reclassify a ranking task as a gate."""
        data = [_record(f"m{i}", {"t": i * 10}) for i in range(4)]
        data.append(_record("wedged", {"t": 0}, complete=False))
        assert scores_by_task(data)["t"] == [0, 10, 20, 30]
        assert distinct_values(data, "t") == 4

    def test_a_gate_that_starts_ranking_is_reported(self):
        data = self._eight_models({"image_real": [0, 25, 50, 75, 100, 60, 40, 20]})
        notes = disagreements(data)
        assert any("image_real" in n and "may have started ranking" in n for n in notes)

    def test_a_ranking_task_that_saturated_is_reported(self):
        data = self._eight_models({"taxes_qa": [100] * 8})
        notes = disagreements(data)
        assert any("taxes_qa" in n and "diluting" in n for n in notes)

    def test_agreement_is_silent(self):
        data = self._eight_models(
            {
                "image_real": [100] * 8,
                "taxes_yoy_narrative": [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )
        assert disagreements(data) == []
