import pytest

from eval.report import (
    compute_score_stats,
    categorize_failures,
    compute_token_estimates,
    compute_verbosity,
    compute_error_rates,
    compute_task_winners,
)


def _make_result(model, task, score, category=None, error=None, content="", messages=None):
    result = {
        "model": model,
        "results": [
            {
                "task": task,
                "quality_score": score,
                "failure_category": category,
                "error": error,
                "result": {"content": content},
            }
        ],
    }
    if messages is not None:
        result["results"][0]["messages"] = messages
    return result


class TestComputeScoreStats:
    def test_single_model_single_score(self):
        results = [_make_result("model_a", "task1", 85)]
        stats = compute_score_stats(results)
        assert stats["model_a"]["mean"] == 85.0
        assert stats["model_a"]["median"] == 85.0
        assert stats["model_a"]["stdev"] == 0
        assert stats["model_a"]["min"] == 85
        assert stats["model_a"]["max"] == 85
        assert stats["model_a"]["count"] == 1

    def test_multiple_scores(self):
        r = {"model": "m1", "results": [
            {"quality_score": 80, "task": "t1"},
            {"quality_score": 90, "task": "t2"},
            {"quality_score": 100, "task": "t3"},
        ]}
        stats = compute_score_stats([r])
        assert stats["m1"]["mean"] == 90.0
        assert stats["m1"]["median"] == 90.0
        assert stats["m1"]["min"] == 80
        assert stats["m1"]["max"] == 100
        assert stats["m1"]["count"] == 3

    def test_empty_scores_skipped(self):
        r = {"model": "m1", "results": []}
        stats = compute_score_stats([r])
        assert "m1" not in stats

    def test_two_models(self):
        r1 = {"model": "m1", "results": [{"quality_score": 90, "task": "t1"}]}
        r2 = {"model": "m2", "results": [{"quality_score": 80, "task": "t1"}]}
        stats = compute_score_stats([r1, r2])
        assert "m1" in stats
        assert "m2" in stats
        assert stats["m1"]["mean"] == 90.0
        assert stats["m2"]["mean"] == 80.0


class TestCategorizeFailures:
    def test_high_score_excluded(self):
        results = [_make_result("m1", "t1", 95, category="INFRA")]
        cats = categorize_failures(results)
        assert len(cats) == 0

    def test_low_score_included(self):
        results = [_make_result("m1", "t1", 50, category="FORMAT")]
        cats = categorize_failures(results)
        assert "FORMAT" in cats
        assert cats["FORMAT"]["count"] == 1

    def test_aggregates_multiple_models(self):
        results = [
            _make_result("m1", "t1", 30, category="FORMAT"),
            _make_result("m2", "t1", 40, category="FORMAT"),
        ]
        cats = categorize_failures(results)
        assert cats["FORMAT"]["count"] == 2
        assert sorted(cats["FORMAT"]["models"]) == ["m1", "m2"]

    def test_default_category_unknown(self):
        results = [_make_result("m1", "t1", 30, category=None)]
        cats = categorize_failures(results)
        assert None in cats

    def test_model_and_task_dedup(self):
        results = [
            _make_result("m1", "t1", 30, category="PARSE"),
            _make_result("m1", "t1", 20, category="PARSE"),
        ]
        cats = categorize_failures(results)
        assert cats["PARSE"]["count"] == 2
        assert len(cats["PARSE"]["models"]) == 1
        assert len(cats["PARSE"]["tasks"]) == 1


class TestComputeTokenEstimates:
    def test_empty_results(self):
        assert compute_token_estimates([]) == {"input": 0, "output": 0, "total": 0}

    def test_output_tokens_only(self):
        results = [{"content": "a" * 100}]
        est = compute_token_estimates(results)
        assert est["output"] == 25
        assert est["total"] == 25

    def test_input_and_output(self):
        results = [{
            "content": "a" * 100,
            "messages": [{"content": "b" * 200}],
        }]
        est = compute_token_estimates(results)
        assert est["output"] == 25
        assert est["input"] == 50
        assert est["total"] == 75

    def test_empty_content_skipped(self):
        results = [{"content": ""}]
        est = compute_token_estimates(results)
        assert est["output"] == 0

    def test_empty_messages_skipped(self):
        results = [{"content": "a" * 40, "messages": []}]
        est = compute_token_estimates(results)
        assert est["output"] == 10
        assert est["input"] == 0


class TestComputeVerbosity:
    def test_basic(self):
        results = [_make_result("m1", "t1", 90, content="hello world")]
        verb = compute_verbosity(results)
        assert verb["m1"]["t1"] == 11

    def test_none_content(self):
        r = {"model": "m1", "results": [{"task": "t1", "result": {"content": None}}]}
        verb = compute_verbosity([r])
        assert verb["m1"]["t1"] == 0

    def test_empty_content(self):
        r = {"model": "m1", "results": [{"task": "t1", "result": {"content": ""}}]}
        verb = compute_verbosity([r])
        assert verb["m1"]["t1"] == 0

    def test_multiple_tasks(self):
        r = {"model": "m1", "results": [
            {"task": "t1", "result": {"content": "abc"}},
            {"task": "t2", "result": {"content": "abcdef"}},
        ]}
        verb = compute_verbosity([r])
        assert verb["m1"]["t1"] == 3
        assert verb["m1"]["t2"] == 6


class TestComputeErrorRates:
    def test_all_success(self):
        results = [_make_result("m1", "t1", 100)]
        rates = compute_error_rates(results)
        assert rates["m1"]["success"] == 1
        assert rates["m1"]["infra"] == 0
        assert rates["m1"]["quality"] == 0
        assert rates["m1"]["success_rate"] == 1.0

    def test_infra_error(self):
        results = [_make_result("m1", "t1", 0, category="INFRA", error="Model not found")]
        rates = compute_error_rates(results)
        assert rates["m1"]["infra"] == 1
        assert rates["m1"]["success"] == 0

    def test_quality_failure(self):
        results = [_make_result("m1", "t1", 30, category="CONTENT")]
        rates = compute_error_rates(results)
        assert rates["m1"]["quality"] == 1
        assert rates["m1"]["success"] == 0

    def test_mixed_results(self):
        r = {"model": "m1", "results": [
            {"quality_score": 100, "failure_category": None, "error": None},
            {"quality_score": 0, "failure_category": "INFRA", "error": "err"},
            {"quality_score": 30, "failure_category": "FORMAT", "error": None},
        ]}
        rates = compute_error_rates([r])
        assert rates["m1"]["success"] == 1
        assert rates["m1"]["infra"] == 1
        assert rates["m1"]["quality"] == 1
        assert rates["m1"]["success_rate"] + rates["m1"]["infra_rate"] + rates["m1"]["quality_rate"] == 1.0


class TestComputeTaskWinners:
    def test_basic_winner(self):
        results = [
            _make_result("m1", "t1", 80),
            _make_result("m2", "t1", 95),
        ]
        winners = compute_task_winners(results)
        assert winners["t1"] == ("m2", 95)

    def test_tie_keeps_first(self):
        results = [
            _make_result("m1", "t1", 90),
            _make_result("m2", "t1", 90),
        ]
        winners = compute_task_winners(results)
        assert winners["t1"] == ("m1", 90)

    def test_multiple_tasks(self):
        results = [
            _make_result("m1", "t1", 80),
            _make_result("m1", "t2", 90),
            _make_result("m2", "t1", 95),
            _make_result("m2", "t2", 70),
        ]
        winners = compute_task_winners(results)
        assert winners["t1"] == ("m2", 95)
        assert winners["t2"] == ("m1", 90)
