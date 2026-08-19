"""A truncated run must not be quotable as a score.

Both abandon paths in `eval/run.py` already print "this is not a quality result".
These tests pin that the JSON on disk and the printed tables agree with the
console, because the console is the one surface the next session does not read.

Every test here was proven able to fail by reverting the guard it covers before
trusting it green (repo rule #2).
"""

import json

import pytest
from eval.completeness import assess, expected_task_names, is_complete, is_runnable
from eval.report_core import compute_score_stats, print_cross_model_comparison, print_score_stats
from eval.report_history import load_historical_stats, save_historical_results


def _tasks(n):
    return {f"task_{i}": {"messages": [{"role": "user", "content": "x"}]} for i in range(n)}


def _results(names, score=100, category=None):
    out = []
    for name in names:
        entry = {"task": name, "quality_score": score, "time": 1.0}
        if category:
            entry["failure_category"] = category
        out.append(entry)
    return out


class TestAssess:
    def test_a_full_run_is_complete(self):
        tasks = _tasks(3)
        meta = assess(tasks, _results(list(tasks)))
        assert meta["complete"] is True
        assert meta["expected"] == 3
        assert meta["completed"] == 3
        assert meta["missing"] == []
        assert meta["reason"] == ""

    def test_the_ornith_shape_is_incomplete(self):
        """11 of 30, which is the run that must never be quoted."""
        tasks = _tasks(30)
        meta = assess(tasks, _results(list(tasks)[:11], score=25, category="TIMEOUT"))
        assert meta["complete"] is False
        assert meta["expected"] == 30
        assert meta["completed"] == 11
        assert len(meta["missing"]) == 19
        assert "TIMEOUT" in meta["reason"]
        assert "19 not run" in meta["reason"]

    def test_a_run_that_completed_nothing_says_so(self):
        meta = assess(_tasks(4), [])
        assert meta["complete"] is False
        assert "no task completed" in meta["reason"]

    def test_a_task_with_no_messages_is_not_expected(self):
        """run_eval skips these, so counting them would call every run truncated."""
        tasks = dict(_tasks(2))
        tasks["unrunnable"] = {"validator": object()}
        assert expected_task_names(tasks) == ["task_0", "task_1"]
        assert assess(tasks, _results(["task_0", "task_1"]))["complete"] is True

    def test_is_runnable_needs_the_messages_key(self):
        assert is_runnable({"messages": []}) is True
        assert is_runnable({"validator": None}) is False
        assert is_runnable(None) is False

    def test_a_record_with_no_metadata_reads_as_complete(self):
        """Every pre-existing record predates this module; they are not truncated."""
        assert is_complete({"model": "m", "results": []}) is True
        assert is_complete({"completeness": {"complete": False}}) is False
        assert is_complete({"completeness": {"complete": True}}) is True


class TestStatsCarryTheVerdict:
    def test_an_incomplete_run_is_marked_in_the_stats(self):
        all_results = [
            {"model": "good", "results": _results(["a", "b"]), "completeness": {"complete": True}},
            {
                "model": "truncated",
                "results": _results(["a"], score=62),
                "completeness": {"complete": False, "reason": "abandoned"},
            },
        ]
        stats = compute_score_stats(all_results)
        assert stats["good"]["complete"] is True
        assert stats["truncated"]["complete"] is False

    def test_the_printed_mean_says_partial(self, capsys):
        """The mean is the number that gets quoted six weeks later."""
        stats = {
            "bonsai": {
                "mean": 62.0,
                "median": 62.0,
                "stdev": 0,
                "min": 62,
                "max": 62,
                "count": 19,
                "complete": False,
            }
        }
        print_score_stats(stats)
        out = capsys.readouterr().out
        assert "partial" in out

    def test_a_complete_mean_is_not_marked(self, capsys):
        stats = {
            "bonsai": {
                "mean": 79.0,
                "median": 79.0,
                "stdev": 0,
                "min": 79,
                "max": 79,
                "count": 30,
                "complete": True,
            }
        }
        print_score_stats(stats)
        assert "partial" not in capsys.readouterr().out


class TestHistoryRefusesToAverageTruncatedRuns:
    def test_truncated_entries_are_written_but_not_averaged(self, tmp_path):
        all_results = [
            {
                "model": "ornith-1.0-9b-mxfp8",
                "results": _results(["a", "b"], score=25),
                "completeness": {"complete": False, "reason": "abandoned after 11 task(s)"},
            }
        ]
        save_historical_results(all_results, {}, {}, eval_dir=tmp_path)

        written = json.loads((tmp_path / "eval_history.json").read_text())
        entries = written["ornith-1.0-9b-mxfp8"]
        assert len(entries) == 2, "entries must be KEPT -- the task that ran, ran"
        assert all(e["complete"] is False for e in entries)

        stats = load_historical_stats(eval_dir=tmp_path)
        assert stats == {}, "nothing countable, so no mean at all"

    def test_a_complete_run_still_averages(self, tmp_path):
        all_results = [
            {
                "model": "muse",
                "results": _results(["a", "b"], score=90),
                "completeness": {"complete": True},
            }
        ]
        save_historical_results(all_results, {}, {}, eval_dir=tmp_path)
        stats = load_historical_stats(eval_dir=tmp_path)
        assert stats["muse"]["mean"] == 90
        assert stats["muse"]["runs"] == 2
        assert stats["muse"]["excluded"] == 0

    def test_the_mean_ignores_the_truncated_half(self, tmp_path):
        """The exact contamination shape: easy tasks from a wedged run drag the mean."""
        save_historical_results(
            [
                {
                    "model": "ornith",
                    "results": _results(["a", "b"], score=100),
                    "completeness": {"complete": True},
                }
            ],
            {},
            {},
            eval_dir=tmp_path,
        )
        save_historical_results(
            [
                {
                    "model": "ornith",
                    "results": _results(["a", "b"], score=0),
                    "completeness": {"complete": False, "reason": "wedged"},
                }
            ],
            {},
            {},
            eval_dir=tmp_path,
        )
        stats = load_historical_stats(eval_dir=tmp_path)
        # Averaging all four would give 50. Only the complete run counts.
        assert stats["ornith"]["mean"] == 100
        assert stats["ornith"]["runs"] == 2
        assert stats["ornith"]["excluded"] == 2

    def test_entries_written_before_this_existed_still_count(self, tmp_path):
        """`complete` is absent on every historical entry; absent means complete."""
        history = {"legacy": [{"date": "2026-01-01", "task": "a", "score": 80}]}
        (tmp_path / "eval_history.json").write_text(json.dumps(history))
        stats = load_historical_stats(eval_dir=tmp_path)
        assert stats["legacy"]["mean"] == 80

    def test_the_exclusion_is_announced(self, tmp_path, capsys):
        """Something that refuses to count data says so when it refuses."""
        save_historical_results(
            [
                {
                    "model": "ornith",
                    "results": _results(["a"], score=25),
                    "completeness": {"complete": False, "reason": "6 timeouts"},
                }
            ],
            {},
            {},
            eval_dir=tmp_path,
        )
        out = capsys.readouterr().out
        assert "ornith" in out and "EXCLUDED" in out
        assert "6 timeouts" in out


class TestComparisonTableIsOrderIndependent:
    @pytest.mark.parametrize("truncated_first", [True, False])
    def test_every_task_gets_a_row_whichever_model_is_first(self, truncated_first, capsys):
        short = {"model": "short", "results": _results(["a"])}
        full = {"model": "full", "results": _results(["a", "b", "c"])}
        order = [short, full] if truncated_first else [full, short]
        print_cross_model_comparison(order)
        out = capsys.readouterr().out
        for task in ("a", "b", "c"):
            assert task in out, f"task {task} lost when truncated_first={truncated_first}"

    def test_a_leading_empty_model_does_not_blank_the_table(self, capsys):
        print_cross_model_comparison(
            [{"model": "empty", "results": []}, {"model": "full", "results": _results(["a"])}]
        )
        assert "a" in capsys.readouterr().out
