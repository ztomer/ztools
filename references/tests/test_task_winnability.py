"""A perfect answer must score 100. If it cannot, the task is broken, not the model.

THE CLASS: the validator's contract must equal the prompt's contract. When they
diverge the validator punishes obedience, and the result is indistinguishable from a
model failure -- which is exactly why it survives. Nobody re-reads a scorer because
several models scored 91.

The instance: `weekend_fixed_mixed` shows the model 12 signal venues and asks it to
"find 10". Recall was computed over all 12, so

    obeying the prompt (10 of 12, no noise)   -> 91, "missed 2/12 signal items"
    ignoring it        (all 12, no noise)     -> 100

All eleven models in the 2026-08-16 sweep scored exactly 91. docs/BACKLOG.md recorded
that as "a fixture or prompt defect rather than eleven coincidences" -- correct, but
the reading was that the models missed two items. They missed nothing. The task paid
them less for doing as they were told, and taught the leaderboard that disobedience
scores better.

These tests construct the ideal answer FROM EACH TASK'S OWN DATA rather than from a
hand-written fixture, so they keep testing the real task after someone edits it.
"""

import pytest
from eval.tasks_core import TASKS
from lib.validators.json_validator import validate_mixed_signal
from lib.validators.prompt_contract import parse_signal_noise, requested_item_count

#: Tasks whose ideal answer can be constructed mechanically: the validator scores a
#: list of names against signal/noise sets parsed from the prompt, so "perfect" is
#: computable. Other validators (prose summaries, filenames) have no mechanical ideal
#: and are covered by their own boundary tests.
MECHANICAL_TASKS = [
    name for name, spec in TASKS.items() if spec["validator"] is validate_mixed_signal
]


def ideal_answer(spec):
    """Exactly what the prompt asks for: its signal items, capped at the count it
    requested, and none of its noise."""
    signal, _noise = parse_signal_noise(spec.get("source", ""))
    asked = requested_item_count(spec.get("source", ""))
    return [{"name": s} for s in (signal[:asked] if asked else signal)]


class TestTheFixtureIsWiredCorrectly:
    """Calibration. If no task were collected, or the ideal answer were empty, every
    assertion below would hold vacuously."""

    def test_there_are_mechanically_scorable_tasks(self):
        assert MECHANICAL_TASKS, "no task uses validate_mixed_signal; the sweep is empty"

    @pytest.mark.parametrize("name", MECHANICAL_TASKS)
    def test_each_task_yields_a_non_empty_ideal_answer(self, name):
        assert ideal_answer(TASKS[name]), f"{name}: could not construct an ideal answer"


class TestEveryMechanicalTaskIsWinnable:
    @pytest.mark.parametrize("name", MECHANICAL_TASKS)
    def test_the_ideal_answer_scores_100(self, name):
        spec = TASKS[name]
        score, failure = validate_mixed_signal(ideal_answer(spec), spec.get("source", ""))
        assert score == 100, (
            f"{name}: an answer that obeys the prompt exactly scores {score} "
            f"({failure!r}). The task is unwinnable -- the validator is asking for "
            f"something different from what the prompt asks for."
        )

    @pytest.mark.parametrize("name", MECHANICAL_TASKS)
    def test_obeying_a_count_is_not_worse_than_ignoring_it(self, name):
        """The specific inversion that shipped: returning everything out-scored
        returning the requested number."""
        spec = TASKS[name]
        src = spec.get("source", "")
        signal, _ = parse_signal_noise(src)
        asked = requested_item_count(src)
        if not asked or asked >= len(signal):
            pytest.skip("task sets no count below the signal total; nothing to invert")
        obedient = validate_mixed_signal([{"name": s} for s in signal[:asked]], src)[0]
        greedy = validate_mixed_signal([{"name": s} for s in signal], src)[0]
        assert obedient >= greedy, (
            f"{name}: obeying the prompt scores {obedient} but ignoring it scores "
            f"{greedy}; the task rewards disobedience"
        )


class TestTheGateStillPunishesRealMistakes:
    """A winnability fix must not become 'everything scores 100'. These are the
    failures the validator exists to catch, and they must still fail."""

    TASK = "weekend_fixed_mixed"

    def _score(self, items):
        spec = TASKS[self.TASK]
        return validate_mixed_signal(items, spec.get("source", ""))[0]

    def test_including_noise_still_costs(self):
        src = TASKS[self.TASK].get("source", "")
        signal, noise = parse_signal_noise(src)
        asked = requested_item_count(src) or len(signal)
        clean = self._score([{"name": s} for s in signal[:asked]])
        dirty = self._score(
            [{"name": s} for s in signal[: asked - 2]] + [{"name": n} for n in noise[:2]]
        )
        assert dirty < clean, "noise items no longer cost anything"

    def test_returning_far_too_few_still_costs(self):
        signal, _ = parse_signal_noise(TASKS[self.TASK].get("source", ""))
        assert self._score([{"name": s} for s in signal[:2]]) < 100

    def test_returning_only_noise_scores_badly(self):
        src = TASKS[self.TASK].get("source", "")
        _, noise = parse_signal_noise(src)
        assert self._score([{"name": n} for n in noise[:4]]) <= 50


class TestTheCountIsReadFromThePrompt:
    """Derived from the prompt text, not configured beside it, so the two cannot
    drift apart when someone edits the wording."""

    def test_the_weekend_fixed_prompt_still_asks_for_a_count(self):
        """If this fails the prompt was reworded; the fix above silently stops
        applying and the task quietly becomes unwinnable again."""
        assert requested_item_count(TASKS["weekend_fixed_mixed"]["source"]) == 10

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("find 10 year-round fixed activities", 10),
            ("Find 5-10 events for the weekend", 5),
            ("list 8 venues", 8),
            ("return 3 to 6 items", 3),
            ("summarise the timeline", None),
            ("", None),
        ],
    )
    def test_count_parsing(self, text, expected):
        assert requested_item_count(text) == expected

    def test_a_range_takes_its_lower_bound(self):
        """The smallest answer that still obeys. Scoring a compliant 5 against a
        denominator of 10 would reintroduce the defect."""
        assert requested_item_count("find 5-10 events") == 5
