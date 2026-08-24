"""Tests for lib.validators.json_validator."""

from pathlib import Path


class TestValidateMixedSignal:
    def test_perfect_signal_passes(self):
        from eval.tasks_core import WEEKEND_USR_TRANSIENT_MIXED
        from lib.validators.json_validator import parse_signal_noise, validate_mixed_signal

        sig, noise = parse_signal_noise(WEEKEND_USR_TRANSIENT_MIXED)
        # Output keeps every signal item, excludes all noise.
        items = [
            {
                "name": s,
                "location": "x",
                "target_ages": "6-13",
                "price": "Free",
                "weather": "indoor",
            }
            for s in sig
        ]
        score, reason = validate_mixed_signal(items, source_text=WEEKEND_USR_TRANSIENT_MIXED)
        assert score >= 90, reason
        assert "noise" not in reason

    def test_noise_included_fails(self):
        from eval.tasks_core import WEEKEND_USR_TRANSIENT_MIXED
        from lib.validators.json_validator import parse_signal_noise, validate_mixed_signal

        sig, noise = parse_signal_noise(WEEKEND_USR_TRANSIENT_MIXED)
        items = [
            {
                "name": s,
                "location": "x",
                "target_ages": "6-13",
                "price": "Free",
                "weather": "indoor",
            }
            for s in sig
        ]
        # Append several noise items that must be excluded.
        for n in noise[:4]:
            items.append(
                {
                    "name": n,
                    "location": "x",
                    "target_ages": "0-100",
                    "price": "Free",
                    "weather": "indoor",
                }
            )
        score, reason = validate_mixed_signal(items, source_text=WEEKEND_USR_TRANSIENT_MIXED)
        assert score < 90, reason
        assert "noise" in reason

    def test_score_capped_at_100(self):
        from eval.tasks_core import WEEKEND_USR_TRANSIENT_MIXED
        from lib.validators.json_validator import parse_signal_noise, validate_mixed_signal

        sig, noise = parse_signal_noise(WEEKEND_USR_TRANSIENT_MIXED)
        # Duplicate every signal item twice — recall must cap at 1.0, score at 100.
        items = []
        for s in sig:
            for _ in range(2):
                items.append(
                    {
                        "name": s,
                        "location": "x",
                        "target_ages": "6-13",
                        "price": "Free",
                        "weather": "indoor",
                    }
                )
        score, _ = validate_mixed_signal(items, source_text=WEEKEND_USR_TRANSIENT_MIXED)
        assert score <= 100, score


class TestValidateMixedSummary:
    def test_clean_summary_passes(self):
        from eval.tasks_core import TWITTER_PROMPT_MIXED
        from lib.validators.text_validator import validate_mixed_summary

        summary = (
            "@TechCrunch announced GPT-5. @Bloomberg: GDP grew. @LocalNews_TOR reopened CN Tower."
        )
        score, reason = validate_mixed_summary(summary, TWITTER_PROMPT_MIXED)
        assert score >= 90, reason

    def test_noise_summary_fails(self):
        from eval.tasks_core import TWITTER_PROMPT_MIXED
        from lib.validators.text_validator import validate_mixed_summary

        summary = (
            "@FakeNews reported aliens landed in Central Park. "
            "lorem ipsum dolor sit amet consectetur adipiscing. "
            "BUY NOW LIMITED TIME OFFER CLICK HERE. "
            "Cryptocurrency price prediction for next week. "
            "Also @LocalNews_TOR reopened CN Tower."
        )
        score, reason = validate_mixed_summary(summary, TWITTER_PROMPT_MIXED)
        assert score < 60, reason
        assert "noise" in reason


class TestValidateMixedFileSummary:
    def test_noise_file_fails(self):
        from eval.tasks_core import FILE_SUMMARY_PROMPT_MIXED
        from lib.validators.text_validator import validate_mixed_file_summary

        project_root = Path(__file__).parent.parent
        out = [
            {"path": str(project_root / "README.md"), "desc": "docs"},
            {"path": "/spam/buy_now/click_here.exe", "desc": "spam"},
        ]
        score, reason = validate_mixed_file_summary(out, FILE_SUMMARY_PROMPT_MIXED)
        assert score < 90, reason
        assert "noise" in reason

    def test_clean_file_summary_passes(self):
        from eval.tasks_core import FILE_SUMMARY_PROMPT_MIXED
        from lib.validators.text_validator import validate_mixed_file_summary

        project_root = Path(__file__).parent.parent
        out = [{"path": str(project_root / "README.md"), "desc": "docs"}]
        score, reason = validate_mixed_file_summary(out, FILE_SUMMARY_PROMPT_MIXED)
        assert "noise" not in reason


class TestValidateMixedFilename:
    def test_noise_filename_fails(self):
        from eval.tasks_core import RENAME_PROMPT_MIXED
        from lib.validators.text_validator import validate_mixed_filename

        out = ["manage_underperformers", "buy_now_click_here", "context_engineering"]
        score, reason = validate_mixed_filename(out, RENAME_PROMPT_MIXED)
        assert score < 90, reason
        assert "noise" in reason

    def test_clean_filename_passes(self):
        from eval.tasks_core import RENAME_PROMPT_MIXED
        from lib.validators.text_validator import validate_mixed_filename

        out = ["manage_underperformers", "scott_adams_essays", "context_engineering"]
        score, reason = validate_mixed_filename(out, RENAME_PROMPT_MIXED)
        assert "noise" not in reason


class TestMixedSignalBoundariesThatSurvivedMutation:
    """Two survivors in `validate_mixed_signal`, both reachable only by inputs the
    existing tests never send.

    Killing them needs the caller shapes the eval itself does not use -- which is the
    point: a function is not only exercised through the one path a task happens to
    take, and a mutation that survives is a change no test would notice.
    """

    SRC = (
        "- Alpha Hall: story time\n"
        "- Beta Park: water play\n"
        "NOISE (ignore these)\n"
        "- Zeta Spam: buy now\n"
    )

    def test_supplying_only_signal_items_still_parses_the_noise(self):
        """`if signal_items is None or noise_items is None` -- `or` becoming `and`
        survived, because every existing caller passes BOTH or NEITHER. With `and`, a
        caller that supplies only the signal list gets `noise_items=None`, so noise
        stops being penalised and a junk answer scores as clean."""
        from lib.validators.json_validator import validate_mixed_signal

        items = [{"name": "Alpha Hall"}, {"name": "Zeta Spam"}]
        score, msg = validate_mixed_signal(
            items, self.SRC, signal_items=["Alpha Hall", "Beta Park"]
        )
        assert "noise" in msg, "noise went unparsed, so including it cost nothing"
        assert score < 100

    def test_supplying_only_noise_items_still_parses_the_signal(self):
        """The mirror case, for the same `or`."""
        from lib.validators.json_validator import validate_mixed_signal

        items = [{"name": "Alpha Hall"}]
        score, _ = validate_mixed_signal(items, self.SRC, noise_items=["Zeta Spam"])
        assert score > 0, "signal went unparsed, so a correct answer scored nothing"

    def test_an_empty_answer_against_an_empty_source_is_not_perfect_precision(self):
        """`(1.0 if tp == 0 and total_signal == 0 else 0.0)` -- the `and` and the
        equality both survived. This branch is only reached when NOTHING matched and
        nothing was expected, and it decides whether producing junk for a source with
        no signal counts as precise."""
        from lib.validators.json_validator import validate_mixed_signal

        # A source with no signal section at all, and an answer full of inventions.
        score, _ = validate_mixed_signal(
            [{"name": "Invented Place"}], "", signal_items=[], noise_items=[]
        )
        assert score == 100, (
            "with nothing to find and nothing marked as noise there is nothing to "
            "get wrong; this pins the branch rather than leaving it unreached"
        )

    def test_inventing_items_when_signal_exists_is_not_precise(self):
        """The discriminating case: signal EXISTS and the answer matched none of it."""
        from lib.validators.json_validator import validate_mixed_signal

        score, _ = validate_mixed_signal(
            [{"name": "Invented Place"}], self.SRC, signal_items=["Alpha Hall"], noise_items=[]
        )
        # Asserted EXACTLY, not `< 100`. The loose bound was satisfied by both the
        # correct 0 and the `and`->`or` mutant's 50, so it could not see the change
        # it was written to catch -- the same outcome-not-boundary mistake this class
        # exists to fix, made while fixing it.
        assert score == 0, (
            "nothing matched and signal existed, so precision is 0 -- scoring this as "
            "precise would mean an answer that invented everything was precise"
        )
