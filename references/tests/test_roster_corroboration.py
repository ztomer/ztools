"""The server's roster is a CLAIM. Disk is the corroboration.

osaurus keeps its model list in memory, so a model deleted from disk stays advertised
by /api/tags until the server restarts. A request for it then hangs or 404s at CALL
time -- strictly worse than failing at selection time, because by then the chance to
pick a different model has gone.

This is not hypothetical and it is not rare: it happened twice in one day, with
qwen3.8-27b-4bit and nemotron-3.5-lightning-30b, and in both cases the roster kept
offering the model as a substitution candidate.

The subtlety that makes a plain file check wrong: `foundation` is Apple's on-device
model. It has no config.json and never will. "No files, therefore gone" would discard
the single most reliable model in the roster -- 0 zeros across all 24 eval tasks and
the fastest thing installed. Corroboration therefore accepts EITHER a config.json on
disk OR a documented context window in conf/models/, and only a model with neither is
treated as stale.
"""

from unittest.mock import patch

import lib.model_resolve as mr


class TestTheDiscriminatorSeparatesTheThreeCases:
    """Calibration first. If corroboration answered the same way for a live model, an
    on-device model and a deleted one, every assertion below would be vacuous."""

    def test_a_model_with_a_config_on_disk_is_corroborated(self):
        with patch("lib.model_caps.probe_context_window", return_value=131072):
            assert mr.disk_corroborated("gemma-4-12b-it-mxfp8") is True

    def test_an_on_device_model_with_no_files_is_still_corroborated(self):
        """foundation: no config.json, but conf/models/foundation.toml documents 4096.
        Dropping it would be the worst possible outcome of this check."""
        with patch("lib.model_caps.probe_context_window", return_value=4096):
            assert mr.disk_corroborated("foundation") is True

    def test_a_model_with_neither_is_not_corroborated(self):
        with patch("lib.model_caps.probe_context_window", return_value=None):
            assert mr.disk_corroborated("qwen3.8-27b-4bit") is False

    def test_an_unreadable_probe_keeps_the_entry(self):
        """Absence of evidence is not evidence of absence. Wrongly dropping a servable
        model is worse than keeping a stale one, which still fails loudly later."""
        with patch("lib.model_caps.probe_context_window", side_effect=OSError("boom")):
            assert mr.disk_corroborated("anything") is True


class TestStaleEntriesAreIdentified:
    ROSTER = [{"model": "live-model"}, {"model": "deleted-model"}, {"model": "foundation"}]

    def _probe(self, name):
        return None if name == "deleted-model" else 4096

    def test_only_the_uncorroborated_entry_is_stale(self):
        with patch("lib.model_caps.probe_context_window", side_effect=self._probe):
            assert mr.stale_roster_entries(self.ROSTER) == ["deleted-model"]

    def test_a_fully_backed_roster_reports_nothing(self):
        with patch("lib.model_caps.probe_context_window", return_value=4096):
            assert mr.stale_roster_entries(self.ROSTER) == []

    def test_an_empty_roster_is_not_an_error(self):
        assert mr.stale_roster_entries([]) == []


class TestFilteringHappensAtTheFetchBoundary:
    """Corroboration belongs where the claim ENTERS the process, not in the selector.

    The first version of this put the filter inside `substitute_model`. That made a
    pure selection function depend on the filesystem: the same roster produced
    different answers on different machines, and four existing unit tests with
    synthetic model names silently began consulting real directories and picking
    `foundation` over their fixtures. Filtering at the fetch keeps selection pure and
    hands every downstream consumer the same trustworthy list.
    """

    ROSTER = [
        {"model": "qwen3.8-27b-mxfp8"},   # on disk
        {"model": "qwen3.8-27b-4bit"},    # deleted, still advertised
    ]

    def _probe(self, name):
        return None if name == "qwen3.8-27b-4bit" else 262144

    def test_a_stale_entry_is_dropped_before_anyone_sees_it(self):
        with patch("lib.model_caps.probe_context_window", side_effect=self._probe):
            kept = mr._drop_uncorroborated(self.ROSTER)
        assert [e["model"] for e in kept] == ["qwen3.8-27b-mxfp8"]

    def test_substitution_downstream_therefore_cannot_pick_it(self):
        """The property that matters, exercised through the real path: filter first,
        then select."""
        with patch("lib.model_caps.probe_context_window", side_effect=self._probe):
            roster = mr._drop_uncorroborated(self.ROSTER)
        pick, reason = mr.substitute_model("qwen3.8-27b-4bit", roster)
        assert pick == "qwen3.8-27b-mxfp8"
        assert reason and "not installed" in reason

    def test_substitute_model_itself_stays_pure(self):
        """It must NOT consult the filesystem. Given a roster it uses that roster,
        whatever is or is not on this machine."""
        with patch("lib.model_caps.probe_context_window",
                   side_effect=AssertionError("substitute_model touched the disk")):
            pick, _ = mr.substitute_model("gone", [{"model": "only-option"}])
        assert pick == "only-option"

    def test_an_entirely_uncorroborated_roster_is_kept_unchanged(self):
        """Far likelier to mean the probe is broken than that the server is serving
        twelve models which do not exist. Emptying the roster would turn a diagnosable
        problem into 'no models installed'."""
        with patch("lib.model_caps.probe_context_window", return_value=None):
            kept = mr._drop_uncorroborated(self.ROSTER)
        assert kept == self.ROSTER


class TestTheAuditReportsStaleSlots:
    def test_a_slot_naming_a_stale_model_is_not_reported_as_healthy(self):
        installed = ["live-model", "deleted-model"]
        with (
            patch("lib.model_resolve.disk_corroborated",
                  side_effect=lambda m: m != "deleted-model"),
            patch("lib.config.get_best_models", return_value={"json": "deleted-model"}),
            patch("lib.config.get_filename_models", return_value=[]),
        ):
            report = mr.audit_configured_models(installed)
        assert "deleted-model" in report["stale"]

    def test_the_message_names_the_model_and_the_remedy(self):
        lines = mr.format_audit(
            {"installed": [], "missing": [], "stale": ["deleted-model"], "unreachable": False}
        )
        joined = "\n".join(lines)
        assert "deleted-model" in joined
        assert "osaurus_one.sh --restart" in joined

    def test_a_clean_report_still_says_nothing(self):
        """Without this the audit could pass by warning unconditionally, which trains
        the reader to ignore it."""
        assert mr.format_audit(
            {"installed": ["a = b"], "missing": [], "stale": [], "unreachable": False}
        ) == []
