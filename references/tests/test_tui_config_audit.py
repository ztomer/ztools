"""The TUI must not open green while conf/config.toml names an unservable model.

This is backlog item 7, and it is the same defect as item 0 seen from the UI side.
Two configured models were deleted from disk; `action_refresh_models` set each
dropdown to `preferred if preferred in models else models[0]` and said nothing, so
the TUI opened "Server Online", showed a plausible model in every dropdown, and gave
no indication that three config slots were dead. The user found out when `wk` failed
with HTTP 404.

The audit itself is NOT mocked here -- these run the real `audit_configured_models`
against the real `conf/config.toml`, because a test that mocks the audit only proves
the status box can render a string someone handed it. Only the server layer is faked,
and the configured names are read from the config rather than hardcoded, so the test
does not rot the next time `best_models` is re-derived.
"""

import asyncio
from unittest.mock import patch

import pytest

import tui.app_actions as actions


class FakeStatusBox:
    def __init__(self):
        self.text = None

    def update(self, message):
        self.text = message


class FakeSelect:
    def __init__(self):
        self.options = None
        self.value = None


class FakeApp:
    """Just enough of the Textual App surface for action_refresh_models."""

    def __init__(self):
        self.status_box = FakeStatusBox()
        self.selects = {
            "#wk-model": FakeSelect(),
            "#tw-model": FakeSelect(),
            "#ev-model": FakeSelect(),
            "#rn-model": FakeSelect(),
        }

    def query_one(self, selector):
        if selector == "#server-status-box":
            return self.status_box
        return self.selects[selector]


def configured_models():
    """Every model name the config routes to, read live."""
    from lib.config import get_best_models, get_filename_models
    from lib.config_core import _auto_load, _config

    _auto_load()
    names = set(get_best_models().values()) | set(get_filename_models())
    if _config.get("default_model"):
        names.add(_config["default_model"])
    return names


def refresh(app, *, running=True, models=()):
    with (
        patch.object(actions, "is_server_running", return_value=running),
        patch.object(actions, "get_models", return_value=list(models)),
    ):
        asyncio.run(actions.action_refresh_models(app))
    return app.status_box.text


class TestTheFixtureIsWiredCorrectly:
    """Calibration. If the config named nothing, or named only models absent from
    every roster below, the assertions in the next class would hold vacuously."""

    def test_the_config_actually_routes_to_some_models(self):
        assert len(configured_models()) >= 2

    def test_the_audit_sees_the_same_config_this_test_reads(self):
        from lib.model_resolve import audit_configured_models

        report = audit_configured_models(list(configured_models()))
        assert report["missing"] == [], (
            "conf/config.toml is already drifted; fix the config, not this test"
        )
        assert report["installed"], "the audit found no slots at all"


class TestDriftIsSurfacedNotHidden:
    def test_a_roster_missing_a_configured_model_warns(self):
        served = sorted(configured_models())
        dropped = served[0]
        text = refresh(FakeApp(), models=[m for m in served if m != dropped] or ["other"])
        assert "not servable" in text
        assert "Online" in text, "the server IS up; the warning must not claim otherwise"

    def test_the_warning_names_the_config_slot_to_edit(self):
        """A count is not actionable. `best_models.summarize` is the line to edit."""
        served = sorted(configured_models())
        text = refresh(FakeApp(), models=[m for m in served if m != served[0]] or ["other"])
        assert any(
            key in text for key in ("default_model", "best_models.", "filename_models[")
        ), text

    def test_a_complete_roster_reports_plain_online(self):
        """Without this the warning above could be firing unconditionally."""
        text = refresh(FakeApp(), models=sorted(configured_models()))
        assert "Online" in text
        assert "not servable" not in text
        assert "substituted" not in text

    def test_a_roster_of_entirely_unrelated_models_warns(self):
        text = refresh(FakeApp(), models=["some-model-nobody-configured"])
        assert "not servable" in text


class TestTheDropdownsStillWork:
    """The audit must not have cost the function its actual job."""

    def test_a_served_preference_is_selected(self):
        from lib.config import get_best_models

        preferred = get_best_models().get("summarize")
        if not preferred:
            pytest.skip("no summarize slot configured")
        app = FakeApp()
        refresh(app, models=sorted(configured_models()))
        assert app.selects["#tw-model"].value == preferred

    def test_every_dropdown_is_offered_the_whole_roster(self):
        served = sorted(configured_models())
        app = FakeApp()
        refresh(app, models=served)
        for select in app.selects.values():
            assert select.options == [(m, m) for m in served]

    def test_an_unservable_preference_falls_back_to_a_servable_model(self):
        """The substitution itself is correct behaviour and must survive -- what was
        wrong was doing it silently."""
        app = FakeApp()
        refresh(app, models=["some-model-nobody-configured"])
        for select in app.selects.values():
            assert select.value == "some-model-nobody-configured"


class TestServerStatesAreDistinguishable:
    def test_an_offline_server_says_offline(self):
        assert "Offline" in refresh(FakeApp(), running=False)

    def test_an_offline_server_is_not_audited(self):
        """There is no roster to audit against, and reporting every slot as missing
        because the server is down would be a false diagnosis."""
        assert "not servable" not in refresh(FakeApp(), running=False)

    def test_a_server_with_an_empty_roster_is_not_left_checking(self):
        """This used to leave the box reading "Checking Osaurus..." indefinitely --
        a stuck spinner reads as a hung app rather than as the diagnosis it is."""
        text = refresh(FakeApp(), models=[])
        assert "Checking" not in text
        assert "no models" in text

    def test_the_three_states_produce_three_different_messages(self):
        offline = refresh(FakeApp(), running=False)
        empty = refresh(FakeApp(), models=[])
        healthy = refresh(FakeApp(), models=sorted(configured_models()))
        assert len({offline, empty, healthy}) == 3


class TestTheAuditReusesTheRosterItWasGiven:
    def test_no_second_roster_request_is_made(self):
        """`audit_configured_models` fetches its own roster when not given one. Doing
        that here would add a network call to a path the suite forbids from reaching a
        server -- which is exactly how this broke 26 tests when first written."""
        with patch("lib.model_resolve.fetch_roster") as fetch:
            refresh(FakeApp(), models=sorted(configured_models()))
        fetch.assert_not_called()
