"""Substituting a servable model when the configured tag has been deleted.

The bug this covers: conf/config.toml named qwen3.6-35b-a3b-mxfp8-mtp for four of
seven tasks long after that model was removed from disk, so `wk` answered every run
with "HTTP 404" and nothing said which config line was stale. These tests pin the
degrade-with-a-stated-reason behaviour and, just as importantly, the cases where it
must NOT fire — a substitution made on no evidence turns a loud error into a quiet
wrong answer.
"""

import pytest
from lib import model_resolve
from lib.model_resolve import (
    audit_configured_models,
    fetch_roster,
    format_audit,
    is_missing_model_error,
    substitute_model,
)

MISSING_BODY = (
    '{"error":{"message":"Model \'qwen3.6-35b-a3b-mxfp8-mtp\' is not installed or '
    'registered with any provider.","type":"invalid_request_error"}}'
)


def entry(model, family="unknown", size=""):
    return {"model": model, "details": {"family": family, "parameter_size": size}}


@pytest.fixture
def fake_config(monkeypatch):
    """Substitute the config slots the audit reads, leaving conf/ untouched.

    The sidecar slots (conf/rename.toml and friends) are stubbed to empty here. The
    audit reads those files from disk, so without this a test that installs three
    synthetic slots would silently be asserting against those three PLUS whatever
    `rn` happens to have configured today -- the same "unit test quietly consults the
    real machine" trap that made the roster filter machine-dependent.
    """

    def install(default_model, best_models, filename_models, sidecar=()):
        import lib.config as cfg
        import lib.config_core as core
        import lib.model_resolve as mr

        monkeypatch.setattr(cfg, "get_best_models", lambda: best_models)
        monkeypatch.setattr(cfg, "get_filename_models", lambda: filename_models)
        monkeypatch.setattr(core, "_auto_load", lambda: None)
        monkeypatch.setattr(core, "_config", {"default_model": default_model})
        monkeypatch.setattr(mr, "_sidecar_model_slots", lambda: list(sidecar))

    return install


ROSTER = [
    entry("foundation", "foundation"),
    entry("bonsai-27b-ternary-jang", "qwen3_5", "27B"),
    entry("gemma-4-12b-it-mxfp8", "gemma4_unified", "12B"),
    # Sorts BEFORE the 12B one as text ("0" < "1") while being six times smaller, so
    # an alphabetical pick and a size pick disagree here. Without such a pair the
    # ordering test passes under either rule and measures nothing.
    entry("gemma-4-02b-it-8bit", "gemma4", "2B"),
    entry("ornith-1.0-35b-jang_4m", "qwen3_5_moe", "35B"),
    entry("potion-base-4m", "unknown", "4M"),
    entry("qwen3.8-27b-mxfp8", "qwen3_5", "27B"),
]


class TestRecognisingTheError:
    def test_a_missing_model_404_is_recognised(self):
        assert is_missing_model_error(404, MISSING_BODY) is True

    def test_a_404_from_a_wrong_path_is_not(self):
        """Substituting on any 404 would swap models whenever a URL was mistyped."""
        assert is_missing_model_error(404, "404 page not found") is False

    def test_a_503_is_not(self):
        assert is_missing_model_error(503, "Server is at capacity") is False

    def test_an_empty_body_is_not(self):
        assert is_missing_model_error(404, "") is False


class TestSubstituting:
    def test_an_installed_model_is_left_alone(self):
        model, reason = substitute_model("gemma-4-12b-it-mxfp8", ROSTER)
        assert model == "gemma-4-12b-it-mxfp8"
        assert reason is None

    def test_an_empty_roster_changes_nothing(self):
        """No roster means no evidence, not evidence of nothing."""
        model, reason = substitute_model("qwen3.6-35b-a3b-mxfp8-mtp", [])
        assert model == "qwen3.6-35b-a3b-mxfp8-mtp"
        assert reason is None

    def test_a_dead_qwen_falls_to_the_installed_qwen(self):
        model, reason = substitute_model("qwen3.6-35b-a3b-mxfp8-mtp", ROSTER)
        assert model == "qwen3.8-27b-mxfp8"
        assert "not installed" in reason
        assert "qwen3.8-27b-mxfp8" in reason

    def test_the_reason_names_the_dead_model(self):
        """The whole point is telling the user which config line to edit."""
        _, reason = substitute_model("qwen3.6-35b-a3b-mxfp8-mtp", ROSTER)
        assert "qwen3.6-35b-a3b-mxfp8-mtp" in reason

    def test_size_beats_alphabetical_order_within_a_family(self):
        """'gemma-4-02b' sorts before 'gemma-4-12b' as text; 12B is the better model."""
        model, _ = substitute_model("gemma-4-99b-it-mxfp8", ROSTER)
        assert model == "gemma-4-12b-it-mxfp8"

    def test_a_family_with_nothing_installed_uses_the_preference_chain(self):
        model, reason = substitute_model("laguna-70b", ROSTER)
        assert model == "foundation"
        assert "laguna-70b" in reason

    def test_a_roster_missing_every_preferred_family_still_picks_something(self):
        roster = [entry("potion-base-4m", "unknown", "4M"), entry("mystery-9b", "x", "9B")]
        model, reason = substitute_model("laguna-70b", roster)
        assert model == "mystery-9b"
        assert reason is not None


@pytest.fixture
def config_dict(monkeypatch):
    """Replace the config mapping that config_getters actually reads.

    Patched on lib.config_core, which is now the only place there is. This fixture
    used to patch lib.config_getters instead, because that module bound `_config` at
    import time and patching the source left the getters reading the real
    conf/config.toml -- while `audit_configured_models`, which imports inside the
    function body, saw the patch. Two bindings, two answers, from one config.

    That split was not merely awkward to test around; it shipped. The TUI reported a
    clean config audit while its own dropdowns silently substituted two slots.
    config_getters reads through the module now (`_cfg`), so patching the source is
    both correct and sufficient. `_config_loaded` has to be pinned alongside it or
    `_auto_load` reloads the real file over the top of the fake.
    """

    def install(mapping):
        import lib.config_core as core

        monkeypatch.setattr(core, "_config", mapping)
        monkeypatch.setattr(core, "_config_loaded", True)

    return install


class TestTheFallbackChain:
    def test_it_comes_from_config(self, config_dict):
        # 'ghost' has no family of its own, so the chain decides — and a chain of
        # ["gemma"] must pick gemma even though foundation is installed and would
        # otherwise lead. If the getter ignored config, this would return foundation.
        config_dict({"model_fallback_chain": ["gemma"]})
        model, _ = substitute_model("ghost-70b", ROSTER)
        assert model == "gemma-4-12b-it-mxfp8"

    def test_a_config_predating_the_key_still_prefers_foundation(self, config_dict):
        """The built-in default, which conf/config.toml currently masks.

        Without it, a config written before model_fallback_chain existed would fall
        through to "biggest model on the roster" — on this machine a 35B MoE, when an
        on-device model was sitting right there.
        """
        config_dict({})
        model, reason = substitute_model("ghost-70b", ROSTER)
        assert model == "foundation"
        assert reason is not None

    def test_the_default_chain_lists_every_known_family_once(self):
        from lib.config_getters import _default_fallback_chain
        from lib.llm.constants import MODEL_FAMILIES

        chain = _default_fallback_chain()
        assert chain[0] == "foundation"
        assert sorted(chain) == sorted(set(MODEL_FAMILIES))


class TestParsingParameterSize:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("27B", 27.0),
            ("4M", 0.004),
            ("", 0.0),
            ("banana", 0.0),
            ("12", 0.0),
            # A known suffix with an unparseable number — the only route to the
            # ValueError branch, since "banana" is rejected earlier by its suffix.
            ("1.2.3B", 0.0),
            ("B", 0.0),
        ],
    )
    def test_sizes_parse_or_degrade_to_zero(self, raw, expected):
        assert model_resolve._parameter_billions(entry("m", size=raw)) == expected

    def test_a_missing_details_dict_is_not_an_error(self):
        """Osaurus omits details for some entries; ranking must not raise on them."""
        assert model_resolve._parameter_billions({"model": "m"}) == 0.0


class TestFetchingTheRoster:
    def test_a_dead_server_yields_an_empty_roster(self, monkeypatch):
        def boom(*a, **k):
            raise OSError("connection refused")

        monkeypatch.setattr(model_resolve.requests, "get", boom)
        assert fetch_roster("localhost", 1337) == []

    def test_a_non_200_yields_an_empty_roster(self, monkeypatch):
        """A server that is up but refusing (401, 503) is still no evidence.

        Distinct from the exception path: requests returns normally here, so without
        the status check the roster would be built from an error body.
        """

        class Resp:
            status_code = 503

            def json(self):
                raise AssertionError("must not parse the body of a non-200")

        monkeypatch.setattr(model_resolve.requests, "get", lambda *a, **k: Resp())
        assert fetch_roster() == []

    def test_entries_without_a_model_id_are_dropped(self, monkeypatch):
        class Resp:
            status_code = 200

            def json(self):
                return {"models": [{"model": "a"}, {"details": {}}, {"model": ""}]}

        monkeypatch.setattr(model_resolve.requests, "get", lambda *a, **k: Resp())
        assert fetch_roster() == [{"model": "a"}]

    def test_a_url_with_a_scheme_is_not_prefixed_again(self, monkeypatch):
        seen = {}

        class Resp:
            status_code = 200

            def json(self):
                return {"models": []}

        def fake_get(url, **k):
            seen["url"] = url
            return Resp()

        monkeypatch.setattr(model_resolve.requests, "get", fake_get)
        fetch_roster("http://127.0.0.1:9999", 1337)
        assert seen["url"] == "http://127.0.0.1:9999/api/tags"


class TestAuditingTheConfig:
    def test_an_unreachable_server_reports_unreachable_not_all_missing(self, monkeypatch):
        monkeypatch.setattr(model_resolve, "fetch_roster", lambda *a, **k: [])
        assert audit_configured_models() == {
            "installed": [],
            "missing": [],
            "unreachable": True,
        }

    def test_missing_entries_are_labelled_by_config_slot(self, fake_config):
        """The label is the deliverable: it names the exact line to edit.

        Asserted against a fixture config rather than the real conf/config.toml —
        reading the live one makes the test flip colour whenever someone edits a
        model name, which is precisely the edit this test exists to survive.
        """
        fake_config(
            default_model="ghost-70b",
            best_models={"summarize": "foundation", "think": "ghost-70b"},
            filename_models=["foundation", "ghost-70b"],
        )
        report = audit_configured_models(installed=["foundation"])
        assert report["unreachable"] is False
        assert report["missing"] == [
            "default_model = ghost-70b",
            "best_models.think = ghost-70b",
            "filename_models[1] = ghost-70b",
        ]

    def test_a_stale_default_model_is_caught(self, fake_config):
        """The regression that started this: default_model outlived its model."""
        fake_config(
            default_model="qwen3.6-35b-a3b-mxfp8-mtp",
            best_models={},
            filename_models=["foundation"],
        )
        report = audit_configured_models(installed=["foundation"])
        assert report["missing"] == ["default_model = qwen3.6-35b-a3b-mxfp8-mtp"]

    def test_format_is_silent_on_a_clean_config(self):
        """`ev` prints this on every run, so a clean config must cost zero lines."""
        assert format_audit({"installed": ["a"], "missing": [], "unreachable": False}) == []

    def test_format_is_silent_when_the_server_is_unreachable(self):
        """`ev` already says the server is down; repeating it is noise, not diagnosis."""
        assert format_audit({"installed": [], "missing": [], "unreachable": True}) == []

    def test_format_names_every_stale_slot(self):
        lines = format_audit(
            {
                "installed": [],
                "missing": ["default_model = ghost-70b", "best_models.think = ghost-70b"],
                "unreachable": False,
            }
        )
        body = "\n".join(lines)
        # Assert the COUNT and the slot names, not the sentence around them. Pinning
        # the exact phrasing made this fail on a reword that changed nothing it cares
        # about ("2 model(s)" -> "2 configured model(s)").
        assert "2" in body
        assert "default_model = ghost-70b" in body
        assert "best_models.think = ghost-70b" in body

    def test_an_installed_model_lands_in_installed(self, fake_config):
        fake_config(
            default_model="foundation", best_models={}, filename_models=["foundation"]
        )
        report = audit_configured_models(installed=["foundation"])
        assert report["installed"] == [
            "default_model = foundation",
            "filename_models[0] = foundation",
        ]
