"""Production LLM calls go through the shared client. Config model slots get audited.

Both halves of this file exist because `rn` sat OUTSIDE the perimeter every other
tool was inside, and the consequences were live rather than theoretical:

  * `is_relevant_with_llm` issued its own POST to /api/chat, so it got no model
    substitution. Its two configured models had been uninstalled for months, every
    call 404'd, and it returned None for EVERY image -- which the caller cannot
    distinguish from "the model had no opinion". The relevance feature was dead and
    silent. Verified against the live server before the fix, and after.

  * `conf/rename.toml` named two more uninstalled models under `vlm_preferred`.
    `audit_configured_models` only read `conf/config.toml`, so neither `ev` nor the
    TUI audit could see them. An audit that covers only the file you remembered is
    how drift stays invisible in the file you did not.

The tests are structural on purpose. Checking that today's call sites happen to be
correct is worth little; the point is that adding a new raw LLM POST, or a new
per-tool config naming models, has to fail loudly.
"""

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
#: Packages that ship to users. `lib/` is the shared client itself and is exempt.
PRODUCTION_PACKAGES = ("rename", "twitter", "weekend", "eval")
#: Endpoints that mean "this is an LLM call". Kept for the message only -- NOT for
#: detection. The first version of this gate skipped any file where none of these
#: appeared as a literal, and `rename/llm.py` referred to its endpoint through an
#: imported constant (API_CHAT_PATH = API_CHAT). The gate therefore read GREEN with
#: two raw POSTs sitting in the very file it was written to catch. Detection now keys
#: on the POST itself, which is the thing that actually bypasses the shared client.
LLM_ENDPOINTS = ("/api/chat", "/v1/chat/completions", "/api/generate")


def python_files():
    for pkg in PRODUCTION_PACKAGES:
        yield from (REPO / pkg).rglob("*.py")


class TestTheFixtureIsWiredCorrectly:
    """Calibration: if no files were collected the rule below would pass vacuously."""

    def test_production_files_are_found(self):
        files = list(python_files())
        assert len(files) > 10, f"only found {len(files)} production files"

    def test_the_shared_client_itself_is_not_scanned(self):
        """lib/ is where the raw POST is SUPPOSED to live."""
        assert not any("/lib/" in str(f) for f in python_files())

    def test_the_detector_would_catch_a_planted_violation(self, tmp_path):
        """Calibration. The first version of this gate could not see a raw POST whose
        endpoint came from an imported constant, and passed while two of them existed.
        A gate that cannot produce a failure is not a gate."""
        planted = tmp_path / "offender.py"
        planted.write_text("import requests\nrequests.post(SOME_CONSTANT, json={})\n")
        src = planted.read_text()
        assert "requests" in src and ".post(" in src


class TestNoProductionCodeCallsAnLlmEndpointDirectly:
    def test_no_raw_llm_post_outside_the_shared_client(self):
        offenders = []
        for path in python_files():
            src = path.read_text(errors="replace")
            if "requests" in src and ".post(" in src:
                offenders.append(str(path.relative_to(REPO)))
        assert offenders == [], (
            f"{offenders} POST to an LLM endpoint directly. Route through "
            "lib.osaurus_lib.call, which supplies model substitution, the streaming "
            "wall-clock deadline, per-model quirks and the Foundation fallback. A raw "
            "call gets none of them and fails silently -- that is how rn's relevance "
            "check stayed dead for months."
        )


class TestEveryConfigFileNamingModelsIsAudited:
    def test_rename_toml_slots_are_covered(self):
        from lib.model_resolve import SIDECAR_MODEL_KEYS

        assert "rename.toml" in SIDECAR_MODEL_KEYS

    def test_a_dead_model_in_a_sidecar_config_is_reported(self):
        from unittest.mock import patch

        from lib.model_resolve import audit_configured_models

        with (
            patch("lib.model_resolve._sidecar_model_slots",
                  return_value=[("rename.toml:vlm_preferred[0]", "long-gone-model")]),
            patch("lib.config.get_best_models", return_value={}),
            patch("lib.config.get_filename_models", return_value=[]),
        ):
            report = audit_configured_models(["live-model"])
        assert any("long-gone-model" in m for m in report["missing"]), report

    def test_the_slot_name_says_which_file_to_edit(self):
        """`best_models.json` and `rename.toml:vlm_preferred[0]` live in different
        files; a bare model name would not tell you where to go."""
        from lib.model_resolve import _sidecar_model_slots

        for slot, _model in _sidecar_model_slots():
            assert ".toml:" in slot, f"{slot!r} does not name its file"

    # NOTE: no "the real configs are currently clean" test here. It needs a live
    # roster, and references/tests/conftest.py blocks real server connections on
    # purpose -- docs/BACKLOG.md item 7 records that pytest cannot be the gate for
    # roster checks. That check runs at runtime, in `ev` and the TUI startup audit.


class TestSidecarParsingHandlesBothShapes:
    def test_a_comma_separated_string_is_split(self, tmp_path, monkeypatch):
        import lib.model_resolve as mr

        monkeypatch.setattr(mr, "SIDECAR_MODEL_KEYS", {"fake.toml": ("models",)})
        monkeypatch.setattr("lib.config_toml.load_config", lambda p: {"models": "a, b"})
        monkeypatch.setattr("lib.paths.conf_path", lambda *a: tmp_path / "fake.toml")
        assert [m for _s, m in mr._sidecar_model_slots()] == ["a", "b"]

    def test_a_missing_file_is_not_an_error(self, monkeypatch):
        import lib.model_resolve as mr

        monkeypatch.setattr(mr, "SIDECAR_MODEL_KEYS", {"nope.toml": ("models",)})
        assert mr._sidecar_model_slots() == []
