"""Deriving what a model IS, instead of guessing it from the name.

Each of these pins a conclusion that was previously reached by hand, in a terminal,
and written into a markdown file that then went stale. The point of the module is
that the derivation can be re-run; the point of these tests is that the derivation
keeps giving the answer the hand investigation gave.
"""

import json

import pytest
from eval import capabilities as caps
from lib import model_caps


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    root = tmp_path / "MLXModels"
    root.mkdir()
    monkeypatch.setattr(model_caps, "MODELS_DIR", root)
    monkeypatch.setattr(model_caps, "HF_CACHE_DIR", tmp_path / "hub")
    for fn in (
        model_caps.model_config_path,
        model_caps.probe_context_window,
        model_caps.is_generative_model,
        model_caps.probe_vision,
        model_caps.model_disk_bytes,
    ):
        fn.cache_clear()
    yield root
    for fn in (
        model_caps.model_config_path,
        model_caps.probe_context_window,
        model_caps.is_generative_model,
        model_caps.probe_vision,
        model_caps.model_disk_bytes,
    ):
        fn.cache_clear()


def write_model(root, org, name, config, shard_sizes=()):
    d = root / org / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config))
    for i, size in enumerate(shard_sizes):
        _write_sparse(d / f"model-{i:05d}.safetensors", size)
    return d


def _write_sparse(path, size):
    """Create a file that REPORTS `size` without occupying it.

    These shards stand in for real model weights, and the realistic sizes are
    enormous: test_memory_estimate_uses_disk_not_the_parameter_count_in_the_name
    asks for a 15GiB shard because that is what the qwen3.8-27b-4bit build
    actually occupies, and the number is the point of the test.

    It used to do `write_bytes(b"\0" * size)`, which both materialised a 15GiB
    bytes object in RAM and wrote 15GiB of real zeros to $TMPDIR -- every run.
    pytest keeps several numbered tmp roots, so this accumulated: 26 of them,
    246GB, which filled the disk to 100% and started failing unrelated builds.

    A sparse file is not a shortcut here, it is exactly equivalent. The code
    under test is model_caps.model_disk_bytes, which sums
    `weights.stat().st_size` and never opens the file. st_size is identical
    either way; only the allocated blocks differ.
    """
    with open(path, "wb") as fh:
        fh.truncate(size)


class TestVisionIsReadFromTheModel:
    def test_a_top_level_vision_config_is_found(self, models_dir):
        write_model(models_dir, "OsaurusAI", "gemma-4-12B-it-MXFP8", {"vision_config": {}})
        assert model_caps.probe_vision("gemma-4-12b-it-mxfp8") is True

    def test_a_nested_vision_config_is_found(self, models_dir):
        """qwen3_5 puts the language side under text_config; vision can nest too."""
        write_model(models_dir, "OsaurusAI", "Nested", {"text_config": {"vision_config": {}}})
        assert model_caps.probe_vision("nested") is True

    def test_a_text_only_model_reports_false(self, models_dir):
        write_model(models_dir, "OsaurusAI", "Nemotron", {"model_type": "nemotron_h"})
        assert model_caps.probe_vision("nemotron") is False

    def test_an_unknown_model_reports_none_not_false(self, models_dir):
        """'Cannot tell' must not read as 'text-only', or foundation loses vision."""
        assert model_caps.probe_vision("not-on-disk") is None

    def test_the_name_heuristic_it_replaces_gets_these_wrong(self, models_dir):
        """The bug this exists to kill, stated as a test.

        DEFAULT_VLM_KEYWORDS is 'vl,vision,qwen,llamavl'. On the 2026-08 roster it
        matches nothing in 'gemma-4-12b-it-mxfp8' though that model has a vision
        tower, and matches nothing to exclude nemotron though nemotron has none.
        """
        from lib.osaurus_models import DEFAULT_VLM_KEYWORDS

        write_model(models_dir, "OsaurusAI", "gemma-4-12B-it-MXFP8", {"vision_config": {}})
        name_says_vision = any(k in "gemma-4-12b-it-mxfp8" for k in DEFAULT_VLM_KEYWORDS)
        probe_says_vision = model_caps.probe_vision("gemma-4-12b-it-mxfp8")

        assert probe_says_vision is True
        assert name_says_vision is False, "if this ever passes, the name heuristic improved"


class TestDiskSizeIsTheUsableInstrument:
    def test_it_sums_only_weight_shards(self, models_dir):
        write_model(models_dir, "OsaurusAI", "Sharded", {}, shard_sizes=(1000, 2000, 3000))
        assert model_caps.model_disk_bytes("sharded") == 6000

    def test_a_model_not_on_disk_is_none(self, models_dir):
        assert model_caps.model_disk_bytes("ghost") is None

    def test_a_config_with_no_shards_is_none_not_zero(self, models_dir):
        """foundation has no weight files here; 0 bytes would read as 'tiny, fits'."""
        write_model(models_dir, "OsaurusAI", "Weightless", {})
        assert model_caps.model_disk_bytes("weightless") is None


class TestFamilyComesFromTheServerNotTheName:
    def test_the_reported_family_wins(self):
        roster = [{"model": "ornith-1.0-9b-mxfp8", "details": {"family": "qwen3_5"}}]
        got = caps.probe_static_capabilities("ornith-1.0-9b-mxfp8", roster)
        assert got["family"] == "qwen3_5"

    def test_a_model_the_server_did_not_list_reports_none(self):
        assert caps.probe_static_capabilities("ghost", [])["family"] is None

    def test_a_populated_roster_that_lacks_the_model_also_reports_none(self):
        """Distinct path from an empty roster, and the likelier one in practice."""
        roster = [{"model": "someone-else", "details": {"family": "gemma4"}}]
        assert caps.roster_entry("ghost", roster) == {}
        assert caps.probe_static_capabilities("ghost", roster)["family"] is None

    def test_the_probe_is_more_specific_than_the_name(self):
        """ornith IS qwen3_5, so it should get conf/models/qwen.toml prompts.

        Asserted as "the probe knows something the name does not" rather than by
        pinning get_model_family's current answer, so that rewiring the matcher onto
        this probe does not require editing this test to keep it honest.
        """
        from lib.config import get_model_family

        roster = [{"model": "ornith-1.0-9b-mxfp8", "details": {"family": "qwen3_5"}}]
        probed = caps.probe_static_capabilities("ornith-1.0-9b-mxfp8", roster)["family"]

        assert probed == "qwen3_5"
        assert get_model_family("ornith-1.0-9b-mxfp8") != "qwen3_5", (
            "name matching cannot reach the real family from 'ornith-*'"
        )


class TestViabilityIsArithmeticNotOpinion:
    def test_required_rate_is_expected_output_over_timeout(self):
        assert caps.required_decode_rate(1536, 600) == pytest.approx(2.56, rel=1e-3)

    def test_a_zero_timeout_yields_none_rather_than_dividing(self):
        assert caps.required_decode_rate(1536, 0) is None

    def test_an_unmeasured_model_is_unknown_not_bad(self):
        """Absence of a measurement must not be read as a failing measurement."""
        got = caps.assess_viability({}, 1536, 600)
        assert got["verdict"] == "unknown"

    def test_a_thrashing_model_is_named_as_such(self):
        """qwen3.8-27b-mxfp8's real number: 0.08 tok/s."""
        got = caps.assess_viability({"decode_tokens_per_sec": 0.08}, 1536, 600)
        assert got["verdict"] == "thrashing"

    def test_sizing_against_max_tokens_would_condemn_a_known_good_model(self):
        """The bug this file's expected_output_tokens() exists to prevent.

        gemma-4-12b measures 18.27 tok/s and scores 100% on summarize. Against the
        16000-token CAP it reads "too_slow"; against ~1000 observed output tokens it
        reads "ok". The second is the true one, so the cap is the wrong divisor --
        an instrument that condemns a known-good model is measuring the wrong thing
        however correct its arithmetic.
        """
        against_cap = caps.assess_viability({"decode_tokens_per_sec": 18.27}, 16000, 600)
        against_observed = caps.assess_viability({"decode_tokens_per_sec": 18.27}, 1536, 600)

        assert against_cap["verdict"] == "too_slow"
        assert against_observed["verdict"] == "ok"

    def test_a_merely_slow_model_is_distinguished_from_a_thrashing_one(self):
        """The remedy differs: a smaller budget vs a smaller quant."""
        got = caps.assess_viability({"decode_tokens_per_sec": 1.5}, 1536, 600)
        assert got["verdict"] == "too_slow"

    def test_a_fast_enough_model_passes(self):
        got = caps.assess_viability({"decode_tokens_per_sec": 35.4}, 1536, 600)
        assert got["verdict"] == "ok"

    def test_the_4bit_and_mxfp8_builds_of_one_model_get_opposite_verdicts(self):
        """The finding that motivated all of this, as a regression test."""
        mxfp8 = caps.assess_viability({"decode_tokens_per_sec": 0.08}, 1536, 600)
        four_bit = caps.assess_viability({"decode_tokens_per_sec": 35.4}, 1536, 600)
        assert mxfp8["verdict"] == "thrashing"
        assert four_bit["verdict"] == "ok"

    def test_expected_output_comes_from_the_measured_production_constant(self):
        """Imported, not copied, so the two cannot drift."""
        from twitter.budget import OUTPUT_RESERVE_TOKENS

        assert caps.expected_output_tokens() == OUTPUT_RESERVE_TOKENS


class TestExplanationsNameTheDecidingNumber:
    def test_thrashing_points_at_the_quant_not_the_kernel(self):
        line = caps.explain_viability(
            "qwen3.8-27b-mxfp8",
            caps.assess_viability({"decode_tokens_per_sec": 0.08}, 1536, 600),
            27 * 1_073_741_824,
        )
        assert "27.0GB" in line
        assert "SMALLER QUANT" in line

    def test_unknown_tells_you_the_command_to_run(self):
        line = caps.explain_viability("m", caps.assess_viability({}, 1536, 600), None)
        assert "ev --model m" in line

    def test_a_passing_model_still_says_what_it_cleared(self):
        """The 'ok' line has to carry the numbers too, or a green verdict is a
        claim with nothing behind it."""
        line = caps.explain_viability(
            "qwen3.8-27b-4bit",
            caps.assess_viability({"decode_tokens_per_sec": 35.4}, 1536, 600),
            15 * 1_073_741_824,
        )
        assert "35.4 tok/s" in line
        assert "2.56" in line

    def test_too_slow_reports_what_the_budget_would_cost(self):
        line = caps.explain_viability(
            "bonsai", caps.assess_viability({"decode_tokens_per_sec": 1.5}, 1536, 600), None
        )
        assert "1024.0s" in line


class TestTheReportReplacesTheHandWrittenTable:
    def test_it_merges_static_probes_with_recorded_measurements(self, models_dir):
        write_model(models_dir, "OsaurusAI", "M", {"vision_config": {}}, shard_sizes=(4096,))
        roster = [{"model": "m", "details": {"family": "qwen3_5", "parameter_size": "9B"}}]
        signals = {"m": {"_capabilities": {"decode_tokens_per_sec": 44.0, "prefill_samples": 2}}}

        (row,) = caps.capability_report(["m"], roster, signals)

        assert row["model"] == "m"
        assert row["family"] == "qwen3_5"
        assert row["parameter_size"] == "9B"
        assert row["vision"] is True
        assert row["disk_bytes"] == 4096
        assert row["decode_tokens_per_sec"] == 44.0
        assert row["prefill_samples"] == 2

    def test_a_model_with_no_measurements_still_gets_a_row(self, models_dir):
        (row,) = caps.capability_report(["ghost"], [], {})
        assert row["model"] == "ghost"
        assert row["decode_tokens_per_sec"] is None


class TestTheProbesAreActuallyWiredIn:
    """The probes existing is not the same as the code using them.

    Each of these covers a production path that used to key on the model name and
    now reads a probed fact. Without them the module would be correct and unused.
    """

    def test_family_routing_prefers_the_recorded_architecture(self, models_dir, monkeypatch):
        """ornith is qwen3_5, so it must reach conf/models/qwen.toml."""
        from lib import config_getters

        monkeypatch.setattr(
            "lib.model_caps.recorded_capability",
            lambda model, key: "qwen3_5" if key == "family" else None,
        )
        assert config_getters.get_model_family("ornith-1.0-9b-mxfp8") == "qwen"

    def test_family_routing_falls_back_to_the_name_when_nothing_was_recorded(
        self, monkeypatch
    ):
        """Never depends on the eval having been run."""
        from lib import config_getters

        monkeypatch.setattr("lib.model_caps.recorded_capability", lambda model, key: None)
        assert config_getters.get_model_family("gemma-3-4b") == "gemma"
        assert config_getters.get_model_family("totally-unknown-7b") == "default"

    def test_an_architecture_with_no_config_file_falls_back_to_the_name(self, monkeypatch):
        """muse_glimmer has no conf/models entry; the mapping must not invent one."""
        from lib import config_getters

        monkeypatch.setattr(
            "lib.model_caps.recorded_capability",
            lambda model, key: "muse_glimmer" if key == "family" else None,
        )
        assert config_getters.get_model_family("muse-glimmer-30b-jang_6m") == "default"

    def test_vlm_selection_reads_the_vision_tower_not_the_name(self, models_dir):
        """The exact roster case the keyword list gets backwards."""
        from lib.osaurus_models import select_best_vlm_model

        write_model(models_dir, "OsaurusAI", "nemotron-30b", {"model_type": "nemotron_h"})
        write_model(models_dir, "OsaurusAI", "gemma-4-12B-it-MXFP8", {"vision_config": {}})

        # nemotron sorts first and has no vision; gemma has one and no matching keyword.
        assert select_best_vlm_model(["nemotron-30b", "gemma-4-12b-it-mxfp8"]) == (
            "gemma-4-12b-it-mxfp8"
        )

    def test_vlm_selection_still_guesses_for_models_not_on_disk(self, models_dir):
        """foundation has no config.json; refusing outright would be worse."""
        from lib.osaurus_models import select_best_vlm_model

        assert select_best_vlm_model(["something-qwen-vl"]) == "something-qwen-vl"

    def test_vlm_selection_returns_none_when_nothing_qualifies(self, models_dir):
        from lib.osaurus_models import select_best_vlm_model

        write_model(models_dir, "OsaurusAI", "textonly", {"model_type": "nemotron_h"})
        assert select_best_vlm_model(["textonly"]) is None

    def test_memory_estimate_uses_disk_not_the_parameter_count_in_the_name(
        self, models_dir
    ):
        """The two qwen3.8 builds are both "27b" and occupy 15GB and 27GB."""
        from eval.cli_runtime import estimate_model_memory

        write_model(
            models_dir, "OsaurusAI", "qwen3.8-27b-4bit", {}, shard_sizes=(15 * 1_073_741_824,)
        )
        assert estimate_model_memory("qwen3.8-27b-4bit") == 15

    def test_memory_estimate_falls_back_to_the_name_off_disk(self, models_dir):
        from eval.cli_runtime import estimate_model_memory

        assert estimate_model_memory("mystery-13b") == 13
        assert estimate_model_memory("no-size-here") == 4


class TestRecordingMakesProbesAvailableOffline:
    def test_it_persists_the_static_facts(self, models_dir, monkeypatch, tmp_path):
        import eval.signals as signals

        monkeypatch.setattr(signals, "EVAL_SIGNALS_PATH", tmp_path / "sig.json")
        write_model(
            models_dir, "OsaurusAI", "M", {"model_type": "qwen3_5", "vision_config": {}},
            shard_sizes=(2048,),
        )
        caps.record_static_capabilities("m")

        stored = signals._load_eval_signals()["m"]["_capabilities"]
        assert stored["family"] == "qwen3_5"
        assert stored["vision"] is True
        assert stored["disk_bytes"] == 2048

    def test_it_overwrites_rather_than_keeping_the_oldest(self, models_dir, monkeypatch, tmp_path):
        """Unlike the rate recorders: family and vision are facts, not samples."""
        import eval.signals as signals

        monkeypatch.setattr(signals, "EVAL_SIGNALS_PATH", tmp_path / "sig.json")
        signals._save_eval_signals({"m": {"_capabilities": {"family": "stale_arch"}}})
        write_model(models_dir, "OsaurusAI", "M", {"model_type": "qwen3_5"})

        caps.record_static_capabilities("m")

        assert signals._load_eval_signals()["m"]["_capabilities"]["family"] == "qwen3_5"

    def test_it_leaves_measured_rates_alone(self, models_dir, monkeypatch, tmp_path):
        import eval.signals as signals

        monkeypatch.setattr(signals, "EVAL_SIGNALS_PATH", tmp_path / "sig.json")
        signals._save_eval_signals({"m": {"_capabilities": {"prefill_chars_per_sec": 900.0}}})
        write_model(models_dir, "OsaurusAI", "M", {"model_type": "qwen3_5"})

        caps.record_static_capabilities("m")

        assert signals._load_eval_signals()["m"]["_capabilities"]["prefill_chars_per_sec"] == 900.0


class TestTheCliSurface:
    """`ev --capabilities` — the command that makes these conclusions reproducible."""

    def printed(self, monkeypatch, roster, signals=None):
        import eval.cli as cli
        import lib.model_resolve as resolve
        from eval import cli_runtime

        lines = []
        monkeypatch.setattr(resolve, "fetch_roster", lambda *a, **k: roster)
        monkeypatch.setattr(
            "eval.signals._load_eval_signals", lambda: signals or {}, raising=False
        )
        monkeypatch.setattr(
            cli_runtime.console, "print", lambda *a, **k: lines.append(str(a[0]) if a else "")
        )
        cli._print_capabilities()
        return "\n".join(lines)

    def test_an_unreachable_server_says_so_instead_of_printing_an_empty_table(
        self, monkeypatch
    ):
        """An empty table would read as "no models installed", which is a different
        and much more alarming claim than "the server is down"."""
        out = self.printed(monkeypatch, roster=[])
        assert "not reachable" in out

    def test_it_prints_a_row_per_served_model(self, monkeypatch):
        roster = [
            {"model": "a", "details": {"family": "qwen3_5", "parameter_size": "9B"}},
            {"model": "b", "details": {"family": "gemma4", "parameter_size": "2B"}},
        ]
        out = self.printed(monkeypatch, roster, {"a": {"_capabilities": {"decode_tokens_per_sec": 40}}})

        assert "qwen3_5" in out and "gemma4" in out
        assert "model" in out and "verdict" in out

    def test_it_also_flags_a_stale_config_slot(self, monkeypatch):
        """The capability probe and the config audit answer one question together:
        can the tools actually run right now."""
        import lib.config as cfg

        monkeypatch.setattr(cfg, "get_best_models", lambda: {"think": "ghost-70b"})
        monkeypatch.setattr(cfg, "get_filename_models", lambda: ["a"])
        out = self.printed(monkeypatch, [{"model": "a", "details": {"family": "qwen3_5"}}])

        assert "ghost-70b" in out


class TestTheTableIsGeneratedNotMaintained:
    def rows(self):
        return [
            {
                "model": "qwen3.8-27b-mxfp8",
                "family": "qwen3_5",
                "parameter_size": "27B",
                "disk_bytes": 27 * 1_073_741_824,
                "vision": True,
                "generative": True,
                "prefill_chars_per_sec": 116.7,
                "decode_tokens_per_sec": 0.08,
            },
            {
                "model": "qwen3.8-27b-4bit",
                "family": "qwen3_5",
                "parameter_size": "27B",
                "disk_bytes": 15 * 1_073_741_824,
                "vision": True,
                "generative": True,
                "prefill_chars_per_sec": 449.5,
                "decode_tokens_per_sec": 26.08,
            },
        ]

    def test_each_model_gets_a_row_with_its_verdict(self):
        lines = caps.format_capability_table(self.rows(), 1536, 600)
        body = "\n".join(lines)
        assert "qwen3.8-27b-mxfp8" in body
        assert "thrashing" in body
        assert "27.0" in body and "15.0" in body

    def test_only_problem_models_get_an_explanatory_note(self):
        """A note per model would bury the one that needs acting on.

        Notes live after the blank separator; table rows also begin with the model
        name, so splitting on that separator is what distinguishes them.
        """
        lines = caps.format_capability_table(self.rows(), 1536, 600)
        notes = lines[lines.index("") + 1 :]

        assert len(notes) == 1
        assert notes[0].startswith("qwen3.8-27b-mxfp8")
        assert not any(n.startswith("qwen3.8-27b-4bit") for n in notes), (
            "a model with an ok verdict needs no note"
        )

    def test_a_non_generative_model_is_called_out_as_unrankable(self):
        rows = [{"model": "potion-base-4m", "generative": False, "vision": False}]
        body = "\n".join(caps.format_capability_table(rows, 1536, 600))
        assert "not a generative model" in body

    def test_unknown_vision_renders_as_a_question_not_a_no(self):
        rows = [{"model": "foundation", "vision": None, "decode_tokens_per_sec": 66.0}]
        body = "\n".join(caps.format_capability_table(rows, 1536, 600))
        assert "  ?" in body
