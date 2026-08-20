"""Tests for detecting broken / defective model artifacts before evaluation."""

import json

import pytest
from eval import capabilities as caps
from eval.cli_runtime import broken_model_refusal
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
        model_caps.probe_model_defects,
    ):
        fn.cache_clear()
    yield root
    for fn in (
        model_caps.model_config_path,
        model_caps.probe_context_window,
        model_caps.is_generative_model,
        model_caps.probe_vision,
        model_caps.model_disk_bytes,
        model_caps.probe_model_defects,
    ):
        fn.cache_clear()


def make_model(root, org, name, config=None, files=None):
    d = root / org / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config or {"model_type": "qwen3_5"}))
    for fname, content in (files or {}).items():
        p = d / fname
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            p.write_text(content)
        else:
            p.write_bytes(content)
    return d


class TestModelDefectDetection:
    def test_clean_model_has_no_defects(self, models_dir):
        make_model(
            models_dir,
            "mlx-community",
            "Qwen3.8-27B-8bit",
            {"model_type": "qwen3_5"},
            {"model-00001-of-00006.safetensors": b"123"},
        )
        defects = model_caps.probe_model_defects("qwen3.8-27b-8bit")
        assert defects == []
        assert broken_model_refusal("qwen3.8-27b-8bit") == ""

    def test_unsupported_mtp_shard_is_detected(self, models_dir):
        jang_cfg = {
            "mtp": {"runtime_available": False},
            "runtime": {"mtp_mode": "preserved_enabled"},
        }
        make_model(
            models_dir,
            "OsaurusAI",
            "Qwen3.8-27B-MXFP8",
            {"model_type": "qwen3_5"},
            {
                "model-mtp-of-00007.safetensors": b"mtp_weights",
                "jang_config.json": json.dumps(jang_cfg),
            },
        )
        defects = model_caps.probe_model_defects("qwen3.8-27b-mxfp8")
        assert len(defects) >= 1
        assert "unsupported MTP speculative shard" in defects[0]
        assert "runtime_available=false" in defects[0]

        refusal = broken_model_refusal("qwen3.8-27b-mxfp8")
        assert "unsupported MTP" in refusal

    def test_unintegrated_standalone_mtp_shard_is_detected(self, models_dir):
        make_model(
            models_dir,
            "OsaurusAI",
            "StandaloneMTP",
            {"model_type": "qwen3_5"},
            {"model-mtp-00001.safetensors": b"mtp_weights"},
        )
        defects = model_caps.probe_model_defects("standalonemtp")
        assert len(defects) >= 1
        assert "unintegrated MTP speculative shard" in defects[0]

    def test_missing_safetensor_shards_detected_from_index(self, models_dir):
        index = {
            "weight_map": {
                "layer.0": "model-00001-of-00002.safetensors",
                "layer.1": "model-00002-of-00002.safetensors",
            }
        }
        make_model(
            models_dir,
            "OsaurusAI",
            "IncompleteShards",
            {"model_type": "qwen3_5"},
            {
                "model.safetensors.index.json": json.dumps(index),
                "model-00001-of-00002.safetensors": b"shard1",
                # shard2 is intentionally missing
            },
        )
        defects = model_caps.probe_model_defects("incompleteshards")
        assert len(defects) >= 1
        assert "missing 1 safetensor shard(s)" in defects[0]
        assert "model-00002-of-00002.safetensors" in defects[0]

    def test_incomplete_download_artifacts_detected(self, models_dir):
        make_model(
            models_dir,
            "OsaurusAI",
            "HalfDownloaded",
            {"model_type": "qwen3_5"},
            {"model-00001.safetensors.incomplete": b"partial"},
        )
        defects = model_caps.probe_model_defects("halfdownloaded")
        assert len(defects) >= 1
        assert "incomplete download artifacts" in defects[0]

    def test_nonexistent_model_has_no_defects(self):
        assert model_caps.probe_model_defects("nonexistent-model") == []
        assert broken_model_refusal("nonexistent-model") == ""

    def test_broken_refusal_respects_allow_flag(self, models_dir):
        make_model(
            models_dir,
            "OsaurusAI",
            "BrokenAllowed",
            {"model_type": "qwen3_5"},
            {"shard.incomplete": b"123"},
        )
        assert broken_model_refusal("brokenallowed", allow=False) != ""
        assert broken_model_refusal("brokenallowed", allow=True) == ""


class TestCapabilitiesBrokenVerdict:
    def test_assess_viability_marks_broken_when_defects_present(self):
        row = {
            "model": "broken-qwen",
            "decode_tokens_per_sec": 15.0,
            "defects": ["unsupported MTP speculative shard"],
        }
        assessment = caps.assess_viability(row, 1000, 600)
        assert assessment["verdict"] == "broken"
        assert assessment["defects"] == ["unsupported MTP speculative shard"]

        explanation = caps.explain_viability("broken-qwen", assessment, 20 * 1024**3)
        assert "BROKEN" in explanation
        assert "unsupported MTP speculative shard" in explanation

    def test_format_capability_table_includes_broken_verdict_and_note(self):
        rows = [
            {
                "model": "broken-model",
                "family": "qwen3_5",
                "parameter_size": "27B",
                "disk_bytes": 27 * 1024**3,
                "vision": True,
                "generative": True,
                "defects": ["missing 2 safetensor shard(s)"],
            }
        ]
        lines = caps.format_capability_table(rows, 1000, 600)
        body = "\n".join(lines)
        assert "broken" in body
        assert "BROKEN — missing 2 safetensor shard(s)" in body
