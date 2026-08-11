"""Reading capabilities off disk, which is where the real answers live.

The gate runs with MLX_MODELS_DIR pointing at nothing, so every disk-probing
path in lib/model_caps.py went unexercised — the module's whole reason to exist.
These tests build model trees in tmp and probe them for real.
"""

import json

import pytest
from lib import model_caps


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    """An MLXModels-shaped tree, with the caches cleared around it."""
    root = tmp_path / "MLXModels"
    root.mkdir()
    monkeypatch.setattr(model_caps, "MODELS_DIR", root)
    monkeypatch.setattr(model_caps, "HF_CACHE_DIR", tmp_path / "hub")
    for fn in (
        model_caps.model_config_path,
        model_caps.probe_context_window,
        model_caps.is_generative_model,
    ):
        fn.cache_clear()
    yield root
    for fn in (
        model_caps.model_config_path,
        model_caps.probe_context_window,
        model_caps.is_generative_model,
    ):
        fn.cache_clear()


def write_model(root, org, name, config):
    d = root / org / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config))
    return d


class TestFindingTheConfigOnDisk:
    def test_served_id_matches_a_differently_cased_directory(self, models_dir):
        """Osaurus serves "gemma-4-12b-it-mxfp8"; the directory keeps its capitals."""
        write_model(models_dir, "Google", "gemma-4-12B-it-MXFP8", {"max_position_embeddings": 131072})

        found = model_caps.model_config_path("gemma-4-12b-it-mxfp8")

        assert found is not None and found.parent.name == "gemma-4-12B-it-MXFP8"

    def test_huggingface_cache_layout_is_searched_too(self, tmp_path, monkeypatch):
        """potion-base-4m exists only in the HF cache, never in MLXModels."""
        hub = tmp_path / "hub"
        snap = hub / "models--minishlab--potion-base-4M" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps({"model_type": "model2vec"}))
        monkeypatch.setattr(model_caps, "MODELS_DIR", tmp_path / "nothing")
        monkeypatch.setattr(model_caps, "HF_CACHE_DIR", hub)
        model_caps.model_config_path.cache_clear()

        assert model_caps.model_config_path("potion-base-4m") is not None

    def test_unknown_model_is_not_found(self, models_dir):
        assert model_caps.model_config_path("no-such-model") is None

    def test_empty_name_is_not_found(self, models_dir):
        assert model_caps.model_config_path("") is None


class TestReadingTheContextWindow:
    def test_top_level_key(self, models_dir):
        write_model(models_dir, "Org", "big-model", {"max_position_embeddings": 262144})
        assert model_caps.probe_context_window("big-model") == 262144

    def test_nested_under_text_config(self, models_dir):
        """Multimodal configs bury the language window one level down."""
        write_model(
            models_dir, "Org", "vision-model", {"text_config": {"max_position_embeddings": 131072}}
        )
        assert model_caps.probe_context_window("vision-model") == 131072

    def test_unknown_model_reports_none_not_a_guess(self, models_dir):
        assert model_caps.probe_context_window("absent-model") is None

    def test_unreadable_config_reports_none(self, models_dir):
        d = models_dir / "Org" / "broken-model"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{not json at all")

        assert model_caps.probe_context_window("broken-model") is None

    def test_config_without_any_length_key_reports_none(self, models_dir):
        write_model(models_dir, "Org", "quiet-model", {"architectures": ["LlamaForCausalLM"]})
        assert model_caps.probe_context_window("quiet-model") is None

    def test_documented_window_from_config_wins_over_disk(self, models_dir, monkeypatch):
        """foundation has no config.json anywhere; the toml is the only source."""
        monkeypatch.setattr(
            model_caps, "_documented_context_window", lambda m: 4096 if m == "foundation" else None
        )
        model_caps.probe_context_window.cache_clear()

        assert model_caps.probe_context_window("foundation") == 4096


class TestNonGenerativeModelsAreIdentifiedByWhatTheyAre:
    """potion-base-4M declares seq_length: 1000000 and cannot generate a word."""

    def test_embedding_model_reports_no_context_window(self, models_dir):
        write_model(
            models_dir, "minishlab", "potion-base-4m", {"model_type": "model2vec", "seq_length": 1000000}
        )

        assert model_caps.probe_context_window("potion-base-4m") is None, (
            "a static embedding model's seq_length is not a context window"
        )

    def test_embedding_model_is_not_generative(self, models_dir):
        write_model(models_dir, "minishlab", "potion-base-4m", {"model_type": "model2vec"})
        assert model_caps.is_generative_model("potion-base-4m") is False

    def test_architecture_also_identifies_it(self, models_dir):
        write_model(models_dir, "Org", "static-model", {"architectures": ["StaticModel"]})
        assert model_caps.is_generative_model("static-model") is False

    def test_a_chat_model_is_generative(self, models_dir):
        write_model(models_dir, "Org", "chat-model", {"architectures": ["Gemma3ForCausalLM"]})
        assert model_caps.is_generative_model("chat-model") is True

    def test_unknown_model_is_assumed_generative(self, models_dir):
        """foundation is not on disk; skipping it would be worse than probing it."""
        assert model_caps.is_generative_model("foundation") is True

    def test_unreadable_config_is_assumed_generative(self, models_dir):
        d = models_dir / "Org" / "broken-model"
        d.mkdir(parents=True)
        (d / "config.json").write_text("{{{")

        assert model_caps.is_generative_model("broken-model") is True


class TestWhatGetsSent:
    def test_the_whole_probed_window_is_used(self, models_dir):
        """No time-based throttle: a big window is a big prompt.

        This used to be capped at MAX_PREFILL_SECONDS x a measured rate, which
        handed back ~46,000 of 262,144 tokens to save time that these tools --
        running every six hours at most -- were never short of.
        """
        write_model(models_dir, "Org", "huge-model", {"max_position_embeddings": 262144})

        assert model_caps.usable_context_window("huge-model", 8192) == 262144

    def test_small_window_is_used_whole(self, models_dir):
        write_model(models_dir, "Org", "small-model", {"max_position_embeddings": 4096})

        assert model_caps.usable_context_window("small-model", 8192) == 4096

    def test_override_wins_over_the_probe(self, models_dir):
        """The one legitimate way to shrink a window: a per-model config entry,
        written down with the evidence that a shorter prompt scored better."""
        write_model(models_dir, "Org", "some-model", {"max_position_embeddings": 131072})

        assert model_caps.usable_context_window("some-model", 8192, override=16384) == 16384

    def test_unprobeable_model_falls_back_to_the_caller_default(self, models_dir):
        assert model_caps.usable_context_window("absent-model", 8192) == 8192
