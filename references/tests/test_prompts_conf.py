"""The shared prompt store (`conf/prompts.toml`) is the single home for prompt
texts that both Rust and Python use. The Rust drift gate lives in the Rust crate;
here we pin the Python-side contract: the file exists, parses, and serves the
twitter summarize instructions the eval harness wraps its timeline into, and a
missing file is a loud error rather than a silent built-in copy."""

import pytest
from lib.prompts_conf import load_prompt, load_prompts_conf


def test_twitter_summarize_instructions_are_served():
    conf = load_prompts_conf()
    instructions = conf["twitter"]["summarize"]["instructions"]
    assert "You are an objective news distillation system" in instructions
    assert "End EVERY bullet with the author handle and timestamp" in instructions
    assert "<timeline>" not in instructions, "the fixture timeline is eval data, not a prompt"
    assert load_prompt("twitter", "summarize") == instructions


def test_missing_file_raises_not_fallback(monkeypatch, tmp_path):
    import lib.paths as paths
    import lib.prompts_conf as pc

    monkeypatch.setattr(paths, "conf_path", lambda *parts: tmp_path / "prompts.toml")
    monkeypatch.setattr(pc, "conf_path", paths.conf_path)
    with pytest.raises(FileNotFoundError):
        load_prompts_conf()
