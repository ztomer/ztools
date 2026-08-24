"""Guards for keeping the model's actual output.

The eval scored each answer and discarded it. When every model failed
`summarize_factual_coverage`, deciding whether that was the models or the
scorer needed one look at what a model had written -- and the only route back
to it was a ten-hour sweep on a machine that runs one model at a time.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import outputs as eo


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.delenv("EVAL_SAVE_OUTPUTS", raising=False)
    return tmp_path / "outputs"


def test_saving_is_on_unless_switched_off(out_dir, monkeypatch):
    """Default-on is the point: the failure mode is silent, permanent loss."""
    assert eo.outputs_enabled() is True
    monkeypatch.setenv("EVAL_SAVE_OUTPUTS", "0")
    assert eo.outputs_enabled() is False


def test_the_content_is_recoverable_with_its_verdict(out_dir):
    """The output alone is not enough -- the score is what you are questioning."""
    path = eo.save_output(
        "gemma-4-12b-it-mxfp8",
        "summarize_factual_coverage",
        {"content": "Shopify posted 40% growth in revenue."},
        16,
        "covered 3/18 key facts",
    )
    assert path is not None and path.exists()
    text = path.read_text()
    assert "Shopify posted 40% growth in revenue." in text
    assert "score: 16" in text
    assert "covered 3/18 key facts" in text


def test_reasoning_is_kept_apart_from_the_answer(out_dir):
    """For thinking models the answer is short and the reasoning explains the
    format failure, so collapsing them together loses the diagnosis."""
    path = eo.save_output(
        "qwen3.6-27b",
        "json",
        {"content": "{}", "reasoning_content": "First I should consider the schema"},
        0,
        "empty object",
    )
    text = path.read_text()
    assert "--- reasoning ---" in text
    assert "First I should consider the schema" in text
    assert text.index("{}") < text.index("--- reasoning ---")


def test_an_error_only_result_is_still_worth_keeping(out_dir):
    """A timeout or 503 is exactly the case someone re-reads later."""
    path = eo.save_output("m", "t", {"content": None, "error": "HTTP 503: at capacity"}, 0, "")
    assert path is not None
    assert "HTTP 503: at capacity" in path.read_text()


def test_nothing_at_all_writes_nothing(out_dir):
    assert eo.save_output("m", "t", {"content": "", "error": ""}, 0, "") is None
    assert eo.save_output("m", "t", None, 0, "") is None


def test_model_identifiers_cannot_escape_the_directory(out_dir):
    """Model names carry dots and slashes; one of them is `../`."""
    path = eo.save_output("../../etc/passwd", "../evil", {"content": "x"}, 0, "")
    assert path is not None
    assert out_dir.resolve() in path.resolve().parents
    assert ".." not in path.parts


def test_a_write_failure_never_takes_down_the_run(tmp_path, monkeypatch):
    """Losing one output is bad. Losing a ten-hour sweep to a full disk is worse."""
    monkeypatch.setenv("EVAL_OUTPUT_DIR", str(tmp_path / "out"))

    def boom(*a, **k):
        raise OSError("No space left on device")

    monkeypatch.setattr(Path, "write_text", boom)
    assert eo.save_output("m", "t", {"content": "x"}, 50, "") is None


def test_enormous_outputs_are_truncated(out_dir, monkeypatch):
    monkeypatch.setenv("EVAL_MAX_SAVED_OUTPUT", "100")
    monkeypatch.setattr(eo, "MAX_SAVED_CHARS", 100)
    path = eo.save_output("m", "t", {"content": "x" * 5000}, 0, "")
    body = path.read_text().split("---\n", 1)[1]
    assert len(body) == 100
    # The true length is still recorded, so truncation is visible, not silent.
    assert "chars: 5000" in path.read_text()


def test_the_suite_cannot_write_into_the_real_config_dir():
    """The autouse gate, checked rather than assumed.

    This feature escaped its sandbox within minutes of landing: every existing
    test that calls run_eval with a fake model wrote real files into
    ~/.config/ztools/outputs. `_saved_outputs_stay_in_tmp` in conftest redirects
    it for the whole session, and this asserts the redirect is actually in force
    -- a sandbox nobody verifies is a sandbox that quietly stops applying.

    Deliberately does NOT use the out_dir fixture: it must observe the session
    gate, not one this test installed.
    """
    from eval.report_core import default_eval_dir

    resolved = eo.outputs_dir().resolve()
    real = (default_eval_dir() / "outputs").resolve()
    assert resolved != real, "saved outputs are pointed at the real config dir"
    assert Path.home() not in resolved.parents or "pytest" in str(resolved), resolved


def test_a_real_run_eval_leaves_the_output_on_disk(out_dir, monkeypatch):
    """Drive run_eval itself, not the helper.

    Asserting that a seam is patchable proves only that the seam exists. What
    has to be true is that a run writes the file, so this runs the real
    `run_eval` against a mocked transport and then reads what landed.
    """
    from unittest.mock import patch

    import eval.run as er
    from eval import run_transport
    from eval.tasks_core import TASKS
    from lib.testing import MockLLM

    mock = MockLLM()
    mock.patch_all()
    try:
        with patch.object(run_transport, "call", mock.call):
            results = er.run_eval(
                "mock-model", tasks={"json": TASKS["json"]}, verbose=False,
                measure_prefill=False,
            )
    finally:
        mock.unpatch()

    saved = out_dir / "mock-model" / "json.txt"
    assert saved.exists(), f"run_eval wrote no output; dir holds {list(out_dir.rglob('*'))}"
    text = saved.read_text()
    assert f"score: {results[0]['quality_score']}" in text
    # The body must be the model's answer, not a summary of it.
    assert text.split("---\n", 1)[1].strip(), "output body is empty"
