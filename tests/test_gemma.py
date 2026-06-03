import pytest


@pytest.mark.skip(reason="Integration test requiring running LLM server")
class TestGemmaModel:
    def test_gemma_call(self):
        from lib.osaurus_lib import call as osaurus_call
        from model_eval import TASKS
        task = TASKS["json"]
        result = osaurus_call(
            model="gemma-4-26b-a4b-it-4bit",
            messages=task["messages"],
            task="json",
            parse_json=True,
            max_retries=0,
        )
        # Real result should be a dict with content, time, and parsed
        assert isinstance(result, dict)
        assert "content" in result
        assert "time" in result
        assert isinstance(result["time"], (int, float))
        assert result["time"] >= 0
