import pytest


@pytest.mark.skip(reason="Integration test requiring running LLM server at localhost:1337")
class TestLlmServer:
    def test_call_llm_api(self):
        from lib.config import get_best_model, Task
        from lib.osaurus_lib import call_llm_api
        model = get_best_model(Task.JSON)
        sys_prompt = "Output JSON now."
        usr_prompt = "Extract popular Vaughan venues for families."
        res = call_llm_api(
            "http://localhost:1337",
            model,
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ],
            temperature=0.1,
            timeout=600,
            parse_json=False,
        )
        assert res is not None
        assert "content" in res
