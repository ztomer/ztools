"""
Mock LLM provider for integration tests.
Provides canned responses for all task types so tests don't need a running server.
"""

import re
import json
import unittest.mock
from typing import Any, Callable, Dict, Optional

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)



def _default_content_for(task: str) -> str:
    if task in ("json", "weekend_transient"):
        return json.dumps([
            {"name": "Spring Festival", "location": "Toronto", "target_ages": "All",
             "price": "Free", "weather": "outdoor", "day": "Saturday"},
            {"name": "Indoor Coding Workshop", "location": "Vaughan", "target_ages": "8-14",
             "price": "$25", "weather": "indoor", "day": "Sunday"},
        ])
    if task in ("filename", "image_rename"):
        return "mock_test_filename"
    if task == "summarize":
        return "## Summary\n- OpenAI announced GPT-5\n- Apple Vision Pro 2 enters production\n- Google unveils Gemini 2.5 Pro\n"
    if task == "file_summary":
        return json.dumps([
            {"path": "eval_lib.py", "desc": "evaluates model quality across tasks"},
            {"path": "validators.py", "desc": "validates JSON output format"},
            {"path": "config.py", "desc": "manages configuration loading"},
            {"path": "osaurus_lib.py", "desc": "LLM API client library"},
        ])
    if task in ("weekend_fixed", "detailed_json"):
        return json.dumps([
            {"name": "Vaughan Sports Arena", "location": "Vaughan", "target_ages": "6-13",
             "price": "$20", "weather": "indoor"},
            {"name": "High Park", "location": "Toronto", "target_ages": "All",
             "price": "Free", "weather": "outdoor"},
        ])
    return "mock content for " + task


def _default_parsed_for(task: str) -> Optional[Any]:
    if task in ("json", "weekend_transient", "weekend_fixed", "detailed_json"):
        return json.loads(_default_content_for(task))
    if task == "file_summary":
        return json.loads(_default_content_for(task))
    return None


class MockLLM:
    """Configurable mock LLM provider for testing.
    
    Patches key LLM functions so tests don't need a running server.
    Call patch_all() before importing modules under test, or use the
    mock_llm pytest fixture from conftest.
    
    Usage:
        mock = MockLLM()
        mock.set_response("json", {"content": "...", "parsed": [...]})
        mock.patch_all()
        from eval.run import run_eval
        run_eval("mock-model", tasks=["json"])
        mock.unpatch()
    """

    def __init__(self):
        self._patches: list[unittest.mock._patch] = []
        self._responses: dict[str, dict] = {}

    def set_response(self, task: str, response: dict):
        self._responses[task] = response

    def set_response_fn(self, task: str, fn: Callable):
        self._responses[task] = fn

    def call(self, model: str = "", messages: list = None,
             task: str = "", **kwargs) -> dict:
        task = task or "json"
        if task in self._responses:
            resp = self._responses[task]
            if callable(resp):
                return resp()
            return resp
        parse_json = kwargs.get("parse_json", False)
        parsed = _default_parsed_for(task) if parse_json else None
        return {
            "content": _default_content_for(task),
            "parsed": parsed,
        }

    def call_llm_api(self, *args, **kwargs) -> dict:
        task = kwargs.get("task", "summarize")
        return {
            "content": _default_content_for(task),
        }

    def get_models(self) -> list[str]:
        return ["mock-model-qwen", "mock-model-gemma"]

    def is_server_running(self, *args, **kwargs) -> bool:
        return True

    def get_best_model(self, *args, **kwargs) -> str:
        return "mock-model"

    def check_llm_availability(self, *args, **kwargs) -> bool:
        return True

    def call_mlx(self, *args, **kwargs) -> str:
        return _default_content_for("json")

    def find_text_mlx_model(self, *args, **kwargs):
        return None

    def find_mlx_model(self, *args, **kwargs):
        return None

    def ensure_server(self, *args, **kwargs) -> None:
        pass

    def strip_thinking(self, content: str) -> str:
        return _THINK_RE.sub('', content).strip()

    def _patch(self, target: str, func: Callable):
        p = unittest.mock.patch(target, func)
        p.start()
        self._patches.append(p)

    def _patch_obj(self, module, name: str, func: Callable):
        p = unittest.mock.patch.object(module, name, func)
        p.start()
        self._patches.append(p)

    def patch_osaurus(self):
        import lib.osaurus_lib as m
        self._patch_obj(m, "call", self.call)
        self._patch_obj(m, "call_llm_api", self.call_llm_api)
        self._patch_obj(m, "get_models", self.get_models)
        self._patch_obj(m, "is_server_running", self.is_server_running)
        self._patch_obj(m, "get_best_model", self.get_best_model)
        self._patch_obj(m, "check_llm_availability", self.check_llm_availability)
        self._patch_obj(m, "ensure_server", self.ensure_server)
        self._patch_obj(m, "panic_dump", lambda *a, **kw: None)
        self._patch_obj(m, "_extract_json_only", lambda c, **kw: json.loads(c) if c else None)

    def patch_mlx(self):
        import lib.mlx_lib as m
        self._patch_obj(m, "call", self.call_mlx)
        self._patch_obj(m, "call_mlx", self.call_mlx)
        self._patch_obj(m, "find_text_mlx_model", self.find_text_mlx_model)
        self._patch_obj(m, "find_mlx_model", self.find_mlx_model)
        self._patch_obj(m, "process_mlx_content", lambda c, **kw: c)

    def patch_config(self):
        import lib.config as m
        self._patch_obj(m, "get_model_prompts_all", lambda *a, **kw: None)
        self._patch_obj(m, "build_tasks_from_model", lambda *a, **kw: None)

    def patch_all(self):
        self.patch_osaurus()
        self.patch_mlx()
        self.patch_config()

    def unpatch(self):
        for p in reversed(self._patches):
            p.stop()
        self._patches.clear()

    def __enter__(self):
        self.patch_all()
        return self

    def __exit__(self, *args):
        self.unpatch()
