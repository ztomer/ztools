"""Tests for lib.testing - MockLLM provider."""



class TestDefaultContentFor:
    def test_json(self):
        from lib.testing import _default_content_for

        result = _default_content_for("json")
        assert "Spring Festival" in result
        assert "Toronto" in result

    def test_weekend_transient(self):
        from lib.testing import _default_content_for

        result = _default_content_for("weekend_transient")
        assert "Spring Festival" in result

    def test_filename(self):
        from lib.testing import _default_content_for

        result = _default_content_for("filename")
        assert "mock_test_filename" in result

    def test_image_rename(self):
        from lib.testing import _default_content_for

        result = _default_content_for("image_rename")
        assert "how_to_manage_underperformers" in result
        result_mixed = _default_content_for("image_rename_mixed")
        assert "how_to_manage_underperformers" in result_mixed

    def test_summarize(self):
        from lib.testing import _default_content_for

        result = _default_content_for("summarize")
        assert "Summary" in result or "OpenAI" in result

    def test_file_summary(self):
        from lib.testing import _default_content_for

        result = _default_content_for("file_summary")
        assert "eval_lib.py" in result or "validators.py" in result

    def test_weekend_fixed(self):
        from lib.testing import _default_content_for

        result = _default_content_for("weekend_fixed")
        assert "Vaughan" in result

    def test_detailed_json(self):
        from lib.testing import _default_content_for

        result = _default_content_for("detailed_json")
        assert "Vaughan" in result

    def test_unknown_task(self):
        from lib.testing import _default_content_for

        result = _default_content_for("unknown_task_xyz")
        assert "mock content for" in result


class TestDefaultParsedFor:
    def test_json(self):
        from lib.testing import _default_parsed_for

        result = _default_parsed_for("json")
        # Default mock: 2 weekend items with full fields
        assert result == [
            {
                "name": "Spring Festival",
                "location": "Toronto",
                "target_ages": "All",
                "price": "Free",
                "weather": "outdoor",
                "day": "Saturday",
            },
            {
                "name": "Indoor Coding Workshop",
                "location": "Vaughan",
                "target_ages": "8-14",
                "price": "$25",
                "weather": "indoor",
                "day": "Sunday",
            },
        ]

    def test_weekend_transient(self):
        """Transient items carry `day` — that is what makes them transient.

        `isinstance(result, list)` held for an empty list and for any wrong
        shape, so the fixture the whole suite depends on was unpinned.
        """
        from lib.testing import _default_parsed_for

        result = _default_parsed_for("weekend_transient")
        assert len(result) == 2
        assert all("day" in item for item in result)
        assert result[0]["name"] == "Spring Festival"

    def test_weekend_fixed(self):
        """Fixed venues carry no `day`: they are available all weekend."""
        from lib.testing import _default_parsed_for

        result = _default_parsed_for("weekend_fixed")
        assert len(result) == 2
        assert all("day" not in item for item in result)
        assert {"name", "location", "target_ages", "price", "weather"} <= set(result[0])

    def test_detailed_json(self):
        from lib.testing import _default_parsed_for

        result = _default_parsed_for("detailed_json")
        assert result == _default_parsed_for("weekend_fixed")

    def test_file_summary(self):
        from lib.testing import _default_parsed_for

        result = _default_parsed_for("file_summary")
        assert isinstance(result, list)

    def test_summarize(self):
        from lib.testing import _default_parsed_for

        result = _default_parsed_for("summarize")
        assert result is None

    def test_filename(self):
        from lib.testing import _default_parsed_for

        result = _default_parsed_for("filename")
        assert result is None

    def test_unknown(self):
        from lib.testing import _default_parsed_for

        result = _default_parsed_for("unknown")
        assert result is None


class TestMockLLM:
    def test_init(self):
        from lib.testing import MockLLM

        m = MockLLM()
        assert m._patches == []
        assert m._responses == {}

    def test_set_response(self):
        from lib.testing import MockLLM

        m = MockLLM()
        m.set_response("json", {"content": "x"})
        assert m._responses["json"] == {"content": "x"}

    def test_set_response_fn(self):
        from lib.testing import MockLLM

        m = MockLLM()

        def fn():
            return {"content": "x"}

        m.set_response_fn("json", fn)
        assert m._responses["json"] is fn

    def test_call_with_set_response(self):
        from lib.testing import MockLLM

        m = MockLLM()
        m.set_response("json", {"content": "specific", "parsed": [1, 2]})
        result = m.call(task="json")
        assert result["content"] == "specific"

    def test_call_with_response_fn(self):
        from lib.testing import MockLLM

        m = MockLLM()
        m.set_response_fn("json", lambda: {"content": "from fn"})
        result = m.call(task="json")
        assert result["content"] == "from fn"

    def test_call_default_task(self):
        from lib.testing import MockLLM

        m = MockLLM()
        result = m.call()
        # No task -> defaults to "json"
        assert "content" in result

    def test_call_with_parse_json(self):
        from lib.testing import MockLLM

        m = MockLLM()
        result = m.call(task="json", parse_json=True)
        # parse_json=True triggers _default_parsed_for → list of 2 items
        assert result["parsed"] == [
            {
                "name": "Spring Festival",
                "location": "Toronto",
                "target_ages": "All",
                "price": "Free",
                "weather": "outdoor",
                "day": "Saturday",
            },
            {
                "name": "Indoor Coding Workshop",
                "location": "Vaughan",
                "target_ages": "8-14",
                "price": "$25",
                "weather": "indoor",
                "day": "Sunday",
            },
        ]

    def test_call_without_parse_json(self):
        from lib.testing import MockLLM

        m = MockLLM()
        result = m.call(task="json", parse_json=False)
        assert result["parsed"] is None

    def test_call_llm_api(self):
        from lib.testing import MockLLM

        m = MockLLM()
        result = m.call_llm_api()
        assert "content" in result

    def test_call_llm_api_with_task(self):
        from lib.testing import MockLLM

        m = MockLLM()
        result = m.call_llm_api(task="json")
        assert "content" in result

    def test_get_models(self):
        from lib.testing import MockLLM

        m = MockLLM()
        models = m.get_models()
        assert "mock-model-qwen" in models
        assert "mock-model-gemma" in models

    def test_is_server_running(self):
        from lib.testing import MockLLM

        m = MockLLM()
        assert m.is_server_running() is True

    def test_get_best_model(self):
        from lib.testing import MockLLM

        m = MockLLM()
        assert m.get_best_model() == "mock-model"

    def test_check_llm_availability(self):
        from lib.testing import MockLLM

        m = MockLLM()
        assert m.check_llm_availability() is True

    def test_call_mlx(self):
        from lib.testing import MockLLM

        m = MockLLM()
        result = m.call_mlx()
        assert "Spring Festival" in result or isinstance(result, str)

    def test_find_text_mlx_model(self):
        from lib.testing import MockLLM

        m = MockLLM()
        assert m.find_text_mlx_model() is None

    def test_find_mlx_model(self):
        from lib.testing import MockLLM

        m = MockLLM()
        assert m.find_mlx_model() is None

    def test_ensure_server(self):
        from lib.testing import MockLLM

        m = MockLLM()
        # No-op: verify it returns None (the mock contract)
        result = m.ensure_server()
        assert result is None
        # Also accept any args/kwargs without raising
        m.ensure_server("any", model="x", timeout=30)

    def test_strip_thinking(self):
        from lib.testing import MockLLM

        m = MockLLM()
        result = m.strip_thinking("hello <think>reasoning</think> world")
        assert "<think>" not in result
        assert "hello" in result
        assert "world" in result

    def test_strip_thinking_no_think(self):
        from lib.testing import MockLLM

        m = MockLLM()
        result = m.strip_thinking("just text")
        assert result == "just text"

    def test_patch_all(self):
        from lib.testing import MockLLM

        m = MockLLM()
        m.patch_all()
        assert len(m._patches) > 0
        m.unpatch()

    def test_unpatch(self):
        from lib.testing import MockLLM

        m = MockLLM()
        m.patch_all()
        m.unpatch()
        assert m._patches == []

    def test_context_manager(self):
        from lib.testing import MockLLM

        with MockLLM() as m:
            assert len(m._patches) > 0
        # After exit, patches are cleared
        assert m._patches == []

    def test_patch_obj(self):
        import lib.osaurus_lib as ol
        from lib.testing import MockLLM

        m = MockLLM()
        m._patch_obj(ol, "call", lambda *a, **kw: {"content": "patched"})
        # The call is patched
        result = ol.call("any", [])
        assert result["content"] == "patched"
        m.unpatch()

    def test_patch_string(self):
        """_patch uses string path patching."""
        from lib.testing import MockLLM

        m = MockLLM()
        # Patch a function via string path
        m._patch("lib.osaurus_lib.is_server_running", lambda *a, **kw: False)
        import lib.osaurus_lib as ol

        assert ol.is_server_running() is False
        m.unpatch()
