"""Tests for img_llm LLM integration functions."""

from pathlib import Path
from unittest.mock import mock_open, patch


class TestEnsureLlmRunning:
    def test_delegates_to_ensure_server(self):
        from rename.llm import ensure_llm_running

        with patch("lib.osaurus_lib.ensure_server", return_value=True) as mock_ensure:
            assert ensure_llm_running() is True
            mock_ensure.assert_called_once()

        with patch("lib.osaurus_lib.ensure_server", return_value=False) as mock_ensure:
            assert ensure_llm_running() is False
            mock_ensure.assert_called_once()


class TestIsRelevantWithLlm:
    """The keep/skip decision, now routed through the shared LLM client.

    These used to mock `requests.Session` and hand-built NDJSON response bodies,
    because this function issued its own raw POST to /api/chat. That raw path is why
    the feature silently died: its two configured models had been uninstalled for
    months, each call 404'd, and the function returned None for EVERY image -- which
    the caller cannot tell apart from "the model had no opinion".

    Routing through `osaurus_lib.call` supplies model substitution, the streaming
    deadline, quirks and the Foundation fallback. The tests that pinned the
    hand-rolled line parsing are gone with the code they described; that parsing now
    lives in the shared client and is tested there.
    """

    def _result(self, content="", error="", **extra):
        return {"content": content, "error": error, "parsed": None, "model": "m", **extra}

    def test_keep_response(self):
        from rename.llm import is_relevant_with_llm

        with patch("lib.osaurus_lib.call", return_value=self._result("keep")):
            assert is_relevant_with_llm("some content", "http://localhost:1337") is True

    def test_skip_response(self):
        from rename.llm import is_relevant_with_llm

        with patch("lib.osaurus_lib.call", return_value=self._result("skip")):
            assert is_relevant_with_llm("some content", "http://localhost:1337") is False

    def test_a_reply_containing_both_words_is_not_a_keep(self):
        """"keep" must not win on a substring when the model also said "skip"."""
        from rename.llm import is_relevant_with_llm

        with patch("lib.osaurus_lib.call",
                   return_value=self._result("I would skip rather than keep this")):
            assert is_relevant_with_llm("x", "http://localhost:1337") is False

    def test_first_model_fails_second_succeeds(self):
        from rename.llm import is_relevant_with_llm

        with (
            patch("rename.llm.filename_models", return_value=["dead", "live"]),
            patch("lib.osaurus_lib.call",
                  side_effect=[self._result(error="HTTP 404"), self._result("keep")]),
        ):
            assert is_relevant_with_llm("x", "http://localhost:1337") is True

    def test_all_models_failing_returns_none(self):
        from rename.llm import is_relevant_with_llm

        with patch("lib.osaurus_lib.call", return_value=self._result(error="HTTP 500")):
            assert is_relevant_with_llm("x", "http://localhost:1337") is None

    def test_an_exception_does_not_escape(self):
        """A relevance check that raises must not take down a rename run."""
        from rename.llm import is_relevant_with_llm

        with patch("lib.osaurus_lib.call", side_effect=RuntimeError("boom")):
            assert is_relevant_with_llm("x", "http://localhost:1337") is None

    def test_a_substitution_is_surfaced_not_swallowed(self):
        """The whole point of routing through the shared client. When a configured
        model is gone, substitution answers anyway AND says so -- the silence is what
        let this feature stay dead."""
        from rename.llm import is_relevant_with_llm

        result = self._result("keep", substituted_from="gone-model",
                              substitution_reason="model 'gone-model' is not installed; using 'live'")
        with (
            patch("lib.osaurus_lib.call", return_value=result),
            patch("rename.llm.logger") as log,
        ):
            assert is_relevant_with_llm("x", "http://localhost:1337") is True
        assert log.warning.called, "substitution happened and nothing was logged"


class TestTheRelevanceModelsAreResolvedAtCallTime:
    """The default must not be a hardcoded pair of tags that can quietly rot.

    The previous default was "qwen3.6-27b-mxfp4,gemma-4-26b-a4b-it-mxfp4". Neither had
    been installed for months, and nothing noticed because the failure mode was a
    None return that reads as "no opinion".
    """

    def test_it_falls_back_to_the_audited_filename_chain(self):
        import rename.llm as rl

        with (
            patch.dict("os.environ", {}, clear=False),
            patch.object(rl, "_RENAME_CFG", {}),
            patch("rename.llm.get_filename_models", return_value=["audited-a", "audited-b"]),
        ):
            import os as _os
            _os.environ.pop("RENAME_RELEVANCE_MODELS", None)
            assert rl.relevance_check_models() == ["audited-a", "audited-b"]

    def test_an_explicit_config_wins(self):
        import rename.llm as rl

        with patch.object(rl, "_RENAME_CFG", {"relevance_check_models": "x, y"}):
            assert rl.relevance_check_models() == ["x", "y"]

    def test_the_env_fallback_bakes_in_no_model_tags(self):
        """The property, not the spelling. A source grep for the old tags cannot tell
        a live default from a comment explaining why it is gone -- the first version
        of this test failed on the docstring above it. What actually matters is that
        when nothing is configured, the answer comes from the audited chain rather
        than from a literal baked into this module.
        """
        import rename.llm as rl

        with (
            patch.object(rl, "_RENAME_CFG", {}),
            patch.dict("os.environ", {"RENAME_RELEVANCE_MODELS": ""}),
            patch("rename.llm.get_filename_models", return_value=["from-the-audit"]),
        ):
            assert rl.relevance_check_models() == ["from-the-audit"]


class TestQueryLlmForFilename:
    def test_successful_query(self):
        from rename.llm import query_llm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "my_cool_photo", "error": ""}
        with (
            patch("rename.llm._shared_call", return_value=_reply),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
        ):
            result = query_llm_for_filename("Test image text", "http://localhost:1337")
            assert result == "my_cool_photo"

    def test_strips_instruction_prefix(self):
        from rename.llm import query_llm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "filename: my_photo", "error": ""}
        with (
            patch("rename.llm._shared_call", return_value=_reply),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
        ):
            result = query_llm_for_filename("Test image text", "http://localhost:1337")
            assert result == "my_photo"

    def test_empty_response(self):
        from rename.llm import query_llm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "", "error": ""}
        with (
            patch("rename.llm._shared_call", return_value=_reply),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
        ):
            assert query_llm_for_filename("Test text", "http://localhost:1337") is None

    def test_http_error_fallback(self):
        from rename.llm import query_llm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        replies = [
            {"content": "", "error": "HTTP 500"},      # first model fails
            {"content": "successful_name", "error": ""},  # second answers
        ]
        with (
            patch("rename.llm._shared_call", side_effect=replies),
            patch("rename.llm.filename_models", return_value=["fail-model", "success-model"]),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
        ):
            result = query_llm_for_filename("Test text", "http://localhost:1337")
            assert result == "successful_name"

    def test_limits_words_to_6(self):
        from rename.llm import query_llm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "one two three four five six seven eight", "error": ""}
        with (
            patch("rename.llm._shared_call", return_value=_reply),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
        ):
            result = query_llm_for_filename("x", "http://localhost:1337")
            assert result == "one_two_three_four_five_six"
            assert "_seven" not in result

    def test_no_alpha_content_returns_none(self):
        from rename.llm import query_llm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "123 456 789", "error": ""}
        with (
            patch("rename.llm._shared_call", return_value=_reply),
            patch("rename.llm.filename_models", return_value=["test-model"]),
        ):
            assert query_llm_for_filename("x", "http://localhost:1337") is None

    def test_invalid_json_in_streaming_response(self):
        """Lines 118-120: invalid JSON in streaming response is skipped."""
        from rename.llm import query_llm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "valid_name", "error": ""}
        with (
            patch("rename.llm._shared_call", return_value=_reply),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
        ):
            result = query_llm_for_filename("x", "http://localhost:1337")
            assert result == "valid_name"

    def test_truncate_long_content(self):
        """Line 133: content longer than 35 chars gets truncated."""
        from rename.llm import query_llm_for_filename

        # A name past the 50-char limit, to exercise the truncation itself.
        _reply = {"content": "x" * 80, "error": ""}
        with (
            patch("rename.llm._shared_call", return_value=_reply),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
        ):
            result = query_llm_for_filename("x", "http://localhost:1337")
            # Truncated to the documented 50-char limit (the prompts promise
            # "under 50"; the code used to cut at 35 and truncate mid-word).
            assert result == "x" * 50
            assert len(result) == 50

    def test_non_alpha_content_skipped(self):
        """Lines 135-136, 138-139: non-alpha content is skipped."""
        from rename.llm import query_llm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "!!! ??? ???", "error": ""}
        with (
            patch("rename.llm._shared_call", return_value=_reply),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
        ):
            assert query_llm_for_filename("x", "http://localhost:1337") is None

    def test_short_content_skipped(self):
        """Line 129: content with no words (after regex) returns None."""
        from rename.llm import query_llm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "!!!", "error": ""}
        with (
            patch("rename.llm._shared_call", return_value=_reply),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
        ):
            assert query_llm_for_filename("x", "http://localhost:1337") is None

    def test_invalid_alpha_pattern(self):
        """Line 135: content with invalid chars (not a-z) is rejected."""
        from rename.llm import query_llm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "TEST NAME", "error": ""}
        with (
            patch("rename.llm._shared_call", return_value=_reply),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
        ):
            # Result is lowercased to "test name" → joined "test_name" → valid
            result = query_llm_for_filename("x", "http://localhost:1337")
            assert result == "test_name"


class TestQueryVlmForFilename:
    def test_successful_vlm_query(self):
        from rename.llm import query_vlm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "white goose grass", "error": ""}
        with (
            patch("builtins.open", mock_open(read_data=b"fake_image_data")),
            patch("rename.llm._shared_call", return_value=_reply),
        ):
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result == "white goose grass"

    def test_api_error(self):
        from rename.llm import query_vlm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "", "error": "HTTP 500"}
        with (
            patch("builtins.open", mock_open(read_data=b"data")),
            patch("rename.llm._shared_call", return_value=_reply),
        ):
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result is None

    def test_file_read_exception(self):
        from rename.llm import query_vlm_for_filename

        with patch("builtins.open", side_effect=Exception("file error")):
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result is None

    def test_with_api_key(self):
        from rename.llm import query_vlm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "output", "error": ""}
        with (
            patch("builtins.open", mock_open(read_data=b"data")),
            patch("rename.llm._shared_call", return_value=_reply) as shared,
        ):
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model", api_key="mykey"
            )
            assert result == "output"
            # The bearer token is now the shared client's job. rn must still HAND it
            # over -- routing through the client dropped it until `call` grew an
            # api_key parameter, which would have broken authenticated servers in
            # silence.
            assert shared.call_args.args[4] == "mykey"

    def test_vlm_done_break(self):
        """Line 210: VLM stream ends with done=true at top level (currently unreachable)."""
        from rename.llm import query_vlm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "firstmore", "error": ""}
        with (
            patch("builtins.open", mock_open(read_data=b"data")),
            patch("rename.llm._shared_call", return_value=_reply),
        ):
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result == "firstmore"

    def test_vlm_invalid_json_continues(self):
        """Lines 211-212: invalid JSON in VLM stream is skipped."""
        from rename.llm import query_vlm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "valid_part", "error": ""}
        with (
            patch("builtins.open", mock_open(read_data=b"data")),
            patch("rename.llm._shared_call", return_value=_reply),
        ):
            result = query_vlm_for_filename(
                Path("/fake/image.png"), "http://localhost:1337", "vlm-model"
            )
            assert result == "valid_part"

    def test_query_llm_with_done_break(self):
        """Line 118: query_llm_for_filename stops on done (at top level of JSON)."""
        from rename.llm import query_llm_for_filename

        # migrated: rn now goes through lib.osaurus_lib.call, not a raw POST
        _reply = {"content": "first_name", "error": ""}
        with (
            patch("rename.llm._shared_call", return_value=_reply),
            patch("rename.llm.filename_models", return_value=["test-model"]),
            patch("rename.llm.default_filename_prompt", return_value="Text: {text}"),
        ):
            result = query_llm_for_filename("x", "http://localhost:1337")
            assert result == "first_name"
