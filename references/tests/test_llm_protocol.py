"""Tests for lib/llm/protocol.py — client factory and call delegation."""

from unittest.mock import patch

import pytest

from lib.llm.protocol import (
    GenericClient,
    MlxClient,
    OsaurusClient,
    create_client,
)


class TestCreateClient:
    def test_creates_osaurus_client(self):
        client = create_client("osaurus")
        assert isinstance(client, OsaurusClient)

    def test_creates_mlx_client(self):
        client = create_client("mlx")
        assert isinstance(client, MlxClient)

    def test_creates_generic_client(self):
        client = create_client("llm")
        assert isinstance(client, GenericClient)

    def test_raises_on_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown client type"):
            create_client("invalid")


class TestOsaurusClient:
    def test_call_delegates_to_osaurus_lib(self):
        client = OsaurusClient()
        msgs = [{"role": "user", "content": "hi"}]
        with patch("lib.osaurus_lib.call", return_value={"response": "ok"}) as mock_call:
            result = client.call("test-model", msgs)
        assert result == {"response": "ok"}
        mock_call.assert_called_once_with(
            "test-model", msgs,
            host="localhost", port=1337, temperature=0.1,
            max_tokens=None, timeout=None, task="think", parse_json=False,
        )


class TestMlxClient:
    def test_call_delegates_to_mlx_lib(self):
        client = MlxClient()
        msgs = [{"role": "user", "content": "hi"}]
        with patch("lib.mlx_lib.call", return_value={"response": "ok"}) as mock_call:
            result = client.call("test-model", msgs, temperature=0.5)
        assert result == {"response": "ok"}
        mock_call.assert_called_once_with(
            "test-model", msgs,
            host="localhost", port=1337, temperature=0.5,
            max_tokens=None, timeout=None, task="think", parse_json=False,
        )


class TestGenericClient:
    def test_call_delegates_to_llm_client(self):
        client = GenericClient()
        msgs = [{"role": "user", "content": "hi"}]
        with patch("lib.llm.client.call", return_value={"response": "ok"}) as mock_call:
            result = client.call("test-model", msgs, parse_json=True)
        assert result == {"response": "ok"}
        mock_call.assert_called_once_with(
            "test-model", msgs,
            host="localhost", port=1337, temperature=0.1,
            max_tokens=None, timeout=None, task="think", parse_json=True,
        )
