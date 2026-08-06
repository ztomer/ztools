from unittest.mock import patch

import lib.llm.client as llm_client
import lib.osaurus_lib as osaurus_lib


def test_global_overrides_applied_in_client():
    llm_client.GLOBAL_OVERRIDES.clear()

    with patch("requests.Session.post") as mock_post:
        mock_response = mock_post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "hello"}}

        llm_client.call("foundation", [{"role": "user", "content": "hi"}])

        args, kwargs = mock_post.call_args
        assert kwargs["json"]["temperature"] == 0.1

    llm_client.GLOBAL_OVERRIDES["temperature"] = 0.8
    llm_client.GLOBAL_OVERRIDES["max_tokens"] = 500

    with patch("requests.Session.post") as mock_post:
        mock_response = mock_post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "hello"}}

        llm_client.call("foundation", [{"role": "user", "content": "hi"}])

        args, kwargs = mock_post.call_args
        assert kwargs["json"]["temperature"] == 0.8
        assert kwargs["json"]["max_tokens"] == 500

    llm_client.GLOBAL_OVERRIDES.clear()

def test_global_overrides_applied_in_osaurus_lib():
    osaurus_lib.GLOBAL_OVERRIDES.clear()

    with patch("requests.Session.post") as mock_post:
        mock_response = mock_post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "hello"}

        osaurus_lib.call("foundation", [{"role": "user", "content": "hi"}])

        args, kwargs = mock_post.call_args
        assert kwargs["json"]["temperature"] == 0.1

    osaurus_lib.GLOBAL_OVERRIDES["temperature"] = 1.2
    osaurus_lib.GLOBAL_OVERRIDES["max_tokens"] = 100

    with patch("requests.Session.post") as mock_post:
        mock_response = mock_post.return_value
        mock_response.status_code = 200
        mock_response.json.return_value = {"content": "hello"}

        osaurus_lib.call("foundation", [{"role": "user", "content": "hi"}])

        args, kwargs = mock_post.call_args
        assert kwargs["json"]["temperature"] == 1.2
        assert kwargs["json"]["max_tokens"] == 100

    osaurus_lib.GLOBAL_OVERRIDES.clear()
