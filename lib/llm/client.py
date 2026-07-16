# LLM Client - Core API calls

import time
from typing import Any, Dict, List, Optional

import requests

from lib.llm.constants import (
    API_CHAT,
    API_TAGS,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TEMPERATURE,
    HTTP_STATUS_OK,
    TASK_THINK,
)
from lib.llm.quirks import apply_model_quirks

# Timeouts for server checks (seconds)
SERVER_CHECK_TIMEOUT = 5
LIST_MODELS_TIMEOUT = 10

GLOBAL_OVERRIDES = {}


class SessionContext:
    """Helper to reuse connection pool in with blocks without closing it (Carmack optimization)"""

    def __init__(self, session, close_on_exit=False):
        self.session = session
        self.close_on_exit = close_on_exit

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.close_on_exit:
            self.session.close()


# Persistent session for HTTP connection pooling
_session = None

MOCK_SPEC_ATTRIBUTE = "mock_add_spec"
MOCK_CLASS_NAMES = ("Mock", "MagicMock")


def _get_session_context():
    global _session
    if (
        hasattr(requests.Session, MOCK_SPEC_ATTRIBUTE)
        or type(requests.Session).__name__ in MOCK_CLASS_NAMES
    ):
        return requests.Session()
    if _session is None:
        _session = requests.Session()
    return SessionContext(_session, close_on_exit=False)


def reset_session():
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
        _session = None


def get_api_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    """Get API URL for host:port.

    Idempotent: if ``host`` already carries a scheme (e.g. the full
    ``http://localhost:1337`` from ``$OLLAMA_BASE_URL``), it is used verbatim
    instead of being prefixed again (which would yield ``http://http://...``).
    """
    if "://" in host:
        return host.rstrip("/")
    return f"http://{host}:{port}"


def get_timeout(task: str) -> int:
    """Get timeout for task type."""
    from lib.config import get_timeout as _gt

    return _gt(task)


def get_max_tokens_for_task(task: str) -> int:
    """Get max tokens for task."""
    from lib.config import get_max_tokens_for_task as _gmt

    return _gmt(task)


def call(
    model: str,
    messages: List[Dict[str, Any]],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    task: str = TASK_THINK,
    parse_json: bool = False,
) -> dict:
    """Call LLM API. Returns dict with content, parsed, time, error."""
    from lib.logging_config import osaurus_logger as logger

    logger.debug(f"Calling {model} for task '{task}' at {host}:{port}")

    # Apply model-specific quirks
    messages = apply_model_quirks(messages, model)

    # Get defaults from config
    max_tokens = max_tokens or get_max_tokens_for_task(task)

    # Apply global overrides
    if "temperature" in GLOBAL_OVERRIDES and temperature == DEFAULT_TEMPERATURE:
        temperature = GLOBAL_OVERRIDES["temperature"]
    if "max_tokens" in GLOBAL_OVERRIDES:
        max_tokens = GLOBAL_OVERRIDES["max_tokens"]

    url = get_api_url(host, port)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if parse_json:
        payload["response_format"] = {"type": "json_object"}

    result = {
        "model": model,
        "time": None,
        "content": None,
        "parsed": None,
        "error": None,
    }

    start = time.time()
    timeout = timeout or get_timeout(task)

    try:
        with _get_session_context() as s:
            response = s.post(
                f"{url}{API_CHAT}",
                json=payload,
                timeout=timeout,
            )
        response.raise_for_status()

        data = response.json()
        result["time"] = round(time.time() - start, 1)

        if "message" in data:
            result["content"] = data["message"].get("content", "")
        elif "content" in data:
            result["content"] = data.get("content", "")

        # Parse JSON if requested
        if parse_json and result["content"]:
            from lib.llm.parsing import extract_json

            result["parsed"] = extract_json(result["content"], model)

    except requests.exceptions.Timeout:
        result["error"] = "Timeout"
    except requests.exceptions.ConnectionError:
        reset_session()
        result["error"] = "Connection failed"
    except Exception as e:
        result["error"] = str(e)

    return result


def is_server_running(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Check if server is running."""
    try:
        with _get_session_context() as s:
            response = s.get(
                f"{get_api_url(host, port)}{API_TAGS}",
                timeout=SERVER_CHECK_TIMEOUT,
            )
        return response.status_code == HTTP_STATUS_OK
    except requests.exceptions.ConnectionError:
        reset_session()
        return False
    except Exception:
        return False


def get_models(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> List[str]:
    """Get list of available models."""
    try:
        with _get_session_context() as s:
            response = s.get(
                f"{get_api_url(host, port)}{API_TAGS}",
                timeout=LIST_MODELS_TIMEOUT,
            )
        data = response.json()
        return [m["model"] for m in data.get("models", [])]
    except requests.exceptions.ConnectionError:
        reset_session()
        return []
    except Exception:
        return []
