#!/usr/bin/env python3
"""
Osaurus Library - Generic LLM utilities.
Shim that re-exports from sub-modules.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

import requests

from lib.llm.quirks import apply_model_quirks

from .config import get_max_tokens_for_task, get_timeout
from .content_processing import remove_thinking_blocks
from .logging_config import osaurus_logger as logger
from .osaurus_models import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    check_llm_availability,
    get_api_url,
    get_available_models,
    get_base_url,
    get_best_model,
    get_models,
    is_server_running,
    select_best_model,
    select_best_vlm_model,
)
from .osaurus_output import (
    KEY_NORMALIZATIONS,
    TOP_LEVEL_KEYS,
    _extract_json_only,
    _extract_plain_list,
    clean_output,
    extract_json,
    filter_json_items,
    fix_json_years,
    merge_flat_dicts,
    normalize_keys,
    normalize_text_output,
)
from .osaurus_server import (
    ensure_server,
    panic_dump,
    restart_server,
    test_connection,
)

# Default parameter values
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TIMEOUT = 600
ERROR_TRUNCATE_LEN = 200


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


def _get_session_context():
    global _session
    if hasattr(requests.Session, "mock_add_spec") or type(requests.Session).__name__ in (
        "Mock",
        "MagicMock",
    ):
        return requests.Session()
    if _session is None:
        _session = requests.Session()
    return SessionContext(_session, close_on_exit=False)


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT",
    "ERROR_TRUNCATE_LEN",
    "get_api_url",
    "get_base_url",
    "get_models",
    "is_server_running",
    "check_llm_availability",
    "get_available_models",
    "get_best_model",
    "select_best_vlm_model",
    "select_best_model",
    "restart_server",
    "ensure_server",
    "test_connection",
    "panic_dump",
    "clean_output",
    "extract_json",
    "normalize_keys",
    "merge_flat_dicts",
    "filter_json_items",
    "fix_json_years",
    "normalize_text_output",
    "_extract_json_only",
    "_extract_plain_list",
    "TOP_LEVEL_KEYS",
    "KEY_NORMALIZATIONS",
    "apply_model_quirks",
    "PROMPTS",
    "call",
    "call_with_prompt",
    "test_model",
    "call_llm_api",
    "extract_thinking",
    "merge_thinking_with_summary",
    "strip_thinking",
]


PROMPTS = {
    "think": {
        "messages": [
            {
                "role": "system",
                "content": "Think step by step if needed. Then provide your answer.",
            },
            {"role": "user", "content": "{prompt}"},
        ]
    },
    "json": {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Output ONLY valid JSON. Start with { or [. "
                    "No markdown, no explanations."
                ),
            },
            {"role": "user", "content": "{prompt}"},
        ]
    },
    "summarize": {
        "messages": [
            {
                "role": "system",
                "content": "Output headers with ## and key facts. No thinking, no markdown.",
            },
            {"role": "user", "content": "{prompt}"},
        ]
    },
    "filename": {
        "messages": [
            {
                "role": "system",
                "content": "Output ONLY a short filename. No explanations. Under 50 chars.",
            },
            {"role": "user", "content": "{prompt}"},
        ]
    },
}


def call(
    model: str,
    messages: List[Dict[str, Any]],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    task: str = "think",
    parse_json: bool = False,
    use_foundation: Optional[bool] = None,
) -> dict:
    logger.debug(f"Calling {model} for task '{task}' at {host}:{port}")
    messages = apply_model_quirks(messages, model)
    max_tokens = max_tokens or get_max_tokens_for_task(task)
    url = get_api_url(host, port)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if parse_json:
        payload["response_format"] = {"type": "json_object"}
    result = {"model": model, "time": None, "content": None, "parsed": None, "error": None}
    start = time.time()

    # Forced Foundation mode: use the on-device model and skip the server.
    if use_foundation is True:
        _try_foundation(True, messages, parse_json, result)
        return result

    try:
        timeout = timeout or get_timeout(task)
        logger.debug(f"Sending request with {len(messages)} messages, timeout={timeout}s")
        with _get_session_context() as s:
            resp = s.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
        result["time"] = round(time.time() - start, 1)
        logger.debug(f"Response received in {result['time']}s")
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:ERROR_TRUNCATE_LEN]}"
            logger.error(f"HTTP error: {result['error']}")
            return result
        resp_data = resp.json()
        if "choices" not in resp_data or not resp_data["choices"]:
            result["error"] = "Empty response from API"
            logger.error(result["error"])
            return result
        message = resp_data["choices"][0].get("message", {})
        content = message.get("content", "")
        result["content"] = clean_output(content)
        logger.debug(f"Extracted {len(content)} chars of content")
        if parse_json and content:
            result["parsed"] = extract_json(content)
            if result["parsed"]:
                logger.debug("JSON parsed successfully")
            else:
                logger.warning("Could not parse JSON from output")
    except requests.exceptions.Timeout:
        result["error"] = "Timeout"
        logger.warning(f"Request timed out after {timeout}s")
    except requests.exceptions.ConnectionError:
        logger.warning(f"Connection error to {url}")
        if _try_foundation(use_foundation, messages, parse_json, result):
            return result
        result["error"] = "Connection failed - is server running?"
    except json.JSONDecodeError as e:
        result["error"] = f"Invalid JSON response: {e}"
        logger.error(f"JSON decode error: {e}")
    except KeyError as e:
        result["error"] = f"Unexpected response format: {e}"
        logger.error(f"Key error in response: {e}")
    except Exception as e:
        result["error"] = f"Error: {type(e).__name__}: {e}"
        logger.exception(f"Unexpected error: {e}")
    return result


def _try_foundation(
    use_foundation: Optional[bool], messages, parse_json: bool, result: dict
) -> bool:
    """Attempt the on-device Foundation Models fallback.

    use_foundation: True = force, False = never, None = only if available.
    On success, fills ``result`` and returns True.
    """
    if use_foundation is False:
        return False
    try:
        from lib.foundation_lib import call_foundation, foundation_available
    except Exception:
        return False
    if not foundation_available():
        return False
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    try:
        raw = call_foundation(system, user, parse_json=parse_json)
    except Exception as e:
        logger.warning(f"Foundation fallback failed: {e}")
        return False
    if not raw:
        return False
    result["content"] = raw
    result["model"] = "foundation"
    result["foundation"] = True
    if parse_json:
        result["parsed"] = extract_json(raw)
    logger.info("Served by on-device Foundation Models")
    return True


def call_with_prompt(
    model: str,
    prompt: str,
    task: str = "think",
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    if task in PROMPTS:
        template = PROMPTS[task]
        messages = []
        for msg in template["messages"]:
            content = msg["content"]
            for placeholder in ["{prompt}", "{text}", "{items}", "{tweets}"]:
                if placeholder in content:
                    content = content.replace(placeholder, prompt)
            messages.append({"role": msg["role"], "content": content})
    else:
        messages = [{"role": "user", "content": prompt}]
    parse_json = task in ("json", "detailed_json")
    return call(
        model,
        messages,
        host,
        port,
        temperature,
        max_tokens,
        timeout=get_timeout(task),
        parse_json=parse_json,
    )


def test_model(
    model: str,
    prompt: str = "Hello",
    task: str = "think",
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> dict:
    return call_with_prompt(model, prompt, task, host, port)


def call_llm_api(
    host: str,
    model: str,
    messages: List[dict],
    api_key: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
    parse_json: bool = False,
) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if parse_json:
        payload["response_format"] = {"type": "json_object"}
    try:
        if host.startswith("http"):
            url = f"{host}/v1/chat/completions"
        else:
            url = f"http://{host}/v1/chat/completions"
        with _get_session_context() as s:
            response = s.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return {
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
            "model": data.get("model", model),
        }
    except Exception as e:
        return {"error": str(e)}


def extract_thinking(text: str) -> tuple[str, str]:
    think_match = re.search(r"<thinking[^>]*>(.+?)</thinking>", text, re.DOTALL)
    if not think_match:
        return "", text
    thinking = think_match.group(1).strip()
    content = remove_thinking_blocks(text)
    return thinking, content


def merge_thinking_with_summary(thinking: str, summary: str) -> str:
    if not thinking:
        return summary
    return f"{summary}\n\n## Analysis\n{thinking}"


def strip_thinking(text: str) -> str:
    return remove_thinking_blocks(text)
