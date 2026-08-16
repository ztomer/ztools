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

from lib.llm.constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
)
from lib.llm.quirks import apply_model_quirks

from .config import get_max_tokens_for_task, get_timeout
from .content_processing import remove_thinking_blocks
from .logging_config import osaurus_logger as logger
from .osaurus_degrade import _streamed_with_guard, _try_foundation
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

ERROR_TRUNCATE_LEN = 200

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


def reset_session():
    global _session
    if _session is not None:
        try:
            _session.close()
        except Exception:
            pass
        _session = None


__all__ = [
    "reset_session",
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
    stream_guard: bool = False,
    _allow_model_substitution: bool = True,
) -> dict:
    logger.debug(f"Calling {model} for task '{task}' at {host}:{port}")
    # Kept unquirked: a substitute model is a different family, so the retry has to
    # re-derive quirks from scratch rather than inherit the dead model's prefixes.
    unquirked_messages = messages
    messages = apply_model_quirks(messages, model)
    max_tokens = max_tokens or get_max_tokens_for_task(task, model)

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
    result = {"model": model, "time": None, "content": None, "parsed": None, "error": None}
    start = time.time()

    # Forced Foundation mode: use the on-device model and skip the server.
    if use_foundation is True:
        _try_foundation(True, messages, parse_json, result)
        return result

    # Streamed, with the reasoning-overrun guard. Placed HERE rather than replacing
    # the transport wholesale so everything around it still applies: quirks were
    # already applied above, and JSON extraction, model substitution and the
    # Foundation fallback all live below.
    #
    # Only the happy path is served from the stream. Any transport error falls
    # through to the blocking request, which is what knows how to substitute a
    # deleted model and how to fall back on-device -- neither of which the stream
    # can do, and both of which matter more than saving a doomed run.
    if stream_guard:
        streamed = _streamed_with_guard(
            model, messages, max_tokens, host, port, temperature, timeout, task
        )
        if streamed is not None:
            result.update(streamed)
            result["time"] = round(time.time() - start, 1)
            if parse_json and result["content"]:
                result["parsed"] = extract_json(result["content"])
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
            body = resp.text
            retried = _retry_with_substitute(
                model=model,
                status_code=resp.status_code,
                body=body,
                allowed=_allow_model_substitution,
                kwargs=dict(
                    messages=unquirked_messages,
                    host=host,
                    port=port,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    task=task,
                    parse_json=parse_json,
                    use_foundation=use_foundation,
                ),
            )
            if retried is not None:
                return retried
            result["error"] = f"HTTP {resp.status_code}: {body[:ERROR_TRUNCATE_LEN]}"
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
        # Reasoning models (the qwen3_5 family here) stream their chain of thought
        # into a SEPARATE field and leave `content` empty until reasoning ends. On a
        # hard prompt they never stop, so the caller sees an empty answer with no
        # indication why. Carrying both fields is what lets a caller tell "the model
        # thought until it ran out of budget" from "the model returned nothing" --
        # two failures with completely different remedies.
        #
        # eval/outputs.py already read result["reasoning_content"]; nothing had ever
        # written it, so every saved output for a reasoning model was blank.
        result["reasoning_content"] = message.get("reasoning_content") or ""
        result["finish_reason"] = resp_data["choices"][0].get("finish_reason") or ""
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
        reset_session()
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




def _retry_with_substitute(
    model: str,
    status_code: int,
    body: str,
    allowed: bool,
    kwargs: dict,
) -> Optional[dict]:
    """Retry once against a servable model when the configured tag is gone.

    Returns the retried result, or None to let the original error stand. None is the
    answer whenever we lack evidence — a non-404, a 404 about something other than a
    missing model, an unreachable roster, or a roster that offers no alternative —
    because rewriting the caller's model on a guess would trade a clear error for a
    silently wrong answer.
    """
    from lib.model_resolve import fetch_roster, is_missing_model_error, substitute_model

    if not allowed or not is_missing_model_error(status_code, body):
        return None
    roster = fetch_roster(kwargs["host"], kwargs["port"])
    substitute, reason = substitute_model(model, roster)
    if not reason:
        return None
    logger.warning(reason)
    retried = call(model=substitute, _allow_model_substitution=False, **kwargs)
    retried["substituted_from"] = model
    retried["substitution_reason"] = reason
    return retried



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
        if isinstance(e, requests.exceptions.ConnectionError):
            reset_session()
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
