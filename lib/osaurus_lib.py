#!/usr/bin/env python3
"""
Osaurus Library - Generic LLM utilities.
Shim that re-exports from sub-modules.
"""

import re
import json
import time
import requests
from typing import Any, Optional, List, Dict

from .content_processing import remove_thinking_blocks
from .logging_config import osaurus_logger as logger
from .config import get_timeout, get_max_tokens_for_task

from lib.llm.quirks import apply_model_quirks

from .osaurus_models import (
    DEFAULT_HOST, DEFAULT_PORT,
    get_api_url, get_base_url, get_models, is_server_running,
    check_llm_availability, get_available_models,
    get_best_model, select_best_vlm_model, select_best_model,
)

from .osaurus_server import (
    restart_server, ensure_server, test_connection, panic_dump,
)

from .osaurus_output import (
    clean_output, extract_json, normalize_keys, merge_flat_dicts,
    filter_json_items, fix_json_years, normalize_text_output,
    _extract_json_only, _extract_plain_list,
    TOP_LEVEL_KEYS, KEY_NORMALIZATIONS,
)

__all__ = [
    "DEFAULT_HOST", "DEFAULT_PORT",
    "get_api_url", "get_base_url", "get_models", "is_server_running",
    "check_llm_availability", "get_available_models",
    "get_best_model", "select_best_vlm_model", "select_best_model",
    "restart_server", "ensure_server", "test_connection", "panic_dump",
    "clean_output", "extract_json", "normalize_keys", "merge_flat_dicts",
    "filter_json_items", "fix_json_years", "normalize_text_output",
    "_extract_json_only", "_extract_plain_list",
    "TOP_LEVEL_KEYS", "KEY_NORMALIZATIONS",
    "apply_model_quirks", "PROMPTS", "call", "call_with_prompt",
    "test_model", "call_llm_api", "extract_thinking",
    "merge_thinking_with_summary", "strip_thinking",
]


PROMPTS = {
    "think": {
        "messages": [
            {"role": "system", "content": "Think step by step if needed. Then provide your answer."},
            {"role": "user", "content": "{prompt}"},
        ]
    },
    "json": {
        "messages": [
            {"role": "system", "content": "Output ONLY valid JSON. Start with { or [. No markdown, no explanations."},
            {"role": "user", "content": "{prompt}"},
        ]
    },
    "summarize": {
        "messages": [
            {"role": "system", "content": "Output headers with ## and key facts. No thinking, no markdown."},
            {"role": "user", "content": "{prompt}"},
        ]
    },
    "filename": {
        "messages": [
            {"role": "system", "content": "Output ONLY a short filename. No explanations. Under 50 chars."},
            {"role": "user", "content": "{prompt}"},
        ]
    },
}


def call(
    model: str,
    messages: List[Dict[str, Any]],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    task: str = "think",
    parse_json: bool = False,
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
    try:
        timeout = timeout or get_timeout(task)
        logger.debug(f"Sending request with {len(messages)} messages, timeout={timeout}s")
        resp = requests.post(
            url, json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        result["time"] = round(time.time() - start, 1)
        logger.debug(f"Response received in {result['time']}s")
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
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
        result["error"] = "Connection failed - is server running?"
        logger.warning(f"Connection error to {url}")
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


def call_with_prompt(
    model: str,
    prompt: str,
    task: str = "think",
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    temperature: float = 0.1,
    max_tokens: int = 16000,
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
    return call(model, messages, host, port, temperature, max_tokens, timeout=get_timeout(task), parse_json=parse_json)


def test_model(model: str, prompt: str = "Hello", task: str = "think", host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> dict:
    return call_with_prompt(model, prompt, task, host, port)


def call_llm_api(
    host: str, model: str, messages: List[dict],
    api_key: str = "", temperature: float = 0.1,
    max_tokens: int = 16000, timeout: int = 600,
    parse_json: bool = False,
) -> dict:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }
    if parse_json:
        payload["response_format"] = {"type": "json_object"}
    try:
        if host.startswith("http"):
            url = f"{host}/v1/chat/completions"
        else:
            url = f"http://{host}/v1/chat/completions"
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
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
