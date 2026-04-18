#!/usr/bin/env python3
"""
Osaurus Library - Generic LLM utilities.
Does NOT contain tool-specific logic.
"""

import os
import re
import json
import time
import requests
from pathlib import Path
from typing import Any, Optional, List, Dict, Union, Callable

from .validators_lib import (
    validate_json,
    validate_detailed_json,
    validate_summary,
    validate_filename,
    VALIDATORS,
)
from .content_processing import clean_model_output, remove_markdown_blocks
from .logging_config import osaurus_logger as logger
from .config import get_timeouts, get_max_tokens, get_best_models, init_config, is_config_loaded

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1337
DEFAULT_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

# ==========================================================
# CONFIG - Now loaded explicitly via init_config()
# ==========================================================

# Default values (used if config not loaded)
DEFAULT_TIMEOUTS = {
    "think": 30,
    "json": 60,
    "summarize": 30,
    "filename": 15,
    "vlm": 45,
}

DEFAULT_MAX_TOKENS = {
    "think": 2000,
    "json": 2000,
    "summarize": 2000,
    "filename": 500,
    "vlm": 3000,
}

DEFAULT_BEST_MODELS = {
    "think": "gemma-4-26b-a4b-it-4bit",
    "json": "gemma-4-26b-a4b-it-4bit",
    "summarize": "gemma-4-26b-a4b-it-4bit",
    "filename": "gemma-4-26b-a4b-it-4bit",  # Changed from foundation
    "vlm": "gemma-4-26b-a4b-it-4bit",
}

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
                "content": "Output ONLY valid JSON. Start with { or [. No markdown, no explanations.",
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

# ==========================================================
# GENERIC LLM UTILITIES
# ==========================================================


def get_api_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}/v1/chat/completions"


def get_base_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}"


def get_models(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> List[str]:
    """Get available models."""
    try:
        resp = requests.get(f"http://{host}:{port}/v1/models", timeout=10)
        if resp.status_code == 200:
            return [m["id"] for m in resp.json().get("data", [])]
    except requests.exceptions.Timeout:
        pass
    except requests.exceptions.ConnectionError:
        pass
    except Exception as e:
        print(f"Warning: get_models failed: {e}")
    return []


def is_server_running(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    """Check if server is available."""
    try:
        resp = requests.get(f"http://{host}:{port}/v1/models", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.Timeout:
        return False
    except requests.exceptions.ConnectionError:
        return False
    except Exception as e:
        print(f"Warning: server check failed: {e}")
        return False


def get_best_model(task: str = None, env_var: str = "OLLAMA_MODEL") -> str:
    """Get best model for task, checking env var first."""
    best_models = get_best_models() if is_config_loaded() else DEFAULT_BEST_MODELS
    if task and task in best_models:
        return os.environ.get(env_var, best_models[task])
    return os.environ.get(env_var, "foundation")


# ==========================================================
# OUTPUT PROCESSING
# ==========================================================


def clean_output(content: str) -> str:
    """Strip thinking tokens, stats, control chars. Returns cleaned text.

    Wrapper for content_processing.clean_model_output() for backward compatibility.
    """
    return clean_model_output(content)


def _extract_json_only(content: str) -> Optional[str]:
    if not content:
        return None

    # Use shared cleanup function
    content = clean_model_output(content)

    content = content.strip()

    # Find { or [ that starts actual JSON
    first_brace = content.find("{")
    first_bracket = content.find("[")
    if first_brace >= 0 or first_bracket >= 0:
        start = min(x for x in [first_brace, first_bracket] if x >= 0)
        content = content[start:]
    else:
        return None

    # Find last } or ] to get full JSON
    last_brace = content.rfind("}")
    last_bracket = content.rfind("]")
    last_json = max(last_brace, last_bracket)

    if last_json > 0:
        content = content[: last_json + 1]

    return content


def extract_json(content: str) -> Union[Dict[str, Any], List[Any], None]:
    """Extract JSON from content - handles plain text lists too."""
    json_result = _extract_json_only(content)

    # If no JSON found, try to parse plain text numbered list
    if json_result is None:
        json_result = _extract_plain_list(content)

    return json_result


def _extract_plain_list(content: str) -> Optional[List[Dict[str, Any]]]:
    """Extract items from plain text like '1. Apple\\n2. Banana'."""
    if not content:
        return None

    items = []
    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Match "1. Item name" or "- Item name" or any line with text
        match = re.match(r"^\d+[\.\)]\s+(.+)$|^- (.+)$", line)
        if match:
            name = (match.group(1) or match.group(2)).strip()
            if name and not name.startswith("#"):
                items.append({"name": name})
        elif line and not line.startswith("#") and len(line) > 1:
            # Just any line item
            items.append({"name": line})

    return items if items else None


# ==========================================================
# QUALITY VALIDATORS
# ==========================================================


# ==========================================================
# CORE API CALL
# ==========================================================


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
    validator: Optional[Callable] = None,
    max_retries: int = 1,
) -> dict:
    """Call LLM API with automatic retry on low quality scores or failures."""
    
    last_result = None
    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.warning(f"Retrying task '{task}' with model {model} (Attempt {attempt+1}/{max_retries+1})...")
            
        result = _call_impl(
            model=model,
            messages=messages,
            host=host,
            port=port,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            task=task,
            parse_json=parse_json,
            validator=validator,
        )
        
        last_result = result
        
        # If there's an error, we might want to retry
        if result.get("error"):
            continue
            
        # If there's a validator and the score is less than 90, retry
        if validator and result.get("quality_score", 0) < 90:
            continue
            
        # If everything is ok, break and return
        break
        
    return last_result

def _call_impl(
    model: str,
    messages: List[Dict[str, Any]],
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    task: str = "think",
    parse_json: bool = False,
    validator: Optional[Callable] = None,
) -> dict:
    """Call LLM API. Returns dict with content, parsed, quality, time, error."""

    logger.debug(f"Calling {model} for task '{task}' at {host}:{port}")

    # Get defaults from config
    max_tokens_config = get_max_tokens() if is_config_loaded() else DEFAULT_MAX_TOKENS
    max_tokens = max_tokens or max_tokens_config.get(task, 2000)

    url = get_api_url(host, port)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    result = {
        "model": model,
        "time": None,
        "content": None,
        "parsed": None,
        "quality_score": 0,
        "error": None,
    }

    start = time.time()

    try:
        timeouts_config = get_timeouts() if is_config_loaded() else DEFAULT_TIMEOUTS
        timeout = timeout or timeouts_config.get(
            task, timeouts_config.get("think", 30))
        logger.debug(
            f"Sending request with {len(messages)} messages, timeout={timeout}s")
        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        result["time"] = round(time.time() - start, 1)
        logger.debug(f"Response received in {result['time']}s")

        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            logger.error(f"HTTP error: {result['error']}")
            return result

        # Safely extract content
        resp_data = resp.json()
        if "choices" not in resp_data or not resp_data["choices"]:
            result["error"] = "Empty response from API"
            logger.error(result["error"])
            return result

        message = resp_data["choices"][0].get("message", {})
        content = message.get("content", "")
        result["content"] = content
        logger.debug(f"Extracted {len(content)} chars of content")

        # Quality validation - run for ALL tasks (JSON and text)
        if validator and content:
            cleaned = clean_output(content)
            # For JSON tasks: extract JSON from raw content
            if parse_json:
                result["parsed"] = extract_json(content)
                if result["parsed"]:
                    logger.debug("JSON parsed successfully")
                    validated = validator(result["parsed"])
                    if isinstance(validated, tuple):
                        result["quality_score"], result["failure_reason"] = validated
                    else:
                        result["quality_score"] = validated
                    logger.info(
                        f"Quality score: {result['quality_score']}/100")
            else:
                # Text tasks - validate cleaned content
                validated = validator(cleaned if cleaned else content)
                if isinstance(validated, tuple):
                    result["quality_score"], result["failure_reason"] = validated
                else:
                    result["quality_score"] = validated
                logger.info(f"Quality score: {result['quality_score']}/100")

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
    max_tokens: int = 2000,
) -> dict:
    """Call model with prompt template."""

    if task in PROMPTS:
        template = PROMPTS[task]
        messages = []
        for msg in template["messages"]:
            content = msg["content"]
            # Replace placeholder
            for placeholder in ["{prompt}", "{text}", "{items}", "{tweets}"]:
                if placeholder in content:
                    content = content.replace(placeholder, prompt)
            messages.append({"role": msg["role"], "content": content})
    else:
        messages = [{"role": "user", "content": prompt}]

    # Determine if we should parse JSON
    parse_json = task in VALIDATORS
    validator = VALIDATORS.get(task) if parse_json else None

    return call(
        model,
        messages,
        host,
        port,
        temperature,
        max_tokens,
        timeout=TIMEOUTS.get(task),
        parse_json=parse_json,
        validator=validator,
    )


def test_model(
    model: str,
    prompt: str = "Hello",
    task: str = "think",
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> dict:
    """Quick test of model."""
    return call_with_prompt(model, prompt, task, host, port)


# ==========================================================
# CONSOLIDATED FUNCTIONS (from multiple scripts)
# ==========================================================


def check_llm_availability(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, api_key: str = ""
) -> bool:
    """
    Check if LLM server is available.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for path in ["/api/tags", "/v1/models"]:
        try:
            response = requests.get(
                f"http://{host}:{port}{path}", headers=headers, timeout=2
            )
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
    return False


def get_available_models(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT
) -> List[str]:
    """Get list of available models from server."""
    try:
        response = requests.get(f"http://{host}:{port}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
    except Exception:
        pass
    return []


def select_best_model(
    available_models: List[str],
    preferred: List[str] = None,
) -> Optional[str]:
    """
    Select best model from available list based on preferred order.
    """
    if preferred is None:
        best_models = get_best_models() if is_config_loaded() else DEFAULT_BEST_MODELS
        preferred = list(best_models.values())

    for pref in preferred:
        for model in available_models:
            if pref.lower() in model.lower():
                return model

    return available_models[0] if available_models else None


def select_best_vlm_model(available_models: List[str]) -> Optional[str]:
    """Select best VLM from available models."""
    vlm_keywords = ["vl", "vision", "qwen", "llamavl"]

    for keyword in vlm_keywords:
        for model in available_models:
            if keyword.lower() in model.lower():
                return model

    return None


def call_llm_api(
    host: str,
    model: str,
    messages: List[dict],
    api_key: str = "",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    timeout: int = 30,
) -> dict:
    """
    Generic LLM API call (Ollama/OpenAI compatible).
    Returns dict with content, usage, etc.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            f"{host}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        return {
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
            "model": data.get("model", model),
        }
    except Exception as e:
        return {"error": str(e)}


def strip_thinking(text: str) -> str:
    """Remove thinking blocks from model output."""
    if not text:
        return ""

    # Remove <think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove XML-like thinking tags
    text = re.sub(r"<\|.*?\|>", "", text)

    # Remove "Thinking Process:" sections
    if "Thinking Process:" in text:
        text = text.split("Thinking Process:")[-1]

    return text.strip()


def parse_llm_response(resp_text: str) -> str:
    """Parse LLM response - extract content from various formats."""
    if not resp_text:
        return ""

    # Already processed by strip_thinking
    text = resp_text.strip()

    # Extract from markdown code blocks
    code_blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", text, re.DOTALL)
    if code_blocks:
        text = code_blocks[-1]

    return text.strip()


def panic_dump(raw_text: str) -> None:
    """Save problematic LLM output for debugging."""
    dump_dir = Path("/tmp/llm_panics")
    dump_dir.mkdir(exist_ok=True)

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dump_file = dump_dir / f"panic_{timestamp}.txt"

    with open(dump_file, "w") as f:
        f.write(raw_text)

    print(f"[PANIC] Dumped to {dump_file}")


# ==========================================================
# SERVER MANAGEMENT
# ==========================================================


def restart_server(
    app_path: str = "/Applications/osaurus.app",
    wait: int = 20,
) -> bool:
    """Restart the Osaurus server app."""
    import subprocess

    try:
        subprocess.run(["pkill", "-f", "osaurus"], stderr=subprocess.DEVNULL)
    except Exception:
        pass

    time.sleep(2)

    try:
        subprocess.Popen(
            ["open", app_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"Failed to restart: {e}")
        return False

    # Wait for server to be ready
    for i in range(wait):
        time.sleep(1)
        if is_server_running():
            return True

    return False


def ensure_server(
    max_retries: int = 3,
    wait: int = 20,
) -> bool:
    """Ensure server is running, restart if needed."""
    for attempt in range(1, max_retries + 1):
        if is_server_running():
            return True
        print(f"Server not responding (attempt {attempt}/{max_retries})")
        if not restart_server(wait=wait):
            if attempt == max_retries:
                print("Server failed to restart")
                return False

    return is_server_running()


def test_connection(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model: str = None,
) -> dict:
    """Test LLM connection and return status."""
    try:
        # Check server
        if not is_server_running(host, port):
            return {"status": "error", "message": "Server not running"}

        # Get models
        models = get_models(host, port)
        if not models:
            return {"status": "error", "message": "No models available"}

        # Test with a simple prompt
        if model is None:
            model = models[0]

        result = call(
            model,
            [{"role": "user", "content": "Hi"}],
            host,
            port,
            timeout=10,
        )

        if result.get("error"):
            return {"status": "error", "message": result["error"]}

        return {
            "status": "ok",
            "models": models,
            "test_model": model,
            "response_preview": result.get("content", "")[:100],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
