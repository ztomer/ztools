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
from typing import Any, Optional, List, Dict, Union

from .content_processing import clean_model_output, remove_markdown_blocks, remove_inline_thinking
from .logging_config import osaurus_logger as logger
from .config import get_timeouts, get_max_tokens, get_best_models, get_timeout, get_max_tokens_for_task, get_best_model

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1337

# ==========================================================
# MODEL FAMILY DETECTION & QUIRKS
# ==========================================================

def get_model_family(model: str) -> str:
    """Detect model family for applying family-specific quirks."""
    model_lower = model.lower()
    if "qwen3.6" in model_lower or "qwen3.5" in model_lower:
        return "qwen"
    elif "gemma-4" in model_lower or "gemma4" in model_lower:
        return "gemma4"
    elif "gemma" in model_lower:
        return "gemma"
    elif "foundation" in model_lower:
        return "foundation"
    return "unknown"


def apply_model_quirks(messages: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """Apply model-specific prompt modifications.
    
    This ensures consistent behavior across all scripts calling the LLM.
    """
    family = get_model_family(model)
    
    # Build updated messages
    updated = []
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "user")
        
        if family == "qwen" and role == "system":
            # Prepend JSON trigger for qwen models to prevent thinking output
            if content and not content.startswith("Output JSON now"):
                content = "Output JSON now.\n\n" + content
                logger.debug(f"Applied qwen JSON trigger for {model}")
        
        elif family == "gemma4":
            if role == "system":
                # Gemma4 needs extraction framing
                if "JSON" in content.upper() and not content.startswith("IMPORTANT"):
                    content = "IMPORTANT: This is DATA EXTRACTION. Output JSON only. " + content
                    logger.debug(f"Applied gemma4 system quirk for {model}")
            elif role == "user":
                # Gemma4 responds badly to "Execute", "Context", "Task" - use "Data" / "Extract"
                if "execute" in content.lower() or "context" in content.lower():
                    content = content.replace("Current Context", "Data")
                    content = content.replace("Execute the task", "Extract to JSON")
                    content = content.replace("Execute the task based on", "Extract")
                    logger.debug(f"Applied gemma4 user quirk for {model}")
        
        updated.append({**msg, "content": content})
    
    return updated
DEFAULT_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}"

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
    from .config import get_best_model as _get_best
    if task:
        return os.environ.get(env_var, _get_best(task))
    return os.environ.get(env_var, "foundation")



# ==========================================================
# OUTPUT PROCESSING
# ==========================================================


# Re-export for backward compatibility
clean_output = clean_model_output


def _extract_json_only(content: str) -> Optional[str]:
    if not content:
        return None

    # Use shared cleanup function
    content = clean_model_output(content)
    # Extra pass: strip verbose inline reasoning (Qwen/Gemma)
    content = remove_inline_thinking(content)

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
    import json
    json_str = _extract_json_only(content)

    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # If no JSON found or parsing fails, try to parse plain text numbered list
    return _extract_plain_list(content)


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
) -> dict:
    """Call LLM API. Returns dict with content, parsed, time, error.
    
    This is a pure transport/parsing layer. Validation and retry logic
    should be handled by the caller (e.g. model_eval.py).
    """

    logger.debug(f"Calling {model} for task '{task}' at {host}:{port}")

    # Apply model-specific quirks (e.g., JSON trigger for qwen)
    messages = apply_model_quirks(messages, model)

    # Get defaults from config.yaml (single source of truth)
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

    result = {
        "model": model,
        "time": None,
        "content": None,
        "parsed": None,
        "error": None,
    }

    start = time.time()

    try:
        timeout = timeout or get_timeout(task)
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
        result["content"] = clean_output(content)
        logger.debug(f"Extracted {len(content)} chars of content")

        # For JSON tasks: extract and parse JSON from raw content
        if parse_json and content:
            result["parsed"] = extract_json(content)
            if result["parsed"]:
                logger.debug("JSON parsed successfully")
            else:
                logger.warning(f"Could not parse JSON from output")

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

    # Determine if we should parse JSON based on task type
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
    """Quick test of model."""
    return call_with_prompt(model, prompt, task, host, port)


# ==========================================================
# CONSOLIDATED FUNCTIONS (from multiple scripts)
# ==========================================================


def check_llm_availability(
    host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, api_key: str = ""
) -> bool:
    """Check if LLM server is available. Alias for is_server_running()."""
    return is_server_running(host, port)


# Alias for backward compatibility
get_available_models = get_models


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
    max_tokens: int = 16000,
    timeout: int = 600,
    parse_json: bool = False,
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
    
    if parse_json:
        payload["response_format"] = {"type": "json_object"}

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
    """Remove thinking blocks. Alias for content_processing.remove_thinking_blocks."""
    from .content_processing import remove_thinking_blocks
    return remove_thinking_blocks(text)


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
