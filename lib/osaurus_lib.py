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

from .content_processing import clean_model_output, remove_markdown_blocks, remove_inline_thinking, extract_content_from_code_blocks
from .logging_config import osaurus_logger as logger
from .config import get_timeouts, get_max_tokens, get_best_models, get_timeout, get_max_tokens_for_task, get_best_model

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1337

# ==========================================================
# MODEL FAMILY DETECTION & QUIRKS
# ==========================================================

# Re-use get_model_family from config
from .config import get_model_family as _get_model_family


def apply_model_quirks(messages: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """Apply model-specific prompt modifications.
    
    This ensures consistent behavior across all scripts calling the LLM.
    """
    family = _get_model_family(model)
    
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

        if role == "user":
            # Models respond badly to "Execute", "Context", "Task" - use "Data" / "Extract"
            if "execute" in content.lower() or "context" in content.lower():
                content = content.replace("Current Context", "Data")
                content = content.replace("Execute the task", "Extract to JSON")
                content = content.replace("Execute the task based on", "Extract")
                logger.debug(f"Applied user quirk for {model}")
        
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


def get_models(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, api_key: str = "") -> List[str]:
    """Get available models."""
    try:
        if host.startswith("http"):
            url = f"{host}/v1/models"
        else:
            url = f"http://{host}:{port}/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = requests.get(url, timeout=10, headers=headers)
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
        if host.startswith("http"):
            url = f"{host}/v1/models"
        else:
            url = f"http://{host}:{port}/v1/models"
        resp = requests.get(url, timeout=3)
        return resp.status_code in (200, 404)
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

    # Try extracting from code blocks first
    code_block = extract_content_from_code_blocks(content)
    if code_block:
        content = code_block
        # Also clean code blocks for stats tokens!
        content = clean_model_output(content)
    else:
        # Use shared cleanup function
        content = clean_model_output(content)
        # Extra pass: strip verbose inline reasoning (Qwen/Gemma)
        content = remove_inline_thinking(content)

    content = content.strip()

    import json

    def find_json(start_char, end_char):
        starts = [i for i, c in enumerate(content) if c == start_char]
        ends = [i for i, c in enumerate(content) if c == end_char]
        ends.reverse()
        
        # Try up to 15 starts and 15 ends (max 225 combinations)
        for s in starts[:15]:
            for e in ends[:15]:
                if e > s:
                    candidate = content[s:e+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        continue
        return None

    # Try JSON object first { ... }
    res = find_json('{', '}')
    if res: return res
    
    # Try JSON array next [ ... ]
    res = find_json('[', ']')
    if res: return res

    return None


def extract_json(content: str, model: str = None) -> Union[Dict[str, Any], List[Any], None]:
    """Extract JSON from content - handles plain text lists, normalizes keys."""
    import json

    # Try JSON first
    json_str = _extract_json_only(content)
    if json_str:
        try:
            data = json.loads(json_str)
            return normalize_keys(data, model)
        except json.JSONDecodeError:
            pass

    # Try plain text list
    data = _extract_plain_list(content)
    if data:
        return normalize_keys(data, model)

    # Try text normalization
    data = normalize_text_output(content)
    if data:
        return normalize_keys(data, model)

    return None


def _extract_plain_list(content: str) -> Optional[List[Dict[str, Any]]]:
    """Extract items from plain text like '1. Apple\n2. Banana'."""
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
# POST-PROCESSING: Model-specific normalizations
# ==========================================================

# Top-level key normalizations (the array container key)
TOP_LEVEL_KEYS = {
    "fixed_activities": "fixed_activities",
    "activities": "fixed_activities",
    "year_round_activities": "fixed_activities",
    "indoor_play_places": "fixed_activities",
    "play_places": "fixed_activities",
    "venues": "fixed_activities",
    "transient_events": "transient_events",
    "events": "transient_events",
    "limited_time_events": "transient_events",
}

# Key normalizations for alternate model schemas
KEY_NORMALIZATIONS = {
    # Alternate identity keys
    "event": "name",
    "title": "activity",
    "place": "name",
    # Alternate location keys
    "venue": "location",
    "address": "location",
    "where": "location",
    # Alternate time keys
    "date": "day",
    "when": "day",
    "time": "duration",
    # Alternate audience keys
    "age_group": "target_ages",
    "ages": "target_ages",
    "audience": "target_ages",
    "who": "target_ages",
    # Alternate price keys
    "cost": "price",
    "pricing": "price",
    # Alternate weather keys
    "type": "weather",
    "setting": "weather",
    "indoor_outdoor": "weather",
}


def normalize_keys(data: Any, model: str = None) -> Any:
    """Normalize alternate key names to standard schema.

    Uses hardcoded defaults plus model-specific overrides from config.
    """
    if not data:
        return data

    # Get model-specific key mappings from config
    from lib.config import get_model_config

    model_mappings = {}
    if model:
        config = get_model_config(model)
        model_mappings = config.get("key_mappings", {})

    # Merge: default mappings + model-specific (model takes priority)
    all_mappings = {**KEY_NORMALIZATIONS, **model_mappings}

    # Handle dict with top-level key (extract array if needed)
    if isinstance(data, dict):
        for old_key, new_key in TOP_LEVEL_KEYS.items():
            if old_key in data:
                arr = data[old_key]
                if isinstance(arr, list):
                    return {new_key: [normalize_keys(item, model) for item in arr]}
                elif isinstance(arr, dict):
                    for k, v in arr.items():
                        if isinstance(v, list):
                            return {new_key: [normalize_keys(item, model) for item in v]}
                return {new_key: arr}

        # Normalize keys within dict
        result = {}
        for key, value in data.items():
            new_key = all_mappings.get(key, key)
            result[new_key] = normalize_keys(value, model)
        return result

    if isinstance(data, list):
        return [normalize_keys(item, model) for item in data]

    return data


def normalize_text_output(text_output: str) -> List[Dict[str, Any]]:
    """Convert formatted text output to JSON objects.

    Handles outputs like:
    - "1. Item - Location - Ages - Price"
    - Markdown tables
    - "Name (Location): details"
    """
    if not text_output:
        return []

    items = []

    # Pattern: Numbered/bulleted items "1. Name: details" or "1. Name - details"
    for line in text_output.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Extract name and details
        match = re.match(r"^\d+[\.\)]\s*([^\-:?]+)(?:\s*[-:]\s*(.+))?$", line)
        if not match:
            match = re.match(r"^-\s*([^\-:?]+)(?:\s*[-:]\s*(.+))?$", line)

        if match:
            name = (match.group(1) or "").strip()
            details = (match.group(2) or "").strip() if match.lastindex and match.group(2) else ""

            if name and not name.startswith("*"):
                item = {"name": name}
                parts = [p.strip() for p in re.split(r"[-:,]", details) if p.strip()] if details else []

                # Map: location, target_ages, price, weather
                if len(parts) > 0 and parts[0]:
                    item["location"] = parts[0]
                if len(parts) > 1 and parts[1]:
                    item["target_ages"] = parts[1]
                if len(parts) > 2 and parts[2]:
                    item["price"] = parts[2]
                if len(parts) > 3 and parts[3]:
                    item["weather"] = parts[3]

                items.append(item)

    return items





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
        if host.startswith("http"):
            url = f"{host}/v1/chat/completions"
        else:
            url = f"http://{host}/v1/chat/completions"
        response = requests.post(
            url,
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


def extract_thinking(text: str) -> tuple[str, str]:
    """Extract thinking block and main content separately.

    Returns (thinking, content) where content excludes the thinking block.
    """
    from .content_processing import remove_thinking_blocks

    # Find thinking block
    think_match = re.search(r"<thinking[^>]*>(.+?)</thinking>", text, re.DOTALL)
    if not think_match:
        return "", text

    thinking = think_match.group(1).strip()
    content = remove_thinking_blocks(text)

    return thinking, content


def merge_thinking_with_summary(thinking: str, summary: str) -> str:
    """Merge thinking with summary - preserve both for max signal.

    Creates a summary that includes the thinking insights as ## Analysis.
    """
    if not thinking:
        return summary

    return f"{summary}\n\n## Analysis\n{thinking}"


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


def panic_dump(content: str) -> None:
    """Dump problematic content to a file for debugging."""
    import tempfile
    from pathlib import Path
    dump_dir = Path.home() / "llm_dumps"
    dump_dir.mkdir(exist_ok=True)
    dump_file = dump_dir / f"panic_{int(time.time())}.txt"
    dump_file.write_text(content or "(empty)")
    logger.warning(f"Dumped problematic output to {dump_file}")


def select_best_model(models: list, preferred: list = None) -> str:
    """Select best model from available models based on preferred list."""
    if not models:
        return None
    if preferred is None:
        preferred = ["foundation", "qwen", "gemma"]
    for pref in preferred:
        for model in models:
            if pref.lower() in model.lower():
                return model
    return models[0] if models else None
