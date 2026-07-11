"""
LLM integration for image renaming - server management, relevance checks, and filename queries.
"""

import json
import re
import base64
import subprocess
import time
from pathlib import Path
from typing import Optional
import requests

from lib.config import get_filename_models, get_model_prompt, Task
from lib.osaurus_lib import check_llm_availability
from lib.mlx_lib import find_mlx_model, find_any_working_mlx_model, process_mlx_content, call_mlx
from lib.tui import WARN, FAIL
from rename.helpers import _strip_instruction_prefix

RELEVANCE_CHECK_PROMPT = """Is this image content useful/interesting enough to keep and rename?
Consider: educational content, useful tips, meaningful information, actionable advice.

Content:
{text}

Answer ONLY one word: "keep" or "skip"."""

PROMPT_IMAGE_TO_FILENAME = "Describe the visual objects in this image using 3 to 4 descriptive nouns and adjectives (e.g., 'white goose grass'). Ignore any text. Do not use words like 'image', 'empty', 'text', 'file', or 'filename'. Output ONLY the descriptive words."

FILENAME_MODELS = get_filename_models()

PROMPT_TEXT_TO_FILENAME = get_model_prompt(FILENAME_MODELS[0], Task.FILENAME) if FILENAME_MODELS else ""

# Load rename config for overridable paths
_RENAME_CONFIG_PATH = Path(__file__).parent.parent / "conf" / "rename.yaml"
try:
    _RENAME_CFG = yaml.safe_load(_RENAME_CONFIG_PATH.read_text()) or {}
except Exception:
    _RENAME_CFG = {}

MLX_MODELS_DIR = Path(_RENAME_CFG.get("mlx_models_dir", str(Path.home() / "MLXModels")))

# Timeouts for LLM API calls (seconds)
RELEVANCE_CHECK_TIMEOUT = 5
FILENAME_QUERY_TIMEOUT = 120
VLM_QUERY_TIMEOUT = 60

# Sleep durations for server management (seconds)
PKILL_WAIT = 2
APP_LAUNCH_WAIT = 15

# Connection, path, limit, and status constants (Mitchell Hashimoto design)
DEFAULT_SERVER_URL = _RENAME_CFG.get("llm_url", "http://localhost:1337")
API_CHAT_PATH = "/api/chat"
TEXT_PREVIEW_LIMIT = 500
RELEVANCE_CHECK_MODELS = ["qwen3.6-27b-mxfp4", "gemma-4-26b-a4b-it-mxfp4"]
MIN_CONTENT_LEN = 2
MAX_FILENAME_WORDS = 6
MAX_FILENAME_LEN = 35
HTTP_STATUS_OK = 200


def ensure_llm_running() -> bool:
    """Detect crash and restart server if needed."""
    from lib.osaurus_lib import ensure_server
    return ensure_server()



def is_relevant_with_llm(text: str, host: str, api_key: str = "") -> Optional[bool]:
    """Ask LLM if image content is relevant worth keeping."""
    prompt = RELEVANCE_CHECK_PROMPT.format(text=text[:TEXT_PREVIEW_LIMIT])
    messages = [{"role": "user", "content": prompt}]

    for model in RELEVANCE_CHECK_MODELS:
        try:
            with requests.Session() as s:
                resp = s.post(
                    f"{host}{API_CHAT_PATH}",
                    json={"model": model, "messages": messages},
                    timeout=RELEVANCE_CHECK_TIMEOUT,
                )
            if resp.status_code != HTTP_STATUS_OK:
                continue
            content = ""
            for line in resp.text.split("\n"):
                if line.strip():
                    try:
                        j = json.loads(line)
                        content = j.get("message", {}).get("content", "").lower()
                        break
                    except Exception:
                        continue

            if "keep" in content and "skip" not in content:
                return True
            elif "skip" in content:
                return False
        except Exception:
            continue

    return None


def query_llm_for_filename(
    text: str, host: str = DEFAULT_SERVER_URL, model: str = "", api_key: str = ""
) -> Optional[str]:
    for m in FILENAME_MODELS:
        try:
            prompt = PROMPT_TEXT_TO_FILENAME.format(text=text)
            messages = [{"role": "user", "content": prompt}]

            with requests.Session() as sess:
                resp = sess.post(
                    f"{host}{API_CHAT_PATH}",
                    json={"model": m, "messages": messages},
                    timeout=FILENAME_QUERY_TIMEOUT,
                )
            if resp.status_code != HTTP_STATUS_OK:
                continue

            content = ""
            for line in resp.text.split("\n"):
                if line.strip():
                    try:
                        j = json.loads(line)
                        content += j.get("message", {}).get("content", "")
                        if j.get("done", False):
                            break
                    except Exception:
                        continue

            if content and len(content) >= MIN_CONTENT_LEN:
                content = content.strip().lower()

                content = _strip_instruction_prefix(content)

                words = re.findall(r'[a-z]+', content)
                if not words:
                    continue

                content = '_'.join(words[:MAX_FILENAME_WORDS])
                if len(content) > MAX_FILENAME_LEN:
                    content = content[:MAX_FILENAME_LEN]

                return content
        except Exception:
            continue

    return None


def query_mlx_for_filename(text: str) -> Optional[str]:
    tried = []
    for model_name in FILENAME_MODELS:
        model_path = find_mlx_model(model_name, MLX_MODELS_DIR)
        if not model_path or model_path in tried:
            continue
        tried.append(model_path)
        prompt = PROMPT_TEXT_TO_FILENAME.format(text=text)
        raw = call_mlx(model_path, prompt)
        if raw:
            content = process_mlx_content(raw)
            if content and len(content) >= MIN_CONTENT_LEN:
                content = content.strip()
                content = _strip_instruction_prefix(content)
                content = re.sub(r"[^\x00-\x7F]", "", content)
                content = re.sub(r"[-\s]+", "_", content)
                content = content.strip("_").lower()
                return content

    fallback = find_any_working_mlx_model()
    if fallback and fallback not in tried:
        prompt = PROMPT_TEXT_TO_FILENAME.format(text=text)
        raw = call_mlx(fallback, prompt)
        if raw:
            content = process_mlx_content(raw)
            if content and len(content) >= MIN_CONTENT_LEN:
                content = content.strip()
                content = _strip_instruction_prefix(content)
                content = re.sub(r"[^\x00-\x7F]", "", content)
                content = re.sub(r"[-\s]+", "_", content)
                content = content.strip("_").lower()
                return content

    return None


def query_vlm_for_filename(
    image_path: Path, host: str, model: str, api_key: str = ""
) -> Optional[str]:
    """Query a Vision Language Model to describe the image using direct HTTP requests."""
    prompt = PROMPT_IMAGE_TO_FILENAME

    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        messages = [{
            "role": "user",
            "content": prompt,
            "images": [base64_image]
        }]

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with requests.Session() as s:
            resp = s.post(
                f"{host}{API_CHAT_PATH}",
                json={"model": model, "messages": messages},
                headers=headers,
                timeout=VLM_QUERY_TIMEOUT,
            )

        if resp.status_code != HTTP_STATUS_OK:
            print(f"{FAIL} VLM API Error: {resp.status_code} - {resp.text}")
            return None

        content = ""
        for line in resp.text.split("\n"):
            if line.strip():
                try:
                    j = json.loads(line)
                    content += j.get("message", {}).get("content", "")
                    if j.get("done", False):
                        break
                except Exception:
                    continue

        return _strip_instruction_prefix(content.strip()) if content else None

    except Exception as e:
        print(f"{FAIL} VLM error: {e}")
        return None
