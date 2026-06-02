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
from lib.mlx_lib import find_mlx_model, process_mlx_content, call_mlx
from lib.tui import WARN, FAIL
from img_helpers import _strip_instruction_prefix

RELEVANCE_CHECK_PROMPT = """Is this image content useful/interesting enough to keep and rename?
Consider: educational content, useful tips, meaningful information, actionable advice.

Content:
{text}

Answer ONLY one word: "keep" or "skip"."""

PROMPT_IMAGE_TO_FILENAME = "Describe the visual objects in this image using 3 to 4 descriptive nouns and adjectives (e.g., 'white goose grass'). Ignore any text. Do not use words like 'image', 'empty', 'text', 'file', or 'filename'. Output ONLY the descriptive words."

FILENAME_MODELS = get_filename_models()

PROMPT_TEXT_TO_FILENAME = get_model_prompt(FILENAME_MODELS[0], Task.FILENAME) if FILENAME_MODELS else ""

MLX_MODELS_DIR = Path.home() / "MLXModels"


def ensure_llm_running() -> bool:
    """Detect crash and restart server if needed."""
    if check_llm_availability("http://localhost:1337"):
        return True

    print(f"{WARN} LLM server not responding, restarting...")

    try:
        subprocess.run(["pkill", "-f", "osaurus"], capture_output=True)
        time.sleep(2)
    except Exception:
        pass

    try:
        subprocess.Popen(["open", "-a", "osaurus"])
        time.sleep(15)
        if check_llm_availability("http://localhost:1337"):
            return True
    except Exception:
        pass

    return False


def is_relevant_with_llm(text: str, host: str, api_key: str = "") -> Optional[bool]:
    """Ask LLM if image content is relevant worth keeping."""
    prompt = RELEVANCE_CHECK_PROMPT.format(text=text[:500])
    messages = [{"role": "user", "content": prompt}]

    for model in ["qwen3.6-27b-mxfp4", "gemma-4-26b-a4b-it-mxfp4"]:
        try:
            resp = requests.post(
                f"{host}/api/chat",
                json={"model": model, "messages": messages},
                timeout=5,
            )
            if resp.status_code != 200:
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
    text: str, host: str = "http://localhost:1337", model: str = "", api_key: str = ""
) -> Optional[str]:
    for m in FILENAME_MODELS:
        try:
            prompt = PROMPT_TEXT_TO_FILENAME.format(text=text)
            messages = [{"role": "user", "content": prompt}]

            resp = requests.post(
                f"{host}/api/chat",
                json={"model": m, "messages": messages},
                timeout=120,
            )
            if resp.status_code != 200:
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

            if content and len(content) >= 2:
                content = content.strip().lower()

                content = _strip_instruction_prefix(content)

                words = re.findall(r'[a-z]+', content)
                if not words:
                    continue

                content = '_'.join(words[:6])
                if len(content) > 35:
                    content = content[:35]

                if not re.match(r"^[a-z_]+$", content):
                    continue

                if not any(c.isalpha() for c in content):
                    continue

                return content
        except Exception:
            continue

    return None


def query_mlx_for_filename(text: str) -> Optional[str]:
    for model_name in FILENAME_MODELS:
        model_path = find_mlx_model(model_name, MLX_MODELS_DIR)
        if not model_path:
            continue

        try:
            prompt = PROMPT_TEXT_TO_FILENAME.format(text=text)
            raw = call_mlx(model_path, prompt)
            if raw:
                content = process_mlx_content(raw)
                if content and len(content) >= 2:
                    content = content.strip()
                    content = _strip_instruction_prefix(content)
                    content = re.sub(r"[^\x00-\x7F]", "", content)
                    content = re.sub(r"[-\s]+", "_", content)
                    content = content.strip("_").lower()
                    return content
        except Exception:
            continue

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

        resp = requests.post(
            f"{host}/api/chat",
            json={"model": model, "messages": messages},
            headers=headers,
            timeout=60,
        )

        if resp.status_code != 200:
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
