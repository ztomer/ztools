"""
LLM integration for image renaming - server management, relevance checks, and filename queries.
"""

import base64
import json
import os
import re
from pathlib import Path
from typing import Optional

import requests
from lib.config import Task, get_filename_models, get_model_prompt
from lib.config_toml import load_config
from lib.llm.constants import API_CHAT
from lib.logging_config import get_logger
from lib.mlx_lib import call_mlx, find_any_working_mlx_model, find_mlx_model, process_mlx_content
from lib.paths import conf_path
from lib.prompt_render import POSITIONAL_SLOT, render_prompt
from lib.tui import FAIL

from rename.helpers import _strip_instruction_prefix

logger = get_logger("rename.llm")

RELEVANCE_CHECK_PROMPT = """Is this image content useful/interesting enough to keep and rename?
Consider: educational content, useful tips, meaningful information, actionable advice.

Content:
{text}

Answer ONLY one word: "keep" or "skip"."""

PROMPT_IMAGE_TO_FILENAME = (
    "Describe the visual objects in this image using 3 to 4 descriptive "
    "nouns and adjectives (e.g., 'white goose grass'). Ignore any text. "
    "Do not use words like 'image', 'empty', 'text', 'file', or "
    "'filename'. Output ONLY the descriptive words."
)

FILENAME_MODELS = get_filename_models()

PROMPT_TEXT_TO_FILENAME = (
    get_model_prompt(FILENAME_MODELS[0], Task.FILENAME) if FILENAME_MODELS else ""
)

# Load rename config for overridable paths
_RENAME_CONFIG_PATH = conf_path("rename.toml")
_RENAME_CFG = load_config(_RENAME_CONFIG_PATH) or {}

MLX_MODELS_DIR = Path(
    _RENAME_CFG.get("mlx_models_dir", str(Path.home() / "MLXModels"))
).expanduser()

# Timeouts for LLM API calls (seconds)
RELEVANCE_CHECK_TIMEOUT = 5
FILENAME_QUERY_TIMEOUT = 120
VLM_QUERY_TIMEOUT = 60

# Sleep durations for server management (seconds)
PKILL_WAIT = 2
APP_LAUNCH_WAIT = 15

# Connection, path, limit, and status constants (Mitchell Hashimoto design)
DEFAULT_SERVER_URL = _RENAME_CFG.get("llm_url", "http://localhost:1337")
API_CHAT_PATH = API_CHAT
TEXT_PREVIEW_LIMIT = 500
_relevance_models_str = _RENAME_CFG.get("relevance_check_models") or os.environ.get(
    "RENAME_RELEVANCE_MODELS",
    "qwen3.6-27b-mxfp4,gemma-4-26b-a4b-it-mxfp4",  # check-ok: env var fallback
)
RELEVANCE_CHECK_MODELS = [m.strip() for m in _relevance_models_str.split(",") if m.strip()]
MIN_CONTENT_LEN = 2
MAX_FILENAME_WORDS = 6
# The prompts promise "under 50 characters"; slicing at 35 truncated mid-word
# ("apple_foldable_iphone_launch_delaye"). One limit, matching what we ask for.
MAX_FILENAME_LEN = 50
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
        except Exception as e:
            logger.warning(f"Relevance check failed for model {model}: {e}")
            continue

    return None



def _truncate_on_word_boundary(name: str, limit: int) -> str:
    """Cut at the last `_` before `limit` so names never end mid-word."""
    if len(name) <= limit:
        return name
    cut = name[:limit]
    boundary = cut.rfind("_")
    # Only honour the boundary if it leaves something substantial.
    return cut[:boundary] if boundary >= limit // 2 else cut


def _filename_prompt(model: str, text: str) -> str:
    """Render the filename template for `model` through the shared renderer.

    The shipped templates come in two shapes — foundation uses the positional
    `{}` slot, others use `{text}` — so `str.format()` raises `IndexError` on the
    former and the per-model `except` downgraded that to a stderr warning, which
    silently killed the whole LLM naming path. Class C1; render_prompt is the one
    renderer that handles both shapes and fails loudly.
    """
    template = get_model_prompt(model, Task.FILENAME) or PROMPT_TEXT_TO_FILENAME
    if POSITIONAL_SLOT in template:
        return render_prompt(template, template_id=f"{model}:filename", positional=text)
    return render_prompt(template, template_id=f"{model}:filename", text=text)


def query_llm_for_filename(
    text: str, host: str = DEFAULT_SERVER_URL, model: str = "", api_key: str = ""
) -> Optional[str]:
    for m in FILENAME_MODELS:
        try:
            prompt = _filename_prompt(m, text)
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

                # Keep digits: years, versions, order numbers and amounts are the
                # most identifying part of a screenshot name. `[a-z]+` deleted every
                # one of them ("Quarterly Revenue 2025" -> quarterly_revenue).
                words = re.findall(r"[a-z0-9]+", content)
                # Digits belong inside a name, but a name that is ONLY digits
                # identifies nothing — keep the original "needs letters" guard.
                if not any(any(c.isalpha() for c in w) for w in words):
                    continue

                content = "_".join(words[:MAX_FILENAME_WORDS])
                content = _truncate_on_word_boundary(content, MAX_FILENAME_LEN)

                return content
        except Exception as e:
            logger.warning(f"Filename query failed for model {m}: {e}")
            continue

    return None


def query_mlx_for_filename(text: str) -> Optional[str]:
    tried = []
    for model_name in FILENAME_MODELS:
        model_path = find_mlx_model(model_name, MLX_MODELS_DIR)
        if not model_path or model_path in tried:
            continue
        tried.append(model_path)
        prompt = _filename_prompt(model_name, text)
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
        prompt = _filename_prompt(FILENAME_MODELS[0] if FILENAME_MODELS else "", text)
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

        messages = [{"role": "user", "content": prompt, "images": [base64_image]}]

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
        logger.warning(f"VLM error for model {model}: {e}")
        print(f"{FAIL} VLM error: {e}")
        return None
