#!/usr/bin/env python3
"""
MLX Library - Utilities for running MLX models directly.
Parallel to osaurus_lib for server-based LLM calls.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .content_processing import clean_model_output, extract_content_from_code_blocks
from .logging_config import mlx_logger as logger

# Get the venv Python with mlx_lm
# For MLX, use uv run to get the right Python environment
UV_RUN = ["rtk", "uv", "run", "--with", "mlx", "--with", "mlx-lm"]

# Timeouts for MLX subprocess calls (seconds)
MLX_GEN_TIMEOUT = 1800
MLX_FALLBACK_TIMEOUT = 600
MLX_VLM_TIMEOUT = 180
# Load-probe timeout: attempt to load the model (weights + tokenizer) and bail
# fast if the installed mlx_lm cannot construct the architecture. Osaurus-format
# models (Gemma4/qwen3_5_moe, MXFP8) fail here with a parameter mismatch.
MLX_LOAD_PROBE_TIMEOUT = int(os.environ.get("MLX_LOAD_PROBE_TIMEOUT", "120"))

# Last failure reason from call_mlx, so callers can surface WHY a model failed
# (load mismatch, timeout, empty output) instead of a generic "failed".
_LAST_MLX_ERROR: Optional[str] = None


def last_mlx_error() -> Optional[str]:
    """Return the reason the most recent call_mlx() failed, or None."""
    return _LAST_MLX_ERROR

# MLX generation parameters
MLX_MAX_TOKENS = 8192
DEFAULT_CTX_LENGTH = 4096

# Unified API defaults (matches osaurus_lib interface)
MLX_DEFAULT_PORT = 1337
MLX_DEFAULT_TEMPERATURE = 0.1

# MLX models directory
MLX_MODELS_DIR = Path(os.environ.get("MLX_MODELS_DIR", Path.home() / "MLXModels"))


# ==========================================================
# MODEL DISCOVERY
# ==========================================================


# Cache of (path -> (loadable, reason)) so we probe each model at most once
# per process. Osaurus-format models fail the probe; re-probing them would waste
# ~minutes per fallback attempt.
_LOAD_PROBE_CACHE: Dict[str, tuple] = {}


def _looks_like_mlx_model(model_path: Path) -> bool:
    """Cheap structural check: a directory with a config.json."""
    return model_path.is_dir() and (model_path / "config.json").exists()


def probe_mlx_loadable(model_path: Path) -> tuple:
    """Attempt a real (but lightweight) load to see if the installed mlx_lm can
    construct this model. Returns (loadable: bool, reason: str).

    This catches Osaurus-format models (mlx-swift quantized Gemma4/qwen3_5_moe)
    that stock Python mlx_lm cannot load, so the fallback can skip them fast
    instead of spending the full generation timeout per model.

    Results are cached per path for the life of the process.
    """
    if not _looks_like_mlx_model(model_path):
        return (False, "not a model directory (no config.json)")

    key = str(model_path)
    if key in _LOAD_PROBE_CACHE:
        return _LOAD_PROBE_CACHE[key]

    model_path_escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    probe = (
        "import sys\n"
        "try:\n"
        "    from mlx_lm import load\n"
        f'    load("{model_path_escaped}")\n'
        '    print("MLX_LOAD_OK", flush=True)\n'
        "except Exception as e:\n"
        '    print(f"MLX_LOAD_FAIL {type(e).__name__}: {e}", flush=True)\n'
        "    sys.exit(1)\n"
    )
    try:
        result = subprocess.run(
            UV_RUN + ["python3", "-c", probe],
            capture_output=True,
            text=True,
            timeout=MLX_LOAD_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        outcome = (False, f"load timed out after {MLX_LOAD_PROBE_TIMEOUT}s")
        _LOAD_PROBE_CACHE[key] = outcome
        return outcome
    except Exception as e:
        outcome = (False, f"probe error: {type(e).__name__}: {e}")
        _LOAD_PROBE_CACHE[key] = outcome
        return outcome

    out = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0 and "MLX_LOAD_OK" in out:
        outcome = (True, "ok")
    else:
        reason = "unknown load failure"
        for line in out.splitlines():
            if line.startswith("MLX_LOAD_FAIL"):
                reason = line[len("MLX_LOAD_FAIL ") :].strip()
                break
        outcome = (False, reason)
    _LOAD_PROBE_CACHE[key] = outcome
    return outcome


def _check_mlx_model_compatible(model_path: Path, deep: bool = False) -> bool:
    """Check whether a path is a usable MLX model.

    Structural check by default (fast). When ``deep`` is True, additionally run
    a real load-probe so models the installed mlx_lm cannot construct are
    rejected before a full generation attempt.
    """
    if not _looks_like_mlx_model(model_path):
        return False
    if deep:
        return probe_mlx_loadable(model_path)[0]
    return True


def find_mlx_model(model_name: str, mlx_dir: Path = MLX_MODELS_DIR) -> Optional[Path]:
    """Find an MLX model by name in the models directory."""
    if not mlx_dir.exists():
        return None

    name_lower = model_name.lower()
    for item in mlx_dir.iterdir():
        if not item.is_dir():
            continue
        if name_lower in item.name.lower():
            return item
        # Check subdirs
        for sub in item.iterdir():
            if sub.is_dir() and name_lower in sub.name.lower():
                return sub

    return None


def find_best_mlx_model(preferred: List[str], deep: bool = False) -> Optional[Path]:
    """Find the best available MLX model from preferred list.

    Skips models that are incompatible with the installed mlx_lm. When
    ``deep`` is True, a real load-probe is run so models that only load in
    Osaurus's mlx-swift runtime (not stock mlx_lm) are skipped fast.
    """
    for name in preferred:
        model = find_mlx_model(name)
        if model and _check_mlx_model_compatible(model, deep=deep):
            return model
    return None


def find_any_working_mlx_model(deep: bool = False) -> Optional[Path]:
    """Find any MLX model that's compatible with the installed mlx_lm.

    When ``deep`` is True, each candidate is load-probed and the first one that
    actually loads is returned (unloadable Osaurus-format models are skipped).
    """
    if not MLX_MODELS_DIR.exists():
        return None
    for item in MLX_MODELS_DIR.iterdir():
        if not item.is_dir():
            continue
        for model_dir in [item] + [sub for sub in item.iterdir() if sub.is_dir()]:
            if _check_mlx_model_compatible(model_dir, deep=deep):
                return model_dir
    return None


def find_text_mlx_model(preferred: List[str] = None) -> Optional[Path]:
    """Find best text generation MLX model."""
    if preferred is None:
        preferred = ["Qwopus", "Qwen3.6", "gemma-4", "MiniMax"]
    found = find_best_mlx_model(preferred)
    if found:
        return found
    return find_any_working_mlx_model()


def get_mlx_context_length(model_path: Path) -> int:
    """Get context length from MLX model's config.json."""
    config_file = model_path / "config.json"
    if not config_file.exists():
        return DEFAULT_CTX_LENGTH


    with open(config_file) as f:
        config = json.load(f)

    return config.get("context_length", config.get("max_position_embeddings", DEFAULT_CTX_LENGTH))


def list_mlx_models(mlx_dir: Path = MLX_MODELS_DIR) -> List[str]:
    """List all available MLX models."""
    if not mlx_dir.exists():
        return []

    models = []
    for item in mlx_dir.iterdir():
        if item.is_dir():
            if (item / "config.json").exists():
                models.append(item.name)
            else:
                for sub in item.iterdir():
                    if sub.is_dir() and (sub / "config.json").exists():
                        models.append(f"{item.name}/{sub.name}")

    return models


def normalize_mlx_model_name(mlx_model: str) -> str:
    """Extract base model name from MLX model ID for matching.

    E.g.: "SomeOrg/SomeModel-mxfp4" -> "somemodel-mxfp4"
    """
    # Remove prefix like "OsaurusAI/" or "mlx-community/"
    if "/" in mlx_model:
        name = mlx_model.split("/")[-1]
    else:
        name = mlx_model
    return name.lower()


# ==========================================================
# MLX MODEL EXECUTION
# ==========================================================


def call_mlx(model_path: Path, prompt: str) -> Optional[str]:
    """Call MLX model for text generation using mlx_lm.generate."""
    global _LAST_MLX_ERROR
    _LAST_MLX_ERROR = None
    if not model_path.exists():
        _LAST_MLX_ERROR = f"model path does not exist: {model_path}"
        logger.warning(_LAST_MLX_ERROR)
        return None

    logger.debug(f"Calling MLX model at {model_path}")

    import tempfile
    import uuid

    model_path_str = str(model_path)
    model_parent = str(model_path.parent)

    # Use temp directory for debugging (no hardcoded paths)

    debug_dir = Path(os.environ.get("MLX_DEBUG_DIR", tempfile.gettempdir())) / "mlx_debug"
    debug_dir.mkdir(exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    prompt_file = str(debug_dir / f"prompt_{uid}.txt")
    script_path = str(debug_dir / f"script_{uid}.py")

    # Escape quotes and backslashes for safe interpolation in python strings
    model_parent_escaped = model_parent.replace("\\", "\\\\").replace('"', '\\"')
    model_path_escaped = model_path_str.replace("\\", "\\\\").replace('"', '\\"')
    prompt_file_escaped = prompt_file.replace("\\", "\\\\").replace('"', '\\"')

    with open(prompt_file, "w") as pf:
        pf.write(prompt)

    with open(script_path, "w") as sf:
        sf.write(f'''
import os, sys, json, traceback
os.chdir("{model_parent_escaped}")
try:
    from mlx_lm import load, stream_generate
    model, tokenizer = load("{model_path_escaped}")
except Exception as e:
    print(f"[MLX LOAD ERROR] {{type(e).__name__}}: {{e}}", flush=True)
    sys.exit(1)

with open("{prompt_file_escaped}", "r") as f:
    prompt = f.read()

# Prepend JSON trigger to avoid thinking
prompt = "Output JSON:\\n" + prompt
text_parts = []
try:
    for r in stream_generate(model, tokenizer, prompt, max_tokens={MLX_MAX_TOKENS}):
        if hasattr(r, "text"):
            text_parts.append(r.text)
        elif isinstance(r, str):
            text_parts.append(r)
except Exception as e:
    print(f"[MLX GENERATE ERROR] {{type(e).__name__}}: {{e}}", flush=True)
    sys.exit(1)

response = "".join(text_parts)
# Strip the trigger from response
if response.startswith("Output JSON:\\n"):
    response = response[13:]
print(response, flush=True)
''')

    try:
        try:
            cmd = UV_RUN + [script_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=MLX_GEN_TIMEOUT,
            )
            stdout = result.stdout
            stderr = result.stderr

            if result.returncode == 0 and stdout.strip():
                return stdout.strip()
            elif stdout.strip() and not stdout.strip().startswith("[MLX"):
                return stdout.strip()
            elif stdout.strip():
                error_msg = stdout.strip()
                _LAST_MLX_ERROR = error_msg
                logger.warning(f"MLX model error: {error_msg}")
                return None
            else:
                _LAST_MLX_ERROR = (
                    f"generate failed (rc={result.returncode}): "
                    f"{stderr[:300] if stderr else 'no output'}"
                )
                logger.warning(f"MLX {_LAST_MLX_ERROR}")
        except subprocess.TimeoutExpired:
            _LAST_MLX_ERROR = f"generate timed out after {MLX_GEN_TIMEOUT}s"
            logger.warning(f"MLX {_LAST_MLX_ERROR} for {model_path.name}")
        except Exception as e:
            _LAST_MLX_ERROR = f"generate failed: {type(e).__name__}: {e}"
            logger.error(f"MLX {_LAST_MLX_ERROR}")

        # Fallback: main.py
        main_py = model_path / "main.py"
        if main_py.exists():
            logger.debug("Trying fallback main.py")
            try:
                result = subprocess.run(
                    ["python3", str(main_py), prompt],
                    capture_output=True,
                    text=True,
                    timeout=MLX_FALLBACK_TIMEOUT,
                )
                if result.returncode == 0:
                    logger.info(f"Fallback successful, got {len(result.stdout)} chars")
                    return result.stdout.strip()
            except Exception as e:
                logger.debug(f"Fallback failed: {e}")

    finally:
        for f in [prompt_file, script_path]:
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    logger.debug(f"All MLX attempts failed for {model_path.name}")
    return None



# ==========================================================
# OUTPUT PROCESSING
# ==========================================================


def process_mlx_content(content: str) -> str:
    """Process MLX output: remove thinking, extract content.

    Uses shared content_processing utilities for consistency with osaurus_lib.
    """
    if not content:
        return ""

    # Try to extract from code blocks first
    extracted = extract_content_from_code_blocks(content)
    if extracted:
        content = extracted

    # Clean all artifacts
    content = clean_model_output(content)

    return content.strip()


# ==========================================================
# UNIFIED API (matches osaurus_lib interface)
# ==========================================================


def call(
    model: str,
    messages: List[Dict[str, Any]],
    host: str = "localhost",
    port: int = MLX_DEFAULT_PORT,
    temperature: float = MLX_DEFAULT_TEMPERATURE,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    task: str = "think",
    parse_json: bool = False,
) -> dict:
    """Call MLX model. Returns dict with content, parsed, time, error.

    This is a pure transport/parsing layer. Validation and retry logic
    should be handled by the caller (e.g. model_eval.py).
    """
    import time

    from .osaurus_lib import apply_model_quirks, extract_json

    # Apply model-specific quirks
    messages = apply_model_quirks(messages, model)

    result = {
        "model": model,
        "time": None,
        "content": None,
        "parsed": None,
        "error": None,
    }

    model_name_for_lookup = model.split("/")[-1] if "/" in model else model
    logger.debug(f"MLX lookup: original={model}, lookup_name={model_name_for_lookup}")
    model_path = find_text_mlx_model([model_name_for_lookup]) or find_mlx_model(
        model_name_for_lookup
    )
    if not model_path:
        result["error"] = f"Model not found: {model}"
        logger.error(f"MLX model not found: {model_name_for_lookup}")
        return result
    logger.debug(f"MLX model found: {model_path}")

    # Format messages to prompt
    system_prompt = "\n".join(m["content"] for m in messages if m["role"] == "system")
    user_prompt = "\n".join(m["content"] for m in messages if m["role"] == "user")
    prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
    logger.debug(f"MLX prompt length: {len(prompt)}")

    start = time.time()
    try:
        logger.debug(f"Calling MLX model: {model_path}")
        content = call_mlx(model_path, prompt)
        logger.debug(f"MLX raw response length: {len(content) if content else 0}")
        result["time"] = round(time.time() - start, 1)

        if content:
            result["content"] = process_mlx_content(content)

            # For JSON tasks: extract and parse JSON from raw content
            if parse_json:
                result["parsed"] = extract_json(content)
                if result["parsed"]:
                    logger.debug("JSON parsed successfully")
                else:
                    logger.warning("Could not parse JSON from output")
        else:
            result["error"] = "Empty response from model"
    except Exception as e:
        result["error"] = f"Error: {type(e).__name__}: {e}"

    return result
