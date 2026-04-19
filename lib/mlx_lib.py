#!/usr/bin/env python3
"""
MLX Library - Utilities for running MLX models directly.
Parallel to osaurus_lib for server-based LLM calls.
"""

import os
import re
import subprocess
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Get the venv Python with mlx_lm
# For MLX, use uv run to get the right Python environment
UV_RUN = ["rtk", "uv", "run", "--with", "mlx", "--with", "mlx-lm"]

from .content_processing import clean_model_output, extract_content_from_code_blocks
from .logging_config import mlx_logger as logger

# MLX models directory
MLX_MODELS_DIR = Path(os.environ.get(
    "MLX_MODELS_DIR", Path.home() / "MLXModels"))


# ==========================================================
# MODEL DISCOVERY
# ==========================================================


def find_mlx_model(model_name: str, mlx_dir: Path = MLX_MODELS_DIR) -> Optional[Path]:
    """Find an MLX model by name in the models directory."""
    if not mlx_dir.exists():
        return None

    for item in mlx_dir.iterdir():
        if not item.is_dir():
            continue
        if model_name.lower() in item.name.lower():
            return item
        # Check subdirs
        for sub in item.iterdir():
            if sub.is_dir() and model_name.lower() in sub.name.lower():
                return sub

    return None


def find_best_mlx_model(preferred: List[str]) -> Optional[Path]:
    """Find the best available MLX model from preferred list."""
    for name in preferred:
        model = find_mlx_model(name)
        if model:
            return model
    return None


def find_text_mlx_model(preferred: List[str] = None) -> Optional[Path]:
    """Find best text generation MLX model."""
    if preferred is None:
        preferred = ["qwen", "llama", "phi", "mistral", "gemma", "airoboros"]
    return find_best_mlx_model(preferred)


def get_mlx_context_length(model_path: Path) -> int:
    """Get context length from MLX model's config.json."""
    config_file = model_path / "config.json"
    if not config_file.exists():
        return 4096

    import json

    with open(config_file) as f:
        config = json.load(f)

    return config.get("context_length", config.get("max_position_embeddings", 4096))


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


# ==========================================================
# MLX MODEL EXECUTION
# ==========================================================


def call_mlx(model_path: Path, prompt: str) -> Optional[str]:
    """Call MLX model for text generation using mlx_lm.generate."""
    if not model_path.exists():
        logger.warning(f"Model path does not exist: {model_path}")
        return None

    logger.debug(f"Calling MLX model at {model_path}")
    
    import tempfile
    import base64
    
    model_path_str = str(model_path)
    model_parent = str(model_path.parent)
    
    # Write prompt to separate temp file to avoid escaping
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as pf:
        pf.write(prompt)
        prompt_file = pf.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as sf:
        sf.write(f'''
import os
os.chdir("{model_parent}")
from mlx_lm import load, stream_generate

model, tokenizer = load("{model_path_str}")

with open("{prompt_file}", "r") as f:
    prompt = f.read()

# Prepend JSON trigger to avoid thinking
prompt = "Output JSON:\\n" + prompt
text_parts = []
for r in stream_generate(model, tokenizer, prompt, max_tokens=2048):
    if hasattr(r, "text"):
        text_parts.append(r.text)
    elif isinstance(r, str):
        text_parts.append(r)
response = "".join(text_parts)
# Strip the trigger from response
if response.startswith("Output JSON:\\n"):
    response = response[13:]
print(response, flush=True)
''')
        script_path = sf.name
    
    # Write prompt file
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    
    try:
        result = subprocess.run(
            UV_RUN + [str(VENV_PYTHON), script_path],
            capture_output=True,
            text=True,
            timeout=600,
        )
        stdout = result.stdout
        stderr = result.stderr
        
        # Debug
        stdout_repr = repr(stdout[:100]) if stdout else "EMPTY"
        logger.debug(f"MLX result: rc={result.returncode}, stdout={stdout_repr}")
        
        if result.returncode == 0 and stdout.strip():
            logger.info(f"MLX generate successful, got {len(stdout.strip())} chars")
            return stdout.strip()
        elif stdout.strip():
            logger.info(f"MLX generated despite rc={result.returncode}, got {len(stdout.strip())} chars")
            return stdout.strip()
        else:
            logger.warning(f"MLX failed (rc={result.returncode}): {stderr[:300] if stderr else 'no output'}")
            return None
    except Exception as e:
        logger.error(f"MLX generate failed: {type(e).__name__}: {e}")
        return None
    finally:
        try:
            os.unlink(script_path)
            os.unlink(prompt_file)
        except:
            pass

    # Fallback: main.py
    main_py = model_path / "main.py"
    if main_py.exists():
        logger.debug("Trying fallback main.py")
        try:
            result = subprocess.run(
                ["python3", str(main_py), prompt],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode == 0:
                logger.info(
                    f"Fallback successful, got {len(result.stdout)} chars")
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Fallback failed: {e}")

    logger.error(f"Failed to call MLX model at {model_path}")
    return None


def run_mlx_vlm(model_path: Path, image_path: Path) -> Optional[str]:
    """Call MLX VLM for image analysis."""
    if not model_path.exists() or not image_path.exists():
        return None

    cmd = [
        "python3",
        "-m",
        "mxrc.vl",
        "--model",
        str(model_path),
        "--image",
        str(image_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0:
            logger.info(f"VLM call successful, got {len(result.stdout)} chars")
            return result.stdout.strip()
        else:
            logger.warning(
                f"VLM command failed with return code {result.returncode}")
    except subprocess.TimeoutExpired:
        logger.error("VLM call timed out after 180s")
    except Exception as e:
        logger.error(f"VLM call failed: {type(e).__name__}: {e}")

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
    port: int = 1337,
    temperature: float = 0.1,
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
    from .osaurus_lib import extract_json, apply_model_quirks

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
    model_path = find_text_mlx_model([model_name_for_lookup]) or find_mlx_model(model_name_for_lookup)
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

