#!/usr/bin/env python3
"""
MLX Library - Utilities for running MLX models directly.
Parallel to osaurus_lib for server-based LLM calls.
"""

import os
import re
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

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
        if item.is_dir() and model_name.lower() in item.name.lower():
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

    if not MLX_MODELS_DIR.exists():
        return None

    for keyword in preferred:
        for item in MLX_MODELS_DIR.iterdir():
            if item.is_dir() and keyword.lower() in item.name.lower():
                return item
            for sub in item.iterdir():
                if sub.is_dir() and keyword.lower() in sub.name.lower():
                    return sub

    return None


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
            models.append(item.name)
            for sub in item.iterdir():
                if sub.is_dir():
                    models.append(f"{item.name}/{sub.name}")

    return models


# ==========================================================
# MLX MODEL EXECUTION
# ==========================================================


def call_mlx(model_path: Path, prompt: str) -> Optional[str]:
    """Call MLX model for text generation."""
    if not model_path.exists():
        logger.warning(f"Model path does not exist: {model_path}")
        return None

    logger.debug(f"Calling MLX model at {model_path}")
    cmd = [
        "python3",
        "-m",
        "mlx.lm",
        "text",
        "--model",
        str(model_path),
        "--prompt",
        prompt,
    ]

    try:
        logger.debug(f"Running command: {' '.join(cmd[:6])}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(model_path.parent),
        )
        if result.returncode == 0:
            logger.info(f"MLX call successful, got {len(result.stdout)} chars")
            return result.stdout.strip()
        else:
            logger.warning(
                f"MLX command failed with return code {result.returncode}")
    except subprocess.TimeoutExpired:
        logger.error("MLX call timed out after 120s")
    except Exception as e:
        logger.debug(f"MLX call failed: {e}")

    # Fallback: main.py
    main_py = model_path / "main.py"
    if main_py.exists():
        logger.debug("Trying fallback main.py")
        try:
            result = subprocess.run(
                ["python3", str(main_py), prompt],
                capture_output=True,
                text=True,
                timeout=120,
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
    temperature: float = 0.1,
    max_tokens: int = 2000,
    timeout: int = 120,
) -> dict:
    """
    Unified MLX API - matches osaurus_lib.call() interface.
    Returns dict with content, quality_score, time, error.
    """
    import json
    import time

    result = {
        "model": model,
        "time": None,
        "content": None,
        "quality_score": 0,
        "error": None,
    }

    # Find the model
    model_path = find_text_mlx_model([model]) or find_mlx_model(model)
    if not model_path:
        result["error"] = f"Model not found: {model}"
        return result

    # Extract prompt from messages
    prompt = messages[-1].get("content", "") if messages else ""

    start = time.time()
    try:
        content = call_mlx(model_path, prompt)
        result["time"] = round(time.time() - start, 1)

        if content:
            result["content"] = process_mlx_content(content)
        else:
            result["error"] = "Empty response from model"
    except Exception as e:
        result["error"] = f"Error: {type(e).__name__}: {e}"

    return result


# Aliases for better API clarity
call_mlx_text = call_mlx  # Call MLX model for text generation
call_mlx_vlm = run_mlx_vlm  # Call MLX model for vision/image analysis
