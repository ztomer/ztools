#!/usr/bin/env python3
"""
MLX Library - Utilities for running MLX models directly.
Parallel to osaurus_lib for server-based LLM calls.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Optional, List

# MLX models directory
MLX_MODELS_DIR = Path(os.environ.get("MLX_MODELS_DIR", Path.home() / "MLXModels"))


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
        return None

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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(model_path.parent),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    # Fallback: main.py
    main_py = model_path / "main.py"
    if main_py.exists():
        try:
            result = subprocess.run(
                ["python3", str(main_py), prompt],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

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
            return result.stdout.strip()
    except Exception:
        pass

    return None


# Aliases
call_mlx_text = call_mlx
call_mlx_vlm = run_mlx_vlm


# ==========================================================
# OUTPUT PROCESSING
# ==========================================================


def process_mlx_content(content: str) -> str:
    """Process MLX output: remove thinking, extract content."""
    if not content:
        return ""

    # Remove thinking blocks
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"<\|.*?\|>", "", content)

    # Extract from code blocks
    code_blocks = re.findall(r"```(?:\w+)?\s*(.*?)```", content, re.DOTALL)
    if code_blocks:
        content = code_blocks[-1]

    return content.strip()
