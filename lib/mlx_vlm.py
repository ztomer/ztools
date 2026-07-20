#!/usr/bin/env python3
"""
MLX-VLM support — loading Osaurus-installed multimodal/MoE models on-device.

The Osaurus-installed models (Gemma4 `gemma4`/`gemma4_unified`, Qwen
`qwen3_5_moe`, MXFP8/4) are standard HF checkpoints, but stock `mlx_lm`
(text-only) cannot construct their vision/audio towers or MoE heads. `mlx-vlm`
from git main loads them (PyPI `mlx-vlm` 0.6.4 predates PR #1523, the audio-conv
weight-layout fix these checkpoints ship in). We invoke it via the uv git+https
URL so the Python MLX route is a real last resort, not a dead end.

Split out of lib/mlx_lib.py to keep both modules under the 500-line limit.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Optional

import lib.mlx_lib as mlx_lib

from .logging_config import mlx_logger as logger
from .mlx_lib import (
    _LOAD_PROBE_CACHE,
    MLX_GEN_TIMEOUT,
    MLX_LOAD_PROBE_TIMEOUT,
    MLX_MAX_TOKENS,
    MLX_VLM_TIMEOUT,
    _looks_like_mlx_model,
    find_mlx_model,
)

# Osaurus-installed models (Gemma4/qwen3_5_moe, MXFP8/4) are standard HF
# checkpoints, but stock `mlx_lm` (text-only) cannot build their vision/audio
# towers or MoE heads. `mlx-vlm` from git main loads them — PyPI `mlx-vlm`
# (0.6.4) predates PR #1523 which fixes the audio-conv weight layout these
# checkpoints ship in. Floating HEAD per user decision.
MLX_VLM_GIT = os.environ.get(
    "MLX_VLM_GIT", "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git"
)
UV_RUN_VLM = ["rtk", "uv", "run", "--with", "mlx", "--with", MLX_VLM_GIT]


def probe_mlx_vlm_loadable(model_path: Path) -> tuple:
    """Deep-probe whether `mlx-vlm` (git main) can LOAD this model.

    Returns (loadable: bool, reason: str). Catches architecture/quant mismatches
    (e.g. an unsupported model_type) so the fallback skips a model fast instead
    of spending a full generation timeout on it. Results cached per path.
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
        "    from mlx_vlm import load\n"
        f'    load("{model_path_escaped}")\n'
        '    print("VLM_LOAD_OK", flush=True)\n'
        "except Exception as e:\n"
        '    print(f"VLM_LOAD_FAIL {type(e).__name__}: {e}", flush=True)\n'
        "    sys.exit(1)\n"
    )
    try:
        result = subprocess.run(
            UV_RUN_VLM + ["python3", "-c", probe],
            capture_output=True,
            text=True,
            timeout=MLX_LOAD_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        outcome = (False, f"vlm load timed out after {MLX_LOAD_PROBE_TIMEOUT}s")
        _LOAD_PROBE_CACHE[key] = outcome
        return outcome
    except Exception as e:
        outcome = (False, f"vlm probe error: {type(e).__name__}: {e}")
        _LOAD_PROBE_CACHE[key] = outcome
        return outcome

    out = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0 and "VLM_LOAD_OK" in out:
        outcome = (True, "ok")
    else:
        reason = "unknown vlm load failure"
        for line in out.splitlines():
            if line.startswith("VLM_LOAD_FAIL"):
                reason = line[len("VLM_LOAD_FAIL ") :].strip()
                break
        outcome = (False, reason)
    _LOAD_PROBE_CACHE[key] = outcome
    return outcome


def find_best_mlx_vlm_model(preferred: List[str], deep: bool = True) -> Optional[Path]:
    """Find the best MLX-VLM-loadable model from preferred list.

    When ``deep`` is True (default), each candidate is load-probed via
    `mlx-vlm` so unloadable model_types are skipped fast.
    """
    for name in preferred:
        model = find_mlx_model(name)
        if model and (not deep or probe_mlx_vlm_loadable(model)[0]):
            return model
    return None


def find_any_working_mlx_vlm_model(deep: bool = True) -> Optional[Path]:
    """Find any model `mlx-vlm` can load under MLX_MODELS_DIR."""
    from .mlx_lib import MLX_MODELS_DIR

    if not MLX_MODELS_DIR.exists():
        return None
    for item in MLX_MODELS_DIR.iterdir():
        if not item.is_dir():
            continue
        for model_dir in [item] + [sub for sub in item.iterdir() if sub.is_dir()]:
            if _looks_like_mlx_model(model_dir) and (
                not deep or probe_mlx_vlm_loadable(model_dir)[0]
            ):
                return model_dir
    return None


def call_mlx_vlm(model_path: Path, prompt: str) -> Optional[str]:
    """Generate text from a multimodal/MoE model via `mlx-vlm` (git main).

    Applies the instruct chat template (matching how Osaurus would format the
    prompt), prepends the JSON trigger to suppress thinking, then returns the
    generated text.     On load or generation failure, sets _LAST_MLX_ERROR and
    returns None.
    """
    mlx_lib._LAST_MLX_ERROR = None
    if not model_path.exists():
        mlx_lib._LAST_MLX_ERROR = f"model path does not exist: {model_path}"
        logger.warning(mlx_lib._LAST_MLX_ERROR)
        return None

    logger.debug(f"Calling MLX-VLM model at {model_path}")

    import tempfile
    import uuid

    model_path_str = str(model_path)
    model_path_escaped = model_path_str.replace("\\", "\\\\").replace('"', '\\"')
    debug_dir = Path(os.environ.get("MLX_DEBUG_DIR", tempfile.gettempdir())) / "mlx_debug"
    debug_dir.mkdir(exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    prompt_file = str(debug_dir / f"vlm_prompt_{uid}.txt")
    script_path = str(debug_dir / f"vlm_script_{uid}.py")

    with open(prompt_file, "w") as pf:
        pf.write(prompt)

    prompt_file_escaped = prompt_file.replace("\\", "\\\\").replace('"', '\\"')

    with open(script_path, "w") as sf:
        sf.write(f'''
import os, sys, json, traceback
os.chdir(os.path.dirname("{model_path_escaped}"))
try:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, processor = load("{model_path_escaped}")
except Exception as e:
    print(f"[VLM LOAD ERROR] {{type(e).__name__}}: {{e}}", flush=True)
    sys.exit(1)

with open("{prompt_file_escaped}", "r") as f:
    user_prompt = f.read()

# Instruct checkpoints need the chat template; prepend JSON trigger to avoid thinking.
try:
    formatted = apply_chat_template(processor, model.config, "Output JSON:\\n" + user_prompt)
except Exception:
    formatted = "Output JSON:\\n" + user_prompt

text_parts = []
try:
    result = generate(
        model, processor, prompt=formatted, max_tokens={MLX_MAX_TOKENS}, verbose=False
    )
    text = result.text if hasattr(result, "text") else str(result)
    if text.startswith("Output JSON:\\n"):
        text = text[13:]
    print(text, flush=True)
except Exception as e:
    print(f"[VLM GENERATE ERROR] {{type(e).__name__}}: {{e}}", flush=True)
    sys.exit(1)
''')

    try:
        result = subprocess.run(
            UV_RUN_VLM + [script_path],
            capture_output=True,
            text=True,
            timeout=MLX_GEN_TIMEOUT,
        )
        stdout = result.stdout
        stderr = result.stderr

        if result.returncode == 0 and stdout.strip():
            return stdout.strip()
        elif stdout.strip() and not stdout.strip().startswith("[VLM"):
            return stdout.strip()
        elif stdout.strip():
            error_msg = stdout.strip()
            mlx_lib._LAST_MLX_ERROR = error_msg
            logger.warning(f"MLX-VLM model error: {error_msg}")
        else:
            mlx_lib._LAST_MLX_ERROR = (
                f"vlm generate failed (rc={result.returncode}): "
                f"{stderr[:300] if stderr else 'no output'}"
            )
            logger.warning(f"MLX-VLM {mlx_lib._LAST_MLX_ERROR}")
    except subprocess.TimeoutExpired:
        mlx_lib._LAST_MLX_ERROR = f"vlm generate timed out after {MLX_GEN_TIMEOUT}s"
        logger.warning(f"MLX-VLM {mlx_lib._LAST_MLX_ERROR} for {model_path.name}")
    except Exception as e:
        mlx_lib._LAST_MLX_ERROR = f"vlm generate failed: {type(e).__name__}: {e}"
        logger.error(f"MLX-VLM {mlx_lib._LAST_MLX_ERROR}")
    finally:
        for f in [prompt_file, script_path]:
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    return None


def run_mlx_vlm(model_path: Path, image_path: Path) -> Optional[str]:
    """Call MLX VLM for image analysis (via the mxrc.vl CLI)."""
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
            timeout=MLX_VLM_TIMEOUT,
        )
        if result.returncode == 0:
            logger.info(f"VLM call successful, got {len(result.stdout)} chars")
            return result.stdout.strip()
        else:
            logger.warning(f"VLM command failed with return code {result.returncode}")
    except subprocess.TimeoutExpired:
        logger.error("VLM call timed out after 180s")
    except Exception as e:
        logger.error(f"VLM call failed: {type(e).__name__}: {e}")

    return None
