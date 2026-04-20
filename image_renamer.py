#!/usr/bin/env python3
# /// script
# dependencies = ["pillow", "pytesseract", "requests", "mlx-lm", "mlx-vlm"]
# ///
"""
Rename images based on their text content.
Reads the first line of text from each image and uses it as the filename.
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image
import pytesseract
import requests

# Use consolidated functions
from lib.osaurus_lib import (
    get_available_models,
    check_llm_availability,
    select_best_model,
    select_best_vlm_model,
    call_llm_api as _call_api,
    strip_thinking,
)
from lib.mlx_lib import (
    find_best_mlx_model,
    find_mlx_model,
    process_mlx_content,
)

# Point pytesseract at Homebrew's tesseract binary if not on PATH
_TESSERACT_BREW = "/opt/homebrew/bin/tesseract"
if Path(_TESSERACT_BREW).exists():
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_BREW

# --- Constants & Prompts ---

PROMPT_TEXT_TO_FILENAME = (
    "You are a file naming assistant. Read the following text extracted from an image "
    "and suggest a short, descriptive filename (without extension). "
    "Use lowercase, underscores for spaces, and no special characters other than hyphens/underscores. "
    "Keep it under 50 characters. "
    "Output ONLY the final filename string, nothing else.\\n\\n"
    "TEXT:\\n{text}"
)

PROMPT_IMAGE_TO_FILENAME = (
    "You are a file naming assistant. Look at this image and suggest a short, descriptive filename (without extension). "
    "Use lowercase, underscores for spaces, and no special characters other than hyphens/underscores. "
    "Keep it under 50 characters. "
    "Output ONLY the final filename string, nothing else."
)


def get_filename_prompt(for_image: bool = False, model: str = None) -> str:
    """Get model-specific filename prompt from config."""
    from lib.config import get_model_prompt

    prompt = get_model_prompt(model, "filename") if model else ""
    if prompt:
        return prompt

    # Fallback
    return PROMPT_IMAGE_TO_FILENAME if for_image else PROMPT_TEXT_TO_FILENAME

VLM_PREFERRED_MODELS = [
    "qwen3.5-27b-claude-4.6-opus-distilled-mlx-4bit",  # Best: 100%
    "gemma-4-e2b-it-4bit",  # Fast
    "foundation",
]

TEXT_PREFERRED_MODELS = [
    "qwen3.5-27b-claude-4.6-opus-distilled-mlx-4bit",  # Best: 100%
    "gemma-4-e2b-it-8bit",  # Fast fallback
    "foundation",
]

MLX_MODELS_DIR = Path.home() / "MLXModels" / "mlx-community"


# --- Helper Functions ---


def clean_filename(text: str, max_length: int = 50) -> str:
    """
    Clean text to create a valid filename.
    """
    # Remove special characters and replace spaces with underscores
    cleaned = re.sub(r"[^\w\s-]", "", text)
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    cleaned = cleaned.strip("_").lower()

    # Truncate to max length
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("_")

    return cleaned if cleaned else "unnamed"


def extract_first_line(image_path: Path) -> Optional[str]:
    """
    Extract the first line of text from an image using OCR.
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        lines = [line.strip() for line in text.split("\\n") if line.strip()]
        if lines:
            return lines[0]
        return None
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return None


def extract_full_text(image_path: Path) -> Optional[str]:
    """
    Extract the full text from an image using OCR.
    """
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else None
    except Exception as e:
        print(f"Error extracting full text from {image_path.name}: {e}")
        return None


# Use consolidated functions from lib.osaurus_lib and mlx_lib


def process_llm_content(content: str) -> Optional[str]:
    """
    Process raw LLM output - delegates to consolidated functions.
    """

    # Try both processing functions
    content = strip_thinking(content)
    content = process_mlx_content(content)

    return content if content else None


def parse_response_content(
    response: requests.Response, endpoint_type: str = "chat"
) -> str:
    """
    Helper to parse standard JSON or NDJSON response from Ollama/OpenAI.
    """
    try:
        # Try standard JSON first
        result = response.json()
        if endpoint_type == "chat":
            # Support both Ollama 'message' and OpenAI 'choices'
            if "choices" in result and result["choices"]:
                return result["choices"][0].get("message", {}).get("content", "")
            return result.get("message", {}).get("content", "")
        else:  # generate
            return result.get("response", "")

    except ValueError:
        # Handle streaming/NDJSON (common with Ollama even when stream=False sometimes)
        full_content = ""
        for line in response.text.strip().split("\\n"):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if endpoint_type == "chat":
                    # OpenAI stream format
                    if "choices" in obj and obj["choices"]:
                        full_content += (
                            obj["choices"][0].get("delta", {}).get("content", "")
                        )
                    elif "message" in obj:  # Ollama stream format
                        full_content += obj["message"].get("content", "")
                else:  # generate
                    if "response" in obj:
                        full_content += obj.get("response", "")
            except Exception:
                pass
        return full_content


def call_llm_api(
    host: str,
    model: str,
    prompt: str,
    images: Optional[List[str]] = None,
    timeout: int = 60,
    api_key: str = "",
) -> Optional[str]:
    """
    Generic function to call LLM/VLM APIs (Chat or Generate).
    Uses consolidated functions from lib.osaurus_lib.
    """
    # Try server API first

    messages = [{"role": "user", "content": prompt}]
    if images:
        messages[0]["images"] = images

    result = _call_api(
        f"http://{host}", model, messages, api_key=api_key, timeout=timeout
    )
    if result and "content" in result:
        return result["content"]

    # Fall back to mlx_lib
    from lib.mlx_lib import call_mlx

    model_path = find_mlx_model(model)
    if model_path and not images:
        raw = call_mlx(model_path, prompt)
        if raw:
            return process_mlx_content(raw)

    return None


def query_llm_for_filename(
    text: str, host: str, model: str = "foundation", api_key: str = ""
) -> Optional[str]:
    """Query the LLM to generate a filename based on the OCR text."""
    from lib.config import get_model_prompt

    # Try config first
    prompt_template = get_model_prompt(model, "filename")
    if prompt_template:
        prompt = prompt_template.format(text=text) if "{text}" in prompt_template else prompt_template
    else:
        prompt = PROMPT_TEXT_TO_FILENAME.format(text=text)

    messages = [{"role": "user", "content": prompt}]
    from lib.osaurus_lib import call_llm_api as _call
    result = _call(host, model, messages, timeout=30, api_key=api_key)
    return result.get("content") if result else None


def query_vlm_for_filename(
    image_path: Path, host: str, model: str, api_key: str = ""
) -> Optional[str]:
    """Query a Vision Language Model to describe the image."""
    from lib.config import get_model_prompt

    # Try config first
    prompt = get_model_prompt(model, "filename")
    if not prompt:
        prompt = PROMPT_IMAGE_TO_FILENAME

    try:
        with open(image_path, "rb") as f:
            import base64

            images = [base64.b64encode(f.read()).decode("utf-8")]
        return call_llm_api(
            host,
            model,
            prompt,
            images=images,
            timeout=60,
            api_key=api_key,
        )
    except Exception as e:
        print(f"VLM error: {e}")
        return None


def test_llm_connection(host: str, model: str, api_key: str = ""):
    """
    Run diagnostic tests on the LLM connection.
    """
    print(f"\n--- LLM Connection Diagnosis ({host}) ---")

    print("\n1. Connectivity Check:")
    if check_llm_availability(host, api_key=api_key):
        print("   [OK] Server is reachable.")
    else:
        print("   [FAIL] Server unreachable.")
        return

    print("\n2. Available Models:")
    models = get_available_models(host, api_key=api_key)
    if models:
        print(f"   [OK] Found {len(models)} models: {', '.join(models[:5])}...")
    else:
        print("   [WARN] Could not list models.")

    print(f"\n3. Inference Test (Model: {model}):")
    result = query_llm_for_filename(
        "test image containing the word hello", host, model, api_key=api_key
    )
    if result:
        print(f"   [OK] Success! Response: {result}")
    else:
        print("   [FAIL] No response generated.")


def get_default_llm_model():
    """Get best model from eval or fallback."""
    import json as json_module

    eval_path = Path.home() / ".config" / "model_eval.json"
    if eval_path.exists():
        try:
            data = json_module.loads(eval_path.read_text())
            best = None
            best_score = -1
            for r in data.get("results", []):
                passed = sum(1 for x in r.get("results", []) if x.get("status") == "ok")
                score = passed / len(r.get("results", [])) if r.get("results") else 0
                if score > best_score:
                    best_score = score
                    best = r.get("model")
            if best and best_score > 0:
                return best
        except Exception:
            pass
    return os.environ.get("OLLAMA_MODEL", TEXT_PREFERRED_MODELS[0])



def rename_image(
    image_path: Path,
    dry_run: bool,
    force: bool,
    llm_host: Optional[str],
    llm_model: Optional[str],
    vlm_model: Optional[str],
    api_key: str,
    mlx_model_path: Optional[Path],
    mlx_vlm_path: Optional[Path],
) -> Tuple[bool, str]:
    if not image_path.exists():
        return False, f"File not found: {image_path.name}"
        
    text = extract_full_text(image_path) or extract_first_line(image_path)
    if not text:
        return False, f"Skipped (No text): {image_path.name}"
        
    new_name = None
    if llm_host and llm_model:
        new_name = query_llm_for_filename(text, llm_host, llm_model, api_key)
    elif mlx_model_path:
        new_name = query_llm_for_filename(text, "localhost", mlx_model_path.name, api_key)
        
    if not new_name:
        new_name = clean_filename(text)
        
    if not new_name:
        return False, f"Could not generate name: {image_path.name}"
        
    new_path = image_path.with_name(f"{new_name}{image_path.suffix}")
    
    if new_path.exists() and not force:
        return False, f"Skipped (Exists): {new_path.name}"
        
    if not dry_run:
        try:
            image_path.rename(new_path)
        except Exception as e:
            return False, f"Error renaming: {e}"
            
    return True, f"Renamed: {image_path.name} -> {new_path.name}"

def main():
    parser = argparse.ArgumentParser(
        description="Rename images based on their text content (LLM or first line)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rename images in current directory
  %(prog)s

  # Test LLM connection
  %(prog)s --test

  # Specify custom LLM host/port
  %(prog)s --llm-host http://localhost:11434
        """,
    )

    parser.add_argument(
        "directory", nargs="?", default=".", help="Directory containing images"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show changes without renaming"
    )
    parser.add_argument("--force", "-f", action="store_true", help="Rename all images")
    parser.add_argument("--pattern", "-p", default="*", help="File pattern")
    parser.add_argument(
        "--max-length", "-m", type=int, default=50, help="Max filename length"
    )
    parser.add_argument(
        "--llm-host", default="http://localhost:1337", help="LLM server URL"
    )
    parser.add_argument(
        "--llm-model", default=get_default_llm_model(), help="LLM model"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OLLAMA_API_KEY", ""),
        help="Bearer token for LLM API",
    )
    parser.add_argument("--test", action="store_true", help="Test connection")

    args = parser.parse_args()

    if args.test:
        test_llm_connection(args.llm_host, args.llm_model, api_key=args.api_key)
        sys.exit(0)

    directory = Path(args.directory)
    if not directory.exists() or not directory.is_dir():
        print(f"Error: Invalid directory '{directory}'")
        sys.exit(1)

    # Setup LLM/VLM
    active_llm_host = None
    active_model = args.llm_model
    active_vlm_model = None
    mlx_model_path = None
    mlx_vlm_path = None

    print(f"Checking LLM at {args.llm_host}...")
    if check_llm_availability(args.llm_host, api_key=args.api_key):
        active_llm_host = args.llm_host
        print("LLM Server found!")

        available_models = get_available_models(active_llm_host, api_key=args.api_key)
        if available_models:
            # Text Model Selection
            if (
                active_model not in available_models
                and f"{active_model}:latest" not in available_models
            ):
                best_model = select_best_model(available_models)
                if best_model:
                    print(
                        f"Requested text model '{active_model}' not found. Auto-selected: {best_model}"
                    )
                    active_model = best_model
            else:
                print(f"Using text model: {active_model}")

            # VLM Selection
            active_vlm_model = select_best_vlm_model(available_models)
            if active_vlm_model:
                print(f"VLM Support enabled using: {active_vlm_model}")
            else:
                print(
                    "No Vision Language Model (VLM) found. Visual renaming will be disabled."
                )
                print(
                    "Recommendation: Install 'llama3.2-vision' or 'minicpm-v' via Ollama."
                )
        else:
            print("Could not fetch model list. Proceeding with default.")
    else:
        print("LLM Server not found. Looking for local MLX models...")
        mlx_model_path = find_best_mlx_model(TEXT_PREFERRED_MODELS)
        mlx_vlm_path = find_best_mlx_model(VLM_PREFERRED_MODELS)
        if mlx_model_path:
            print(f"Using local MLX text model: {mlx_model_path.name}")
        if mlx_vlm_path:
            print(f"Using local MLX vision model: {mlx_vlm_path.name}")
        if not mlx_model_path and not mlx_vlm_path:
            print(
                "No local MLX models found. Falling back to simple first-line extraction."
            )

    # Find Images
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    image_files = []

    # Simple glob handling
    
    # If user provided specific pattern like "*.jpg", use it. If "*", use all extensions.

    # Actually, previous logic was: if pattern is "*", check all ext. If pattern is specific, check it.
    if args.pattern == "*":
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
    else:
        image_files.extend(directory.glob(args.pattern))

    # De-duplicate and filter
    image_files = list(
        set([f for f in image_files if f.suffix.lower() in image_extensions])
    )

    if not image_files:
        print(f"No images found in '{directory}' matching pattern '{args.pattern}'")
        sys.exit(0)

    print(f"Found {len(image_files)} image(s)")
    if args.dry_run:
        print("DRY RUN MODE - No files will be renamed\n")
    print()

    # Process
    stats = {"renamed": 0, "skipped": 0, "errors": 0}

    for image_path in sorted(image_files):
        success, message = rename_image(
            image_path,
            dry_run=args.dry_run,
            force=args.force,
            llm_host=active_llm_host,
            llm_model=active_model,
            vlm_model=active_vlm_model,
            api_key=args.api_key,
            mlx_model_path=mlx_model_path,
            mlx_vlm_path=mlx_vlm_path,
        )
        print(message)

        if success:
            stats["renamed"] += 1
        elif "Skipped" in message:
            stats["skipped"] += 1
        else:
            stats["errors"] += 1

    print(f"\n{'=' * 60}")
    print(
        f"Summary: {stats['renamed']} renamed, {stats['skipped']} skipped, {stats['errors']} errors"
    )
    if args.dry_run and stats["renamed"] > 0:
        print("\nRun without --dry-run to actually rename the files")


if __name__ == "__main__":
    main()
