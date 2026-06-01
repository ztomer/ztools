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
import base64
from pathlib import Path
from rich.console import Console
import requests
from PIL import Image
import pytesseract

console = Console()
from typing import List, Tuple, Optional
from lib.config import get_filename_models, get_filename_prompt, get_model_prompt, Task

# Use consolidated functions
from lib.osaurus_lib import (
    get_available_models,
    check_llm_availability,
    select_best_model,
    select_best_vlm_model,
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


# --- Server Management ---
def ensure_llm_running() -> bool:
    """Detect crash and restart server if needed."""
    import subprocess
    import time

    if check_llm_availability("http://localhost:1337"):
        return True

    print("[WARN] LLM server not responding, restarting...")

    try:
        subprocess.run(["pkill", "-f", "osaurus"], capture_output=True)
        time.sleep(2)
    except:
        pass

    try:
        subprocess.Popen(["open", "-a", "osaurus"])
        time.sleep(15)
        if check_llm_availability("http://localhost:1337"):
            print("[OK] Server restarted")
            return True
    except:
        pass

    return False


# --- Relevance Check Prompt ---
RELEVANCE_CHECK_PROMPT = """Is this image content useful/interesting enough to keep and rename?
Consider: educational content, useful tips, meaningful information, actionable advice.

Content:
{text}

Answer ONLY one word: "keep" or "skip"."""

# FIX 1: Restrictive VLM prompt
PROMPT_IMAGE_TO_FILENAME = "Describe the visual objects in this image using 3 to 4 descriptive nouns and adjectives (e.g., 'white goose grass'). Ignore any text. Do not use words like 'image', 'empty', 'text', 'file', or 'filename'. Output ONLY the descriptive words."

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
                    except:
                        continue

            if "keep" in content and "skip" not in content:
                return True
            elif "skip" in content:
                return False
        except:
            continue

    return None

FILENAME_MODELS = get_filename_models()

PROMPT_TEXT_TO_FILENAME = get_model_prompt(FILENAME_MODELS[0], Task.FILENAME) if FILENAME_MODELS else ""

MLX_MODELS_DIR = Path.home() / "MLXModels"


# --- Helper Functions ---

def clean_filename(text: str, max_length: int = 50) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text)
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    cleaned = cleaned.strip("_").lower()

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("_")

    return cleaned if cleaned else "unnamed"


def extract_first_line(image_path: Path) -> Optional[str]:
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            return lines[0]
        return None
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return None


def extract_full_text(image_path: Path) -> Optional[str]:
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else None
    except Exception as e:
        print(f"Error extracting full text from {image_path.name}: {e}")
        return None


def is_meaningful_text(text: str, min_word_count: int = 2) -> bool:
    if not text:
        return False

    text = text.strip()
    if not text:
        return False

    words = text.split()
    if len(words) == 1 and len(text) > 8:
        if text.isalnum() and text[:2].isupper():
            return False

    if text.isupper() and len(text) > 4 and " " not in text:
        return False

    word_like = sum(1 for w in words if len(w) > 2 and any(c.isalpha() for c in w))
    return word_like >= min_word_count


def is_non_human_readable(text: str) -> bool:
    if not text:
        return True

    text = text.strip()
    if len(text) < 3:
        return True

    if re.match(r'^HF[A-Za-z0-9]{7,}$', text) or re.match(r'^HH[A-Za-z0-9]{7,}$', text):
        return True

    if text.startswith("@") and "_" not in text and len(text) > 1:
        return True

    if len(text) <= 3 and text.isupper():
        return True

    if " " not in text:
        if text.isupper() and any(c.isdigit() for c in text):
            return True

    return False


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
                    except:
                        continue

            if content and len(content) >= 2:
                content = content.strip().lower()

                # Strip common instruction/explanation prefixes from LLM outputs
                for prefix in ("here is a filename:", "here is the filename:", "here is:",
                               "here's the filename:", "the filename is:", "filename:",
                               "rename to:", "output:"):
                    if content.startswith(prefix):
                        content = content[len(prefix):].strip()
                        break

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
        except:
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
    # FIX 2: Override config prompt
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
            print(f"VLM API Error: {resp.status_code} - {resp.text}")
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

        return content.strip() if content else None

    except Exception as e:
        print(f"VLM error: {e}")
        return None


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
    mlx_mode: bool = False,
) -> Tuple[bool, str]:
    if not image_path.exists():
        return False, f"File not found: {image_path.name}"

    text = extract_full_text(image_path) or extract_first_line(image_path)
    new_name = None

    needs_vlm = False
    if not text:
        needs_vlm = True
    elif is_non_human_readable(text) or not is_meaningful_text(text, min_word_count=2):
        needs_vlm = True

    if needs_vlm:
        if vlm_model and llm_host:
            print(f"   [INFO] No readable text found. Falling back to VLM ({vlm_model})...")
            new_name = query_vlm_for_filename(image_path, llm_host, vlm_model, api_key)
            if new_name:
                new_name = clean_filename(new_name)

                # FIX 3: Aggressive validation gate
                bad_names = ("text", "file", "image", "unnamed", "output", "filename", "none", "empty", "blank")
                if any(bad_word in new_name for bad_word in bad_names) or len(new_name) < 4:
                    print(f"   [WARN] Generic VLM result rejected: {new_name}")
                    new_name = None
        else:
            return False, f"Skipped (No text & no VLM fallback configured): {image_path.name}"
    else:
        if force and llm_host:
            relevant = is_relevant_with_llm(text, llm_host, api_key)
            if relevant is False:
                return False, f"Skipped (Not relevant): {image_path.name}"
            elif relevant is True:
                print(f"   [RELEVANT] {image_path.name}")

        if mlx_mode:
            try:
                new_name = query_mlx_for_filename(text)
            except Exception as e:
                print(f"   [WARN] MLX failed: {e}, using fallback")
                new_name = None
        elif llm_host and llm_model:
            try:
                new_name = query_llm_for_filename(text, llm_host, llm_model, api_key)
            except Exception as e:
                print(f"   [WARN] LLM failed: {e}, using fallback")
                new_name = None

            if new_name:
                GENERIC_NAMES = {"text", "file", "image", "unnamed", "output", "filename",
                                 "none", "screenshot", "document", "note",
                                 "filename_txt", "file_txt", "text_txt",
                                 "output_txt", "note_txt", "document_txt",
                                 "screenshot_png", "image_png", "unnamed_png"}
                if new_name in GENERIC_NAMES:
                    print(f"   [WARN] Generic LLM result: {new_name}, using fallback")
                    new_name = None
                elif len(new_name) < 4:
                    print(f"   [WARN] Too short: {new_name}, using fallback")
                    new_name = None

    if not new_name and text and not needs_vlm:
        new_name = clean_filename(text)

    if new_name in ("text", "file", "image", "unnamed", "output", None):
        if not new_name:
            return False, f"Could not generate name: {image_path.name}"
        return False, f"Skipped (Generic name): {image_path.name}"

    new_path = image_path.with_name(f"{new_name}{image_path.suffix}")

    if not new_name or len(new_name) < 1:
        return False, f"Error: empty name for {image_path.name}"

    counter = 1
    original_name = new_name
    while new_path.exists():
        new_name = f"{original_name}_{counter}"
        new_path = image_path.with_name(f"{new_name}{image_path.suffix}")
        counter += 1
        if counter > 100:
            return False, f"Too many duplicates: {image_path.name}"

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
    )

    parser.add_argument("directory", nargs="?", default="", help="Directory containing images")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Show changes without renaming")
    parser.add_argument("--force", "-f", action="store_true", help="Check relevance before rename")
    parser.add_argument("--pattern", "-p", default="*", help="File pattern")
    parser.add_argument("--max-length", "-m", type=int, default=50, help="Max filename length")
    parser.add_argument("--llm-host", default="http://localhost:1337", help="LLM server URL")
    parser.add_argument("--llm-model", default=FILENAME_MODELS[0] if FILENAME_MODELS else "foundation", help="LLM model")
    parser.add_argument("--vlm-model", default="", help="VLM model to use when no text is found")
    parser.add_argument("--api-key", default="", help="Bearer token for LLM API")
    parser.add_argument("--test", action="store_true", help="Test connection")
    parser.add_argument("--mlx-mode", action="store_true", help="Use MLX models directly, skip server")

    args = parser.parse_args()

    directory = Path(args.directory) if args.directory else Path.cwd()

    if not directory.exists() or not directory.is_dir():
        print(f"Error: Invalid directory '{directory}'")
        sys.exit(1)

    directory = directory.resolve()

    use_mlx = args.mlx_mode
    active_llm_host = args.llm_host
    active_model = args.llm_model

    if use_mlx:
        print("MLX MODE - Using direct MLX calls")
    else:
        print(f"Checking LLM at {args.llm_host}...")
        if check_llm_availability(args.llm_host, api_key=args.api_key):
            print("LLM Server found!")
        else:
            print("ERROR: LLM server not responding")
            print("Use --llm-host to specify a different server")
            sys.exit(1)

        print(f"Using model: {active_model}")

    print("Finding images...")
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    image_files = []

    if args.pattern == "*":
        for ext in image_extensions:
            image_files.extend(directory.glob(f"*{ext}"))
    else:
        image_files.extend(directory.glob(args.pattern))

    image_files = list(set([f for f in image_files if f.suffix.lower() in image_extensions]))

    if not image_files:
        print(f"No images found in '{directory}' matching pattern '{args.pattern}'")
        sys.exit(0)

    print(f"Found {len(image_files)} image(s)")
    if args.dry_run:
        print("DRY RUN MODE - No files will be renamed\n")

    mlx_model_path = None
    mlx_vlm_path = None

    active_vlm_model = args.vlm_model
    if not active_vlm_model:
        active_vlm_model = active_model
    print(f"Using VLM fallback model: {active_vlm_model}")

    stats = {"renamed": 0, "skipped": 0, "errors": 0}
    print(f"Processing {len(image_files)} images...")

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
            mlx_mode=use_mlx,
        )
        print(message)

        if success:
            stats["renamed"] += 1
        elif "Skipped" in message:
            stats["skipped"] += 1
        else:
            stats["errors"] += 1

    print(f"\n{'=' * 60}")
    print(f"Summary: {stats['renamed']} renamed, {stats['skipped']} skipped, {stats['errors']} errors")
    if args.dry_run and stats["renamed"] > 0:
        print("\nRun without --dry-run to actually rename the files")

if __name__ == "__main__":
    import sys
    sys.stdout = sys.stderr
    main()
