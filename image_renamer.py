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
from rich.console import Console

console = Console()
from typing import List, Tuple, Optional
from lib.config import get_filename_models, get_filename_prompt

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


# --- Server Management ---
def ensure_llm_running() -> bool:
    """Detect crash and restart server if needed."""
    import subprocess
    import time
    
    # Check if running
    if check_llm_availability("http://localhost:1337"):
        return True
    
    print("[WARN] LLM server not responding, restarting...")
    
    # Kill any stale processes
    try:
        subprocess.run(["pkill", "-f", "osaurus"], capture_output=True)
        time.sleep(2)
    except:
        pass
    
    # Restart
    try:
        subprocess.Popen(["open", "-a", "osaurus"])
        time.sleep(15)  # Wait for startup
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

def is_relevant_with_llm(text: str, host: str, api_key: str = "") -> Optional[bool]:
    """Ask LLM if image content is relevant worth keeping."""
    import requests
    
    prompt = RELEVANCE_CHECK_PROMPT.format(text=text[:500])  # Limit text
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
    
    return None  # Can't determine

FILENAME_MODELS = get_filename_models()

# Prompt loaded from config
PROMPT_TEXT_TO_FILENAME = get_filename_prompt()

# Use server for filename generation (MLX direct is broken)
MLX_MODELS_DIR = Path.home() / "MLXModels"


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


def is_meaningful_text(text: str, min_word_count: int = 2) -> bool:
    """Check if extracted text is meaningful (human-readable) vs random characters."""
    if not text:
        return False
    
    # Remove common OCR artifacts
    text = text.strip()
    
    # Check for patterns that indicate non-meaningful text:
    # 1. All uppercase alphanumeric strings (like HFyWGG4XIAAvWXG)
    # 2. Very short random-looking strings
    # 3. Strings with no spaces between random characters
    
    if not text:
        return False
    
    # If text is mostly uppercase letters with no spaces, it's likely an ID not readable text
    # e.g., "HFYAG4XIAAVWXg" or "HAPPYBIRTHDAY"
    words = text.split()
    if len(words) == 1 and len(text) > 8:
        # Single long word - check if it's alphanumeric ID
        if text.isalnum() and text[:2].isupper():
            return False
    
    # If all uppercase with no spaces and length > 4, likely not readable prose
    if text.isupper() and len(text) > 4 and " " not in text:
        return False
    
    # Count actual word-like sequences (letters > 2 chars)
    word_like = sum(1 for w in words if len(w) > 2 and any(c.isalpha() for c in w))
    
    return word_like >= min_word_count


def is_non_human_readable(text: str) -> bool:
    """Check if text appears to be non-human-readable (IDs, random chars, codes)."""
    if not text:
        return True
    
    text = text.strip()
    
    # Skip empty or very short
    if len(text) < 3:
        return True
    
    import re
    
    # Pattern 1: HuggingFace-style IDs
    # Like: HFyWGG4XIAAvWXG, HHRqeLwXUAAQmG6
    # Starts with HF or HH, followed by 7+ alphanumeric chars
    if re.match(r'^HF[A-Za-z0-9]{7,}$', text) or re.match(r'^HH[A-Za-z0-9]{7,}$', text):
        return True
    
    # Pattern 2: Twitter-like handles (@something without spaces)
    if text.startswith("@") and "_" not in text and len(text) > 1:
        return True
    
    # Pattern 3: Very short codes (likely OCR mistakes or headers)
    if len(text) <= 3 and text.isupper():
        return True
    
    # Pattern 4: Single word that's all uppercase with digits mixed in
    # Like: ABC123, TEST99 - these look like IDs/codes
    if " " not in text:
        if text.isupper() and any(c.isdigit() for c in text):
            return True
    
    return False


# Use consolidated functions from lib.osaurus_lib and mlx_lib


def query_llm_for_filename(
    text: str, host: str = "http://localhost:1337", model: str = "", api_key: str = ""
) -> Optional[str]:
    """Query for filename using server models."""
    
    # Use server models
    import requests
    
    for m in FILENAME_MODELS:
        try:
            prompt = PROMPT_TEXT_TO_FILENAME.format(text=text)
            messages = [{"role": "user", "content": prompt}]
            
            resp = requests.post(
                f"{host}/api/chat",
                json={"model": m, "messages": messages},
                timeout=120,  # 2 min timeout
            )
            if resp.status_code != 200:
                continue
            
            # Accumulate all chunks until done=true
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
                content = content.strip()
                content = content.lower()
                
                # Take first 4-6 words, but limit to ~30 chars
                import re
                words = re.findall(r'[a-z]+', content)
                if not words:
                    continue
                
                # Build filename, limiting to ~30 chars
                content = '_'.join(words[:6])
                if len(content) > 35:
                    content = content[:35]
                
                # Must be letters and underscores only  
                if not re.match(r"^[a-z_]+$", content):
                    continue
                    
                # Must have at least one letter
                if not any(c.isalpha() for c in content):
                    continue
                
                return content
                
        except:
            continue
    
    return None


def query_mlx_for_filename(text: str) -> Optional[str]:
    """Query MLX model directly for filename."""
    from lib.mlx_lib import find_mlx_model, call_mlx, process_mlx_content
    
    # Try each model from config
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
                    # Clean
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
    """Query a Vision Language Model to describe the image."""
    from lib.config import get_model_prompt, Task

    # Try config first
    prompt = get_model_prompt(model, Task.FILENAME)
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
        ensure_llm_running()  # Try to restart
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
    mlx_mode: bool = False,
) -> Tuple[bool, str]:
    if not image_path.exists():
        return False, f"File not found: {image_path.name}"
        
    text = extract_full_text(image_path) or extract_first_line(image_path)
    
    # Check if text is human-readable - use LLM if available, else fallback to heuristics
    if not text:
        return False, f"Skipped (No text): {image_path.name}"
    
    # Use heuristics to check readability
    if is_non_human_readable(text):
        return False, f"Skipped (Non-human-readable): {image_path.name}"

    if not is_meaningful_text(text, min_word_count=2):
        return False, f"Skipped (No meaningful text): {image_path.name}"
    
    # In force mode, check relevance with LLM
    if force and llm_host:
        relevant = is_relevant_with_llm(text, llm_host, api_key)
        if relevant is False:
            return False, f"Skipped (Not relevant): {image_path.name}"
        elif relevant is True:
            print(f"   [RELEVANT] {image_path.name}")
        
    new_name = None
    
    # Query for new name - MLX mode or server mode
    if mlx_mode:
        # Use MLX directly
        try:
            new_name = query_mlx_for_filename(text)
        except Exception as e:
            print(f"   [WARN] MLX failed: {e}, using fallback")
            new_name = None
    elif llm_host and llm_model:
        # Use server
        try:
            new_name = query_llm_for_filename(text, llm_host, llm_model, api_key)
        except Exception as e:
            print(f"   [WARN] LLM failed: {e}, using fallback")
            new_name = None
        
        # Validate LLM result - accept if it looks like a filename
        if new_name:
            # Reject if it's clearly not a filename
            if new_name in ("text", "file", "image", "unnamed", "output", "filename", "none"):
                print(f"   [WARN] Generic LLM result: {new_name}, using fallback")
                new_name = None
            elif len(new_name) < 4:
                print(f"   [WARN] Too short: {new_name}, using fallback")
                new_name = None
        
    if not new_name:
        new_name = clean_filename(text)
        
    # If cleaned is still generic, keep original (avoid data loss)
    if new_name in ("text", "file", "image", "unnamed", "output"):
        return False, f"Skipped (Generic name): {image_path.name}"
        
    if not new_name:
        return False, f"Could not generate name: {image_path.name}"
        new_name = query_llm_for_filename(text, "localhost", mlx_model_path.name, api_key)
        
    if not new_name:
        new_name = clean_filename(text)
        print(f"   [FALLBACK] Using cleaned: {new_name}")
        
    if not new_name:
        return False, f"Could not generate name: {image_path.name}"
        
    new_path = image_path.with_name(f"{new_name}{image_path.suffix}")
    
    # Validate new_path
    if not new_name or len(new_name) < 1:
        return False, f"Error: empty name for {image_path.name}"
    
    # Prevent duplicate names - add suffix if exists
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
        "directory", nargs="?", default="", help="Directory containing images"
    )
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show changes without renaming"
    )
    parser.add_argument("--force", "-f", action="store_true", 
        help="Check relevance before rename, skip low-value content")
    parser.add_argument("--pattern", "-p", default="*", help="File pattern")
    parser.add_argument(
        "--max-length", "-m", type=int, default=50, help="Max filename length"
    )
    parser.add_argument(
        "--llm-host", default="http://localhost:1337", help="LLM server URL"
    )
    parser.add_argument(
        "--llm-model", default=FILENAME_MODELS[0] if FILENAME_MODELS else "foundation", help="LLM model"
    )
    parser.add_argument(
        "--api-key", default="", help="Bearer token for LLM API",
    )
    parser.add_argument("--test", action="store_true", help="Test connection")
    parser.add_argument(
        "--mlx-mode", action="store_true", 
        help="Use MLX models directly, skip server",
    )
    
    args = parser.parse_args()

    directory = Path(args.directory) if args.directory else Path.cwd()
    
    if not directory.exists() or not directory.is_dir():
        print(f"Error: Invalid directory '{directory}'")
        sys.exit(1)
    
    # Resolve to absolute path
    directory = directory.resolve()
    all_files = list(directory.glob('*'))
    
    # Check LLM availability (skip if mlx-mode)
    use_mlx = args.mlx_mode
    
    if use_mlx:
        print("MLX MODE - Using direct MLX calls")
    else:
        print(f"Checking LLM at {args.llm_host}...")
        if check_llm_availability(args.llm_host, api_key=args.api_key):
            active_llm_host = args.llm_host
            print("LLM Server found!")
        else:
            print("ERROR: LLM server not responding")
            print("Use --llm-host to specify a different server")
            sys.exit(1)
        
        active_llm_host = args.llm_host
        active_model = args.llm_model
        print(f"Using model: {active_model}")
    
# Find Images
    print("Finding images...")
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    image_files = []

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
    
    # Initialize None vars to avoid errors
    active_vlm_model = None
    mlx_model_path = None
    mlx_vlm_path = None
    
# Process
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
    print(
        f"Summary: {stats['renamed']} renamed, {stats['skipped']} skipped, {stats['errors']} errors"
    )
    if args.dry_run and stats["renamed"] > 0:
        print("\nRun without --dry-run to actually rename the files")


if __name__ == "__main__":
    import sys
    sys.stdout = sys.stderr  # Unbuffered
    main()
