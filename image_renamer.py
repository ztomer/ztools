#!/usr/bin/env python3
# /// script
# dependencies = ["pillow", "pytesseract", "requests", "mlx-lm", "mlx-vlm"]
# ///
"""
Rename images based on their text content.
Reads the first line of text from each image and uses it as the filename.
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

from img_helpers import (
    clean_filename,
    extract_first_line,
    extract_full_text,
    is_meaningful_text,
    is_non_human_readable,
    _strip_instruction_prefix,
    _GENERIC_BASES,
    _GENERIC_EXTENSIONS,
    _GENERIC_NAMES,
    image_extensions,
)
from img_llm import (
    ensure_llm_running,
    is_relevant_with_llm,
    query_llm_for_filename,
    query_mlx_for_filename,
    query_vlm_for_filename,
    FILENAME_MODELS,
    RELEVANCE_CHECK_PROMPT,
    PROMPT_TEXT_TO_FILENAME,
    PROMPT_IMAGE_TO_FILENAME,
    MLX_MODELS_DIR,
)
from lib.osaurus_lib import check_llm_availability
from lib.tui import STEP, FAIL

__all__ = [
    "ensure_llm_running",
    "is_relevant_with_llm",
    "query_llm_for_filename",
    "query_mlx_for_filename",
    "query_vlm_for_filename",
    "clean_filename",
    "extract_first_line",
    "extract_full_text",
    "is_meaningful_text",
    "is_non_human_readable",
    "_strip_instruction_prefix",
    "_GENERIC_BASES",
    "_GENERIC_EXTENSIONS",
    "_GENERIC_NAMES",
    "image_extensions",
    "FILENAME_MODELS",
    "RELEVANCE_CHECK_PROMPT",
    "PROMPT_TEXT_TO_FILENAME",
    "PROMPT_IMAGE_TO_FILENAME",
    "MLX_MODELS_DIR",
    "rename_image",
    "main",
]


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
            print(f"  {STEP} No readable text, using VLM")
            new_name = query_vlm_for_filename(image_path, llm_host, vlm_model, api_key)
            if new_name:
                new_name = clean_filename(new_name)
                if new_name in _GENERIC_NAMES or len(new_name) < 4:
                    print(f"  {FAIL} Generic VLM result: {new_name}")
                    new_name = None
        else:
            return False, f"Skipped (No text & no VLM fallback configured): {image_path.name}"
    elif text:
        if force and llm_host:
            relevant = is_relevant_with_llm(text, llm_host, api_key)
            if relevant is False:
                return False, f"Skipped (Not relevant): {image_path.name}"

        if mlx_mode:
            try:
                new_name = query_mlx_for_filename(text)
            except Exception as e:
                print(f"  {FAIL} MLX failed: {e}")

            if new_name:
                new_name = clean_filename(new_name)
                if new_name in _GENERIC_NAMES:
                    print(f"  {FAIL} Generic MLX result: {new_name}")
                    new_name = None
                elif len(new_name) < 4:
                    print(f"  {FAIL} Too short: {new_name}")
                    new_name = None
        elif llm_host and llm_model:
            try:
                new_name = query_llm_for_filename(text, llm_host, llm_model, api_key)
            except Exception as e:
                print(f"  {FAIL} LLM failed: {e}")

            if new_name:
                if new_name in _GENERIC_NAMES:
                    print(f"  {FAIL} Generic LLM result: {new_name}")
                    new_name = None
                elif len(new_name) < 4:
                    print(f"  {FAIL} Too short: {new_name}")
                    new_name = None
        else:
            return False, f"Could not generate name: {image_path.name}"

    if not new_name and text and not needs_vlm:
        new_name = clean_filename(text)

    if new_name is None:
        return False, f"Could not generate name: {image_path.name}"
    if new_name in _GENERIC_NAMES:
        return False, f"Skipped (Generic name): {image_path.name}"

    new_path = image_path.with_name(f"{new_name}{image_path.suffix}")

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

    return True, ""


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
        print(f"{STEP} MLX mode")
    else:
        if not check_llm_availability(args.llm_host, api_key=args.api_key):
            print(f"{FAIL} LLM server not responding")
            print("  Use --llm-host to specify a different server")
            sys.exit(1)

    print(f"{STEP} {active_model}")
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

    print(f"{STEP} {len(image_files)} images")
    if args.dry_run:
        print(f"{STEP} Dry run")

    mlx_model_path = None
    mlx_vlm_path = None

    active_vlm_model = args.vlm_model
    if not active_vlm_model:
        active_vlm_model = active_model

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
            mlx_mode=use_mlx,
        )
        if success:
            stats["renamed"] += 1
        elif "Skipped" in message:
            stats["skipped"] += 1
            print(message)
        else:
            stats["errors"] += 1
            print(message)

    print(f"{STEP} {stats['renamed']} renamed, {stats['skipped']} skipped, {stats['errors']} errors")
    if args.dry_run and stats["renamed"] > 0:
        print("  Run without --dry-run to rename")


if __name__ == "__main__":
    import sys
    sys.stdout = sys.stderr
    main()
