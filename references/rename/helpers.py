"""
Helper utilities for image processing and filename generation.
"""

import re
import sys
from pathlib import Path
from typing import Optional

from lib.config_toml import load_config
from lib.paths import conf_path
from lib.tui import FAIL
from PIL import Image

# Load tesseract path from rename config, fall back to Homebrew default
_RENAME_CONFIG_PATH = conf_path("rename.toml")
_RENAME_CFG = load_config(_RENAME_CONFIG_PATH) or {}
_TESSERACT_BREW = _RENAME_CFG.get("tesseract_cmd", "/opt/homebrew/bin/tesseract")
_TESSERACT_INIT = False

# Generic LLM outputs that should be rejected (not real filenames)
_GENERIC_BASES = frozenset(
    {
        "text",
        "file",
        "image",
        "unnamed",
        "output",
        "filename",
        "none",
        "screenshot",
        "document",
        "note",
        "empty",
        "blank",
    }
)
_GENERIC_EXTENSIONS = frozenset(
    {
        "txt",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "tiff",
        "tif",
        "webp",
    }
)
_GENERIC_NAMES = _GENERIC_BASES | frozenset(
    f"{base}_{ext}" for base in _GENERIC_BASES for ext in _GENERIC_EXTENSIONS
)

image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"}


def clean_filename(text: str, max_length: int = 50) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", text)
    cleaned = re.sub(r"[-\s]+", "_", cleaned)
    cleaned = cleaned.strip("_").lower()

    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("_")

    return cleaned if cleaned else "unnamed"


def ocr_available() -> bool:
    """Whether OCR can actually run — checked before a batch, not per image."""
    return _get_tesseract() is not None


def _get_tesseract():
    """Return the pytesseract module, or None with a stated reason.

    A silent None turns this tool into a picture-describer: every screenshot
    OCRs to nothing, routes to the VLM, and the only trace is a per-image
    "No readable text" that is indistinguishable from an image with no text.
    """
    global _TESSERACT_INIT
    if not _TESSERACT_INIT:
        try:
            import pytesseract
        except ImportError:
            pytesseract = None
        if pytesseract and Path(_TESSERACT_BREW).exists():
            pytesseract.pytesseract.tesseract_cmd = _TESSERACT_BREW
        elif pytesseract is None:
            print(
                f"{FAIL} OCR unavailable: pytesseract is not installed — every image "
                "would be sent to the VLM. Install it with: uv sync",
                file=sys.stderr,
            )
        elif not Path(_TESSERACT_BREW).exists():
            print(
                f"{FAIL} OCR unavailable: the tesseract binary is missing at "
                f"{_TESSERACT_BREW} — every image would be sent to the VLM.",
                file=sys.stderr,
            )
        _TESSERACT_INIT = True
    else:
        pytesseract = sys.modules.get("pytesseract")
    return pytesseract


def extract_first_line(image_path: Path, pytesseract=None) -> Optional[str]:
    if pytesseract is None:
        pytesseract = _get_tesseract()
    if pytesseract is None:
        return None
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            return lines[0]
        return None
    except Exception as e:
        print(f"{FAIL} {image_path.name}: {e}")
        return None


def extract_full_text(image_path: Path, pytesseract=None) -> Optional[str]:
    if pytesseract is None:
        pytesseract = _get_tesseract()
    if pytesseract is None:
        return None
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip() if text.strip() else None
    except Exception as e:
        print(f"{FAIL} {image_path.name}: {e}")
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

    if re.match(r"^HF[A-Za-z0-9]{7,}$", text) or re.match(r"^HH[A-Za-z0-9]{7,}$", text):
        return True

    if text.startswith("@") and "_" not in text and len(text) > 1:
        return True

    if len(text) <= 3 and text.isupper():
        return True

    if " " not in text:
        if text.isupper() and any(c.isdigit() for c in text):
            return True

    return False


def _strip_instruction_prefix(content: str) -> str:
    return re.sub(
        r"^\s*(?:(?:here(?: is|'s)?(?: a| the)?|the|suggested|renamed? to)?\s*"
        r"(?:filename|file|name|output|result|response)?(?:\s+is)?\s*:\s*)",
        "",
        content,
        count=1,
        flags=re.IGNORECASE,
    ).strip()
