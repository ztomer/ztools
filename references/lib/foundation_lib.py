"""
Bridge to Apple's on-device Foundation Models (macOS 26+).

When Osaurus/Ollama is not available, ZTools falls back to the device's
built-in Foundation model — no server, no model download. Backed by Apple's
official ``apple-fm-sdk`` (Apache-2.0).
"""

import asyncio
from typing import Optional

from .logging_config import osaurus_logger as logger

try:
    import apple_fm_sdk as fm
except Exception as e:  # pragma: no cover - optional dependency
    fm = None
    logger.debug(f"apple-fm-sdk unavailable: {e}")


def foundation_available() -> bool:
    """True if the apple-fm-sdk is importable and a model is available on-device."""
    if fm is None:
        return False
    try:
        model = fm.SystemLanguageModel()
        available, _reason = model.is_available()
        return bool(available)
    except Exception as e:
        logger.debug(f"Foundation availability check failed: {e}")
        return False


def call_foundation(
    system_prompt: str, user_prompt: str, parse_json: bool = False
) -> Optional[str]:
    """Call the on-device Foundation Model. Returns response text, or None on failure."""
    if fm is None:
        return None
    try:
        model = fm.SystemLanguageModel()
        available, reason = model.is_available()
        if not available:
            logger.warning(f"Foundation model unavailable: {reason}")
            return None
        session = fm.LanguageModelSession(model=model)
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        response = asyncio.run(session.respond(prompt=full_prompt))
        return response.strip() if isinstance(response, str) else None
    except Exception as e:
        logger.warning(f"Foundation call error: {e}")
        return None
