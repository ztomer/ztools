# LLM quirks - Model-specific prompt modifications

from typing import Any, Dict, List

from lib.llm.constants import (
    DEFAULT_FAMILY,
    GEMMA4_FAMILY,
    GEMMA4_PREFIX_IMPORTANT,
    GEMMA4_PREFIX_KW,
    GEMMA4_TRIGGER_TEXT,
    MODEL_FAMILIES,
    NO_JSON_KW,
    PLAIN_TEXT_KW,
    QWEN_FAMILY,
    QWEN_TRIGGER_PREFIX,
    QWEN_TRIGGER_TEXT,
    REPLACE_SRC_CONTEXT,
    REPLACE_SRC_TASK,
    REPLACE_SRC_TASK_BASED,
    REPLACE_TGT_CONTEXT,
    REPLACE_TGT_TASK,
    REPLACE_TGT_TASK_BASED,
    ROLE_SYSTEM,
    ROLE_USER,
    USER_REPLACE_CONTEXT,
    USER_REPLACE_EXECUTE,
)
from lib.logging_config import lib_logger as logger


def _get_model_family(model: str) -> str:
    """Extract model family from full model name."""
    if not model:
        return DEFAULT_FAMILY

    model_lower = model.lower()

    for family in MODEL_FAMILIES:
        if family in model_lower:
            return family

    return DEFAULT_FAMILY


def apply_model_quirks(messages: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """Apply model-specific prompt modifications."""
    family = _get_model_family(model)

    updated = []
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", ROLE_USER)

        if family == QWEN_FAMILY and role == ROLE_SYSTEM:
            # Prepend JSON trigger for qwen models to prevent thinking output
            # Skip if content already says no JSON or plain text
            if content and not content.startswith(QWEN_TRIGGER_PREFIX):
                if NO_JSON_KW not in content.lower() and PLAIN_TEXT_KW not in content.lower():
                    content = QWEN_TRIGGER_TEXT + content
                    logger.debug(f"Applied qwen JSON trigger for {model}")

        elif family == GEMMA4_FAMILY:
            if role == ROLE_SYSTEM:
                # Gemma4 needs extraction framing
                if GEMMA4_PREFIX_KW in content.upper() and not content.startswith(
                    GEMMA4_PREFIX_IMPORTANT
                ):
                    content = GEMMA4_TRIGGER_TEXT + content
                    logger.debug(f"Applied gemma4 system quirk for {model}")

        if role == ROLE_USER:
            # Models respond badly to "Execute", "Context", "Task" - use "Data" / "Extract"
            if USER_REPLACE_EXECUTE in content.lower() or USER_REPLACE_CONTEXT in content.lower():
                content = content.replace(REPLACE_SRC_CONTEXT, REPLACE_TGT_CONTEXT)
                content = content.replace(REPLACE_SRC_TASK, REPLACE_TGT_TASK)
                content = content.replace(REPLACE_SRC_TASK_BASED, REPLACE_TGT_TASK_BASED)
                logger.debug(f"Applied user quirk for {model}")

        updated.append({**msg, "content": content})

    return updated
