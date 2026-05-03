# LLM quirks - Model-specific prompt modifications

import re
from typing import List, Dict, Any

from lib.llm.constants import MODEL_FAMILIES


def _get_model_family(model: str) -> str:
    """Extract model family from full model name."""
    if not model:
        return "default"
    
    model_lower = model.lower()
    
    for family in MODEL_FAMILIES:
        if family in model_lower:
            return family
    
    return "default"


def apply_model_quirks(messages: List[Dict[str, Any]], model: str) -> List[Dict[str, Any]]:
    """Apply model-specific prompt modifications."""
    family = _get_model_family(model)
    
    updated = []
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "user")
        
        if family == "qwen" and role == "system":
            # Prepend JSON trigger for qwen models to prevent thinking output
            # Skip if content already says no JSON or plain text
            if content and not content.startswith("Output JSON now"):
                if "no JSON" not in content.lower() and "plain text" not in content.lower():
                    content = "Output JSON now.\n\n" + content
        
        elif family == "gemma4":
            if role == "system":
                # Gemma4 needs extraction framing
                if "JSON" in content.upper() and not content.startswith("IMPORTANT"):
                    content = "IMPORTANT: This is DATA EXTRACTION. Output JSON only. " + content
        
        if role == "user":
            # Models respond badly to "Execute", "Context", "Task" - use "Data" / "Extract"
            if "execute" in content.lower() or "context" in content.lower():
                content = content.replace("Current Context", "Data")
        
        updated.append({**msg, "content": content})
    
    return updated