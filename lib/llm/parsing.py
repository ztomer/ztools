# LLM Response Parsing

import json
import re
from typing import Optional, Dict, Any, List

# Pre-compiled regular expressions for performance (John Carmack optimization)
MARKDOWN_JSON_BLOCK_RE = re.compile(r'```(?:json)?\s*([\s\S]*?)```')
JSON_CONTAINER_RE = re.compile(r'(\{[\s\S]*\}|\[[\s\S]*\])')
THINKING_BLOCK_RE = re.compile(r'<think>[\s\S]*?</think>')
ANY_CODE_BLOCK_RE = re.compile(r'```[\s\S]*?```')


def extract_json(content: str, model: str = None) -> Optional[Any]:
    """Extract JSON from model response."""
    if not content:
        return None
    
    # Try direct parse first
    try:
        return json.loads(content)
    except Exception:
        pass
    
    # Try extracting from markdown code blocks
    match = MARKDOWN_JSON_BLOCK_RE.search(content)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    
    # Try finding JSON array or object
    match = JSON_CONTAINER_RE.search(content)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    
    return None


def safe_content(result: dict) -> str:
    """Safely extract content from result dict."""
    content = result.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        return str(content)
    return content


def clean_output(text: str) -> str:
    """Clean model output text."""
    if not text:
        return ""
    
    # Remove thinking blocks
    text = THINKING_BLOCK_RE.sub('', text)
    
    # Remove markdown code blocks
    text = ANY_CODE_BLOCK_RE.sub('', text)
    
    # Remove backticks
    text = text.strip('`').strip()
    
    return text.strip()