# LLM Response Parsing

import json
import re
from typing import Optional, Dict, Any, List


def extract_json(content: str, model: str = None) -> Optional[Any]:
    """Extract JSON from model response."""
    if not content:
        return None
    
    # Try direct parse first
    try:
        return json.loads(content)
    except:
        pass
    
    # Try extracting from markdown code blocks
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # Try finding JSON array or object
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', content)
    if match:
        try:
            return json.loads(match.group(1))
        except:
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
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    
    # Remove markdown code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # Remove backticks
    text = text.strip('`').strip()
    
    return text.strip()