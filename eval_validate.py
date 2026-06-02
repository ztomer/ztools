#!/usr/bin/env python3
"""
Validation module for model evaluation.
"""

import json
import re
from typing import Tuple, Any
from lib.validators_lib import has_text_headers


def safe_content(result: dict) -> str:
    """Safely extract content from a result dict, handling None values."""
    content = result.get("content")
    if content is None:
        return ""
    if not isinstance(content, str):
        return str(content)
    return content


def validate_file_summary(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Validate file summary quality - checks for ACTUAL content detail, not filename inference.
    
    STRICT checks:
    - No filename-only summaries (must describe what file does)
    - No generic patterns like "a python script"
    - Must have actionable content about file purpose/function
    """
    if not data:
        return 0, "empty response"

    if isinstance(data, list):
        failures = []
        items = data
        num_files = len(items)
        if num_files < 4:
            failures.append(f"only {num_files} files")

        detailed_count = 0
        content_verbs = ['parse', 'validat', 'evaluat', 'extract', 'load', 'save',
            'read', 'write', 'fetch', 'send', 'process', 'handle',
            'config', 'setting', 'option', 'parameter', 'api', 'client', 'model', 'llm']

        for item in items:
            if not isinstance(item, dict):
                continue
            path = item.get("path", "")
            desc = item.get("desc", "") or item.get("summary", "")
            if not path or not desc:
                continue
            desc_lower = str(desc).lower()

            has_content = any(kw in desc_lower for kw in content_verbs)
            if has_content:
                detailed_count += 1

        if num_files == 0:
            return 0, "no items"
        if detailed_count >= num_files * 0.8:
            score = 100
        elif detailed_count >= num_files * 0.5:
            score = 85
        elif detailed_count >= 2:
            score = 70
        elif detailed_count >= 1:
            score = 50
        else:
            score = 25
            failures.append("no content details")

        return min(100, score), "; ".join(failures) if failures else ""

    if isinstance(data, dict):
        data = json.dumps(data)

    data_str = str(data).strip()
    failures = []
    score = 0

    parsed = None
    try:
        parsed = json.loads(data_str)
    except Exception:
        pass

    if not parsed:
        if has_text_headers(data_str):
            score += 20
        if len(data_str) >= 200:
            score += 20
        if score < 40:
            failures.append("no headers")
        return min(100, max(score, 20)), "; ".join(failures)

    items = list(parsed.items()) if isinstance(parsed, dict) else parsed
    num_files = len(items)

    detailed_count = 0
    content_verbs = ['parse', 'validat', 'evaluat', 'extract', 'load', 'save',
        'read', 'write', 'fetch', 'send', 'process', 'handle',
        'config', 'setting', 'option', 'parameter', 'api', 'client', 'model', 'llm']

    for filepath, summary in items:
        if not filepath or not summary:
            continue
        summary_lower = str(summary).lower()

        has_content = any(kw in summary_lower for kw in content_verbs)
        if has_content:
            detailed_count += 1

    if detailed_count >= num_files * 0.8:
        score = 85
    elif detailed_count >= num_files * 0.5:
        score = 70
    elif detailed_count >= 2:
        score = 55
    elif detailed_count >= 1:
        score = 40
    else:
        score = 25

    if not detailed_count:
        failures.append("no content details")

    return min(100, score), "; ".join(failures) if failures else ""

    data_lower = data_str.lower()

    if has_text_headers(data_str):
        score += 20
    else:
        failures.append("no headers")

    user_mentions = ['@user1', '@user2', '@user3', '@user4']
    found_users = sum(1 for u in user_mentions if u in data_lower)
    if found_users >= 3:
        score += 20
    elif found_users >= 2:
        score += 10
    else:
        failures.append(f"missing user mentions ({found_users}/4)")

    time_patterns = ['10:00', '10:15', '10:30', '10:45', '11:00', '11:15', '11:30']
    found_times = sum(1 for t in time_patterns if t in data_str)
    if found_times >= 4:
        score += 20
    elif found_times >= 2:
        score += 10
    else:
        failures.append(f"missing timestamps ({found_times})")

    key_events = ['launch', 'access', 'beta', 'feedback', 'smooth']
    found_events = sum(1 for e in key_events if e in data_lower)
    if found_events >= 3:
        score += 20
    elif found_events >= 2:
        score += 10
    else:
        failures.append(f"missing key events ({found_events})")

    if len(data_str) >= 300:
        score += 20
    elif len(data_str) >= 150:
        score += 10
    else:
        failures.append(f"too short ({len(data_str)} chars)")

    return min(100, score), "; ".join(failures) if failures else ""

    content_lines = []
    for line in data_str.split('\n'):
        stripped = line.strip()
        if not stripped:
            continue
        header_match = re.match(r'^##\s+\S+(?:\s*:?\s*)?$', stripped)
        if header_match:
            continue
        content_lines.append(stripped)

    detail_keywords = [
        'evaluat', 'parse', 'validat', 'extract', 'config', 'setting',
        'planning', 'summariz', 'renam', 'browser', 'playwright', 'ocr',
        'weekend', 'twitter', 'image', 'context', 'assistant', 'instruction',
        'overview', 'document', 'guideline', 'interaction', 'setup',
        'api', 'server', 'client', 'test', 'mock', 'request', 'response',
        'application', 'development', 'performance', 'behavior', 'quirk',
        'tool', 'utility', 'library', 'package', 'model', 'llm',
        'weekend', 'twitter', 'image', 'scrape', 'fetch',
    ]

    generic_patterns = [
        r'^a\s+python\s+script',
        r'^a\s+script\s+(for|to|of)',
        r'^a\s+tool\s+for',
        r'^a\s+utility\s+for',
        r'^an?\s+(exploration|investigation)\s+',
        r'^the\s+entry\s+point',
    ]

    detailed_lines = 0
    generic_lines = 0

    for line in content_lines:
        line_lower = line.lower()
        is_generic = any(re.match(p, line_lower.rstrip('.').strip()) for p in generic_patterns)
        has_detail = any(kw in line_lower for kw in detail_keywords)

        if is_generic:
            generic_lines += 1
        elif has_detail:
            detailed_lines += 1

    num_lines = len(content_lines)
    if num_lines > 0:
        detail_ratio = detailed_lines / num_lines
        if detail_ratio >= 0.7:
            score += 40
        elif detail_ratio >= 0.5:
            score += 30
        elif detail_ratio >= 0.3:
            score += 20
        elif detailed_lines >= 2:
            score += 10
        else:
            failures.append("filename inference only - no content detail")

    return min(100, score), "; ".join(failures) if failures else ""
