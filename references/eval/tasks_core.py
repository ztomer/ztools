"""Evaluation tasks and test cases for model evaluation.

Shim: re-exports prompts from eval/tasks_prompts.py, defines TASKS dict and
_extract_items_from_text helper.
"""

import re
from typing import Dict, List

from eval.tasks_prompts import (
    CONTRADICTION_PHRASE,
    FALSEHOOD_PHRASES,
    FILE_SUMMARY_PROMPT,
    FILE_SUMMARY_PROMPT_MIXED,
    IMAGE_RENAME_PROMPT,
    IMAGE_RENAME_PROMPT_MIXED,
    KEY_FACTS,
    RENAME_PROMPT,
    RENAME_PROMPT_MIXED,
    TWITTER_PROMPT,
    TWITTER_PROMPT_ACCURACY,
    TWITTER_PROMPT_CONTRADICTION,
    TWITTER_PROMPT_MIXED,
    WEEKEND_USR_FIXED_MIXED,
    WEEKEND_USR_TRANSIENT_MIXED,
)
from lib.eval_data import (
    WEEKEND_SYS_FIXED,
    WEEKEND_SYS_TRANSIENT,
    WEEKEND_USR_FIXED,
    WEEKEND_USR_TRANSIENT,
)
from lib.validators.json_validator import (
    validate_detailed_json,
    validate_mixed_signal,
)
from lib.validators.text_validator import (
    validate_factual_accuracy,
    validate_factual_coverage,
    validate_file_summary,
    validate_filename,
    validate_mixed_file_summary,
    validate_mixed_filename,
    validate_mixed_summary,
    validate_no_contradiction,
    validate_no_leak,
    validate_strict_schema,
    validate_summary,
)


def _extract_items_from_text(text: str) -> List[Dict]:
    items = []
    table_pattern = r"\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
    tables = re.findall(table_pattern, text)
    if tables and len(tables) >= 2:
        header = tables[0]
        is_header_row = (
            "---" in header[0].lower()
            or "---" in header[1].lower()
            or not any(c.isalnum() for c in header[0])
            or not any(c.isalnum() for c in header[1])
        )
        data_rows = tables[1:] if is_header_row else tables
        if data_rows:
            key1 = header[0].strip().lower()
            key2 = header[1].strip().lower()
            header_map = {
                "name": "name", "event": "name", "title": "name", "activity": "name",
                "location": "location", "venue": "location", "place": "place",
                "where": "location", "day": "day", "date": "day", "when": "day",
                "time": "time",
            }
            field1 = header_map.get(key1, "name")
            field2 = header_map.get(key2, key2)
            for row in data_rows:
                if "---" in row[0].lower() or "---" in row[1].lower():
                    continue
                if not row[0].strip() or not row[1].strip():
                    continue
                row0_clean = row[0].strip().lower()
                row1_clean = row[1].strip().lower()
                header_names = {
                    "name", "event", "title", "activity",
                    "location", "venue", "place", "where",
                }
                if row0_clean in header_names or row1_clean in header_names:
                    continue
                item = {field1: row[0].strip(), field2: row[1].strip()}
                items.append(item)
            if items:
                return items
    bullet_pattern = r"^[•\-]\s*(.+?)(?:\n|$)"
    bullets = re.findall(bullet_pattern, text, re.MULTILINE)
    for bullet in bullets:
        bullet = bullet.strip()
        if bullet and len(bullet) > 2:
            parts = bullet.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                val = parts[1].strip()
                field_map = {
                    "name": "name", "event": "name", "title": "name", "activity": "name",
                    "location": "location", "venue": "location", "place": "location",
                }
                field = field_map.get(key.lower(), key.lower())
                items.append({field: val})
            else:
                sep_match = re.match(r"^([^,\-]+)[,\-](.+)$", bullet)
                if sep_match:
                    items.append({
                        "name": sep_match.group(1).strip(),
                        "location": sep_match.group(2).strip(),
                    })
                else:
                    items.append({"name": bullet})
    return items


TASKS = {
    "weekend_transient": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_TRANSIENT},
            {"role": "user", "content": WEEKEND_USR_TRANSIENT},
        ],
        "validator": validate_detailed_json,
        "parse_json": True,
        "source": WEEKEND_USR_TRANSIENT,
    },
    "weekend_fixed": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_FIXED},
            {"role": "user", "content": WEEKEND_USR_FIXED},
        ],
        "validator": validate_detailed_json,
        "parse_json": True,
        "source": WEEKEND_USR_FIXED,
    },
    "weekend_transient_mixed": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_TRANSIENT},
            {"role": "user", "content": WEEKEND_USR_TRANSIENT_MIXED},
        ],
        "validator": validate_mixed_signal,
        "parse_json": True,
        "source": WEEKEND_USR_TRANSIENT_MIXED,
    },
    "weekend_fixed_mixed": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_FIXED},
            {"role": "user", "content": WEEKEND_USR_FIXED_MIXED},
        ],
        "validator": validate_mixed_signal,
        "parse_json": True,
        "source": WEEKEND_USR_FIXED_MIXED,
    },
    "filename": {
        "messages": [{"role": "user", "content": RENAME_PROMPT}],
        "validator": validate_filename,
        "parse_json": False,
    },
    "image_rename": {
        "messages": [{"role": "user", "content": IMAGE_RENAME_PROMPT}],
        "validator": validate_mixed_filename,
        "parse_json": True,
        "source": IMAGE_RENAME_PROMPT,
    },
    "summarize": {
        "messages": [{"role": "user", "content": TWITTER_PROMPT}],
        "validator": validate_summary,
        "parse_json": False,
    },
    "file_summary": {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Output JSON now. No preamble, no markdown.\n\n"
                    'Required format: {"path": "description", ...} OR '
                    '[{"path": "x", "desc": "y"}, ...]\n\n'
                    "Summarize each file in one line. Be specific - mention "
                    "actual functionality, not just file type."
                ),
            },
            {"role": "user", "content": FILE_SUMMARY_PROMPT},
        ],
        "validator": validate_file_summary,
        "parse_json": True,
    },
    "rename_mixed": {
        "messages": [{"role": "user", "content": RENAME_PROMPT_MIXED}],
        "validator": validate_mixed_filename,
        "parse_json": True,
        "source": RENAME_PROMPT_MIXED,
    },
    "summarize_mixed": {
        "messages": [{"role": "user", "content": TWITTER_PROMPT_MIXED}],
        "validator": validate_mixed_summary,
        "parse_json": False,
        "source": TWITTER_PROMPT_MIXED,
    },
    "file_summary_mixed": {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Output JSON now. No preamble, no markdown.\n\n"
                    'Required format: {"path": "description", ...} OR '
                    '[{"path": "x", "desc": "y"}, ...]\n\n'
                    "Summarize each file in one line. Be specific - mention "
                    "actual functionality, not just file type."
                ),
            },
            {"role": "user", "content": FILE_SUMMARY_PROMPT_MIXED},
        ],
        "validator": validate_mixed_file_summary,
        "parse_json": False,
        "source": FILE_SUMMARY_PROMPT_MIXED,
    },
    "filename_mixed": {
        "messages": [{"role": "user", "content": RENAME_PROMPT_MIXED}],
        "validator": validate_mixed_filename,
        "parse_json": True,
        "source": RENAME_PROMPT_MIXED,
    },
    "image_rename_mixed": {
        "messages": [{"role": "user", "content": IMAGE_RENAME_PROMPT_MIXED}],
        "validator": validate_mixed_filename,
        "parse_json": True,
        "source": IMAGE_RENAME_PROMPT_MIXED,
    },
    "weekend_transient_schema": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_TRANSIENT},
            {"role": "user", "content": WEEKEND_USR_TRANSIENT},
        ],
        "validator": validate_strict_schema,
        "parse_json": False,
        "validator_kwargs": {"kind": "json"},
    },
    "summarize_contradiction": {
        "messages": [{"role": "user", "content": TWITTER_PROMPT_CONTRADICTION}],
        "validator": validate_no_contradiction,
        "parse_json": False,
        "validator_kwargs": {"contradiction_phrase": CONTRADICTION_PHRASE},
    },
    "filename_leak": {
        "messages": [{"role": "user", "content": RENAME_PROMPT}],
        "validator": validate_no_leak,
        "parse_json": False,
    },
    "summarize_factual_accuracy": {
        "messages": [{"role": "user", "content": TWITTER_PROMPT_ACCURACY}],
        "validator": validate_factual_accuracy,
        "parse_json": False,
        "validator_kwargs": {"falsehood_phrases": FALSEHOOD_PHRASES},
    },
    "summarize_factual_coverage": {
        "messages": [{"role": "user", "content": TWITTER_PROMPT}],
        "validator": validate_factual_coverage,
        "parse_json": False,
        "validator_kwargs": {"key_facts": KEY_FACTS},
    },
}

TASKS["json"] = dict(TASKS["weekend_transient"])
TASKS["detailed_json"] = dict(TASKS["weekend_fixed"])
