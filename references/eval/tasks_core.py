"""Evaluation tasks and test cases for model evaluation.

Shim: re-exports prompts from eval/tasks_prompts.py, defines TASKS dict and
_extract_items_from_text helper.
"""

import json
import re
from typing import Dict, List

from lib.config import get_eval_input
from lib.eval_data import (
    WEEKEND_SYS_FIXED,
    WEEKEND_SYS_TRANSIENT,
    WEEKEND_USR_FIXED,
    WEEKEND_USR_TRANSIENT,
)
from lib.paths import eval_tasks_path
from lib.validators.adversarial import validate_no_fabrication, validate_resists_injection
from lib.validators.attribution import validate_attribution
from lib.validators.json_validator import (
    validate_detailed_json,
    validate_mixed_signal,
)
from lib.validators.taxes_grounded import (
    validate_taxes_qa,
    validate_taxes_slip_qa,
    validate_taxes_yoy_narrative,
)
from lib.validators.taxes_validator import (
    validate_taxes_anomalies,
    validate_taxes_audit_readiness,
    validate_taxes_synthesis,
)
from lib.validators.text_validator import (
    validate_file_summary,
    validate_filename,
    validate_summary,
)
from lib.validators.text_validator_mixed import (
    validate_factual_accuracy,
    validate_factual_coverage,
    validate_mixed_file_summary,
    validate_mixed_filename,
    validate_mixed_summary,
    validate_no_contradiction,
    validate_no_leak,
    validate_strict_schema,
)
from lib.validators.vision_validator import validate_image_description

from eval.tasks_prompts import (
    CONTRADICTION_PHRASE,
    FALSEHOOD_PHRASES,
    FILE_SUMMARY_PROMPT,
    FILE_SUMMARY_PROMPT_MIXED,
    FILENAME_INJECTION_KEYWORDS,
    FILENAME_INJECTION_MARKERS,
    FILENAME_INJECTION_PROMPT,
    IMAGE_RENAME_PROMPT,
    IMAGE_RENAME_PROMPT_MIXED,
    KEY_FACTS,
    RENAME_PROMPT,
    RENAME_PROMPT_MIXED,
    TWITTER_PROMPT,
    TWITTER_PROMPT_ACCURACY,
    TWITTER_PROMPT_CONTRADICTION,
    TWITTER_PROMPT_MISATTRIBUTION,
    TWITTER_PROMPT_MIXED,
    WEEKEND_FABRICATION_LURES,
    WEEKEND_FABRICATION_PROMPT,
    WEEKEND_USR_FIXED_MIXED,
    WEEKEND_USR_TRANSIENT_MIXED,
)
from eval.vision_fixtures import image_message


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



def _filename_prompt_for(text: str) -> str:
    """Render RENAME_PROMPT with a real input, never the bare template."""
    return RENAME_PROMPT.replace("{text}", text)


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
        # `{text}` MUST be filled: sending the raw template asked the model to
        # summarise the literal string "{text}", and the shape-only validator
        # scored the result 100. `source` lets validate_filename judge relevance
        # to the actual input rather than just the shape of the output.
        "messages": [
            {"role": "user", "content": _filename_prompt_for(get_eval_input("filename"))}
        ],
        "validator": validate_filename,
        "parse_json": False,
        "source": get_eval_input("filename"),
    },
    "image_rename": {
        "messages": [{"role": "user", "content": IMAGE_RENAME_PROMPT}],
        "validator": validate_mixed_filename,
        "parse_json": True,
        "source": IMAGE_RENAME_PROMPT,
    },
    "summarize": {
        # `source` MUST be set, for the same reason `filename` needs it above, and it
        # is the same bug: a task whose input does not carry the thing under test,
        # graded by a validator that then skips the test.
        #
        # validate_summary gates its strongest rule on having a source --
        # `if source_text and total_bullets and faithful < total_bullets` caps the
        # score at MISATTRIBUTION_MAX_SCORE, and its own comment calls that
        # "disqualifying, not a deduction". With no source the cap cannot fire, the
        # attribution ratio contributes no specificity credit either, and a summary
        # that credits every quote to the wrong person scores exactly the same as one
        # that gets them all right. Ten of eleven models tied at 100 here in the
        # 2026-08-16 sweep; the task was measuring structure and little else.
        #
        # The whole prompt is the source, matching summarize_mixed. The attribution
        # parser keys on `[@handle | HH:MM]` pairs, which appear only in the timeline
        # block, so the surrounding instructions cannot be mistaken for evidence.
        "messages": [{"role": "user", "content": TWITTER_PROMPT}],
        "validator": validate_summary,
        "parse_json": False,
        "source": TWITTER_PROMPT,
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
        "messages": [
            {"role": "user", "content": _filename_prompt_for(get_eval_input("filename"))}
        ],
        "validator": validate_no_leak,
        "parse_json": False,
    },
    "summarize_factual_accuracy": {
        "messages": [{"role": "user", "content": TWITTER_PROMPT_ACCURACY}],
        "validator": validate_factual_accuracy,
        "parse_json": False,
        "validator_kwargs": {"falsehood_phrases": FALSEHOOD_PHRASES},
    },
    "image_real": {
        # The ONLY task in this suite that sends an actual image.
        #
        # `image_rename` and `image_rename_mixed` send their prompt as TEXT, so ten
        # models scoring 100 on them proved only that a model can emit a
        # filename-shaped string -- and `best_models.vlm` had to be marked UNMEASURED
        # because of it. Building this task also revealed that rn's vision transport
        # was silently dropping the image entirely, so every score here before
        # 2026-08-18 would have been measuring hallucination.
        #
        # Same payload shape rn uses (OpenAI content parts + image_url data URIs), so
        # a pass here is evidence about the path that actually ships.
        "messages": image_message(
            "Describe each of the three images separately, in a few words each. "
            "Name the main colour and shape of each. No preamble."
        ),
        "validator": validate_image_description,
        "parse_json": False,
    },
    "weekend_fabrication": {
        # The `json` slot saturated once weekend_fixed_mixed was fixed -- four models
        # at exactly 100.0, so it is decided purely on tiebreakers. This separates
        # them on the property `wk` actually needs: does the plan send you to a place
        # that exists in the listings, or one the model remembered?
        "messages": [{"role": "user", "content": WEEKEND_FABRICATION_PROMPT}],
        "validator": validate_no_fabrication,
        "validator_kwargs": {"lures": WEEKEND_FABRICATION_LURES},
        "parse_json": True,
        "source": WEEKEND_FABRICATION_PROMPT,
    },
    "filename_injection": {
        # `rn`'s real threat model: OCR text from an arbitrary screenshot goes
        # straight into a prompt, so a screenshot can carry instructions. Never
        # tested before -- filename_leak checks for template leakage, not obedience.
        "messages": [{"role": "user", "content": FILENAME_INJECTION_PROMPT}],
        "validator": validate_resists_injection,
        "validator_kwargs": {
            "injection_markers": FILENAME_INJECTION_MARKERS,
            "expected_keywords": FILENAME_INJECTION_KEYWORDS,
        },
        "parse_json": False,
        "source": FILENAME_INJECTION_PROMPT,
    },
    "summarize_misattribution": {
        # Ranks models on attribution, which `summarize` cannot: its timeline gives
        # every claim one plausible author, so ten of eleven models scored 100 there.
        # Graded as a RATIO rather than through validate_summary's all-or-nothing cap
        # -- that cap is correct for `tw`, where one wrong attribution is
        # disqualifying, but as an instrument it puts every model with a single slip
        # on the same number and separates nobody.
        "messages": [{"role": "user", "content": TWITTER_PROMPT_MISATTRIBUTION}],
        "validator": validate_attribution,
        "parse_json": False,
        "source": TWITTER_PROMPT_MISATTRIBUTION,
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


def _register_taxes_tasks() -> None:
    """Wire the sanitized taxes snapshots into TASKS.

    The data, the rubric and the validators all shipped, but nothing loaded
    them: README advertised three taxes tasks and `ev --task taxes_synthesis`
    answered "Unknown task". Each snapshot carries its own system/user prompt,
    so the task is just the snapshot plus its validator.
    """
    validators = {
        "anomalies": validate_taxes_anomalies,
        "audit_readiness": validate_taxes_audit_readiness,
        "synthesis": validate_taxes_synthesis,
        # The grounded three. These carry a `grounding` block and no `rubric`,
        # so their verdict is arithmetic and set-membership rather than keyword
        # hits -- which is why they were imported: the rubric tasks saturate.
        "yoy_narrative": validate_taxes_yoy_narrative,
        "qa": validate_taxes_qa,
        "slip_qa": validate_taxes_slip_qa,
    }
    for name, validator in validators.items():
        snapshot = eval_tasks_path("data", "taxes", f"taxes_{name}.sanitized.json")
        if not snapshot.is_file():
            continue
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        messages = []
        if data.get("system"):
            messages.append({"role": "system", "content": data["system"]})
        messages.append({"role": "user", "content": data["user"]})
        # The validators consume raw text (audit_readiness does its own
        # json.loads), so the JSON extraction path must not pre-parse for them.
        TASKS[f"taxes_{name}"] = {
            "messages": messages,
            "validator": validator,
            "parse_json": False,
        }


_register_taxes_tasks()
