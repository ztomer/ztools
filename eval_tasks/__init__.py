# Task definitions for model evaluation
# Import from model_eval for backwards compatibility

import json
from pathlib import Path
from typing import Dict, Any, List


def _load_taxes_snapshot(short_name: str) -> Dict[str, Any]:
    """Load a sanitized taxes-* prompt snapshot ported from
    github.com/ztomer/Taxes (scripts/snapshot_eval_prompts.py
    with --sanitize). Returns the rendered (system, user) pair +
    the rubric the source repo's eval applies."""
    fp = Path(__file__).resolve().parent / "data" / "taxes" / f"taxes_{short_name}.sanitized.json"
    if not fp.exists():
        return {}
    return json.loads(fp.read_text(encoding="utf-8"))

# Weekend planner prompts (hardcoded for eval)
WEEKEND_SYS_TRANSIENT = """You are a helpful Weekend Activity Planner for a family with young children.

Find events, activities for {date_range} in {location}. Kids ages {age_range}.

Use EXACT schema:
{{"transient_events": [{{"name": "str", "location": "str", "target_ages": "str", "price": "str", "duration": "str", "weather": "str", "day": "str"}}]}}"""

WEEKEND_USR_TRANSIENT = """Find weekend activities in NYC for ages 3-7.

Use ONLY these specific events:
- Music Festival at Central Park, Saturday 10am-6pm
- Food Fair at Downtown, Saturday 11am-8pm
- Art Show at Museum, Saturday-Sunday 9am-5pm
- Concert at Stadium, Sunday 7pm-11pm

Find additional events for Saturday and Sunday."""

WEEKEND_SYS_FIXED = """You are a helpful Weekend Activity Planner for a family with young children.

Find popular family-friendly venues in {location} for kids ages {age_range}.

Use EXACT schema: {{"fixed_activities": [{{"name": "str", "location": "str", "target_ages": "str", "price": "str", "weather": "str"}}]}}"""

WEEKEND_USR_FIXED = """Find venues in NYC for ages 3-7.

Venues should be different from the transient events (festivals, concerts, fairs).
Focus on: museums, parks, libraries, rec centers, playgrounds."""

# Image renamer prompts
FILENAME_SYS = "You are a file naming assistant. Output ONLY the new filename, no code blocks, no quotes, no explanation."
FILENAME_USR = "Rename this file to a descriptive name: image_20240101_123456.jpg"

# Summarize prompts
SUMMARIZE_SYS = "You are a helpful assistant that summarizes content."
SUMMARIZE_USR = "Summarize: Test content for validation."

# Build tasks dict
TASKS = {
    "weekend_transient": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_TRANSIENT},
            {"role": "user", "content": WEEKEND_USR_TRANSIENT},
        ],
        "validator": None,  # Set at runtime
        "parse_json": True,
        "source": WEEKEND_USR_TRANSIENT,
    },
    "weekend_fixed": {
        "messages": [
            {"role": "system", "content": WEEKEND_SYS_FIXED},
            {"role": "user", "content": WEEKEND_USR_FIXED},
        ],
        "validator": None,
        "parse_json": True,
        "source": WEEKEND_USR_FIXED,
    },
    "filename": {
        "messages": [
            {"role": "system", "content": FILENAME_SYS},
            {"role": "user", "content": FILENAME_USR},
        ],
        "validator": None,
        "parse_json": False,
    },
    "summarize": {
        "messages": [
            {"role": "system", "content": SUMMARIZE_SYS},
            {"role": "user", "content": SUMMARIZE_USR},
        ],
        "validator": None,
        "parse_json": False,
    },
    "file_summary": {
        "messages": [
            {"role": "user", "content": "Files in directory:\n- alpha.py: core logic\n- beta.py: utilities\n- gamma.py: settings\n\nDO NOT infer from filename. Describe each file based ONLY on its actual purpose (infer: parse, validate, convert, etc). Output as JSON: {\"filename\": \"description\"}."},
        ],
        "validator": None,
        "parse_json": False,
    },
}


# ─── Taxes tasks (ported 2026-05-17) ──────────────────────────────────────
#
# Three real-world tax-prep prompts from github.com/ztomer/Taxes,
# sanitized (dollar amounts bucketed, no PII) and vendored under
# eval_tasks/data/taxes/. Substantially harder than the other tasks
# here: 2.7-7.5kB user prompts, dense tax-domain context, expect
# specific cross-border findings (T1135, Form 106, box 38, etc.).
#
# These exercise long-context reasoning + domain-knowledge grounding
# in a way the existing tasks don't. Good for filtering "this model
# might be useful for a real workload" from "this model can summarize
# four bullet points."
#
# To add tasks dynamically (so model_eval --list-tasks finds them):
for _t in ("anomalies", "audit_readiness", "synthesis"):
    _snap = _load_taxes_snapshot(_t)
    if not _snap:
        continue
    TASKS[f"taxes_{_t}"] = {
        "messages": [
            {"role": "system", "content": _snap["system"]},
            {"role": "user",   "content": _snap["user"]},
        ],
        "validator": None,  # set in load_tasks_from_config
        "parse_json": _t == "audit_readiness",  # only audit returns JSON
        "source": _snap["user"][:120] + "…",
    }


def load_tasks_from_config(model: str) -> Dict[str, Any]:
    """Load tasks from config.yaml or use defaults."""
    from pathlib import Path
    import yaml
    from lib.validators_lib import validate_detailed_json, validate_summary, validate_filename, validate_file_summary
    from lib.validators.taxes_validator import (
        validate_taxes_anomalies, validate_taxes_audit_readiness,
        validate_taxes_synthesis,
    )

    tasks = TASKS.copy()

    # Try to load from config
    config_path = Path(__file__).resolve().parent.parent / "conf" / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}

            taxes_validators = {
                "taxes_anomalies":       validate_taxes_anomalies,
                "taxes_audit_readiness": validate_taxes_audit_readiness,
                "taxes_synthesis":       validate_taxes_synthesis,
            }
            for task_name, task_cfg in tasks.items():
                if task_name in taxes_validators:
                    validator = taxes_validators[task_name]
                elif task_cfg.get("parse_json"):
                    validator = validate_detailed_json
                elif task_name == "filename":
                    validator = validate_filename
                elif task_name == "file_summary":
                    validator = validate_file_summary
                else:
                    validator = validate_summary

                tasks[task_name]["validator"] = validator
        except Exception:
            pass

    return tasks


def get_tasks(model: str = None) -> Dict[str, Any]:
    """Get tasks dict with validators set."""
    return load_tasks_from_config(model or "default")