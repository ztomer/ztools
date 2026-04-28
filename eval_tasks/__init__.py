# Task definitions for model evaluation
# Import from model_eval for backwards compatibility

from typing import Dict, Any, List

# Weekend planner prompts (hardcoded for eval)
WEEKEND_SYS_TRANSIENT = """You are a helpful Weekend Activity Planner for a family with young children.

Find events, activities for {date_range} in {location}. Kids ages {age_range}.

Use EXACT schema:
{{"transient_events": [{{"name": "str", "location": "str", "target_ages": "str", "price": "str", "duration": "str", "weather": "str", "day": "str"}}]}}"""

WEEKEND_USR_TRANSIENT = """Find weekend activities in NYC for ages 3-7.

Saturday: Music Festival at Central Park, Food Fair at Downtown, Art Show at Museum
Sunday: Concert at Stadium, Picnic at Riverside, Swimming at Aquatics Center"""

WEEKEND_SYS_FIXED = """You are a helpful Weekend Activity Planner for a family with young children.

Find popular family-friendly venues in {location} for kids ages {age_range}.

Use EXACT schema: {{"fixed_activities": [{{"name": "str", "location": "str", "target_ages": "str", "price": "str", "weather": "str"}}]}}"""

WEEKEND_USR_FIXED = """Find venues in NYC for ages 3-7."""

# Image renamer prompts
FILENAME_SYS = "You are a file naming assistant."
FILENAME_USR = "Rename: image_20240101_123456.jpg"

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
            {"role": "user", "content": "Files in directory:\n- eval_lib.py: evaluates models\n- validators.py: validation logic\n- config.py: configuration\n\nSummarize what each file does."},
        ],
        "validator": None,
        "parse_json": False,
    },
}


def load_tasks_from_config(model: str) -> Dict[str, Any]:
    """Load tasks from config.yaml or use defaults."""
    from pathlib import Path
    import yaml
    from lib.validators_lib import validate_detailed_json, validate_summary, validate_filename, validate_file_summary
    
    tasks = TASKS.copy()
    
    # Try to load from config
    config_path = Path("conf/config.yaml")
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            
            for task_name, task_cfg in tasks.items():
                validator = None
                if task_cfg.get("parse_json"):
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