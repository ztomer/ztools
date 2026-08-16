"""Config getters - all config lookup functions."""

import sys
from typing import Dict, List

from .config_core import (
    _FALLBACK_MAX_TOKENS,
    _FALLBACK_MODEL,
    _FALLBACK_TIMEOUT,
    Task,
    _auto_load,
    _config,
    _model_configs_cache,
)
from .config_toml import load_config as _load_config_toml
from .paths import conf_path


def get_timeouts() -> Dict[str, int]:
    _auto_load()
    return _config.get("timeouts", {})


def get_max_tokens() -> Dict[str, int]:
    _auto_load()
    return _config.get("max_tokens", {})


def get_best_models() -> Dict[str, str]:
    _auto_load()
    return _config.get("best_models", {})


def get_best_model(task: Task) -> str:
    models = get_best_models()
    task_key = task.value if isinstance(task, Task) else task
    return models.get(task_key, _config.get("default_model", _FALLBACK_MODEL))


def get_timeout(task: Task) -> int:
    timeouts = get_timeouts()
    task_key = task.value if isinstance(task, Task) else task
    return timeouts.get(task_key, _FALLBACK_TIMEOUT)


def get_max_tokens_for_task(task: Task) -> int:
    tokens = get_max_tokens()
    task_key = task.value if isinstance(task, Task) else task
    return tokens.get(task_key, _FALLBACK_MAX_TOKENS)


def _config_family_for(architecture: str) -> str:
    """The conf/models/<family>.toml that serves an architecture, or "".

    Architectures carry version and variant suffixes ("<fam>3_5_moe",
    "<fam>4_unified", "<fam>_h") while the config files are named for the bare
    family, so the two have to be reconciled. Done by trimming one trailing segment
    at a time and taking the first name that has a file --

        <fam>3_5_moe -> <fam>3_5 -> <fam>3 -> <fam>   (conf/models/<fam>.toml)
        <fam>_variant -> <fam> -> ""                  (no file: genuinely unserved)

    rather than by a hand-written architecture-to-family table, which would need
    editing every time a vendor ships a new suffix and would silently mis-serve
    until someone noticed.
    """
    import re

    candidate = (architecture or "").lower()
    while candidate:
        for suffix in ("", "_versions"):
            if conf_path("models", f"{candidate}{suffix}.toml").exists():
                return candidate
        trimmed = re.sub(r"([_.-][^_.-]*|\d+)$", "", candidate)
        if trimmed == candidate:
            return ""
        candidate = trimmed
    return ""


def get_model_family(model: str) -> str:
    """Which conf/models/*.toml drives this model's prompts and quirks.

    Prefers the architecture `ev --capabilities` recorded, because the NAME does not
    reliably encode it. Vendors ship models under brand names that share an
    architecture with a differently-named family, and no substring of such a name
    reaches the config file written for it — so name matching sent those models to
    the built-in fallback prompts while the right config sat unused.

    Falls back to name matching when nothing has been recorded — an unprobed model,
    or any caller running without the signals file — so this never depends on the
    eval having been run.
    """
    if not model:
        return "default"

    from .model_caps import recorded_capability

    architecture = recorded_capability(model, "family")
    if architecture:
        mapped = _config_family_for(architecture)
        if mapped:
            return mapped

    model_lower = model.lower()
    if "qwopus" in model_lower:
        return "qwopus"
    elif "qwen" in model_lower:
        return "qwen"
    elif "gemma" in model_lower:
        return "gemma"
    elif "nemotron" in model_lower:
        return "nemotron"
    elif "laguna" in model_lower:
        return "laguna"
    elif "foundation" in model_lower:
        return "foundation"
    else:
        return "default"


def clear_model_config_cache():
    _model_configs_cache.clear()


def get_model_config(model: str) -> Dict:
    family = get_model_family(model)
    version = model.replace(family + "-", "") if family in model else ""
    if family in _model_configs_cache:
        family_config = _model_configs_cache[family]
        if version and "models" in family_config:
            version_config = family_config["models"].get(model, {})
            if version_config:
                merged = {k: v for k, v in family_config.items() if k != "models"}
                merged.update(version_config)
                merged["version"] = version
                return merged
        return family_config
    version_config_path = conf_path("models", f"{family}_versions.toml")
    config_path = conf_path("models", f"{family}.toml")
    if version_config_path.exists():
        loaded = _load_config_toml(version_config_path) or {}
        _model_configs_cache[family] = loaded
        if "models" in loaded:
            version_specific = loaded["models"].get(model, {})
            if version_specific:
                merged = {k: v for k, v in loaded.items() if k != "models"}
                merged.update(version_specific)
                merged["version"] = version
                return merged
        return loaded
    elif config_path.exists():
        _model_configs_cache[family] = _load_config_toml(config_path) or {}
    else:
        print(
            f"Warning: No model config found for '{family}', using built-in fallback prompts",
            file=sys.stderr,
        )
        _model_configs_cache[family] = {
            "name": family,
            "timeout": 300,
            "prompts": {
                "json": "Output JSON now. Use EXACT schema.",
                "weekend_fixed": (
                    "Output JSON now. Use EXACT schema: "
                    '{"fixed_activities": [{"name": "str", "location": "str", '
                    '"target_ages": "str", "price": "str", "weather": "str"}]}'
                    "\n\nExtract and format the family-friendly venues from this "
                    "list as JSON. Return ALL venues.\n\n{}\n\nUse the values "
                    "from the source data as-is. Output ONLY JSON. No extra text."
                ),
                "weekend_transient": (
                    "Output JSON now. Schema: "
                    '{"transient_events": [{"name": "str", "location": "str", '
                    '"target_ages": "str", "price": "str", "duration": "str", '
                    '"weather": "str", "day": "str"}]}\n\nFind events. Use exact '
                    "fields. Output ONLY JSON."
                ),
                "summarize": (
                    "Create a structured summary of this timeline. Start with a "
                    "brief TL;DR paragraph that captures the overall narrative. "
                    "Then organize events into topic sections with ## headers, "
                    "using bullet points. Include who (@user mentions), what "
                    "happened, and when. Use natural connecting language between "
                    "related events.\n\n{}\n"
                ),
                "filename": (
                    "Output ONLY the filename string (no JSON, no code "
                    "blocks). Use lowercase, underscores for spaces, no "
                    "special characters. Keep it under 50 characters.\n\n"
                    "TEXT: {}"
                ),
                "file_summary": "Output JSON array with path and desc fields.",
            },
            "key_mappings": {
                "event": "name",
                "title": "name",
                "activity": "name",
                "venue": "location",
                "address": "location",
                "place": "location",
                "age_group": "target_ages",
                "ages": "target_ages",
                "age_range": "target_ages",
                "cost": "price",
                "pricing": "price",
                "fee": "price",
                "type": "weather",
                "category": "weather",
            },
            "quirks": [
                {"type": "prefix", "pattern": "Output JSON now.", "reason": "Ensures clean JSON"}
            ],
            "top_keys": {
                "fixed": ["fixed_activities", "venues", "places", "activities", "items"],
                "transient": ["transient_events", "events", "activities", "recommendations"],
            },
            "field_mapping": {},
        }
    return _model_configs_cache.get(
        family, {"name": family, "prompts": {}, "key_mappings": {}, "quirks": []}
    )


def get_model_field_mapping(model: str) -> Dict[str, str]:
    config = get_model_config(model)
    return config.get("field_mapping", {})


def get_model_top_keys(model: str) -> Dict[str, List[str]]:
    config = get_model_config(model)
    return config.get(
        "top_keys",
        {
            "fixed": [
                "fixed_activities",
                "year_round_fixed_activities",
                "venues",
                "places",
                "activities",
                "items",
            ],
            "transient": ["transient_events", "events", "activities", "recommendations"],
        },
    )


def get_model_quirks(model: str) -> List[Dict]:
    config = get_model_config(model)
    return config.get("quirks", [])


def get_model_prompt(model: str, task: Task) -> str:
    config = get_model_config(model)
    prompts = config.get("prompts", {})
    task_key = task.value if isinstance(task, Task) else task
    return prompts.get(task_key, "")


def get_model_prompts_all(model: str) -> Dict[str, str]:
    config = get_model_config(model)
    return config.get("prompts", {})


def get_filename_models() -> List[str]:
    _auto_load()
    models = _config.get("filename_models", [])
    return models if models else ["foundation"]


def _default_fallback_chain() -> List[str]:
    """The known families, on-device first.

    Assembled from MODEL_FAMILIES rather than written out again so there is one list of
    families in the codebase, not two that drift. conf/config.toml overrides it; this
    exists so a config predating the key still resolves through the families instead of
    falling through to "biggest model on the roster".
    """
    from lib.llm.constants import DEFAULT_MODEL, MODEL_FAMILIES

    return [DEFAULT_MODEL] + [f for f in MODEL_FAMILIES if f != DEFAULT_MODEL]


def get_model_fallback_chain() -> List[str]:
    """Families lib/model_resolve.py tries when the configured model is gone."""
    _auto_load()
    chain = _config.get("model_fallback_chain", [])
    return list(chain) if chain else _default_fallback_chain()


def get_filename_prompt() -> str:
    _auto_load()
    prompts = _config.get("prompts", {})
    return prompts.get("filename", "Give a short summary of: {text}")
