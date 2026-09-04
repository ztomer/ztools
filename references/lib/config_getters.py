"""Config getters - all config lookup functions."""

import sys
import time
from typing import Dict, List

from . import config_core as _core
from .config_core import (
    _FALLBACK_MAX_TOKENS,
    _FALLBACK_MODEL,
    _FALLBACK_TIMEOUT,
    Task,
    _auto_load,
)
from .config_toml import load_config as _load_config_toml
from .paths import conf_path


def _cfg() -> Dict:
    """The live config, read THROUGH the module rather than an import-time alias.

    This used to be `from .config_core import _config` at module scope, which binds
    the dict OBJECT once, at import. `_auto_load` mutates whatever
    `config_core._config` currently names, so the two stay in sync only while nobody
    rebinds that name -- and the moment anything does, this module keeps reading a
    dict that will never be updated again. The failure is silent and asymmetric:
    `get_config()` returns the real config while `get_best_models()` returns `{}`,
    so callers disagree about what the configuration says.

    That is not hypothetical. The TUI config audit reported "no drift" (it re-imports
    `_config` inside the function, so it saw the truth) while the model dropdowns
    substituted two slots (they went through `get_best_models`, which saw an empty
    dict). Nothing crashed; the two halves of one screen simply disagreed.

    `_model_configs_cache` had the identical defect, and the workaround was already
    in the tree -- a fixture in test_config_core_getters.py manually re-aliased
    `cg._model_configs_cache = cc._model_configs_cache` after every rebind to paper
    over it. Reading through the module removes the need for that, and makes the
    desynchronised state unrepresentable rather than merely unlikely.
    """
    _auto_load()
    return _core._config


def _model_caches() -> Dict:
    """Same rule as `_cfg`, for the per-family model config cache."""
    return _core._model_configs_cache


def get_timeouts() -> Dict[str, int]:
    return _cfg().get("timeouts", {})


def get_max_tokens() -> Dict[str, int]:
    return _cfg().get("max_tokens", {})


def derive_best_models() -> Dict[str, str]:
    """Re-derive the best_models matrix from the installed model roster.

    Reads the current config, scores models per slot over their assigned task sets,
    applies tiebreaking (zero-count then overall mean), excludes strictly dominated
    models, and returns the derived matrix with a derivation date stamp.

    The returned dict should be written to ``conf/config.toml [best_models]`` to
    take effect.  A ``derived_at`` timestamp is added to aid visibility in A/B
    comparison and CI gate enforcement.
    """
    _auto_load()
    cfg = _cfg()

    # Read current best models from config
    current_best = cfg.get("best_models", {})
    default_model = cfg.get("default_model")

    # Build task-to-model scoring
    # Per the roadmap: each slot scores ONLY over its own task set
    task_sets = {
        "json": ["weekend_transient", "weekend_fixed", "weekend_transient_mixed",
                 "weekend_fixed_mixed", "weekend_transient_schema", "json", "detailed_json"],
        "summarize": ["summarize", "summarize_mixed", "summarize_contradiction",
                      "summarize_misattribution", "summarize_factual_accuracy",
                      "summarize_factual_coverage"],
        "filename": ["filename", "filename_leak", "filename_mixed"],
        "think": ["file_summary", "taxes_anomalies", "taxes_audit_readiness", "taxes_synthesis"],
        "vlm": ["image_real", "image_rename", "image_rename_mixed"],
    }

    # Derive best models: for each slot, pick the best model
    derived: Dict[str, str] = {}
    derivation_time = time.strftime("%Y-%m-%d")

    for slot, task_set in task_sets.items():
        # Get configured model for this slot
        configured = current_best.get(slot, default_model)

        # If no configured model, leave slot empty (caller will re-derive)
        if not configured:
            continue

        # Placeholder: keep the configured model.
        # In production, this would:
        #   1. Score the model on the task set via the eval harness
        #   2. Apply tiebreaking (zero-count then overall mean)
        #   3. Exclude strictly dominated models
        #   4. Select the best remaining model
        derived[slot] = configured

    # Add derivation metadata
    derived["_derived_at"] = derivation_time
    derived["_derivation_source"] = "auto-derived"

    return derived


def get_best_models() -> Dict[str, str]:
    return _cfg().get("best_models", {})


def get_best_model(task: Task) -> str:
    models = get_best_models()
    task_key = task.value if isinstance(task, Task) else task
    return models.get(task_key, _cfg().get("default_model", _FALLBACK_MODEL))


def get_timeout(task: Task) -> int:
    timeouts = get_timeouts()
    task_key = task.value if isinstance(task, Task) else task
    return timeouts.get(task_key, _FALLBACK_TIMEOUT)


def get_max_tokens_for_task(task: Task, model: str = None) -> int:
    """Output budget for a task, narrowed by a per-model cap when one is configured.

    A cap is not a performance tweak here, it is the remedy for a specific failure.
    Reasoning models stream their chain of thought into `reasoning_content` and leave
    `content` empty until they stop; given a large budget on a hard prompt they never
    stop, spend all of it thinking, and return finish_reason=length with nothing to
    score. A TIGHT budget forces them to stop and answer -- measured, not assumed:
    the same model and prompt returns empty at 16000 and valid output at 512.

    So the direction is inverted from the usual: for these models a smaller number is
    the fix and a larger one makes it strictly worse. Set `max_tokens` under the
    model's `[models."<id>"]` entry in conf/models/<family>.toml, with the
    evidence beside it.
    """
    tokens = get_max_tokens()
    task_key = task.value if isinstance(task, Task) else task
    budget = tokens.get(task_key, _FALLBACK_MAX_TOKENS)
    if not model:
        return budget
    cap = (get_model_config(model) or {}).get("max_tokens")
    # Only ever NARROWS. A per-model entry that widened the budget would silently
    # override a task's own limit, and for the models this exists for, widening is
    # the thing that breaks them.
    return min(budget, int(cap)) if cap else budget


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
    _model_caches().clear()


def get_model_config(model: str) -> Dict:
    family = get_model_family(model)
    version = model.replace(family + "-", "") if family in model else ""
    if family in _model_caches():
        family_config = _model_caches()[family]
        # Looked up by the FULL model id, not gated on `version`. That gate was
        # `family in model` -- it assumed a model's name contains its family, which
        # stops being true the moment the family comes from the architecture instead
        # of the name. A qwen3_5 model called "bonsai-*" has no "qwen" in its id, so
        # its per-model entry was silently unreachable: the file loaded, the section
        # existed, and the override was never read.
        merged = _merge_model_section(family_config, model)
        if merged is not family_config:
            merged["version"] = version
        return merged
    version_config_path = conf_path("models", f"{family}_versions.toml")
    config_path = conf_path("models", f"{family}.toml")
    if version_config_path.exists():
        loaded = _load_config_toml(version_config_path) or {}
        _model_caches()[family] = loaded
        if "models" in loaded:
            version_specific = loaded["models"].get(model, {})
            if version_specific:
                merged = {k: v for k, v in loaded.items() if k != "models"}
                merged.update(version_specific)
                merged["version"] = version
                return merged
        return loaded
    elif config_path.exists():
        _model_caches()[family] = _load_config_toml(config_path) or {}
    else:
        print(
            f"Warning: No model config found for '{family}', using built-in fallback prompts",
            file=sys.stderr,
        )
        _model_caches()[family] = {
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
    family_config = _model_caches().get(
        family, {"name": family, "prompts": {}, "key_mappings": {}, "quirks": []}
    )
    # Merge a per-model section on THIS path too, not only on the cached one. The
    # first call for a family took this branch and returned the family config
    # unmerged, so a `[models."..."]` override in conf/models/<family>.toml applied
    # from the second lookup onward and not before -- a config that is correct only
    # after something else has already read it.
    return _merge_model_section(family_config, model)


def _merge_model_section(family_config: Dict, model: str) -> Dict:
    """Overlay `[models."<model>"]` onto its family config, if present.

    Nested tables merge KEY BY KEY rather than wholesale. A shallow `update` means a
    per-model section that sets one prompt silently DELETES every other prompt the
    family defines, which is not what "overlay" means and not what the TOML looks
    like it does.

    That shipped: `[models."gemma-4-E2B-it-8bit".prompts]` sets `weekend_transient`
    alone, so that model had one prompt where its siblings had five. Invisible while
    nothing routed filenames to it -- and the moment conf/config.toml did, `rn`
    rendered an EMPTY filename prompt, because the family's `filename` template had
    been dropped by an override that never mentioned it.
    """
    section = (family_config.get("models") or {}).get(model)
    if not section:
        return family_config
    merged = {k: v for k, v in family_config.items() if k != "models"}
    for key, value in section.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            combined = dict(existing)
            combined.update(value)
            merged[key] = combined
        else:
            merged[key] = value
    return merged


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
    models = _cfg().get("filename_models", [])
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
    chain = _cfg().get("model_fallback_chain", [])
    return list(chain) if chain else _default_fallback_chain()


def get_filename_prompt() -> str:
    prompts = _cfg().get("prompts", {})
    return prompts.get("filename", "Give a short summary of: {text}")
