"""The multi-phase weekend pipeline: extract -> draft -> refine -> structure.

Extracted from weekend/llm.py to stay under the repo's 500-line cap.
weekend/llm.py re-exports these names, so existing imports keep working (the
same shim pattern as lib/config.py).

Class C2c: these four phases form a CARRIER CHAIN. Each one must pass
DATES/PRICE/AGES/LOCATION through verbatim -- the predecessor narrowed the
payload at every step ("name, location, description", then "name +
description"), so an event's real dates were gone two phases before the schema
that asked for them, and every date column rendered blank.
"""

import json


def _llm():
    """Lazy accessor for weekend.llm.

    weekend.llm re-exports these phase functions, so a module-level import here
    would be circular. The file already used function-local imports for
    weekend.prompts; this keeps that style.
    """
    from weekend import llm

    return llm

def _load_extract_signals():
    try:
        if _llm().EXTRACT_SIGNALS_PATH.exists():
            return json.loads(_llm().EXTRACT_SIGNALS_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_extract_signals(signals):
    _llm().EXTRACT_SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _llm().EXTRACT_SIGNALS_PATH.write_text(json.dumps(signals, indent=2, sort_keys=True))


def _extract_group_prompt(lines, source_type):
    from weekend.prompts import build_source_extract_prompt

    return build_source_extract_prompt("\n".join(lines), source_type)


def extract_sources(raw_text, source_type, model_name="default", use_foundation=False):
    signals = _load_extract_signals()
    per_type = signals.setdefault(model_name or "default", {}).setdefault(source_type, {})
    batch_size = per_type.get("batch_size", _llm().DEFAULT_BATCH_SIZE)
    phase_key = f"extract_sources/{source_type}"

    lines = [line for line in raw_text.split("\n") if line.startswith("- ")]
    if not lines:
        return raw_text

    results = []
    streak = 0
    i = 0
    while i < len(lines):
        chunk = lines[i : i + batch_size]
        prompt = _extract_group_prompt(chunk, source_type)
        result = _llm()._call_llm(
            "",
            prompt,
            timeout=_llm().PHASE_TIMEOUT_EXTRACT,
            phase_key=phase_key,
            use_foundation=use_foundation,
        )

        if result:
            results.append(result)
            streak += 1
            i += batch_size
            if streak >= _llm().BATCH_GROWTH_STREAK_LIMIT and batch_size < _llm().MAX_BATCH_SIZE:
                batch_size += 1
                per_type["batch_size"] = batch_size
                _save_extract_signals(signals)
        else:
            per_type["timeouts"] = per_type.get("timeouts", 0) + 1
            batch_size = max(batch_size // 2, 1)
            per_type["batch_size"] = batch_size
            _save_extract_signals(signals)
            streak = 0
            if batch_size == 1:
                results.append(chunk[0])
                i += 1
                batch_size = per_type.get("batch_size", _llm().DEFAULT_BATCH_SIZE)

    if not results:
        return raw_text
    return "\n".join(results)


def draft_activities(
    weather_condensed,
    cleaned_sources,
    source_type,
    location,
    age_range,
    date_range,
    use_foundation=False,
    year=None,
):
    from weekend.prompts import build_draft_prompt

    prompt = build_draft_prompt(
        weather_condensed, cleaned_sources, source_type, location, age_range, date_range,
        year=year,
    )
    return _llm()._call_llm(
        "",
        prompt,
        timeout=_llm().PHASE_TIMEOUT_DRAFT,
        phase_key=f"draft_activities/{source_type}",
        use_foundation=use_foundation,
    )


def refine_draft(draft_text, use_foundation=False):
    from weekend.prompts import build_refine_prompt

    result = _llm()._call_llm(
        "",
        build_refine_prompt(draft_text),
        timeout=_llm().PHASE_TIMEOUT_REFINE,
        phase_key="refine_draft",
        use_foundation=use_foundation,
    )
    return result if result else draft_text


def structure_to_json(
    text, source_type, age_range, weather_condensed="", use_foundation=False, year=None
):
    from weekend.prompts import build_structure_system_prompt, build_structure_user_prompt

    sys_prompt = build_structure_system_prompt(
        source_type, age_range, weather_condensed, year=year
    )
    usr_prompt = build_structure_user_prompt(text)
    return _llm()._call_llm(
        sys_prompt,
        usr_prompt,
        timeout=_llm().PHASE_TIMEOUT_STRUCTURE,
        parse_json=True,
        phase_key=f"structure_to_json/{source_type}",
        use_foundation=use_foundation,
    )

