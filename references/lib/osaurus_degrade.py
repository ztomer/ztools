"""Degrade paths for a model call, each with a stated reason.

Split out of osaurus_lib to keep that file under the repo's 500-line limit. Both
functions answer the same question in different ways: the dependency did not do what
was asked, so what is the honest next move?

    _streamed_with_guard   the model is reasoning past the point of no return
    _try_foundation        the server is unreachable, but an on-device model is not

They are imported by NAME into osaurus_lib and called there as bare globals, so
`patch("lib.osaurus_lib._try_foundation", ...)` still binds what the call site reads.
Moving a function out of a module silently breaks that: the moved function resolves
its own globals in the new module, and a patch applied to the old name binds
something nobody reads. Hence the re-import rather than a qualified call.
"""

from typing import Optional

from lib.logging_config import osaurus_logger as logger
from lib.osaurus_output import clean_output, extract_json  # noqa: F401

from .config import get_timeout
from .osaurus_models import FALLBACK_MODEL


def _streamed_with_guard(model, messages, max_tokens, host, port, temperature, timeout, task):
    """Stream one completion under the reasoning-overrun guard.

    Returns the fields to merge into a result, or None to let the caller fall back to
    the blocking request. None is returned for ANY transport error, because the
    blocking path is the one that knows how to substitute a deleted model and how to
    fall back to on-device Foundation -- capabilities the stream does not have and
    which matter more than the time the guard would save.
    """
    from lib.llm.streaming import stream_with_overrun_guard

    streamed = stream_with_overrun_guard(
        model,
        messages,
        max_tokens=max_tokens,
        host=host,
        port=port,
        temperature=temperature,
        timeout=timeout or get_timeout(task),
    )
    if streamed.get("error"):
        return None
    fields = {
        "content": clean_output(streamed.get("content") or ""),
        "reasoning_content": streamed.get("reasoning_content") or "",
        "finish_reason": streamed.get("finish_reason") or "",
        "error": None,
    }
    if streamed.get("aborted"):
        # An abort is a FAILURE with a stated reason, not an empty answer. Recording
        # it as an error is what stops a scorer treating the blank content as the
        # model's considered response and scoring it as bad output.
        fields["aborted"] = True
        fields["error"] = f"Reasoning overrun: {streamed.get('abort_reason', '')}"
    return fields



def _try_foundation(
    use_foundation: Optional[bool], messages, parse_json: bool, result: dict
) -> bool:
    """Attempt the on-device Foundation Models fallback.

    use_foundation: True = force, False = never, None = only if available.
    On success, fills ``result`` and returns True.
    """
    if use_foundation is False:
        return False
    try:
        from lib.foundation_lib import call_foundation, foundation_available
    except Exception:
        return False
    if not foundation_available():
        return False
    system = next((m["content"] for m in messages if m.get("role") == "system"), "")
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    try:
        raw = call_foundation(system, user, parse_json=parse_json)
    except Exception as e:
        logger.warning(f"Foundation fallback failed: {e}")
        return False
    if not raw:
        return False
    result["content"] = raw
    result["model"] = FALLBACK_MODEL
    result["served_by_foundation"] = True
    if parse_json:
        result["parsed"] = extract_json(raw)
    logger.info("Served by on-device Foundation Models")
    return True

