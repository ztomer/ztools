"""Keep what the model actually said, not just the score it earned.

The eval recorded a score, a failure reason and a one-line evidence string, and
threw the output away. That makes every question about a SCORER unanswerable
without re-running the model.

It cost a full day. `summarize_factual_coverage` was failed by all five models
that produced results, and deciding whether that was the models or the metric
needed one look at what a model had written. The output was gone, and the only
way back to it was another sweep -- ten hours of GPU on a machine that can run
exactly one model at a time. The diagnosis had to be reconstructed from
synthetic controls instead.

A few KB per task removes that entire category of dead end, so this is on by
default. The prompts are fixtures in eval/tasks_prompts.py and sanitized taxes
snapshots, so nothing written here is the user's own data.
"""

import os
import re
from pathlib import Path

from eval.report_core import default_eval_dir

# Enough to diagnose a scorer; short of a model that emits a novel of reasoning.
MAX_SAVED_CHARS = int(os.environ.get("EVAL_MAX_SAVED_OUTPUT", "200000"))


def outputs_enabled() -> bool:
    """On unless explicitly disabled, because the failure mode is silent loss."""
    return os.environ.get("EVAL_SAVE_OUTPUTS", "1") not in ("0", "false", "no")


def outputs_dir(eval_dir: Path = None) -> Path:
    """Where saved outputs live. Overridable so tests never touch the real one."""
    override = os.environ.get("EVAL_OUTPUT_DIR")
    if override:
        return Path(override)
    return (eval_dir or default_eval_dir()) / "outputs"


def _safe(name: str) -> str:
    """Model identifiers carry dots and slashes; keep them out of the path."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._-") or "unnamed"


def save_output(
    model: str,
    task: str,
    result: dict,
    score: int,
    failure_reason: str = "",
    eval_dir: Path = None,
) -> Path | None:
    """Write one model's raw answer for one task, with its verdict in the header.

    Returns the path written, or None when saving is off or there was nothing to
    save. Never raises: losing an output is bad, but failing a ten-hour eval run
    because a disk was full would be worse.
    """
    if not outputs_enabled():
        return None
    content = (result or {}).get("content")
    reasoning = (result or {}).get("reasoning_content") or ""
    error = (result or {}).get("error") or ""
    if not content and not reasoning and not error:
        return None

    try:
        target = outputs_dir(eval_dir) / _safe(model)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{_safe(task)}.txt"
        body = str(content or "")[:MAX_SAVED_CHARS]
        header = [
            f"model: {model}",
            f"task: {task}",
            f"score: {score}",
            f"failure: {failure_reason}",
            f"error: {error}",
            f"chars: {len(str(content or ''))}",
        ]
        if reasoning:
            # Kept separate: for thinking models the visible answer is often
            # short and the reasoning is where a format failure is explained.
            header.append(f"reasoning_chars: {len(reasoning)}")
        text = "\n".join(header) + "\n---\n" + body
        if reasoning:
            text += "\n--- reasoning ---\n" + str(reasoning)[:MAX_SAVED_CHARS]
        path.write_text(text, encoding="utf-8")
        return path
    except OSError:
        return None
