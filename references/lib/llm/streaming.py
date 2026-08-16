"""Watch a generation as it arrives, and stop only the runs that cannot finish.

Reasoning models stream chain of thought into `reasoning_content` and leave `content`
empty until they close the think block. On an easy prompt they close it after a few
hundred tokens and answer. On a hard one some never close it: they spend the entire
budget thinking and return finish_reason=length with nothing to score.

The blunt fix is to cap the budget so they are forced to stop early. That works, and
it is wrong: it removes thinking from every request that was already succeeding,
including the ones where thinking is what makes the answer good. It also settles by
configuration a question that should be settled by measurement -- is this model better
with thinking on this task?

So the rule here is deliberately narrow. Abort when the model has spent
REASONING_OVERRUN_FRACTION of its budget thinking and produced NO content, because at
that point the remaining budget cannot hold an answer and the run is already lost.
Below that line, think as long as you like.

That distinction is the whole design:

    reasoning for 6,000 of 16,000 tokens, then answers   -> allowed, and useful
    reasoning past 12,000 of 16,000 with content empty   -> aborted, already doomed

An abort is RECORDED, not swallowed. "This model overran on this task" is an eval
result; discovering it again on every run is not.
"""

import json
import os
from typing import Callable, Dict, List, Optional

import requests

from lib.llm.constants import DEFAULT_HOST, DEFAULT_PORT

#: Fraction of the output budget a model may spend on reasoning before an empty
#: `content` is treated as terminal. Not a tuning knob for speed: below this line a
#: run can still recover, above it the remaining budget cannot hold an answer.
REASONING_OVERRUN_FRACTION = float(os.environ.get("LLM_REASONING_OVERRUN_FRACTION", "0.75"))

#: Rough chars-per-token, only ever used to turn streamed characters into a token
#: estimate for the fraction above. An estimate is adequate because the threshold is
#: a fraction of a budget, not a boundary anything is scored against.
CHARS_PER_TOKEN = int(os.environ.get("TWITTER_CHARS_PER_TOKEN", "3"))

SSE_DATA_PREFIX = "data: "
SSE_DONE = "[DONE]"


def _deltas(line: str) -> Optional[Dict]:
    """Parse one SSE line into its delta dict, or None if it carries no data."""
    if not line or not line.startswith(SSE_DATA_PREFIX):
        return None
    payload = line[len(SSE_DATA_PREFIX):].strip()
    if not payload or payload == SSE_DONE:
        return None
    try:
        chunk = json.loads(payload)
    except ValueError:
        return None
    choices = chunk.get("choices") or []
    if not choices:
        return None
    return choices[0]


def stream_with_overrun_guard(
    model: str,
    messages: List[Dict],
    max_tokens: int,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    temperature: float = 0.0,
    timeout: int = 600,
    session_factory: Optional[Callable] = None,
) -> Dict:
    """Stream a completion, aborting a reasoning overrun as soon as it is certain.

    Returns the usual result shape plus:
        aborted            True when the guard stopped the run
        abort_reason       why, in a sentence, when it did
        reasoning_content  what the model was thinking about when it did

    Never raises for a transport problem: it reports the error the same way `call`
    does, because a failed request during a ten-hour sweep must not end the sweep.
    """
    url = host.rstrip("/") if "://" in host else f"http://{host}:{port}"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    result = {
        "model": model,
        "content": "",
        "reasoning_content": "",
        "finish_reason": "",
        "error": None,
        "aborted": False,
        "abort_reason": "",
    }
    budget_chars = max(1, int(max_tokens * REASONING_OVERRUN_FRACTION * CHARS_PER_TOKEN))
    make_session = session_factory or requests.Session

    try:
        with make_session() as session:
            response = session.post(
                f"{url}/v1/chat/completions",
                json=payload,
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=timeout,
            )
            if response.status_code != 200:
                result["error"] = f"HTTP {response.status_code}"
                return result

            for raw in response.iter_lines(decode_unicode=True):
                choice = _deltas(raw if isinstance(raw, str) else (raw or b"").decode())
                if choice is None:
                    continue
                delta = choice.get("delta") or {}
                result["content"] += delta.get("content") or ""
                result["reasoning_content"] += delta.get("reasoning_content") or ""
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]

                # The only abort condition. Content having arrived means the model
                # closed its think block, so however long it thought, it is answering
                # and must be left alone.
                if not result["content"] and len(result["reasoning_content"]) > budget_chars:
                    spent = len(result["reasoning_content"]) // CHARS_PER_TOKEN
                    result["aborted"] = True
                    result["finish_reason"] = "aborted_reasoning_overrun"
                    result["abort_reason"] = (
                        f"~{spent} tokens of reasoning with no content, past "
                        f"{REASONING_OVERRUN_FRACTION:.0%} of a {max_tokens}-token "
                        f"budget: the rest cannot hold an answer"
                    )
                    response.close()
                    break
    except requests.exceptions.Timeout:
        result["error"] = "Timeout"
    except requests.exceptions.ConnectionError:
        result["error"] = "Connection failed"
    except Exception as exc:  # noqa: BLE001 - reported, never raised into a sweep
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
