"""Measure how fast a model ingests a prompt, per model, on this host.

Every context budget in the suite derives from this number, and getting it
honestly took three attempts. 40 chars/sec was assumed (35-90x too low). 85
chars/sec was derived from whole-call time (17x too low: decode dominates a
generation-heavy task, so that timing measures the wrong quantity). Then the
`max_tokens=1` probe read 1,322 and later 3,789 chars/sec for the same model on
the same host -- because it sent byte-identical filler every time and the server
served it from its prefix cache. Measured against identical text the "rate"
climbs to 140,000 chars/sec, which is not a speed, it is a cache hit.

So the probe must be unrepeatable: a nonce leads the prompt, and since a prefix
cache keys on the prefix, nothing downstream can be reused. With that, gemma-4-12b
measures 1,045-1,237 chars/sec across repeated probes instead of climbing.
"""

import os
import time
import uuid

from lib.model_caps import CHARS_PER_TOKEN, probe_context_window
from lib.osaurus_lib import call

from eval.signals import DEFAULT_EVAL_TIMEOUT, _load_eval_signals, _save_eval_signals

# Prefill probe sizing. Big enough that per-request overhead does not dominate,
# small enough to cost a few seconds per model.
PREFILL_PROBE_CHARS = int(os.environ.get("EVAL_PREFILL_PROBE_CHARS", "20000"))
_PROBE_LINE = "[@SomeHandle | 08:15]: A reasonably typical sentence about a launch today.\n"
# Above this, the transport did not really ingest anything: a mock, a stub, or a
# prefix-cache hit. The fastest genuine measurement on this host is ~3,500
# chars/sec (the 35B MoE); a cache hit returns 65,000-140,000. The bound sits
# well above any real model and well below any cache hit, and only ever
# discards a measurement -- it never invents one.
MAX_PLAUSIBLE_PREFILL_RATE = int(os.environ.get("EVAL_MAX_PLAUSIBLE_PREFILL", "20000"))


def _probe_size_for(model: str) -> int:
    """Probe length that fits inside this model's own window.

    A 20K-char probe is 6.6K tokens, which foundation (4096, and that covers
    output too) answers with an HTTP 500. Sizing the probe to the model keeps
    the measurement possible on small-window models instead of returning None
    and quietly falling back to the global floor.
    """
    window = probe_context_window(model)
    if not window:
        return PREFILL_PROBE_CHARS
    return max(2000, min(PREFILL_PROBE_CHARS, int(window * CHARS_PER_TOKEN * 0.6)))


def measure_prefill_rate(model: str, host: str, port: int, transport=None) -> float | None:
    """Characters per second this model ingests, measured with max_tokens=1.

    Whole-call timing cannot answer this: on a generation-heavy task decode
    dominates, and the rate derived that way came out 17x below what the same
    model measured in isolation. Only `max_tokens=1` isolates ingestion, which
    is the quantity a context budget actually depends on.

    `transport` is the caller's own `call`, so a test that patches `eval.run.call`
    covers the probe too. Importing `call` here by value created a second seam
    that mocks did not reach, and eighteen mocked tests silently hit the live
    server through it until the conftest gate started failing on connections.

    Returns None when the probe cannot run; callers fall back to their floor.
    """
    send = transport or call
    size = _probe_size_for(model)
    # The nonce goes FIRST: a prefix cache matches from the start of the prompt,
    # so a leading unique token makes every byte after it new work. Identical
    # filler measured 130x faster than the same model measured honestly.
    nonce = f"[run {uuid.uuid4().hex}]\n"
    filler = nonce + (_PROBE_LINE * (size // len(_PROBE_LINE) + 1))[: size - len(nonce)]
    started = time.monotonic()
    try:
        result = send(
            model,
            messages=[{"role": "user", "content": filler}],
            host=host,
            port=port,
            max_tokens=1,
            timeout=DEFAULT_EVAL_TIMEOUT,
        )
    except Exception:
        return None
    elapsed = time.monotonic() - started
    if not result or result.get("error") or elapsed <= 0:
        return None
    rate = size / elapsed
    # An instant answer is not a measurement. A mocked or cached transport
    # returns in microseconds, and recording that as throughput would inflate
    # every downstream budget with a number no hardware produced.
    if rate > MAX_PLAUSIBLE_PREFILL_RATE:
        return None
    return round(rate, 1)


def record_prefill_rate(model: str, rate: float | None) -> None:
    """Store a measured prefill rate as a per-model capability.

    Kept at the model level, not per task: it is a property of the model and
    the host. The slowest observation wins, because the context budget has to
    hold on a bad run rather than a lucky one.
    """
    if not rate or rate <= 0:
        return
    signals = _load_eval_signals()
    caps = signals.setdefault(model, {}).setdefault("_capabilities", {})
    previous = caps.get("prefill_chars_per_sec")
    caps["prefill_chars_per_sec"] = min(previous, rate) if previous else rate
    caps["prefill_samples"] = caps.get("prefill_samples", 0) + 1
    _save_eval_signals(signals)
