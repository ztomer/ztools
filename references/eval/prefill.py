"""Measure how fast a model ingests a prompt, per model, on this host.

Used to size request TIMEOUTS so a large prompt is not killed mid-flight. It is
explicitly NOT used to decide how much context to send: these tools run every
six hours at most, so ingestion time is free and trading context for it buys
nothing. An earlier version did exactly that and cost output quality.

Measuring it honestly took four attempts, and three of them produced a
confident, wrong number:

  assumed 40 chars/sec              never measured at all
  derived from whole-call time      decode dominates a generation-heavy task,
                                    so this times the wrong quantity (17x low)
  max_tokens=1, identical filler    the server's prefix cache: the same model
                                    climbed 1,322 -> 3,789 -> 140,281 across
                                    successive probes
  no warmup                         the probe is an eval's FIRST request, so it
                                    timed 27GB of model load as throughput --
                                    bonsai-27b read 59 chars/sec against
                                    ornith-9b's 1,803, almost entirely weights
                                    moving into memory

Hence: warm the model first, then lead the timed prompt with a nonce so no
prefix can be reused. A rate that changes every time you measure it, or that
tracks model size rather than model speed, is not a rate.
"""

import os
import time
import uuid

from lib.model_caps import CHARS_PER_TOKEN, probe_context_window
from lib.osaurus_lib import call

from eval.samples import add_sample, migrate_scalar
from eval.signals import DEFAULT_EVAL_TIMEOUT, _load_eval_signals, _save_eval_signals

# Prefill probe sizing. Big enough that per-request overhead does not dominate,
# small enough to cost a few seconds per model.
PREFILL_PROBE_CHARS = int(os.environ.get("EVAL_PREFILL_PROBE_CHARS", "20000"))
_PROBE_LINE = "[@SomeHandle | 08:15]: A reasonably typical sentence about a launch today.\n"
# Above this, the transport did not really ingest anything: a mock, a stub, or a
# prefix-cache hit. The bound only ever DISCARDS a measurement -- it never invents
# one -- so it must sit above every genuine reading, or it silently converts a real
# measurement into "never measured" and the caller falls back to the pessimistic
# 200 chars/sec default.
#
# Both populations, measured on this host rather than assumed:
#
#   genuine, 2026-08-15, all installed models x2, one server, idle machine
#     gemma-4-e2b-it-8bit (2B)     23,063   <- the fastest, and the reason for 40,000
#     gemma-4-e4b-it-8bit (4B)      7,183 / 6,553
#     foundation                    4,006 / 3,224
#     ornith-35b / nemotron-30b     2,856 - 2,740
#     gemma-4-12b                     805 /   714
#     qwen3.8-27b-mxfp8               128 /   117
#   prefix-cache hits (historical)  65,000 - 140,281
#
# 40,000 sits 1.7x above the fastest genuine reading and 1.6x below the slowest
# cache hit. Two earlier values were both set from a guess about which model would
# be fastest and both were wrong: ~3,500 ("the 35B MoE") and then 20,000. The
# fastest ingester is the SMALLEST model, not the biggest or the MoE one -- prefill
# is compute-bound and parallel, so a 2B model shreds a 20K-char prompt in under a
# second. gemma-4-e2b sat at "unmeasured" for exactly this reason, its real 23,063
# discarded by a bound set to 20,000.
MAX_PLAUSIBLE_PREFILL_RATE = int(os.environ.get("EVAL_MAX_PLAUSIBLE_PREFILL", "40000"))
# The warmup asks for enough tokens to time generation, not just to load weights.
WARMUP_TOKENS = int(os.environ.get("EVAL_WARMUP_TOKENS", "64"))
WARMUP_PROMPT = "Count from one to twenty in words, one per line."


def _probe_size_for(model: str) -> int:
    """Probe length that fits inside this model's own window.

    A 20K-char probe is 6.6K tokens, which foundation (4096, and that covers
    output too) answers with an HTTP 500. Sizing the probe to the model keeps
    the measurement possible on small-window models instead of returning None.
    """
    window = probe_context_window(model)
    if not window:
        return PREFILL_PROBE_CHARS
    return max(2000, min(PREFILL_PROBE_CHARS, int(window * CHARS_PER_TOKEN * 0.6)))


def measure_prefill_rate(model: str, host: str, port: int, transport=None) -> float | None:
    """Characters per second this model ingests, measured with max_tokens=1.

    Only `max_tokens=1` isolates ingestion from generation, and only a warmed
    model isolates ingestion from loading.

    `transport` is the caller's own `call`, so a test that patches `eval.run.call`
    covers the probe too. Importing `call` here by value created a second seam
    that mocks did not reach, and eighteen mocked tests silently hit the live
    server through it until the conftest gate started failing on connections.

    Returns None when the probe cannot run, so a caller can tell 'not measured'
    from a measurement.
    """
    send = transport or call
    size = _probe_size_for(model)

    # Load the model BEFORE timing anything. The probe is the first request an
    # eval makes, so without this it times model load as though it were
    # per-character throughput: bonsai-27b (27GB, loading from disk) measured 59
    # chars/sec against ornith-9b's 1,803, a difference that was almost entirely
    # weights moving into memory. A tiny prompt loads the model without
    # meaningfully warming any prefix the real probe will reuse.
    # THREE calls, one quantity each. Sharing a call between two measurements is
    # how every previous version of this got a wrong number: whole-call timing
    # blamed prefill for decode, and an unwarmed probe blamed prefill for model
    # load. Timing the load call for decode speed would have repeated the same
    # error one function later.
    #
    # 1. LOAD. The first request after a server restart pays the cold start.
    #    Timing it turns COLD_START into a measurement instead of a guess.
    try:
        load_started = time.monotonic()
        loaded = send(model, messages=[{"role": "user", "content": WARMUP_PROMPT}],
                      host=host, port=port, max_tokens=1,
                      timeout=DEFAULT_EVAL_TIMEOUT)
        load_elapsed = time.monotonic() - load_started
    except Exception:
        return None
    if not loaded or loaded.get("error"):
        return None
    _record_cold_start(model, load_elapsed)

    # 2. DECODE, now that the weights are resident. Same tiny prompt, so
    #    ingestion is negligible and the elapsed time is generation.
    try:
        decode_started = time.monotonic()
        generated = send(model, messages=[{"role": "user", "content": WARMUP_PROMPT}],
                         host=host, port=port, max_tokens=WARMUP_TOKENS,
                         timeout=DEFAULT_EVAL_TIMEOUT)
        decode_elapsed = time.monotonic() - decode_started
    except Exception:
        generated, decode_elapsed = None, 0
    if generated and not generated.get("error") and decode_elapsed > 0:
        _record_decode_rate(model, WARMUP_TOKENS / decode_elapsed)

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
    # returns in microseconds, and recording that as throughput would produce a
    # timeout far too short for the request it is meant to cover.
    if rate > MAX_PLAUSIBLE_PREFILL_RATE:
        return None
    return round(rate, 1)


def _record_cold_start(model: str, seconds: float) -> None:
    """Store how long this model takes to become ready, measured not assumed."""
    _record_sample("cold_start_seconds", model, seconds)


def _record_decode_rate(model: str, rate: float) -> None:
    """Store measured generation speed (tokens/sec)."""
    _record_sample("decode_tokens_per_sec", model, rate)


def _record_sample(key: str, model: str, value: float) -> None:
    """Add one observation of `key` for `model` and re-derive the estimate.

    These three recorders used to keep the EXTREME observation -- slowest rate,
    longest cold start. Right for a timeout, fatal for a measurement: the extreme is
    exactly what a contended machine produces, and nothing could ever displace it. A
    leaked daemon held 31GB for a day and the resulting numbers (nemotron at 0.68
    tok/s against ~33 actual) became permanent, reaching conf/config.toml and a
    default_model choice. Re-measuring on a healthy machine changed nothing.

    Now: median of recent samples taken while the machine was quiet. A bad reading is
    outvoted instead of enshrined. See eval/samples.py.
    """
    if not value or value <= 0:
        return
    signals = _load_eval_signals()
    caps = signals.setdefault(model, {}).setdefault("_capabilities", {})
    migrate_scalar(caps, key)
    add_sample(caps, key, value)
    _save_eval_signals(signals)


def record_prefill_rate(model: str, rate: float | None) -> None:
    """Store a measured prefill rate as a per-model capability.

    Kept at the model level, not per task: it is a property of the model and the host.
    """
    if not rate or rate <= 0:
        return
    _record_sample("prefill_chars_per_sec", model, rate)
    signals = _load_eval_signals()
    caps = signals.setdefault(model, {}).setdefault("_capabilities", {})
    caps["prefill_samples"] = caps.get("prefill_samples", 0) + 1
    _save_eval_signals(signals)
