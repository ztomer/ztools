"""Prompt-budget and timeout model for the twitter summarizer.

Split out of summarize.py for the repo's 500-line limit. Two questions live
here: how much timeline fits a given model's context window, and how long the
call should be allowed to take.
"""

import os

CHARS_PER_TOKEN = int(os.environ.get("TWITTER_CHARS_PER_TOKEN", "3"))
# Observed summaries run ~900-1100 tokens; reserving 4096 took half of an 8192
# window away from the timeline for output that never needed it.
OUTPUT_RESERVE_TOKENS = int(os.environ.get("TWITTER_OUTPUT_RESERVE", "1536"))

# Dynamic timeout model (seconds):
#   timeout = cold_start_load + input_chars/prefill_rate + output_tokens/decode_rate
# COLD_START_BASE is the fixed model-load cost (large for 35B models on first hit).
# TWITTER_PREFILL_CHARS_PER_SEC is how fast the server ingests the prompt.
# TWITTER_DECODE_TOKENS_PER_SEC is generation throughput; combined with the output
# reserve it bounds the time spent producing the summary.
COLD_START_BASE = int(os.environ.get("TWITTER_COLD_START_BASE", "120"))
MAX_TIMEOUT = int(os.environ.get("TWITTER_MAX_TIMEOUT", "1800"))
MIN_TIMEOUT = int(os.environ.get("TWITTER_MIN_TIMEOUT", "60"))
# Fallback only: `_prefill_rate_for_model` prefers the rate `ev` measured for the
# specific model. Measured with cache-defeating probes, gemma-4-12b ingests
# ~1,100 chars/sec; the old 40 overestimated prefill time by ~26x, so
# `_estimate_timeout` clamped every large prompt to MAX_TIMEOUT and made a
# generous budget look unaffordable. This default sits below the slowest honest
# measurement for margin.
PREFILL_CHARS_PER_SEC = int(os.environ.get("TWITTER_PREFILL_CHARS_PER_SEC", "800"))
DECODE_TOKENS_PER_SEC = int(os.environ.get("TWITTER_DECODE_TOKENS_PER_SEC", "8"))

# Default context window size for Osaurus models (tokens)
OSAURUS_CONTEXT_WINDOW = int(os.environ.get("TWITTER_CONTEXT_WINDOW", "8192"))

# On-device Apple Foundation Models have a much smaller context window than the
# server models. Sizing the prompt to OSAURUS_CONTEXT_WINDOW and re-sending it to
# `foundation` overflows its window and returns HTTP 500. Size per-model instead.
FOUNDATION_CONTEXT_WINDOW = int(os.environ.get("TWITTER_FOUNDATION_CONTEXT_WINDOW", "4096"))


def _context_window_for_model(model: str) -> int:
    """Return the token context window to size the prompt against for a model.

    A flat 8192 for every server model cut a 24h timeline (200-800 tweets) down
    to ~60-90 before the model ever saw it, on models serving far more. The
    per-model `context_window` key in conf/models/*.toml is authoritative;
    OSAURUS_CONTEXT_WINDOW remains the floor for models that declare nothing.
    """
    if model and "foundation" in model.lower():
        return FOUNDATION_CONTEXT_WINDOW
    try:
        from lib.config import get_model_config
        from lib.model_caps import usable_context_window

        # Ask the model, do not assume: the installed models report 131072 or
        # 262144 in their own config.json, so both the old flat 8192 and the
        # hand-written per-family guesses threw most of the window away. A
        # `context_window` entry in conf/models/*.toml still wins as an override.
        override = (get_model_config(model) or {}).get("context_window")
        return usable_context_window(model, OSAURUS_CONTEXT_WINDOW, override)
    except Exception:
        return OSAURUS_CONTEXT_WINDOW


def _ctx_chars_for_model(model: str) -> int:
    """Prompt character budget for a model, derived from its context window.

    The output reserve is capped at half the window so a small window (e.g.
    Foundation's) still leaves room for the input prompt instead of collapsing
    the budget to zero.
    """
    window = _context_window_for_model(model)
    reserve = min(OUTPUT_RESERVE_TOKENS, window // 2)
    return (window - reserve) * CHARS_PER_TOKEN


def _output_tokens_for_model(model: str) -> int:
    """Estimated output tokens to budget generation time against."""
    window = _context_window_for_model(model)
    return min(OUTPUT_RESERVE_TOKENS, window // 2)


# Quality thresholds



def _prefill_rate_for_model(model: str = None) -> float:
    """Ingestion rate to budget time against: this model's, or the fallback.

    `ev` measures every model it evaluates, and the models differ by more than
    2x (1,322 chars/sec on the dense 12B vs 3,515 on the 35B MoE). Budgeting
    both against one constant gives the fast model a timeout far longer than it
    needs and the slow model one that is too tight.
    """
    try:
        from lib.model_caps import measured_prefill_rate

        return measured_prefill_rate(model or "") or PREFILL_CHARS_PER_SEC
    except Exception:
        return PREFILL_CHARS_PER_SEC


def _estimate_timeout(prompt: str, model: str = None) -> int:
    """Dynamic request timeout scaled to input size + expected output.

    timeout = cold-start load + prefill(input) + decode(output), clamped to
    [MIN_TIMEOUT, MAX_TIMEOUT]. Larger timelines and slower models get more time
    instead of a flat cap that a big prompt on a 35B model blows through.
    """
    prefill = len(prompt) / max(1, _prefill_rate_for_model(model))
    decode = _output_tokens_for_model(model) / max(1, DECODE_TOKENS_PER_SEC)
    estimate = COLD_START_BASE + prefill + decode
    return int(max(MIN_TIMEOUT, min(MAX_TIMEOUT, estimate)))
