#!/usr/bin/env python3
"""
LLM summarization helpers for twitter_summarizer.
"""

import os
import re
import shutil
import textwrap
from pathlib import Path
from typing import Optional

from lib.config import Task, get_model_prompt
from lib.llm.fallback import call_with_fallback
from lib.mlx_lib import (
    _check_mlx_model_compatible,
    call_mlx,
    find_any_working_mlx_model,
    find_best_mlx_model,
    find_mlx_model,
    get_mlx_context_length,
    last_mlx_error,
    process_mlx_content,
)
from lib.mlx_vlm import (
    call_mlx_vlm,
    find_any_working_mlx_vlm_model,
    find_best_mlx_vlm_model,
    probe_mlx_vlm_loadable,
)
from lib.osaurus_lib import (
    call_llm_api,
    ensure_server,
    extract_thinking,
    get_available_models,
    get_best_model,
    merge_thinking_with_summary,
    restart_server,
    select_best_model,
    strip_thinking,
)
from lib.tui import STEP, WARN

_mlx_preferred_str = os.environ.get(
    "TWITTER_MLX_PREFERRED", "Qwopus3.6-27B-v2-MLX-4bit,Qwen3.6,gemma-4,MiniMax"
)
MLX_PREFERRED = [m.strip() for m in _mlx_preferred_str.split(",") if m.strip()]

_PROMPT_RULES = """
- Use headers starting with ##
- Use bullet points for facts
- Keep it concise and factual
"""

CHARS_PER_TOKEN = int(os.environ.get("TWITTER_CHARS_PER_TOKEN", "3"))
OUTPUT_RESERVE_TOKENS = int(os.environ.get("TWITTER_OUTPUT_RESERVE", "4096"))

# Dynamic timeout model (seconds):
#   timeout = cold_start_load + input_chars/prefill_rate + output_tokens/decode_rate
# COLD_START_BASE is the fixed model-load cost (large for 35B models on first hit).
# TWITTER_PREFILL_CHARS_PER_SEC is how fast the server ingests the prompt.
# TWITTER_DECODE_TOKENS_PER_SEC is generation throughput; combined with the output
# reserve it bounds the time spent producing the summary.
COLD_START_BASE = int(os.environ.get("TWITTER_COLD_START_BASE", "120"))
MAX_TIMEOUT = int(os.environ.get("TWITTER_MAX_TIMEOUT", "1800"))
MIN_TIMEOUT = int(os.environ.get("TWITTER_MIN_TIMEOUT", "60"))
PREFILL_CHARS_PER_SEC = int(os.environ.get("TWITTER_PREFILL_CHARS_PER_SEC", "40"))
DECODE_TOKENS_PER_SEC = int(os.environ.get("TWITTER_DECODE_TOKENS_PER_SEC", "8"))

# Default context window size for Osaurus models (tokens)
OSAURUS_CONTEXT_WINDOW = int(os.environ.get("TWITTER_CONTEXT_WINDOW", "8192"))

# On-device Apple Foundation Models have a much smaller context window than the
# server models. Sizing the prompt to OSAURUS_CONTEXT_WINDOW and re-sending it to
# `foundation` overflows its window and returns HTTP 500. Size per-model instead.
FOUNDATION_CONTEXT_WINDOW = int(os.environ.get("TWITTER_FOUNDATION_CONTEXT_WINDOW", "4096"))


def _context_window_for_model(model: str) -> int:
    """Return the token context window to size the prompt against for a model."""
    if model and "foundation" in model.lower():
        return FOUNDATION_CONTEXT_WINDOW
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
MIN_BULLET_COUNT = 3
MIN_SUMMARY_CHARS = 100
BUDGET_MARGIN = 200
TERMINAL_WIDTH_LIMIT = 100


def _check_summary_quality(summary: str) -> tuple[list[str], bool]:
    if not summary:
        return (["Summary is empty"], True)
    warnings = []
    lines = summary.strip().splitlines()
    header_count = 0
    bullet_count = 0
    char_count = 0
    for line in lines:
        stripped = line.strip()
        char_count += len(stripped)
        if re.match(r"^#{2,}\s+\w+", stripped):
            header_count += 1
        elif stripped.startswith(("- ", "* ", "  - ", "  * ")):
            bullet_count += 1
    if header_count == 0:
        warnings.append("No ## headers — may be unstructured")
    if bullet_count < MIN_BULLET_COUNT:
        warnings.append(f"Only {bullet_count} bullet points — may lack detail")
    if char_count < MIN_SUMMARY_CHARS:
        warnings.append(f"Very short ({char_count} chars)")
    critical = header_count == 0 and bullet_count == 0
    return (warnings, critical)


def _build_prompt(
    tweets: list[dict], max_chars: int, for_mlx: bool = False, model: str = None
) -> tuple[str, int]:
    budget = max_chars - BUDGET_MARGIN
    lines = []
    used = 0
    width = min(TERMINAL_WIDTH_LIMIT, shutil.get_terminal_size().columns) if not for_mlx else 120
    for t in reversed(tweets):
        parts = [f"@{t['screen_name']} | {t['created_at'].strftime('%H:%M')}"]
        fav = t.get("favorite_count", 0)
        rt = t.get("retweet_count", 0)
        if fav or rt:
            parts.append(f"{fav} favs, {rt} RTs")
        reply_to = t.get("in_reply_to_screen_name")
        if reply_to:
            parts.append(f"\u2192 @{reply_to}")
        prefix = f"[{' | '.join(parts)}]: "
        text = t["text"].strip()
        line = textwrap.fill(
            f"{prefix}{text}",
            width=width,
            subsequent_indent=" " * len(prefix),
            replace_whitespace=False,
        )
        if used + len(line) + 1 > budget:
            break
        lines.append(line)
        used += len(line) + 1
    lines.reverse()
    timeline = "\n".join(lines)

    prompt_template = get_model_prompt(model, Task.SUMMARIZE)
    if not prompt_template:
        prompt_template = (
            "Create a structured summary of this Twitter timeline.\n"
            "Start with a brief overall paragraph capturing the main narrative.\n"
            "Then organize events into topic sections with ## headers, using bullet points.\n"
            "Include who (@user mentions), what happened, and when.\n"
            "Use connecting phrases and narrative verbs to show how events relate.\n"
            "After the topic sections, list the 5 most notable tweets "
            "(highest engagement or most impactful) with their full text and URL.\n"
        )
    prompt_template += _PROMPT_RULES

    if "{}" in prompt_template:
        prompt = prompt_template.replace("{}", timeline)
    else:
        prompt = f"{prompt_template}\n<timeline>\n{timeline}\n</timeline>"

    return prompt, len(lines)


def _estimate_timeout(prompt: str, model: str = None) -> int:
    """Dynamic request timeout scaled to input size + expected output.

    timeout = cold-start load + prefill(input) + decode(output), clamped to
    [MIN_TIMEOUT, MAX_TIMEOUT]. Larger timelines and slower models get more time
    instead of a flat cap that a big prompt on a 35B model blows through.
    """
    prefill = len(prompt) / max(1, PREFILL_CHARS_PER_SEC)
    decode = _output_tokens_for_model(model) / max(1, DECODE_TOKENS_PER_SEC)
    estimate = COLD_START_BASE + prefill + decode
    return int(max(MIN_TIMEOUT, min(MAX_TIMEOUT, estimate)))


def _summarize_with_model(
    tweets: list[dict],
    base_url: str,
    api_key: str,
    ctx_chars: int,
    models: list[str],
    try_model: str,
) -> Optional[str]:
    if models and try_model not in models:
        return None
    ctx_chars = _ctx_chars_for_model(try_model)
    prompt, n = _build_prompt(tweets, max_chars=ctx_chars, model=try_model)
    try:
        print(f"{STEP} Trying {try_model} ({n} tweets)...")
        result = call_llm_api(
            f"{base_url.rstrip('/')}",
            try_model,
            [{"role": "user", "content": prompt}],
            api_key=api_key,
            timeout=_estimate_timeout(prompt, try_model),
        )
        if result and "content" in result:
            thinking, cleaned = extract_thinking(result["content"])
            if thinking:
                print(f"{STEP} {try_model}: included thinking block")
                warnings, critical = _check_summary_quality(cleaned)
                if critical:
                    for w in warnings:
                        print(f"   {WARN} {w}")
                    return None
                return merge_thinking_with_summary(thinking, cleaned)
            cleaned = strip_thinking(cleaned)
            warnings, critical = _check_summary_quality(cleaned)
            if critical:
                for w in warnings:
                    print(f"   {WARN} {w}")
                return None
            return cleaned
        elif result and "error" in result:
            print(f"{WARN} {try_model} error: {result['error']}")
    except Exception as e:
        print(f"{WARN} {try_model} failed: {str(e)[:50]}")
    return None


def _restart_and_retry_server(
    tweets: list[dict], base_url: str, api_key: str, target_model: str
) -> Optional[str]:
    """Restart Osaurus and retry the server once.

    The installed models only run in Osaurus's mlx-swift runtime, so restarting
    the server and retrying is the genuine last resort (a timeout/500 is often a
    wedged server, not a bad model). Returns a summary or None.
    """
    print(f"{STEP} Restarting Osaurus server and retrying {target_model} ...")
    if not restart_server():
        print(f"{WARN} Osaurus restart failed.")
        return None
    models = get_available_models(base_url)
    retry_model = target_model
    if models and target_model not in models:
        retry_model = select_best_model(models) or target_model
    ctx_chars = _ctx_chars_for_model(retry_model)
    return _summarize_with_model(tweets, base_url, api_key, ctx_chars, models, retry_model)


def _direct_mlx_fallback(tweets: list[dict]) -> Optional[str]:
    """Try Python-loadable MLX models directly, mlx-vlm first then stock mlx_lm.

    The Osaurus-installed models (Gemma4/qwen3_5_moe, MXFP8) load via `mlx-vlm`
    (git main), which is the primary Python path. Stock `mlx_lm` only handles
    plain-text checkpoints, so it is the secondary path. Each candidate is
    load-probed, so unloadable architectures are skipped fast and the real
    failure reason is surfaced instead of a generic message.
    """
    # Stage 1: mlx-vlm (multimodal/MoE models the server also runs).
    vlm_paths = []
    first = find_mlx_model(target_model_env())
    if first and probe_mlx_vlm_loadable(first)[0]:
        vlm_paths.append(first)
    vlm_paths.extend(
        m
        for m in [find_best_mlx_vlm_model(MLX_PREFERRED), find_any_working_mlx_vlm_model()]
        if m and m not in vlm_paths
    )
    for mlx_path in vlm_paths:
        summary = _generate_via_mlx_vlm(tweets, mlx_path)
        if summary:
            return summary

    # Stage 2: stock mlx_lm (plain-text checkpoints only).
    lm_paths = []
    vlm_loadable = probe_mlx_vlm_loadable(first)[0] if first else False
    if first and not vlm_loadable and _check_mlx_model_compatible(first, deep=True):
        lm_paths.append(first)
    stock_candidates = [
        find_best_mlx_model(MLX_PREFERRED, deep=True),
        find_any_working_mlx_model(deep=True),
    ]
    lm_paths.extend(m for m in stock_candidates if m and m not in lm_paths)
    for mlx_path in lm_paths:
        mlx_ctx = get_mlx_context_length(mlx_path)
        mlx_prompt_chars = (mlx_ctx - OUTPUT_RESERVE_TOKENS) * CHARS_PER_TOKEN
        prompt, n = _build_prompt(
            tweets, max_chars=mlx_prompt_chars, for_mlx=True, model=mlx_path.name
        )
        print(f"{STEP} Sending {n}/{len(tweets)} tweets to MLX model {mlx_path.name} ...")
        raw = call_mlx(mlx_path, prompt)
        if raw and not raw.startswith("[LLM error"):
            cleaned = process_mlx_content(raw)
            warnings, critical = _check_summary_quality(cleaned)
            if warnings:
                for w in warnings:
                    print(f"   {WARN} MLX summary: {w}")
            if critical:
                print("MLX returned unusable summary, aborting.")
                return None
            return cleaned
        reason = last_mlx_error() or "unknown error"
        print(f"{WARN} MLX model {mlx_path.name} failed: {reason[:120]}")
    return None


def _generate_via_mlx_vlm(tweets: list[dict], mlx_path: Path) -> Optional[str]:
    """Generate a summary from one mlx-vlm-loadable model, or None."""
    mlx_ctx = get_mlx_context_length(mlx_path)
    mlx_prompt_chars = (mlx_ctx - OUTPUT_RESERVE_TOKENS) * CHARS_PER_TOKEN
    prompt, n = _build_prompt(
        tweets, max_chars=mlx_prompt_chars, for_mlx=True, model=mlx_path.name
    )
    print(f"{STEP} Sending {n}/{len(tweets)} tweets to MLX-VLM model {mlx_path.name} ...")
    raw = call_mlx_vlm(mlx_path, prompt)
    if raw and not raw.startswith("[VLM"):
        cleaned = process_mlx_content(raw)
        warnings, critical = _check_summary_quality(cleaned)
        if warnings:
            for w in warnings:
                print(f"   {WARN} MLX-VLM summary: {w}")
        if critical:
            print("MLX-VLM returned unusable summary, aborting.")
            return None
        return cleaned
    reason = last_mlx_error() or "unknown error"
    print(f"{WARN} MLX-VLM model {mlx_path.name} failed: {reason[:120]}")
    return None


def target_model_env() -> str:
    """Preferred model name for direct-MLX lookup (env override or best model)."""
    return os.environ.get("OLLAMA_MODEL", "") or get_best_model(Task.SUMMARIZE)


def summarize_with_llm(tweets: list[dict], base_url: str, model: str, api_key: str = "") -> str:
    target_model = model if model else get_best_model(Task.SUMMARIZE)
    models = get_available_models(base_url)
    if not models:
        print(f"{WARN} Osaurus server not responding — trying to start it...")
        ensure_server()
        models = get_available_models(base_url)

    if models and target_model not in models:
        target_model = select_best_model(models) or target_model

    _fallback_names = os.environ.get(
        "TWITTER_FALLBACK_MODELS", "foundation"  # check-ok: env var fallback
    ).split(",")
    fallback_models = list(
        dict.fromkeys([target_model] + [m.strip() for m in _fallback_names if m.strip()])
    )
    # Drop server models the live server does not actually serve so a stale id
    # does not waste a full request/timeout cycle. `foundation` is the on-device
    # model and is always kept as a last-resort server fallback.
    if models:
        fallback_models = [
            m for m in fallback_models if m in models or "foundation" in m.lower()
        ]

    def call_fn(m: str) -> Optional[str]:
        ctx_chars = _ctx_chars_for_model(m)
        return _summarize_with_model(tweets, base_url, api_key, ctx_chars, models, m)

    def mlx_fn() -> Optional[str]:
        # Priority: Osaurus server first. If it is up, the server path runs it
        # (it is the runtime that produced these models and is usually warmed).
        # On failure we restart+retry the server. Only after the server is
        # exhausted/down do we fall back to Python `mlx-vlm`, which can load the
        # same checkpoints on-device as a genuine last resort.
        result = _restart_and_retry_server(tweets, base_url, api_key, target_model)
        if result:
            return result
        return _direct_mlx_fallback(tweets)

    result = call_with_fallback(
        fallback_models,
        call_fn,
        mlx_fn=mlx_fn,
        max_server_retries=0,
        label="model",
    )

    if result:
        return result

    print(f"{WARN} All server models failed: {fallback_models}")
    return "[LLM error: both local MLX and server failed]"
