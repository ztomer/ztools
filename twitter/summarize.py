#!/usr/bin/env python3
"""
LLM summarization helpers for twitter_summarizer.
"""

import os
import re
import shutil
import textwrap
from typing import Optional

from lib.config import Task, get_model_prompt
from lib.llm.fallback import call_with_fallback
from lib.mlx_lib import (
    call_mlx,
    find_any_working_mlx_model,
    find_best_mlx_model,
    find_mlx_model,
    get_mlx_context_length,
    process_mlx_content,
)
from lib.osaurus_lib import (
    call_llm_api,
    ensure_server,
    extract_thinking,
    get_available_models,
    get_best_model,
    merge_thinking_with_summary,
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
COLD_START_BASE = int(os.environ.get("TWITTER_COLD_START_BASE", "120"))
MAX_TIMEOUT = int(os.environ.get("TWITTER_MAX_TIMEOUT", "600"))

# Default context window size for Osaurus models (tokens)
OSAURUS_CONTEXT_WINDOW = int(os.environ.get("TWITTER_CONTEXT_WINDOW", "8192"))

# Timeout estimation: chars per second processing rate
CHARS_PER_SECOND = 25

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


def _estimate_timeout(prompt: str) -> int:
    return max(COLD_START_BASE, min(MAX_TIMEOUT, len(prompt) // CHARS_PER_SECOND))


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
    prompt, n = _build_prompt(tweets, max_chars=ctx_chars, model=try_model)
    try:
        print(f"{STEP} Trying {try_model} ({n} tweets)...")
        result = call_llm_api(
            f"{base_url.rstrip('/')}",
            try_model,
            [{"role": "user", "content": prompt}],
            api_key=api_key,
            timeout=_estimate_timeout(prompt),
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


def summarize_with_llm(tweets: list[dict], base_url: str, model: str, api_key: str = "") -> str:
    target_model = model if model else get_best_model(Task.SUMMARIZE)
    models = get_available_models()
    if not models:
        print(f"{WARN} Osaurus server not responding — trying to start it...")
        ensure_server()
        models = get_available_models()

    if models and target_model not in models:
        target_model = select_best_model(models) or target_model

    ctx_chars = (OSAURUS_CONTEXT_WINDOW - OUTPUT_RESERVE_TOKENS) * CHARS_PER_TOKEN

    _fallback_names = os.environ.get(
        "TWITTER_FALLBACK_MODELS", "qwen3.6-35b-a3b-mxfp4,foundation"  # check-ok: env var fallback
    ).split(",")
    fallback_models = list(
        dict.fromkeys([target_model] + [m.strip() for m in _fallback_names if m.strip()])
    )

    def call_fn(m: str) -> Optional[str]:
        return _summarize_with_model(tweets, base_url, api_key, ctx_chars, models, m)

    def mlx_fn() -> Optional[str]:
        mlx_paths = []
        first = find_mlx_model(target_model)
        if first:
            mlx_paths.append(first)
        mlx_paths.extend(
            m
            for m in [find_best_mlx_model(MLX_PREFERRED), find_any_working_mlx_model()]
            if m and m not in mlx_paths
        )

        for mlx_path in mlx_paths:
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
            print(f"{WARN} MLX model {mlx_path.name} failed, trying next...")
        return None

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
