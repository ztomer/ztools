#!/usr/bin/env python3
"""
LLM summarization helpers for twitter_summarizer.
"""

import re
import shutil
import textwrap

from lib.config import get_model_prompt, Task
from lib.osaurus_lib import (
    call_llm_api, strip_thinking, get_available_models,
    get_best_model, select_best_model, merge_thinking_with_summary, extract_thinking,
    ensure_server,
)
from lib.mlx_lib import (
    find_mlx_model, find_best_mlx_model, find_any_working_mlx_model,
    get_mlx_context_length, call_mlx, process_mlx_content,
)
from lib.tui import STEP, WARN

MLX_PREFERRED = [
    "Qwopus3.6-27B-v2-MLX-4bit",
    "Qwen3.6",
    "gemma-4",
    "MiniMax",
]

_PROMPT_RULES = """
- Use headers starting with ##
- Use bullet points for facts
- Keep it concise and factual
"""

CHARS_PER_TOKEN = 3
OUTPUT_RESERVE_TOKENS = 4096


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
        if re.match(r'^#{2,}\s+\w+', stripped):
            header_count += 1
        elif stripped.startswith(("- ", "* ", "  - ", "  * ")):
            bullet_count += 1
    if header_count == 0:
        warnings.append("No ## headers — may be unstructured")
    if bullet_count < 3:
        warnings.append(f"Only {bullet_count} bullet points — may lack detail")
    if char_count < 100:
        warnings.append(f"Very short ({char_count} chars)")
    critical = header_count == 0 and bullet_count == 0
    return (warnings, critical)


def _build_prompt(
    tweets: list[dict], max_chars: int, for_mlx: bool = False, model: str = None
) -> tuple[str, int]:
    budget = max_chars - 200
    lines = []
    used = 0
    width = min(100, shutil.get_terminal_size().columns) if not for_mlx else 120
    for t in reversed(tweets):
        prefix = f"[@{t['screen_name']} | {t['created_at'].strftime('%H:%M')}]: "
        text = t['text'].strip()
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
        prompt_template = "Summarize this timeline:\n\n{}\n\nUse ## headers for topics."

    if "{}" in prompt_template:
        prompt = prompt_template.replace("{}", timeline)
    else:
        prompt = prompt_template

    return prompt, len(lines)


def summarize_with_llm(
    tweets: list[dict], base_url: str, model: str, api_key: str = ""
) -> str:
    target_model = model if model else get_best_model(Task.SUMMARIZE)
    models = get_available_models()
    if not models:
        print(f"{WARN} Osaurus server not responding — trying to start it...")
        ensure_server()
        models = get_available_models()

    if models and target_model not in models:
        target_model = select_best_model(models) or target_model

    ctx_chars = (8192 - OUTPUT_RESERVE_TOKENS) * CHARS_PER_TOKEN
    prompt, n = _build_prompt(tweets, max_chars=ctx_chars, model=target_model)

    fallback_models = list(dict.fromkeys([target_model, "qwen3.6-35b-a3b-mxfp4", "foundation"]))
    tried = set()
    attempted_models = []

    for try_model in fallback_models:
        if try_model in tried:
            continue
        if models and try_model not in models:
            tried.add(try_model)
            continue
        tried.add(try_model)
        attempted_models.append(try_model)

        prompt, n = _build_prompt(tweets, max_chars=ctx_chars, model=try_model)

        try:
            print(f"{STEP} Trying {try_model} ({n} tweets)...")
            result = call_llm_api(
                f"{base_url.rstrip('/')}",
                try_model,
                [{"role": "user", "content": prompt}],
                api_key=api_key,
                timeout=120,
            )
            if result and "content" in result:
                thinking, cleaned = extract_thinking(result["content"])
                if thinking:
                    print(f"{STEP} {try_model}: included thinking block")
                    warnings, critical = _check_summary_quality(cleaned)
                    if critical:
                        for w in warnings:
                            print(f"   {WARN} {w}")
                        continue
                    return merge_thinking_with_summary(thinking, cleaned)
                cleaned = strip_thinking(cleaned)
                warnings, critical = _check_summary_quality(cleaned)
                if critical:
                    for w in warnings:
                        print(f"   {WARN} {w}")
                    continue
                return cleaned
            elif result and "error" in result:
                print(f"{WARN} {try_model} error: {result['error'][:50]}")
        except Exception as e:
            print(f"{WARN} {try_model} failed: {str(e)[:50]}")
            continue

    print("Server models failed, trying MLX...")

    mlx_paths = []
    first = find_mlx_model(target_model)
    if first:
        mlx_paths.append(first)
    mlx_paths.extend(m for m in [find_best_mlx_model(MLX_PREFERRED), find_any_working_mlx_model()] if m and m not in mlx_paths)

    for mlx_path in mlx_paths:
        mlx_ctx = get_mlx_context_length(mlx_path)
        mlx_prompt_chars = (mlx_ctx - OUTPUT_RESERVE_TOKENS) * CHARS_PER_TOKEN
        prompt, n = _build_prompt(
            tweets, max_chars=mlx_prompt_chars, for_mlx=True, model=mlx_path.name)
        print(
            f"{STEP} Sending {n}/{len(tweets)} tweets to MLX model {mlx_path.name} ..."
        )
        raw = call_mlx(mlx_path, prompt)
        if raw and not raw.startswith("[LLM error"):
            cleaned = process_mlx_content(raw)
            warnings, critical = _check_summary_quality(cleaned)
            if warnings:
                for w in warnings:
                    print(f"   {WARN} MLX summary: {w}")
            if critical:
                print("MLX returned unusable summary, aborting.")
                return "[LLM error: both local MLX and server failed]"
            return cleaned
        print(f"{WARN} MLX model {mlx_path.name} failed, trying next...")

    print(f"{WARN} All server models failed: {attempted_models}")
    return "[LLM error: both local MLX and server failed]"
