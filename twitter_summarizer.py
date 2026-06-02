#!/usr/bin/env python3
# /// script
# dependencies = ["playwright", "requests", "cryptography", "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git", "transformers", "pyyaml"]
# ///
"""
twitter_summarizer.py — Fetch your X/Twitter home timeline via browser automation,
summarize with a local LLM, and export to Markdown.

Shim that re-exports from smaller sub-modules.
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
except ImportError:
    Cipher = algorithms = modes = None

try:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
except ImportError:
    sync_playwright = PWTimeout = None

from lib import init_config
from lib.osaurus_lib import get_best_model
from lib.config import Task
from lib.tui import STEP, WARN

from twit_cookies import (
    CHROME_COOKIES_DB,
    _get_chrome_keychain_key,
    _decrypt_cookie,
    get_chrome_cookies,
)
from twit_browser import (
    MAX_SCROLLS,
    SCROLL_PAUSE_MS,
    parse_tweets_from_response,
    collect_tweets_via_browser,
)
from twit_summarize import (
    MLX_PREFERRED,
    _PROMPT_RULES,
    CHARS_PER_TOKEN,
    OUTPUT_RESERVE_TOKENS,
    _check_summary_quality,
    _build_prompt,
    summarize_with_llm,
)
from twit_output import (
    STATE_FILE,
    DEBUG_CACHE_FILE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OLLAMA_URL,
    load_state,
    save_state,
    load_debug_cache,
    save_debug_cache,
    print_to_stdout,
    write_markdown,
    clean_folder,
)

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", get_best_model(Task.SUMMARIZE))

__all__ = [
    "MLX_PREFERRED",
    "STATE_FILE",
    "DEBUG_CACHE_FILE",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_MODEL",
    "CHROME_COOKIES_DB",
    "MAX_SCROLLS",
    "SCROLL_PAUSE_MS",
    "CHARS_PER_TOKEN",
    "OUTPUT_RESERVE_TOKENS",
    "_PROMPT_RULES",
    "_check_summary_quality",
    "_get_chrome_keychain_key",
    "_decrypt_cookie",
    "get_chrome_cookies",
    "load_state",
    "save_state",
    "load_debug_cache",
    "save_debug_cache",
    "parse_args",
    "resolve_since_time",
    "parse_tweets_from_response",
    "collect_tweets_via_browser",
    "_build_prompt",
    "summarize_with_llm",
    "print_to_stdout",
    "write_markdown",
    "clean_folder",
    "main",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize your X/Twitter home timeline with a local LLM."
    )
    p.add_argument(
        "--since",
        default=None,
        help="Override start time. Accepts ISO 8601 or relative like '24h'.",
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL),
        help="Model name",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        help="Ollama base URL",
    )
    p.add_argument(
        "--api-key", default=os.environ.get("OLLAMA_API_KEY", ""), help="API key"
    )
    p.add_argument(
        "--debug", action="store_true", help="Show browser window and verbose output"
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete all .md files in the target output and exit",
    )
    p.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached tweets from last run instead of fetching new ones",
    )
    return p.parse_args()


def resolve_since_time(args_since: str | None, state: dict) -> datetime:
    if args_since:
        m = re.fullmatch(r"(\d+)h", args_since.strip())
        if m:
            return datetime.now(timezone.utc) - timedelta(hours=int(m.group(1)))
        try:
            dt = datetime.fromisoformat(args_since)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            print(
                f"{WARN} Cannot parse --since '{args_since}'. Using last run or 24h.")
    if "last_run" in state:
        return datetime.fromisoformat(state["last_run"])
    return datetime.now(timezone.utc) - timedelta(hours=24)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).expanduser()

    model = args.model or os.environ.get('OLLAMA_MODEL', 'default')
    print(f"{STEP} Using model: {model}")

    if args.clean:
        clean_folder(output_dir)

    state = load_state()
    since_time = resolve_since_time(args.since, state)
    until_time = datetime.now(timezone.utc)

    if args.use_cache:
        tweets = load_debug_cache()
        if tweets:
            print(f"{STEP} Using {len(tweets)} cached tweets")
        else:
            print(f"{WARN} No cached tweets found. Run without --use-cache first.")
            sys.exit(1)
    else:
        tweets = collect_tweets_via_browser(since_time, debug=args.debug)
        if not tweets:
            print(f"{WARN} No tweets found.")
            sys.exit(0)

    summary = summarize_with_llm(
        tweets, args.base_url, args.model, api_key=args.api_key
    )
    if summary.startswith("[LLM error"):
        print(f"{WARN} {summary}\n{WARN} Aborting.")
        sys.exit(1)

    out_path, content = write_markdown(
        tweets, summary, since_time, until_time, output_dir
    )
    print_to_stdout(content)
    print(f"{STEP} Summary written to: {out_path}")
    save_state({"last_run": until_time.isoformat()})


if __name__ == "__main__":
    init_config()
    main()
