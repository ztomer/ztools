#!/usr/bin/env python3
"""CLI for twitter timeline summarizer — parse_args, resolve_since_time, main."""

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lib import osaurus_server
from lib.config import Task
from lib.llm.constants import DEFAULT_HOST, DEFAULT_PORT
from lib.osaurus_lib import get_best_model
from lib.signal_handling import is_shutdown_requested, setup_signals
from lib.tui import STEP, WARN

from twitter.browser import (
    collect_tweets_via_browser,
)
from twitter.output import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_OUTPUT_DIR,
    clean_folder,
    load_debug_cache,
    load_state,
    print_to_stdout,
    save_debug_cache,
    save_state,
    write_markdown,
)
from twitter.session import login as run_login
from twitter.summarize import (
    summarize_with_llm,
)

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", get_best_model(Task.SUMMARIZE))


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
        "--host",
        default=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL),
        help=f"Osaurus/Ollama server URL (default: $OLLAMA_BASE_URL or http://{DEFAULT_HOST}:{DEFAULT_PORT})",
    )
    p.add_argument("--api-key", default=os.environ.get("OLLAMA_API_KEY", ""), help="API key")
    p.add_argument("--debug", action="store_true", help="Show browser window and verbose output")
    p.add_argument(
        "--login",
        action="store_true",
        help="Open a browser window to sign in to x.com and save the session, "
        "then exit. Only needed when no installed browser is signed in.",
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
    p.add_argument(
        "--fetch-only",
        action="store_true",
        help="Collect tweets, save them to the cache, and exit without summarizing",
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
            print(f"{WARN} Cannot parse --since '{args_since}'. Using last run or 24h.")
    if "last_run" in state:
        try:
            last_run = datetime.fromisoformat(state["last_run"])
            # If last run was less than 4 hours ago, use default 24h lookback
            # to avoid tiny 10-second timeline windows on repeated runs.
            if (datetime.now(timezone.utc) - last_run) > timedelta(hours=4):
                return last_run
        except Exception:
            pass
    return datetime.now(timezone.utc) - timedelta(hours=24)


def main() -> None:
    # drain=True: Ctrl+C stops the timeline scroll but keeps the tweets already
    # collected so the run still produces a summary. Ctrl+C twice force-quits.
    setup_signals(drain=True)
    args = parse_args()

    if args.login:
        # Sign-in only. Exits before touching the LLM server or output dir.
        sys.exit(0 if run_login() else 1)

    output_dir = Path(args.output).expanduser()

    model = args.model or os.environ.get("OLLAMA_MODEL", "default")
    print(f"{STEP} Using model: {model}")

    if args.clean:
        clean_folder(output_dir)

    osaurus_server.check_server_or_die(args.host, DEFAULT_PORT, model)

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
        save_debug_cache(tweets)
        if args.fetch_only:
            print(f"{STEP} --fetch-only provided. Exiting.")
            sys.exit(0)
        if is_shutdown_requested():
            print(f"{WARN} Interrupted — summarizing the {len(tweets)} tweets collected so far.")

    summary = summarize_with_llm(tweets, args.host, args.model, api_key=args.api_key)
    if summary.startswith("[LLM error"):
        print(f"{WARN} {summary}\n{WARN} Aborting.")
        sys.exit(1)

    out_path, content = write_markdown(tweets, summary, since_time, until_time, output_dir)
    print_to_stdout(content)
    print(f"{STEP} Summary written to: {out_path}")
    save_state({"last_run": until_time.isoformat()})
