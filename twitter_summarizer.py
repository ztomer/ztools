#!/usr/bin/env python3
# /// script
# dependencies = ["playwright", "requests", "cryptography", "mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git", "transformers", "pyyaml"]
# ///
"""
twitter_summarizer.py — Fetch your X/Twitter home timeline via browser automation,
summarize with a local LLM, and export to Markdown.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Try to import optional dependencies
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
from lib.mlx_lib import (
    find_mlx_model,
    find_best_mlx_model,
)

# MLX model preferences (prefer smaller, faster models)
MLX_PREFERRED = [
    "qwen2.5-0.5b",
    "qwen2.5-1b",
    "llama-1b",
    "phi-4",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_FILE = Path.home() / ".twitter_summary_state.json"
DEBUG_CACHE_FILE = Path.home() / ".twitter_summary_debug_cache.json"
DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "twitter_summaries"
DEFAULT_OLLAMA_URL = "http://localhost:1337"

MODELS = [
    "qwen3.5-27b-claude-4.6-opus-distilled-mlx-4bit",  # Best: 100% in 228s
    "gemma-4-e2b-it-8bit",  # Fast: 80% in 15s
    "foundation",  # Fast but low quality
]


# Use get_best_model from consolidated osaurus_lib
from lib.config import Task
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", get_best_model(Task.SUMMARIZE))
CHROME_COOKIES_DB = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Google"
    / "Chrome"
    / "Default"
    / "Cookies"
)
MAX_SCROLLS = 1200
SCROLL_PAUSE_MS = 1800

# ---------------------------------------------------------------------------
# Chrome cookie extraction (macOS)
# ---------------------------------------------------------------------------


def _get_chrome_keychain_key() -> bytes:
    password = subprocess.check_output(
        ["security", "find-generic-password", "-w", "-s", "Chrome Safe Storage"],
        stderr=subprocess.DEVNULL,
    ).strip()
    return hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1003, dklen=16)


def _decrypt_cookie(encrypted_value: bytes, key: bytes) -> str:
    if not encrypted_value or encrypted_value[:3] != b"v10":
        return encrypted_value.decode("utf-8", errors="replace")

    if not all((Cipher, algorithms, modes)):
        return encrypted_value.decode("utf-8", errors="replace")

    try:
        iv = encrypted_value[3:19]
        ciphertext = encrypted_value[19:]
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        raw = decryptor.update(ciphertext) + decryptor.finalize()
        pad = raw[-1]
        return raw[16:-pad].decode("utf-8", errors="replace")
    except Exception:
        return encrypted_value.decode("utf-8", errors="replace")


def get_chrome_cookies(
    domains: tuple[str, ...] = (".twitter.com", ".x.com"),
) -> list[dict]:
    if not CHROME_COOKIES_DB.exists():
        print(f"[!] Chrome Cookies DB not found at {CHROME_COOKIES_DB}")
        sys.exit(1)

    tmp_db = Path(tempfile.mktemp(suffix=".db"))
    shutil.copy2(CHROME_COOKIES_DB, tmp_db)

    try:
        key = _get_chrome_keychain_key()
        domain_clauses = " OR ".join(f"host_key LIKE '%{d}'" for d in domains)
        conn = sqlite3.connect(f"file:{tmp_db}?mode=ro", uri=True)
        rows = conn.execute(
            f"SELECT name, encrypted_value, value, path, host_key, expires_utc, is_secure "
            f"FROM cookies WHERE {domain_clauses}"
        ).fetchall()
        conn.close()
    finally:
        tmp_db.unlink(missing_ok=True)

    cookies = []
    for name, enc_val, plain_val, path, host_key, expires_utc, is_secure in rows:
        value = _decrypt_cookie(enc_val, key) if enc_val else plain_val
        if not value:
            continue
        cookie: dict = {
            "name": name,
            "value": value,
            "domain": host_key,
            "path": path or "/",
            "secure": bool(is_secure),
        }
        if expires_utc:
            unix_ts = int((expires_utc / 1_000_000) - 11_644_473_600)
            if unix_ts > 0:
                cookie["expires"] = unix_ts
        cookies.append(cookie)
    return cookies


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


_PROMPT_RULES = """
- Use headers starting with ##
- Use bullet points for facts
- Keep it concise and factual
"""
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_debug_cache() -> list[dict]:
    if DEBUG_CACHE_FILE.exists():
        try:
            tweets = json.loads(DEBUG_CACHE_FILE.read_text())
            for tweet in tweets:
                if "created_at" in tweet and isinstance(tweet["created_at"], str):
                    tweet["created_at"] = datetime.fromisoformat(tweet["created_at"])
            return tweets
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_debug_cache(tweets: list[dict]) -> None:
    def serialize(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    DEBUG_CACHE_FILE.write_text(json.dumps(tweets, indent=2, default=serialize))


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


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
                f"[!] Cannot parse --since '{args_since}'. Using last run or 24h.")
    if "last_run" in state:
        return datetime.fromisoformat(state["last_run"])
    return datetime.now(timezone.utc) - timedelta(hours=24)


# ---------------------------------------------------------------------------
# Tweet parsing
# ---------------------------------------------------------------------------


def parse_tweets_from_response(data: dict) -> list[dict]:
    tweets = []
    try:
        instructions = (
            data.get("data", {})
            .get("home", {})
            .get("home_timeline_urt", {})
            .get("instructions", [])
        )
        for instruction in instructions:
            if instruction.get("type") != "TimelineAddEntries":
                continue
            for entry in instruction.get("entries", []):
                content = entry.get("content", {})
                item_content = content.get("itemContent", {})
                if item_content.get("itemType") != "TimelineTweet":
                    continue
                tweet_result = item_content.get(
                    "tweet_results", {}).get("result", {})
                if tweet_result.get("__typename") == "TweetWithVisibilityResults":
                    tweet_result = tweet_result.get("tweet", tweet_result)

                legacy = tweet_result.get("legacy", {})
                user_result = (
                    tweet_result.get("core", {})
                    .get("user_results", {})
                    .get("result", {})
                )
                user_core = user_result.get("core", {})
                user_legacy = user_result.get("legacy", {})
                full_text = legacy.get("full_text", "")
                created_at_str = legacy.get("created_at", "")
                screen_name = (
                    user_core.get("screen_name")
                    or user_legacy.get("screen_name")
                    or "unknown"
                )

                if not full_text or not created_at_str:
                    continue

                try:
                    created_at = datetime.strptime(
                        created_at_str, "%a %b %d %H:%M:%S +0000 %Y"
                    )
                    created_at = created_at.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

                tweets.append(
                    {
                        "screen_name": screen_name,
                        "text": full_text,
                        "created_at": created_at,
                    }
                )
    except Exception as e:
        if os.environ.get("DEBUG"):
            print(f"[parse] Error: {e}")
    return tweets


# ---------------------------------------------------------------------------
# Browser automation
# ---------------------------------------------------------------------------


def collect_tweets_via_browser(since_time: datetime, debug: bool) -> list[dict]:
    print("[cookies] Extracting Twitter/X cookies from Chrome profile ...")
    cookies = get_chrome_cookies()
    if not cookies:
        print("[!] No Twitter/X cookies found. Are you logged in to x.com in Chrome?")
        sys.exit(1)

    all_tweets: list[dict] = []
    oldest_seen: datetime | None = None

    def handle_response(response):
        nonlocal oldest_seen
        url = response.url
        if "HomeTimeline" not in url and "HomeLatestTimeline" not in url:
            return
        try:
            body = response.json()
        except Exception:
            return
        batch = parse_tweets_from_response(body)
        for tweet in batch:
            all_tweets.append(tweet)
            if oldest_seen is None or tweet["created_at"] < oldest_seen:
                oldest_seen = tweet["created_at"]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not debug)
        context = browser.new_context()
        for cookie in cookies:
            try:
                context.add_cookies([cookie])
            except Exception:
                pass

        page = context.new_page()
        page.on("response", handle_response)

        try:
            page.goto(
                "https://x.com/home", wait_until="domcontentloaded", timeout=30000
            )
        except PWTimeout:
            pass

        time.sleep(3)
        if any(
            kw in page.title().lower()
            for kw in ("log in", "login", "sign in", "signin")
        ):
            print(
                "[!] Not logged in — cookies may be stale. Log into x.com in Chrome and retry."
            )
            sys.exit(1)

        try:
            following_tab = page.locator(
                '[role="tab"]', has_text="Following").first
            following_tab.click(timeout=5000)
            time.sleep(2)
        except Exception:
            pass

        print(
            f"[browser] Scrolling timeline (collecting tweets since {since_time.isoformat()}) ..."
        )
        scrolls = 0
        while scrolls < MAX_SCROLLS:
            page.evaluate("window.scrollBy(0, window.innerHeight * 2)")
            time.sleep(SCROLL_PAUSE_MS / 1000)
            scrolls += 1
            if oldest_seen and oldest_seen < since_time:
                break

        context.close()
        browser.close()

    filtered = [t for t in all_tweets if t["created_at"] >= since_time]
    rt_prefix = re.compile(r"^RT @\w+: ")
    seen_exact = set()
    seen_content = set()
    unique = []
    for t in filtered:
        exact_key = (t["screen_name"], t["text"][:80])
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)

        content_key = rt_prefix.sub("", t["text"])[:100]
        if content_key in seen_content:
            continue
        seen_content.add(content_key)

        unique.append(t)

    unique.sort(key=lambda t: t["created_at"])
    return unique


# ---------------------------------------------------------------------------
# LLM summarization
# ---------------------------------------------------------------------------


CHARS_PER_TOKEN = 3
OUTPUT_RESERVE_TOKENS = 4096
MLX_MODELS_DIR = Path.home() / "MLXModels"


# Use consolidated functions from osaurus_lib and mlx_lib
# Remove duplicates - use library functions instead


def _build_prompt(
    tweets: list[dict], max_chars: int, for_mlx: bool = False, model: str = None
) -> tuple[str, int]:
    from lib.config import get_model_prompt, Task

    # Build timeline content first
    budget = max_chars - 200
    lines = []
    used = 0
    for t in reversed(tweets):
        line = f"[@{t['screen_name']} | {t['created_at'].strftime('%H:%M')}]: {t['text'].strip()}"
        if used + len(line) + 1 > budget:
            break
        lines.append(line)
        used += len(line) + 1
    lines.reverse()
    timeline = "\n".join(lines)

    # Get prompt from config and inject timeline
    prompt_template = get_model_prompt(model, Task.SUMMARIZE)
    if not prompt_template:
        prompt_template = "Summarize this timeline:\n\n{}\n\nUse ## headers for topics."

    if "{}" in prompt_template:
        prompt = prompt_template.format(timeline)
    else:
        prompt = prompt_template

    return prompt, len(lines)


def summarize_with_llm(
    tweets: list[dict], base_url: str, model: str, api_key: str = ""
) -> str:
    # Try Osaurus server first
    from lib.osaurus_lib import (
        call_llm_api,
        strip_thinking,
        get_available_models,
        get_best_model,
        select_best_model,
    )

    models = get_available_models()
    target_model = model if model else get_best_model("summarize")
    if models and target_model not in models:
        target_model = select_best_model(models) or target_model

    ctx_chars = (8192 - OUTPUT_RESERVE_TOKENS) * CHARS_PER_TOKEN
    prompt, n = _build_prompt(tweets, max_chars=ctx_chars, model=target_model)

    # Try models in order of preference
    fallback_models = [target_model, "qwen3.6-35b-a3b-mxfp4", "foundation"]
    tried = set()
    failed_models = []

    for try_model in fallback_models:
        if try_model in tried or try_model not in models:
            tried.add(try_model)
            continue
        tried.add(try_model)
        failed_models.append(try_model)

        # Rebuild prompt for each model
        prompt, n = _build_prompt(tweets, max_chars=ctx_chars, model=try_model)

        try:
            print(f"[llm] Trying {try_model} ({n} tweets)...")
            result = call_llm_api(
                f"{base_url.rstrip('/')}",
                try_model,
                [{"role": "user", "content": prompt}],
                api_key=api_key,
                timeout=120,
            )
            if result and "content" in result:
                from lib.osaurus_lib import merge_thinking_with_summary, extract_thinking
                thinking, cleaned = extract_thinking(result["content"])
                if thinking:
                    print(f"[llm] {try_model}: included thinking block")
                    return merge_thinking_with_summary(thinking, cleaned)
                return cleaned
            elif result and "error" in result:
                print(f"[llm] {try_model} error: {result['error'][:50]}")
        except Exception as e:
            print(f"[llm] {try_model} failed: {str(e)[:50]}")
            continue

    # All models failed, try MLX fallback
    print("[llm] Server models failed, trying MLX...")

    # Fall back to MLX
    mlx_path = find_mlx_model(target_model) or find_best_mlx_model(MLX_PREFERRED)

    if mlx_path:
        from lib.mlx_lib import get_mlx_context_length, call_mlx, process_mlx_content

        mlx_ctx = get_mlx_context_length(mlx_path)
        mlx_prompt_chars = (mlx_ctx - OUTPUT_RESERVE_TOKENS) * CHARS_PER_TOKEN
        prompt, n = _build_prompt(
            tweets, max_chars=mlx_prompt_chars, for_mlx=True, model=mlx_path.name)
        print(
            f"[llm] Sending {n}/{len(tweets)} tweets to local MLX model {mlx_path.name} ..."
        )
        raw = call_mlx(mlx_path, prompt)
        if raw and not raw.startswith("[LLM error"):
            return process_mlx_content(raw)
        print(f"[llm] MLX error: {raw}")

    print(f"[llm] All server models failed: {failed_models}")
    return "[LLM error: both local MLX and server failed]"


def print_to_stdout(content: str) -> None:
    if shutil.which("bat"):
        try:
            subprocess.run(
                ["bat", "-l", "md", "--color", "always", "--style", "plain"],
                input=content,
                text=True,
                check=True,
            )
        except Exception:
            print(content)
    else:
        print(content)


def write_markdown(
    tweets: list[dict],
    summary: str,
    since_time: datetime,
    until_time: datetime,
    output_dir: Path,
) -> tuple[Path, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = "%Y-%m-%d_%H%M"
    filename = f"{since_time.strftime(fmt)}_to_{until_time.strftime(fmt)}.md"
    out_path = output_dir / filename
    unique_authors = len({t["screen_name"] for t in tweets})
    period_str = f"{since_time.strftime('%Y-%m-%d %H:%M')} → {until_time.strftime('%Y-%m-%d %H:%M')} UTC"

    content = f"""# Twitter Timeline Summary

**Period:** {period_str}
**Tweets:** {len(tweets)} from {unique_authors} accounts

## Summary

{summary}

---
*Generated by twitter_summarizer.py on {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}*
"""
    out_path.write_text(content)
    return out_path, content


def clean_folder(output_dir: Path) -> None:
    """Deletes all .md files in the specified directory and terminates execution."""
    if not output_dir.exists():
        print(
            f"[clean] Directory {output_dir} does not exist. Nothing to clean. Exiting."
        )
        sys.exit(0)

    print(f"[clean] Removing existing .md files in {output_dir} ...")
    for md_file in output_dir.glob("*.md"):
        try:
            md_file.unlink()
        except OSError as e:
            print(f"[!] Failed to delete {md_file}: {e}")

    print("[clean] Cleanup complete. Exiting.")
    sys.exit(0)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output).expanduser()

    if args.clean:
        clean_folder(output_dir)

    state = load_state()
    since_time = resolve_since_time(args.since, state)
    until_time = datetime.now(timezone.utc)

    if args.use_cache:
        tweets = load_debug_cache()
        if tweets:
            print(f"[cache] Using {len(tweets)} cached tweets")
        else:
            print("[!] No cached tweets found. Run without --use-cache first.")
            sys.exit(1)
    else:
        tweets = collect_tweets_via_browser(since_time, debug=args.debug)
        if not tweets:
            print("[!] No tweets found.")
            sys.exit(0)

    summary = summarize_with_llm(
        tweets, args.base_url, args.model, api_key=args.api_key
    )
    if summary.startswith("[LLM error"):
        print(f"[!] {summary}\n[!] Aborting.")
        sys.exit(1)

    out_path, content = write_markdown(
        tweets, summary, since_time, until_time, output_dir
    )
    print("-" * 40)
    print_to_stdout(content)
    print("-" * 40)
    print(f"[ok] Summary written to: {out_path}")
    save_state({"last_run": until_time.isoformat()})


if __name__ == "__main__":
    init_config()
    main()
