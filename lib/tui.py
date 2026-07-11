import os
import sys
from pathlib import Path

_NO_COLOR = os.environ.get("NO_COLOR") == "1" or not sys.stdout.isatty()

_cfg = {}
_zstyle_path = os.environ.get("ZSTYLE_CONFIG", str(Path.home() / ".config" / "zstyle"))
try:
    with open(_zstyle_path) as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                _cfg[k.strip()] = v.strip().strip("\"'").replace(r"\033", "\033")
except OSError:
    pass

STEP = _cfg.get("ICON_STEP", "·")
WARN = _cfg.get("ICON_WARN", "⚠")
FAIL = _cfg.get("ICON_ERR", "✗")
OK = _cfg.get("ICON_OK", "✓")
DEBUG = False


def _style(text, code):
    return f"\033[{code}m{text}\033[0m" if not _NO_COLOR else text


def info(text):
    print(f"{STEP} {text}")


def ok(text):
    print(f"{OK} {text}")


def warn(text):
    print(f"{WARN} {_style(text, '33')}")


def err(text):
    print(f"{FAIL} {_style(text, '31')}")


def die(text, code=1):
    err(text)
    sys.exit(code)


def section(text):
    print(f"\n{STEP} {text}")


def hr():
    print("\u2500" * 60)


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
