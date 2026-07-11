import os

_cfg = {}
try:
    _zstyle_path = os.environ.get("ZSTYLE_CONFIG", os.path.expanduser("~/.config/zstyle"))
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
DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
