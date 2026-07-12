"""TOML config loader."""
import tomllib
from pathlib import Path
from typing import Any, Dict


def _try_toml(path: Path) -> dict | None:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return None


def load_config(path: str | Path, default: dict | None = None) -> Dict[str, Any]:
    p = Path(path)
    toml_path = p.with_suffix(".toml") if p.suffix != ".toml" else p
    data = _try_toml(toml_path)
    if data is not None:
        return data
    if default is not None:
        return default
    return {}
