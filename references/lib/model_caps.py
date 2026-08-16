"""What a served model can actually do, read from the model itself.

The context window used to be a hardcoded 8192 for every server model, then a
set of hand-written per-family guesses. Both were wrong in the same direction:
the installed models report 131072 or 262144 in their own `config.json`, so a
guess of 32768 still threw away three quarters of the available window.

Nothing in the Osaurus API reports context length — `/v1/models` and `/api/tags`
carry family, parameter size and quantization only — but the models are
safetensors on disk and every one of them states `max_position_embeddings` in
its config. That is the authoritative answer, so ask it.

Probed values are the CEILING, not the target: see `usable_context_window`.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

__all__ = [
    "MODELS_DIR",
    "measured_prefill_rate",
    "is_generative_model",
    "model_config_path",
    "model_disk_bytes",
    "probe_context_window",
    "probe_vision",
    "usable_context_window",
]

MODELS_DIR = Path(os.environ.get("MLX_MODELS_DIR", str(Path.home() / "MLXModels")))
# Osaurus serves from MLXModels, but models pulled through huggingface_hub live
# in its cache instead — potion-base-4M is only there.
HF_CACHE_DIR = Path(
    os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
) / "hub"

# A static embedding model is not a chat model, and its config lies about
# context: potion-base-4M declares `seq_length: 1000000`, which would be read as
# a one-million-token window by anything that trusted the field. Identify what a
# model IS before believing what it claims.
_NON_GENERATIVE_TYPES = frozenset({"model2vec", "sentence-transformer", "static"})
_NON_GENERATIVE_ARCHITECTURES = frozenset({"staticmodel", "sentencetransformer"})

# How much context to send is a QUALITY question, not a speed one. `tw` runs
# every six hours and `wk` once a day, so seconds spent ingesting a prompt are
# free -- there is nothing to trade them against.
#
# This file used to cap context at MAX_PREFILL_SECONDS (120) x a measured rate,
# with an 800 chars/sec floor for unmeasured models. Both numbers were invented
# here, not derived from anything, and together they threw away most of a
# 131072-token window to buy time nobody was spending. They are gone.
#
# The measured rate still has one honest use: sizing REQUEST TIMEOUTS so a large
# prompt is not killed mid-flight. That lives in twitter/budget.py. Note the
# direction flips -- for a timeout, an unknown rate must be assumed SLOW so the
# wait is long enough, whereas a throttle would assume slow to send less.
CHARS_PER_TOKEN = int(os.environ.get("TWITTER_CHARS_PER_TOKEN", "3"))


def recorded_capability(model: str, key: str):
    """A capability `ev` probed and wrote to conf/eval_signals.json, or None.

    Read from DISK, never from the server. Two reasons: production paths like
    `get_model_family` are called constantly and must not do network I/O, and the
    test suite forbids reaching a live server at all. The probe runs once, in `ev`;
    everything else consumes what it recorded.
    """
    try:
        from eval.signals import _load_eval_signals

        caps = (_load_eval_signals().get(model) or {}).get("_capabilities") or {}
        return caps.get(key)
    except Exception:
        return None


def measured_prefill_rate(model: str) -> float | None:
    """Chars/sec measured for this model by the eval, or None if never run.

    Used to size timeouts, never to decide how much context to send.
    """
    try:
        from eval.signals import _load_eval_signals

        caps = (_load_eval_signals().get(model) or {}).get("_capabilities") or {}
        rate = caps.get("prefill_chars_per_sec")
        return float(rate) if rate else None
    except Exception:
        return None


_CONTEXT_KEYS = ("max_position_embeddings", "max_seq_len", "n_positions", "seq_length")
_NESTED_KEYS = ("text_config", "llm_config", "language_config")


def _context_from_config(cfg: dict) -> int | None:
    """Context length from a HuggingFace-style config, including nested ones."""
    for key in _CONTEXT_KEYS:
        value = cfg.get(key)
        if isinstance(value, int) and value > 0:
            return value
    for sub in _NESTED_KEYS:
        nested = cfg.get(sub)
        if isinstance(nested, dict):
            found = _context_from_config(nested)
            if found:
                return found
    return None


@lru_cache(maxsize=64)
def model_config_path(model: str) -> Path | None:
    """The on-disk config.json for a served model id, if it can be found.

    Served ids are lowercased ("gemma-4-12b-it-mxfp8") while directories keep
    their original case ("gemma-4-12B-it-MXFP8"), so match case-insensitively.
    """
    if not model:
        return None
    target = model.strip().lower()
    for root in (MODELS_DIR, HF_CACHE_DIR):
        if not root.is_dir():
            continue
        for config in root.rglob("config.json"):
            # MLXModels: <Org>/<Model>/config.json
            if config.parent.name.lower() == target:
                return config
            # HF cache: models--<org>--<model>/snapshots/<sha>/config.json
            for part in config.parts:
                if part.startswith("models--") and part.split("--")[-1].lower() == target:
                    return config
    return None


def _documented_context_window(model: str) -> int | None:
    """A window declared in conf/models/<family>.toml rather than read off disk.

    Some models are not files we own -- Apple's on-device `foundation` has no
    config.json anywhere -- so the number has to be written down. It lives in
    config beside the per-model override, cited and dated there, instead of in a
    model-name table in this file: one mechanism, one place to look, and a stale
    entry is visible rather than silently authoritative.
    """
    try:
        from lib.config import get_model_config

        window = (get_model_config(model) or {}).get("context_window")
        return int(window) if window else None
    except Exception:
        return None


@lru_cache(maxsize=64)
def probe_context_window(model: str) -> int | None:
    """The model's real context length, or None when it cannot be determined.

    None means "unknown", never a guessed number: a caller that cannot find out
    should fall back to its own documented default and say so, rather than
    inherit a fabricated capability.
    """
    documented = _documented_context_window(model)
    if documented:
        return documented
    config = model_config_path(model)
    if config is None:
        return None
    try:
        cfg = json.loads(config.read_text())
    except (OSError, ValueError):
        return None
    if not _is_generative(cfg):
        return None
    return _context_from_config(cfg)


def _is_generative(cfg: dict) -> bool:
    """Whether this config describes a model that generates text at all."""
    if str(cfg.get("model_type", "")).lower() in _NON_GENERATIVE_TYPES:
        return False
    arches = {str(a).lower() for a in cfg.get("architectures") or []}
    return not (arches & _NON_GENERATIVE_ARCHITECTURES)


@lru_cache(maxsize=64)
def is_generative_model(model: str) -> bool:
    """Whether a served model can generate text, judged by its config.

    The eval harness skips `potion-base-4m` by matching its NAME. That works
    until the next embedding model arrives under a different one.
    """
    config = model_config_path(model)
    if config is None:
        # Unknown on disk: assume generative rather than silently skipping a
        # model the user installed. foundation lands here and is generative.
        return True
    try:
        return _is_generative(json.loads(config.read_text()))
    except (OSError, ValueError):
        return True


@lru_cache(maxsize=64)
def probe_family(model: str) -> str | None:
    """The model's architecture, read from its own config.json `model_type`.

    Deliberately read from DISK rather than from the server's `/api/tags`, even
    though that endpoint reports the same strings -- checked across the roster,
    `model_type` and `details.family` agree exactly ("qwen3_5", "gemma4_unified",
    "qwen3_5_moe", "nemotron_h", "muse_glimmer").

    Reading it offline matters more than it looks. The alternative put a second HTTP
    request into `ev`'s startup path, where the test suite forbids one, and where
    `fetch_roster`'s catch-all would have swallowed the block and carried on --
    which conftest correctly fails the run for. Disk has no such problem, works when
    the server is down, and costs nothing.

    None means the model is not on disk (foundation), never a guessed family.
    """
    config = model_config_path(model)
    if config is None:
        return None
    try:
        family = json.loads(config.read_text()).get("model_type")
    except (OSError, ValueError):
        return None
    return str(family) if family else None


_VISION_KEYS = ("vision_config", "vision_tower", "has_vision", "vision_start_token_id")


def _has_vision(cfg: dict) -> bool:
    """Whether this config describes a model with an image tower."""
    if any(key in cfg for key in _VISION_KEYS):
        return True
    for sub in _NESTED_KEYS:
        nested = cfg.get(sub)
        if isinstance(nested, dict) and _has_vision(nested):
            return True
    return False


@lru_cache(maxsize=64)
def probe_vision(model: str) -> bool | None:
    """Whether a served model can take images, read from its own config.json.

    None means "cannot tell" -- no config on disk -- never a guessed False, so a
    caller can distinguish "text-only" from "unknown" instead of silently ruling
    out a model it failed to find.

    This replaces guessing from the NAME. `lib/osaurus_models.DEFAULT_VLM_KEYWORDS`
    matches "vl,vision,qwen,llamavl", which on the roster this was written against
    finds the qwens and misses gemma, ornith, bonsai and muse-glimmer -- every one
    of which has a vision tower -- while failing to exclude nemotron, the only
    text-only server model. A name has never been evidence of a capability, and
    `ev --capabilities` prints the current answer rather than this frozen one.
    """
    config = model_config_path(model)
    if config is None:
        return None
    try:
        return _has_vision(json.loads(config.read_text()))
    except (OSError, ValueError):
        return None


@lru_cache(maxsize=64)
def model_disk_bytes(model: str) -> int | None:
    """Total size of a model's weight files, or None if not found on disk.

    The number that decides whether a model is usable on a given machine, and the
    one nothing was reading. osaurus mmaps weights, so a model larger than the page
    cache can hold is re-read from SSD every token: qwen3.8-27b at 27GB measured
    0.08 tok/s against 80.6 for the LARGER-parameter ornith-35b at 18GB. RSS does
    not show this -- it read 3.5GB for the 27GB model -- so parameter count and
    resident size are both the wrong instrument. On-disk bytes is the right one.

    Counts weight shards only. Tokenizers and configs are noise at this scale, and
    counting the whole directory would include any stray download artifacts.
    """
    config = model_config_path(model)
    if config is None:
        return None
    directory = config.parent
    total = 0
    try:
        for weights in directory.glob("*.safetensors"):
            total += weights.stat().st_size
    except OSError:
        return None
    return total or None


def usable_context_window(model: str, default: int, override: int | None = None) -> int:
    """How many tokens to size a prompt against.

    Precedence: an explicit `override` (a per-model config entry), then the
    window probed from the model's own config.json, then `default`.

    Deliberately NOT capped by how long the prompt takes to ingest. That cap
    existed, and it cost quality for a speed nobody needed. If a shorter prompt
    ever turns out to produce BETTER output -- long-context attention really
    does degrade -- that is a finding for the eval to make and for a
    `context_window` entry in conf/models/*.toml to encode, per model, with the
    evidence beside it. It is not something to assume here.
    """
    if override:
        return int(override)
    probed = probe_context_window(model)
    if probed:
        return probed
    return default
