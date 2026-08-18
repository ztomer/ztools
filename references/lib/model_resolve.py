"""Resolve a configured model name against the roster the server can actually serve.

`conf/config.toml` names models by their server tag (``qwen3.6-35b-a3b-mxfp8-mtp``).
That tag is not a stable identity: models get deleted, upgraded and renamed on disk
underneath a config that still names the old one. When that happens the server
answers every request with

    HTTP 404 {"error": {"message": "Model 'X' is not installed or registered ..."}}

and every tool that routed to X dies with a message about HTTP status codes rather
than about the real problem, which is that the config is stale.

That is the same failure shape the Foundation fallback already handles for a dead
server (``lib.osaurus_lib._try_foundation``): a dependency's fatal path is reachable,
so probe it and degrade with a *stated reason* instead of dying. This module is the
probe-and-degrade half; the retry is wired into the two ``call`` sites.

Nothing here rewrites the user's config. A substitution is a stopgap that says so out
loud on every use — the fix is to re-derive ``best_models`` from an eval sweep.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from lib.llm.constants import API_TAGS, DEFAULT_HOST, DEFAULT_PORT

ROSTER_TIMEOUT = 10

#: Substring of the server's 404 body that identifies a stale model tag specifically,
#: as opposed to a 404 from a wrong URL path.
MISSING_MODEL_MARKERS = ("is not installed", "not registered with any provider")


def is_missing_model_error(status_code: int, body: str) -> bool:
    """True when a 404 means "that model tag is gone", not "wrong endpoint"."""
    if status_code != 404:
        return False
    lowered = (body or "").lower()
    return any(marker in lowered for marker in MISSING_MODEL_MARKERS)


def fetch_roster(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> List[Dict]:
    """Return ``/api/tags`` entries, or ``[]`` if the server cannot be asked.

    ``[]`` is deliberately indistinguishable from "server down": both mean we have no
    evidence about what is installed, and callers must treat no-evidence as "change
    nothing" rather than as "nothing is installed".
    """
    url = host.rstrip("/") if "://" in host else f"http://{host}:{port}"
    try:
        resp = requests.get(f"{url}{API_TAGS}", timeout=ROSTER_TIMEOUT)
        if resp.status_code != 200:
            return []
        entries = [m for m in resp.json().get("models", []) if m.get("model")]
    except Exception:
        return []
    return _drop_uncorroborated(entries)


def _drop_uncorroborated(entries: List[Dict]) -> List[Dict]:
    """Remove roster entries with nothing on disk behind them.

    Done HERE, at the boundary where the claim enters the process, rather than inside
    `substitute_model`. Putting it in the selector made a pure function depend on the
    filesystem: the same roster produced different answers on different machines, and
    a unit test with synthetic model names silently started consulting real
    directories. Filtering at the fetch keeps selection pure and gives every
    downstream consumer -- substitution, the audit, the TUI -- the same trustworthy
    list.

    If NOTHING is corroborated the original list is returned unchanged. That case is
    far more likely to mean the probe is broken than that the server is serving twelve
    models which do not exist, and silently emptying the roster would turn a
    diagnosable problem into "no models installed".
    """
    kept = [e for e in entries if disk_corroborated(e.get("model", ""))]
    return kept if kept else entries


def _parameter_billions(entry: Dict) -> float:
    """Parse ``details.parameter_size`` ("27B", "4M", "") into billions, 0.0 if absent."""
    raw = (entry.get("details") or {}).get("parameter_size") or ""
    raw = raw.strip().upper()
    if not raw:
        return 0.0
    scale = {"B": 1.0, "M": 0.001, "K": 0.000001}.get(raw[-1:])
    if scale is None:
        return 0.0
    try:
        return float(raw[:-1]) * scale
    except ValueError:
        return 0.0


def _name_family(model: str) -> str:
    from lib.config import get_model_family

    return get_model_family(model)


def _rank(entry: Dict) -> Tuple[float, str]:
    """Sort key: biggest model first, then name, so the pick is deterministic.

    Size is the tiebreak rather than the name because a name-sorted pick silently
    tracks version-string formatting ("qwen3.10" sorts below "qwen3.8"), which is a
    property of ASCII rather than of the model.
    """
    return (-_parameter_billions(entry), entry.get("model", ""))


def substitute_model(
    configured: str, roster: Sequence[Dict]
) -> Tuple[str, Optional[str]]:
    """Pick a servable stand-in for ``configured``.

    Returns ``(model, reason)``. ``reason`` is None when nothing was substituted —
    either because ``configured`` is installed, or because the roster is empty and we
    have no grounds to override the caller. It is a human-readable sentence otherwise,
    and every caller is expected to surface it rather than swallow it.
    """
    if not roster:
        return configured, None
    installed = {e["model"] for e in roster}
    if configured in installed:
        return configured, None

    family = _name_family(configured)
    if family != "default":
        same = [e for e in roster if _name_family(e["model"]) == family]
        if same:
            pick = sorted(same, key=_rank)[0]["model"]
            return pick, (
                f"model '{configured}' is not installed; using '{pick}' "
                f"(largest installed '{family}' model). Re-derive best_models."
            )

    from lib.config import get_model_fallback_chain

    for preferred in get_model_fallback_chain():
        matches = [e for e in roster if preferred in e["model"].lower()]
        if matches:
            pick = sorted(matches, key=_rank)[0]["model"]
            return pick, (
                f"model '{configured}' is not installed and no '{family}' model is "
                f"either; falling back to '{pick}'. Re-derive best_models."
            )

    pick = sorted(roster, key=_rank)[0]["model"]
    return pick, (
        f"model '{configured}' is not installed and nothing in the preference chain "
        f"is either; falling back to '{pick}'. Re-derive best_models."
    )


#: conf/<tool>.toml keys that name model tags. `rn` keeps its own preferences here
#: rather than in conf/config.toml, so the audit did not look at them -- and both
#: entries under `vlm_preferred` named models that had been uninstalled for months
#: with nothing reporting it. An audit that covers only the file you remembered is
#: how drift stays invisible in the file you did not.
SIDECAR_MODEL_KEYS = {
    "rename.toml": ("vlm_preferred", "text_preferred", "relevance_check_models"),
}


def _sidecar_model_slots() -> List[Tuple[str, str]]:
    """(slot, model) pairs from the per-tool config files."""
    from lib.config_toml import load_config
    from lib.paths import conf_path

    out: List[Tuple[str, str]] = []
    for filename, keys in SIDECAR_MODEL_KEYS.items():
        try:
            cfg = load_config(conf_path(filename)) or {}
        except Exception:
            continue
        for key in keys:
            value = cfg.get(key)
            if isinstance(value, str):
                value = [v.strip() for v in value.split(",") if v.strip()]
            for i, model in enumerate(value or []):
                if isinstance(model, str) and model:
                    out.append((f"{filename}:{key}[{i}]", model))
    return out


def audit_configured_models(
    installed: Optional[Iterable[str]] = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
) -> Dict[str, List[str]]:
    """Report which configured model names the server cannot serve.

    Returns ``{"installed": [...], "missing": [...], "unreachable": bool}`` keyed by
    the config slot ("best_models.summarize", "default_model", "filename_models[1]")
    so the caller can name the exact line to edit.

    Takes model IDs rather than full roster entries, and callers that already hold a
    roster should pass it: a caller which has just listed the models does not need a
    second request, and issuing one anyway put a network call into a code path the
    test suite deliberately forbids from reaching a server.
    """
    from lib.config import get_best_models, get_filename_models
    from lib.config_core import _auto_load, _config

    if installed is None:
        installed = [e["model"] for e in fetch_roster(host, port)]
    installed = set(installed)
    if not installed:
        return {"installed": [], "missing": [], "unreachable": True}

    _auto_load()
    slots: List[Tuple[str, str]] = []
    default_model = _config.get("default_model")
    if default_model:
        slots.append(("default_model", default_model))
    for task, model in sorted(get_best_models().items()):
        slots.append((f"best_models.{task}", model))
    for i, model in enumerate(get_filename_models()):
        slots.append((f"filename_models[{i}]", model))
    slots.extend(_sidecar_model_slots())

    ok = [f"{slot} = {model}" for slot, model in slots if model in installed]
    bad = [f"{slot} = {model}" for slot, model in slots if model not in installed]
    # A slot naming a model the roster advertises but disk does not back is NOT
    # "installed" -- it is a call-time failure waiting to happen, and reporting it as
    # healthy is exactly how a deleted model keeps its slot until someone runs the tool.
    stale = sorted({
        model for _slot, model in slots
        if model in installed and not disk_corroborated(model)
    })
    return {"installed": ok, "missing": bad, "stale": stale, "unreachable": False}


def format_audit(report: Dict[str, List[str]]) -> List[str]:
    """Render an audit as lines for the console. Empty list means nothing to say.

    Silent on a clean config and silent on an unreachable server — `ev` already tells
    the user the server is down, and a second message saying the roster could not be
    read is noise on top of a diagnosis they have.
    """
    lines: List[str] = []
    if report.get("stale"):
        lines.append(
            "the server advertises "
            f"{len(report['stale'])} model(s) with nothing on disk behind them; its "
            "roster is cached and a restart is overdue:"
        )
        lines.extend(f"  {name}" for name in report["stale"])
        lines.append("Fix with: ./tools/osaurus_one.sh --restart")
    if report.get("unreachable") or not report.get("missing"):
        return lines
    lines.append(
        f"{len(report['missing'])} configured model(s) the server cannot serve; each "
        "falls back at call time with a warning. The slot name gives the file:"
    )
    lines.extend(f"  {slot}" for slot in report["missing"])
    lines.append("Re-derive these from an eval sweep rather than editing them by hand.")
    return lines


def disk_corroborated(model: str) -> bool:
    """Does anything on this machine back up the roster's claim to serve `model`?

    `/api/tags` is a CLAIM, not proof. osaurus keeps its roster in memory, so a model
    deleted from disk stays advertised until the server is restarted -- and a request
    for it then hangs or 404s rather than failing at selection time, which is the
    point where a stand-in could still have been chosen. That happened twice in one
    day, with qwen3.8-27b-4bit and nemotron.

    Corroboration is `probe_context_window(model) is not None`, which is true by two
    independent routes and false only when neither holds:

        a config.json on disk        every model osaurus serves from MODELS_DIR
        a documented context window  conf/models/<family>.toml, for on-device models

    Deliberately NOT "the model reports a context window": an embedding model has
    files and a config but no max_position_embeddings, and treating that as absent
    drops a model that is plainly installed.

    That second route is why this is not a plain file check. `foundation` is Apple's
    on-device model: it has no config.json and never will, so "no files therefore
    gone" would discard the single most reliable model in the roster. Measured:

        foundation             no config.json, documented 4096  -> corroborated
        gemma-4-12b-it-mxfp8   config.json found                -> corroborated
        qwen3.8-27b-4bit       neither (deleted)                -> NOT corroborated
    """
    from lib.model_caps import _documented_context_window, model_config_path

    try:
        # "Has a config.json on disk" OR "is documented in conf/models/". NOT
        # `probe_context_window is not None`, which is how this was first written and
        # is a different question: it additionally requires the config to CONTAIN a
        # context window. `potion-base-4m` is an embedding model -- 15MB and a
        # config.json on disk, no max_position_embeddings -- so it was dropped from
        # the roster as absent. Exactly the false positive this function's own
        # docstring promises not to make.
        # bool(), not the raw `or` result: the annotation says bool, and returning
        # 4096 or None instead would make `is True` / `is False` checks lie.
        return bool(model_config_path(model) is not None or _documented_context_window(model))
    except Exception:
        # An unreadable probe is not evidence of absence. Keep the entry: wrongly
        # dropping a servable model is worse than keeping a stale one, because the
        # stale one still degrades loudly at call time.
        return True


def stale_roster_entries(roster: Iterable[Dict]) -> List[str]:
    """Roster entries with nothing on disk behind them, i.e. a restart is overdue."""
    return [
        entry["model"]
        for entry in (roster or [])
        if entry.get("model") and not disk_corroborated(entry["model"])
    ]
