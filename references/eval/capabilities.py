"""What each installed model IS, probed rather than guessed, and recorded.

Every fact in here was at some point a name match, a hand-written table, or a
conclusion someone reached once in a terminal and wrote into a markdown file. All
three rot the moment the roster changes, and the roster changed underneath this repo
without anyone noticing: four of seven configured tasks pointed at deleted models and
answered HTTP 404 for an unknown length of time.

So the rule here is: a capability is DERIVED, from the model or the server, by code
that can be re-run. If it cannot be derived it is reported as None -- never as a
plausible default, because a fabricated capability is worse than a missing one.

The three sources, in order of authority:

  the model's own config.json   vision tower, context window, whether it generates
                                text at all, and its weight-file size
  the server's /api/tags        the real architecture family (`qwen3_5`,
                                `gemma4_unified`, `muse_glimmer`), parameter size,
                                quantization -- none of which the NAME reliably
                                encodes
  a measurement                 prefill, decode, cold start (eval/prefill.py)

`viability` is the one derived judgement, and it is arithmetic rather than opinion:
a model that cannot emit a TYPICAL answer inside its configured timeout cannot do the
task, however good it is. That is what made a 27GB qwen3.8 useless here while a 15GB
build of the SAME model is fine.

The word "typical" is load-bearing and was got wrong first time round. Sizing against
`max_tokens` (16000, a runaway ceiling) rather than observed output (~1000) declared
gemma-4-12b unusable -- a model that scores 100% on summarize. An instrument that
condemns a known-good model is measuring the wrong quantity, however correct its
arithmetic.
"""

from typing import Dict, List, Optional

from lib.model_caps import (
    is_generative_model,
    model_disk_bytes,
    probe_context_window,
    probe_family,
    probe_model_defects,
    probe_vision,
)

#: Below this many tokens/sec, treat a decode measurement as evidence the model is
#: thrashing rather than merely slow. Not a preference: 1 tok/s cannot produce even a
#: short answer inside any timeout this repo configures.
THRASHING_DECODE_TOKENS_PER_SEC = 1.0


def roster_entry(model: str, roster: List[Dict]) -> Dict:
    """The /api/tags record for a model, or {} when the server did not list it."""
    for entry in roster or []:
        if entry.get("model") == model:
            return entry
    return {}


def probe_static_capabilities(model: str, roster: Optional[List[Dict]] = None) -> Dict:
    """Everything derivable WITHOUT running a task against the model.

    Cheap: reads files and one already-fetched roster. Safe to call while another
    model is loaded, because it never sends a generation request.
    """
    details = (roster_entry(model, roster or []).get("details") or {}) if roster else {}
    return {
        # The real architecture, which is the correct key for per-family prompt
        # config: matching on the model NAME calls bonsai and ornith "default" when
        # both are qwen3_5, sending them built-in fallback prompts while
        # conf/models/qwen.toml sits unused.
        #
        # The server and the model's own config.json report identical strings here,
        # so prefer the roster when a caller already has one and read disk otherwise.
        # The disk path is what makes this callable with no server at all.
        "family": details.get("family") or probe_family(model),
        "parameter_size": details.get("parameter_size") or None,
        "quantization": details.get("quantization_level") or None,
        "vision": probe_vision(model),
        "generative": is_generative_model(model),
        "context_window": probe_context_window(model),
        "disk_bytes": model_disk_bytes(model),
        "defects": probe_model_defects(model),
    }


def expected_output_tokens() -> int:
    """How many tokens a task actually produces, not the cap it is allowed.

    This distinction is the whole difference between a useful viability test and a
    false alarm. `max_tokens` in conf/config.toml is 16000 — a CEILING that stops a
    runaway generation, not an expectation. Sizing against it declared gemma-4-12b
    (18.3 tok/s) unusable, a model that scores 100% on summarize, because 16000
    tokens at 18.3 tok/s exceeds a 600s timeout. No summarize task has ever emitted
    16000 tokens.

    twitter/budget.py already carries the measured figure and cites it: "observed
    summaries run ~900-1100 tokens". Imported rather than copied so the two cannot
    drift into disagreeing about the same quantity.
    """
    from twitter.budget import OUTPUT_RESERVE_TOKENS

    return OUTPUT_RESERVE_TOKENS


def required_decode_rate(output_tokens: int, timeout_seconds: int) -> Optional[float]:
    """Tokens/sec a model must sustain to emit `output_tokens` before the timeout.

    The whole viability test, in one division. Pass the EXPECTED output, not the cap.
    """
    if not output_tokens or not timeout_seconds or timeout_seconds <= 0:
        return None
    return output_tokens / timeout_seconds


def assess_viability(capabilities: Dict, output_tokens: int, timeout_seconds: int) -> Dict:
    """Can this model finish its configured task, given what was measured?

    Returns a verdict plus the arithmetic behind it, because a bare "unusable" is
    the kind of claim that gets argued with rather than acted on.

    "unknown" is a real answer: an unmeasured model is not assumed good OR bad.
    """
    defects = capabilities.get("defects") or []
    decode = capabilities.get("decode_tokens_per_sec")
    needed = required_decode_rate(output_tokens, timeout_seconds)

    if defects:
        return {
            "verdict": "broken",
            "measured_decode": decode,
            "required_decode": round(needed, 2) if needed else None,
            "seconds_for_output": round(output_tokens / decode, 1) if decode else None,
            "defects": defects,
        }

    if decode is None or needed is None:
        return {"verdict": "unknown", "measured_decode": decode, "required_decode": needed}

    if decode < THRASHING_DECODE_TOKENS_PER_SEC:
        verdict = "thrashing"
    elif decode < needed:
        verdict = "too_slow"
    else:
        verdict = "ok"
    return {
        "verdict": verdict,
        "measured_decode": decode,
        "required_decode": round(needed, 2),
        "seconds_for_output": round(output_tokens / decode, 1) if decode else None,
    }


def explain_viability(model: str, assessment: Dict, disk_bytes: Optional[int]) -> str:
    """One line a human can act on, naming the number that decided it."""
    verdict = assessment.get("verdict")
    decode = assessment.get("measured_decode")
    needed = assessment.get("required_decode")
    gb = f"{disk_bytes / 1_073_741_824:.1f}GB" if disk_bytes else "unknown size"

    if verdict == "broken":
        defects = "; ".join(assessment.get("defects") or ["unidentified packaging defect"])
        return f"{model}: BROKEN — {defects}. Remove broken artifact or pull clean build."
    if verdict == "unknown":
        return f"{model}: never measured — run `ev --model {model}` before trusting it"
    if verdict == "thrashing":
        return (
            f"{model}: {decode} tok/s at {gb} — the weights do not fit this machine's "
            f"page cache, so they are re-read from disk every token. A SMALLER QUANT "
            f"fixes this; a faster kernel does not exist to be found."
        )
    if verdict == "too_slow":
        return (
            f"{model}: {decode} tok/s but a typical answer needs {needed} tok/s "
            f"({assessment.get('seconds_for_output')}s for a typical answer) — it will "
            f"be killed mid-generation and produce nothing"
        )
    return f"{model}: {decode} tok/s, clears the {needed} tok/s a typical answer needs"


def capability_report(models: List[str], roster: List[Dict], signals: Dict) -> List[Dict]:
    """Assemble static probes + recorded measurements into one row per model.

    The report the roster table in docs/MODEL_QUIRKS.md was written by hand, twice,
    and was stale both times.
    """
    rows = []
    for model in models:
        caps = probe_static_capabilities(model, roster)
        measured = (signals.get(model) or {}).get("_capabilities") or {}
        caps.update(
            {
                "prefill_chars_per_sec": measured.get("prefill_chars_per_sec"),
                "decode_tokens_per_sec": measured.get("decode_tokens_per_sec"),
                "cold_start_seconds": measured.get("cold_start_seconds"),
                "prefill_samples": measured.get("prefill_samples"),
            }
        )
        caps["model"] = model
        rows.append(caps)
    return rows


#: Static probes worth persisting, so offline callers can consume them without a
#: server. Measured rates are written separately by eval/prefill.py.
_PERSISTED = ("family", "parameter_size", "quantization", "vision", "generative", "disk_bytes")


def record_static_capabilities(model: str, roster: Optional[List[Dict]] = None) -> Dict:
    """Probe a model and persist the result beside its measured rates.

    This is what turns a probe into something the rest of the codebase can USE.
    `get_model_family`, VLM selection and the memory estimate all run in contexts
    with no server and no permission to reach one, so they read what this wrote
    rather than probing themselves.

    Unlike the rate recorders, this OVERWRITES: `family` and `vision` are facts
    about the model, not samples of a noisy quantity, so the newest reading is
    simply the correct one. Keeping the "slowest wins" policy here would pin a
    model to whatever it looked like the first time it was seen.
    """
    from eval.signals import _load_eval_signals, _save_eval_signals

    probed = probe_static_capabilities(model, roster)
    signals = _load_eval_signals()
    caps = signals.setdefault(model, {}).setdefault("_capabilities", {})
    for key in _PERSISTED:
        value = probed.get(key)
        if value is not None:
            caps[key] = value
    _save_eval_signals(signals)
    return probed


def _gb(value: Optional[int]) -> str:
    return f"{value / 1_073_741_824:.1f}" if value else "-"


def format_capability_table(
    rows: List[Dict], output_tokens: int, timeout_seconds: int
) -> List[str]:
    """The roster table, generated instead of hand-maintained.

    Every column here replaced a guess: family replaced a name prefix, vision
    replaced a keyword list, disk replaced a parameter count, and the verdict
    replaced a person reading numbers and forming an opinion in a terminal.
    """
    header = (
        f"{'model':38} {'family':16} {'size':>6} {'disk GB':>8} {'vis':>4} "
        f"{'prefill':>8} {'decode':>7} {'verdict':>9}"
    )
    lines = [header, "-" * len(header)]
    notes = []
    for row in sorted(rows, key=lambda r: r["model"]):
        assessment = assess_viability(row, output_tokens, timeout_seconds)
        vision = {True: "yes", False: "no", None: "?"}[row.get("vision")]
        lines.append(
            f"{row['model']:38} {(row.get('family') or '-'):16} "
            f"{(row.get('parameter_size') or '-'):>6} {_gb(row.get('disk_bytes')):>8} "
            f"{vision:>4} {str(row.get('prefill_chars_per_sec') or '-'):>8} "
            f"{str(row.get('decode_tokens_per_sec') or '-'):>7} "
            f"{assessment['verdict']:>9}"
        )
        if assessment["verdict"] in ("broken", "thrashing", "too_slow", "unknown"):
            notes.append(explain_viability(row["model"], assessment, row.get("disk_bytes")))
        if row.get("generative") is False:
            notes.append(f"{row['model']}: not a generative model — cannot be ranked")
    if notes:
        lines.append("")
        lines.extend(notes)
    return lines
