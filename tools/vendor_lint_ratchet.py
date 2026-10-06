#!/usr/bin/env python3
"""Shrink-only ratchet over the vendored `camoufox` crate's clippy findings.

WHAT THIS IS. `vendor/camoufox-rs` is a third-party crate carried in-tree as a
path dependency of `rust/Cargo.toml`; 13,892 of its lines go into the shipped
`ztools` binary. `.gatesrc` sets `GOH_EXCLUDE='vendor/'`, which `rust_gate.sh`
forwards to clippy and to `check_no_allow.py`, so the crate is outside both.
That exclusion is correct -- a lint sweep is not ours to run on code we do not
own -- but "we do not enforce it" and "it may get worse forever" are different
statements, and only the first is true here. This gate makes the second false:
the finding count per lint may fall and may never rise.

WHY A RATCHET AND NOT MEMBERSHIP. Measured 2026-10-04, at `-D warnings`:

  clippy::all alone            1 finding   (after the fixes below: 0)
  all + pedantic + nursery      365 unique findings, 668 records, 28 files

So the crate CAN meet `clippy::all` at `-D warnings` and CANNOT meet the
parent's full policy. Membership would enforce the first and be silent about the
second, and it would also pull the crate into `coverage_gate.sh --lang rust`,
which enumerates `cargo metadata --no-deps` packages and runs `cargo llvm-cov -p`
on each -- i.e. 13,892 third-party lines into a 95% floor built around ztools.
This gate reaches the crate without either cost.

THE THREE FAILURE MODES THIS HAS TO NOT HAVE.

1. A measurement that did not measure. Cargo replays cached diagnostics only for
   units it rebuilds, so a warm target dir yields far fewer findings rather than
   an error -- a ratchet over that reports a huge phantom SHRINK. So the run is
   preceded by `cargo clean -p camoufox`, and the gate refuses unless at least
   one camoufox unit was actually compiled in this run (`fresh: false`).
2. A measurement that failed. clippy is invoked with `-W`, never `-D`, so its own
   exit status is about compiling, not about lint debt: a non-zero status means
   the crate did not build and the count is unknown, which is exit 2, not 0.
3. A ceiling that silently absorbed growth. Every baseline entry carries a
   `why`; a lint with no entry fails as NEW; a count above its entry fails;
   an entry whose lint no longer fires at all fails as stale, so the baseline
   cannot describe a debt that has been paid.

Sibling, not a copy: `gates_of_heck/checks/check_baseline_ratchet.py` encodes
the same shrink-only contract, and its line/JSON formats parse `{key: number}`.
That format cannot carry a per-entry justification, which this baseline is
required to have (an allowlist entry is an argument, not a number), so the
comparison is done here rather than by vendoring that file into the repo.

    tools/vendor_lint_ratchet.py                    # the gate
    tools/vendor_lint_ratchet.py --record           # re-seed after a shrink
    tools/vendor_lint_ratchet.py --json             # machine-readable counts

Exit codes: 0 clean · 1 violation · 2 precondition (measurement unusable).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "tools" / "vendor_lint_baseline.jsonc"
DEFAULT_MANIFEST = REPO_ROOT / "vendor" / "camoufox-rs" / "Cargo.toml"

# The house policy the parent crate is held to (rust/Cargo.toml [lints.clippy]),
# measured on the vendored crate so the baseline is comparable to the 365 that
# policy produced there. Declared here rather than read from the crate's own
# manifest on purpose: that manifest states the policy the crate MEETS (`all`),
# and a ratchet that only measured that would never notice the 208 pedantic and
# nursery findings being added to. A lint group added to the baseline policy
# starts unbaselined, so widening it fails loudly instead of silently.
POLICY = ("clippy::all", "clippy::pedantic", "clippy::nursery")
PACKAGE = "camoufox"


def _tui():
    """The house TUI (icons, restrained colour, NO_COLOR/non-tty aware).

    Falls back to plain prints with the same glyphs when the house checkout is
    not reachable, so a missing sibling gate directory can never stop this one
    from reporting a verdict.
    """
    goh = os.environ.get("GOH_DIR") or str(Path.home() / "Projects" / "gates_of_heck")
    if (Path(goh) / "tui" / "lib.py").is_file():
        sys.path.insert(0, goh)
        try:
            from tui.lib import err, info, ok, section, warn  # noqa: PLC0415

            return info, ok, err, warn, section
        except ImportError:
            pass

    def info(m):
        print(f"→ {m}", flush=True)

    def ok(m):
        print(f"✓ {m}", flush=True)

    def err(m):
        print(f"✗ {m}", file=sys.stderr, flush=True)

    def warn(m):
        print(f"⚠ {m}", file=sys.stderr, flush=True)

    def section(t):
        print(f"\n→ {t}", flush=True)

    return info, ok, err, warn, section


INFO, OK, ERR, WARN, SECTION = _tui()


class Precondition(Exception):
    """The gate could not measure. Never reported as a pass."""


def _run(cmd, target_dir, timeout):
    env = dict(os.environ, CARGO_TARGET_DIR=str(target_dir))
    try:
        return subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        raise Precondition(f"timed out after {timeout}s: {' '.join(cmd[:4])}…") from None
    except FileNotFoundError:
        raise Precondition(f"{cmd[0]} not found on PATH") from None


def measure(manifest: Path, target_dir: Path, timeout: int) -> tuple[Counter, int]:
    """Per-lint finding counts for the crate, deduplicated by (file, line, lint).

    Dedup matters: `--all-targets` compiles `src/` three times (lib, lib test,
    integration test) and once more for the example, so the raw diagnostic
    stream counts the same finding 3-4 times. A ceiling over the raw stream
    would be a ceiling on cargo's target list rather than on the code.

    Returns (counts, units_compiled).
    """
    # WHY THE CLEAN: cargo re-emits a unit's diagnostics only when it rebuilds
    # that unit, so without this a warm target dir silently reports a fraction
    # of the findings and every number below reads as a shrink.
    #
    # WHY `--manifest-path` AND WHY ITS STATUS IS CHECKED. Both were bugs here
    # first. `cargo clean -p camoufox` resolves the package from the CURRENT
    # directory, and the gate runs from the repo root, which has no Cargo.toml:
    # it exited 101, its status was ignored, nothing was cleaned, and the run
    # that followed replayed a fully cached build and reported 0 findings
    # against a baseline of 208 -- a total phantom collapse that only the
    # "no unit was rebuilt" precondition caught. An ignored non-zero here is
    # exactly the failure-path-no-op class.
    cleaned = _run(
        ["cargo", "clean", "-p", PACKAGE, "--manifest-path", str(manifest)], target_dir, timeout
    )
    if cleaned.returncode != 0:
        raise Precondition(
            f"cargo clean -p {PACKAGE} exited {cleaned.returncode}; the following clippy run "
            "would replay cached diagnostics instead of measuring this source. Refusing."
        )

    # ORDER MATTERS: `--message-format=json` is a CARGO flag and the lint flags
    # are DRIVER flags, so the explicit `--` separator has to sit between them.
    # Without it cargo rejects the `-W`s itself ("unexpected argument '-W'"),
    # which is loud — but the reason this comment exists is that a silent
    # mis-ordering here would look like a measurement of zero.
    cmd = [
        "cargo",
        "clippy",
        "--manifest-path",
        str(manifest),
        "--all-targets",
        "--all-features",
        "--message-format=json",
        "--",
    ]
    for group in POLICY:
        cmd += ["-W", group]
    proc = _run(cmd, target_dir, timeout)

    counts: Counter = Counter()
    keys: set[tuple[str, int, str]] = set()
    rebuilt = 0
    for line in proc.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("reason") == "compiler-artifact":
            if PACKAGE in (msg.get("package_id") or "") and not msg.get("fresh", True):
                rebuilt += 1
            continue
        if msg.get("reason") != "compiler-message":
            continue
        diag = msg.get("message") or {}
        if diag.get("level") not in ("warning", "error"):
            continue
        primary = [s for s in diag.get("spans") or [] if s.get("is_primary")]
        if not primary:
            continue
        lint = (diag.get("code") or {}).get("code") or "unnamed-diagnostic"
        keys.add((primary[0].get("file_name", "?"), primary[0].get("line_start", 0), lint))

    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
        raise Precondition(
            f"cargo clippy exited {proc.returncode} for {manifest.name} -- the crate did "
            f"not build, so the finding count is UNKNOWN (not zero):\n{tail}"
        )
    if rebuilt == 0:
        raise Precondition(
            f"no {PACKAGE} unit was compiled (every unit was cached), so clippy reported "
            "nothing about this source. Refusing to read that as 0 findings."
        )

    for _, _, lint in keys:
        counts[lint] += 1
    return counts, rebuilt


def read_jsonc(path: Path) -> dict:
    """Parse a baseline that carries `//` comment lines above a strict-JSON body.

    The extension is `.jsonc` and not `.json` for a reason that is not taste:
    `.gitignore` has a blanket `*.json` (whitelisting only conf/, eval_tasks/,
    docs/, tests/fixtures/), so a `tools/*.json` baseline is untrackable -- and a
    ratchet whose ceiling cannot be committed re-seeds itself on every clone,
    which is a gate that measures nothing. Comments only count when they are the
    FIRST thing on a line, so a `//` occurring inside a JSON string is untouched;
    the reasons quote paths and `///` doc-comment markers and must survive.
    """
    body = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("//")
    ]
    try:
        return json.loads("\n".join(body))
    except json.JSONDecodeError as exc:
        raise Precondition(
            f"{path} is not valid JSON (after stripping // comments): {exc}"
        ) from None


def load_baseline(path: Path) -> dict:
    if not path.is_file():
        raise Precondition(
            f"baseline {path} does not exist. Create it from today's truth with "
            f"--record, then commit it -- the committed ceiling is the gate."
        )
    data = read_jsonc(path)
    lints = data.get("lints")
    if not isinstance(lints, dict) or not lints:
        raise Precondition(f"{path} has no non-empty 'lints' object")
    for name, entry in lints.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("count"), int):
            raise Precondition(f"{path}: lints[{name!r}] needs an integer 'count'")
        if not (entry.get("why") or "").strip():
            raise Precondition(
                f"{path}: lints[{name!r}] has no 'why'. A tolerated finding is an "
                "argument, and an unreasoned entry is how a baseline rots."
            )
    return data


def compare(baseline: dict, current: Counter) -> dict:
    """Split the measurement into the four verdicts. All are printed; three fail.

    The `stale` case is "recorded a positive ceiling, measured none" -- an entry
    still describing debt that has been paid. It is NOT the same as `shrank`: a
    partial shrink (7 -> 3) is progress to bank, while 7 -> 0 means the entry
    itself is the thing that is wrong. The first version of this function tested
    only `measured == 0` and put a positive ceiling into `shrank`, which made
    `stale` unreachable and the check dead; the D-calibration (a baseline entry
    for a lint the crate does not trip) is what caught it, and it is why that
    calibration exists rather than a green run being taken as proof.
    """
    lints = baseline["lints"]
    rose, new, shrank, stale = [], [], [], []
    for name, entry in sorted(lints.items()):
        ceiling = entry["count"]
        measured = current.get(name, 0)
        if measured > ceiling:
            rose.append((name, ceiling, measured))
        elif measured == 0 and ceiling > 0:
            stale.append((name, ceiling))
        elif measured < ceiling:
            shrank.append((name, ceiling, measured))
    for name, measured in sorted(current.items()):
        if name not in lints:
            new.append((name, measured))
    return {"rose": rose, "new": new, "shrank": shrank, "stale": stale}


def record(
    baseline_path: Path, current: Counter, manifest: Path, rebuilt: int, measured: str | None
) -> int:
    """Rewrite the baseline from a live measurement, printing every change.

    Preserves an existing entry's `why` -- re-recording must never silently drop
    the reason a finding was tolerated. Only counts and the measurement stamp
    change.
    """
    try:
        previous = load_baseline(baseline_path)["lints"]
    except Precondition:
        previous = {}
    entries, carried = {}, 0
    for name in sorted(current):
        was = previous.get(name, {})
        entries[name] = {
            "count": current[name],
            "why": was.get("why") or "FILL IN: why this finding may stand.",
        }
        carried += 1 if was.get("why") and "FILL IN" not in was["why"] else 0
    data = {
        "crate": PACKAGE,
        "manifest": str(manifest.relative_to(REPO_ROOT)),
        "measured": measured or date.today().isoformat(),
        "policy": list(POLICY),
        "units_compiled": rebuilt,
        "lints": entries,
    }
    # The `//` header is carried across verbatim: it is the baseline's rationale,
    # and a re-record that dropped it would delete the argument for the thing it
    # is re-arguing.
    header = []
    if baseline_path.is_file():
        header = [
            line
            for line in baseline_path.read_text(encoding="utf-8").splitlines()
            if line.lstrip().startswith("//")
        ]
    payload = "\n".join(header + ([""] if header else []) + [json.dumps(data, indent=2)]) + "\n"
    baseline_path.write_text(payload, encoding="utf-8")
    OK(
        f"recorded {len(entries)} lint kind(s) into {baseline_path} "
        f"({carried} existing reason(s) carried over, {len(header)} header line(s) kept)"
    )
    WARN(f"{len(entries) - carried} entry/entries still carry a FILL IN reason — write them")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Shrink-only clippy ratchet for vendor/camoufox-rs")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--target-dir", type=Path, default=None, help="cargo target dir to measure in")
    ap.add_argument("--record", action="store_true", help="re-seed the baseline from a live count")
    ap.add_argument("--json", action="store_true", help="print counts as JSON and exit")
    ap.add_argument(
        "--measured",
        default=None,
        help="stamp to write into the baseline on --record (default: today)",
    )
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    target_dir = args.target_dir or REPO_ROOT / "vendor" / "camoufox-rs" / "target"
    try:
        current, rebuilt = measure(args.manifest, target_dir, args.timeout)
        if args.json:
            print(json.dumps({"units_compiled": rebuilt, "counts": dict(current)}, indent=2))
            return 0
        if args.record:
            return record(args.baseline, current, args.manifest, rebuilt, args.measured)
        baseline = load_baseline(args.baseline)
    except Precondition as exc:
        ERR(f"[vendor_lint] measurement unusable for crate {PACKAGE!r}: {exc}")
        return 2

    verdict = compare(baseline, current)
    rel_manifest = args.manifest.relative_to(REPO_ROOT)
    INFO(f"crate {PACKAGE!r} ({rel_manifest}) vs baseline {args.baseline.relative_to(REPO_ROOT)}")
    INFO(
        f"policy {' + '.join(POLICY)}; {rebuilt} unit(s) rebuilt; "
        f"{sum(current.values())} finding(s)"
    )
    SECTION("per-lint counts compared (lint: baseline -> measured)")
    for name, entry in sorted(baseline["lints"].items()):
        measured = current.get(name, 0)
        ceiling = entry["count"]
        mark = "==" if measured == ceiling else ("<" if measured < ceiling else ">")
        print(f"    {name:44s} {ceiling:4d} {mark} {measured:<4d}")
    for name, measured in verdict["new"]:
        print(f"    {name:44s} {'—':>4s} NEW {measured:<4d}")

    total_ceiling = sum(e["count"] for e in baseline["lints"].values())
    if verdict["rose"] or verdict["new"]:
        ERR(
            f"[vendor_lint] {len(verdict['rose'])} ceiling(s) exceeded, "
            f"{len(verdict['new'])} new lint kind(s) — this gate is shrink-only:"
        )
        for name, ceiling, measured in verdict["rose"]:
            ERR(f"    {name}: {ceiling} -> {measured}  (+{measured - ceiling})")
        for name, measured in verdict["new"]:
            ERR(f"    {name}: NEW at {measured} (absent from the baseline)")
        ERR("  Fix the finding, or re-record deliberately in the same commit after")
        ERR("  writing the new entry's `why`. Do not raise a ceiling to absorb it.")
        return 1

    if verdict["stale"]:
        ERR(
            f"[vendor_lint] {len(verdict['stale'])} baseline entry/entries no longer occur — "
            "a stale entry describes debt that has been paid:"
        )
        for name, ceiling in verdict["stale"]:
            ERR(f"    {name}: recorded {ceiling}, measured 0")
        ERR(f"  Re-record with --record so the baseline matches reality: {args.baseline}")
        return 1

    detail = ""
    if verdict["shrank"]:
        detail = " (" + ", ".join(f"{n} {c}->{m}" for n, c, m in verdict["shrank"]) + ")"
    OK(
        f"[vendor_lint] OK — {len(baseline['lints'])} lint kind(s) within ceilings; "
        f"{sum(current.values())} of {total_ceiling} recorded findings remain{detail}"
    )
    if verdict["shrank"]:
        INFO(f"  re-record to bank the shrink: {args.baseline} --record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
