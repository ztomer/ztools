#!/usr/bin/env python3
"""mutate.py — break the code one edit at a time and report what the tests miss.

A test that never fails proves nothing, and neither counting patch sites nor
grepping for missing assertions can tell you which tests those are. Only breaking
the code can. Two tests written and reviewed carefully during this repo's own work
turned out to pass for the wrong reason -- an ordering test whose fixture sorted the
same way under both rules, and a recursion-guard test whose scenario converged before
reaching the guard. Both read perfectly; both were found this way.

Usage:
  python3 tools/mutate.py --module lib/quality_scorers.py --tests test_quality_scorers.py
  python3 tools/mutate.py --preset scorers          # every scorer + its tests
  python3 tools/mutate.py --preset validators
  python3 tools/mutate.py --preset scorers --limit 40

A SURVIVOR is a mutation the tests did not notice. That is the output; everything
else is bookkeeping. Survivors are not automatically bugs -- some mutate genuinely
unreachable or cosmetic code -- but each one is a place where the suite would not
have told you.
"""

import argparse
import ast
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
REFS = ROOT / "references"

PRESETS = {
    "scorers": (
        ["lib/quality_scorers.py", "lib/quality_scorers_core.py", "lib/scorers_file.py",
         "lib/scorers_filename.py", "lib/scorers_summarize.py", "lib/quality_weekend_scorers.py"],
        ["test_quality_scorers.py", "test_quality_weekend_scorers.py",
         "test_scorer_discrimination.py", "test_benchmark_quality.py"],
    ),
    "validators": (
        ["lib/validators/text_validator.py", "lib/validators/json_validator.py",
         "lib/validators/text_match.py", "lib/validators/attribution.py",
         "lib/validators/helpers.py", "lib/validators_lib.py"],
        ["test_text_validator.py", "test_json_validator.py", "test_validators.py",
         "test_validators_helpers.py", "test_faithfulness_validators.py",
         "test_validator_ordering.py", "test_factual_coverage.py"],
    ),
}

# Each mutation is a (regex, replacement, label). Deliberately small and local: a
# mutation big enough to break imports tells you nothing about assertions.
MUTATIONS = [
    (r"(?<![<>=!])>=(?!=)", "> ", "boundary >= to >"),
    (r"(?<![<>=!])<=(?!=)", "< ", "boundary <= to <"),
    (r"(?<![<>=!])==(?!=)", "!=", "equality flipped"),
    (r"(?<![<>=!])!=(?!=)", "==", "inequality flipped"),
    (r"\band\b", "or", "and to or"),
    (r"\bor\b", "and", "or to and"),
    (r"\bnot\s+", "", "not removed"),
    (r"\bTrue\b", "False", "True to False"),
    (r"\bFalse\b", "True", "False to True"),
    (r"\bmin\(", "max(", "min to max"),
    (r"\bmax\(", "min(", "max to min"),
]


def mutable_lines(path):
    """Line numbers that are real code: not comments, docstrings, or imports.

    Mutating a docstring changes nothing and would report a false survivor for every
    sentence containing the word "not".
    """
    src = path.read_text()
    tree = ast.parse(src)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
           and isinstance(node.value.value, str):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                doc_lines.add(ln)
    out = []
    for i, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or i in doc_lines:
            continue
        if stripped.startswith(("import ", "from ")):
            continue
        out.append(i)
    return out


def run_tests(test_files, timeout=600):
    cmd = [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-x", "-q",
           "-p", "no:cacheprovider", "--no-header", "-o", "addopts="]
    cmd += [str(REFS / "tests" / t) for t in test_files]
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False  # a hang is a detection, not a survivor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", action="append", default=[])
    ap.add_argument("--tests", action="append", default=[])
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--limit", type=int, default=0, help="max mutations per module")
    args = ap.parse_args()

    modules, tests = list(args.module), list(args.tests)
    if args.preset:
        pmods, ptests = PRESETS[args.preset]
        modules += pmods
        tests += ptests
    if not modules or not tests:
        ap.error("need --module/--tests or --preset")

    modules = [m for m in modules if (REFS / m).exists()]
    tests = [t for t in tests if (REFS / "tests" / t).exists()]

    print(f"baseline: running {len(tests)} test file(s) unmutated...", flush=True)
    started = time.time()
    if not run_tests(tests):
        print("BASELINE FAILS — fix the suite before mutating; survivors would be meaningless")
        return 2
    print(f"baseline green in {time.time() - started:.1f}s\n", flush=True)

    survivors, killed, total = [], 0, 0
    for mod in modules:
        path = REFS / mod
        original = path.read_text()
        lines = original.splitlines(keepends=True)
        candidates = []
        for lineno in mutable_lines(path):
            text = lines[lineno - 1]
            for pattern, repl, label in MUTATIONS:
                if re.search(pattern, text):
                    candidates.append((lineno, pattern, repl, label))
        if args.limit:
            step = max(1, len(candidates) // args.limit)
            candidates = candidates[::step][: args.limit]

        print(f"{mod}: {len(candidates)} mutation(s)", flush=True)
        for lineno, pattern, repl, label in candidates:
            mutated = list(lines)
            mutated[lineno - 1] = re.sub(pattern, repl, mutated[lineno - 1], count=1)
            if mutated[lineno - 1] == lines[lineno - 1]:
                continue
            total += 1
            path.write_text("".join(mutated))
            try:
                passed = run_tests(tests)
            finally:
                path.write_text(original)
            if passed:
                survivors.append((mod, lineno, label, lines[lineno - 1].strip()[:88]))
                snippet = lines[lineno - 1].strip()[:70]
                print(f"  SURVIVED  L{lineno:<5} {label:22} {snippet}", flush=True)
            else:
                killed += 1
    print(f"\n{'=' * 78}")
    print(f"mutations: {total}   killed: {killed}   SURVIVED: {len(survivors)}")
    if total:
        print(f"detection rate: {killed / total:.0%}")
    if survivors:
        print("\nSurvivors — each is a change the tests did not notice:\n")
        for mod, lineno, label, text in survivors:
            print(f"  {mod}:{lineno}  [{label}]\n      {text}")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
