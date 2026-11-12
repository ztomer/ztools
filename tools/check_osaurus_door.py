#!/usr/bin/env python3
"""Refuse a bare `osaurus` CLI call in a tracked shell script outside tools/osaurus_door.sh.

ZT.OSAURUSSHELL. The osaurus CLI opens a second Osaurus app when the server is slow to answer
its 0.5 s probe; every shell tool must reach it through the health-checked door. A command is
`osaurus` in command position: at line start, after `;`, `&`, `|`, `(`, `$(`, `` ` ``, or after
a launcher word (nohup, exec, command, time, xargs, env, then, do, else, if, !). Comments are
skipped. Usage: check_osaurus_door.py [ROOT]  (default: the repo; git ls-files decides).
"""

import re
import subprocess
import sys
from pathlib import Path

DOOR = "tools/osaurus_door.sh"
LAUNCH = r"(?:nohup|exec|command|time|xargs|env|then|do|else|if|!|sudo)"
CALL = re.compile(rf"(?:^|[;&|(`]|\$\(|\b{LAUNCH}\s)\s*(?:{LAUNCH}\s+)*osaurus(?=\s|$|;|\)|`)")


def shell_files(root):
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=False,
        stdin=subprocess.DEVNULL,
    ).stdout.decode()
    for rel in filter(None, out.split("\0")):
        p = root / rel
        if rel == DOOR or not p.is_file():
            continue
        if rel.endswith((".sh", ".bash", ".zsh")):
            yield rel, p
            continue
        try:
            head = p.open("rb").readline()
        except OSError:
            continue
        if head.startswith(b"#!") and re.search(rb"\b(ba|z)?sh\b", head):
            yield rel, p


def violations(root):
    found = []
    files = list(shell_files(root))
    if not files:
        return [f"{root}: no tracked shell script to inspect (an empty scope proves nothing)"]
    for rel, p in files:
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            # Drop quoted text so a message naming the CLI is not a call -- but keep a
            # double-quoted string holding a command substitution, which still runs.
            code = re.sub(r"'[^']*'", "''", line)
            code = re.sub(r'"(?:[^"\\$`]|\\.)*"', '""', code)
            if CALL.search(code):
                found.append(f"{rel}:{n}: {line.strip()}")
    return found


def main(argv):
    root = Path(argv[1] if len(argv) > 1 else Path(__file__).resolve().parent.parent)
    bad = violations(root)
    for b in bad:
        print(f"bare osaurus CLI call outside {DOOR}: {b}", file=sys.stderr)
    if bad:
        print("route it through osaurus_door / osaurus_door_serve, or HTTP", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
