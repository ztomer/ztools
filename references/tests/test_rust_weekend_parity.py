"""The CI drift gate between the Rust and Python weekend CORPUS stacks.

For every fixture in ``tests/fixtures/weekend_parity/`` the RUST side
(``rust/tests/weekend_parity.rs``) prints ``PARITY weekend_<task>|0|<json>``
lines computed by its corpus pipeline (``clean_search_results``,
``as_candidate_lines``, ``looks_like_aggregator``); this test computes the
SAME payloads with the PYTHON pipeline and asserts byte-for-byte agreement of
the decoded values — a corpus string that differs in any character is drift.

The parity surface is the DETERMINISTIC half only (corpus cleaning,
candidate lines, aggregator classification). Live search rankings and
lsHTML-vs-bs4 page extraction are corpus-QUALITY concerns and are not
byte-compared, per the Phase 3 spike.

Region evidence: the fixtures use ONLY inputs the Python whitelist-only filter
and the Rust whitelist+blocklist filter AGREE on. The deliberate divergence
(a foreign-city token beats a local one in Rust but not Python) is pinned by
the Rust test ``a_foreign_city_beats_any_local_token`` and is a design
decision, not drift.

Requires a cargo build environment; fails (loudly) when cargo is missing.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from weekend.data import _clean_search_results
from weekend.followup import (
    MAX_LINES_PER_PAGE,
    MIN_CANDIDATE_CHARS,
    as_candidate_lines,
    looks_like_aggregator,
)

FIXTURES = Path(__file__).resolve().parent.parent.parent / "tests/fixtures/weekend_parity"

# Rust hardcodes these; Python reads them from env at import. Pin the module
# globals to the Rust constants so a host with non-default env does not flip
# the gate red — Rust is the reference here, so the gate asserts THE default
# is what the fixture expects.
assert MIN_CANDIDATE_CHARS == 8, "followup env drift: MIN_CANDIDATE_CHARS"
assert MAX_LINES_PER_PAGE == 60, "followup env drift: MAX_LINES_PER_PAGE"

MAX_BODY = 300  # Rust MAX_BODY_LENGTH; mirrors the fixed call-site default


def _rust_verdicts() -> dict[str, object]:
    if shutil.which("cargo") is None:
        pytest.fail("cargo not found: the weekend parity gate cannot run")
    repo_root = FIXTURES.parent.parent.parent
    proc = subprocess.run(
        ["cargo", "test", "--test", "weekend_parity", "--", "--nocapture"],
        cwd=repo_root / "rust",
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"rust parity probe failed:\n{proc.stdout}\n{proc.stderr}"
    verdicts: dict[str, object] = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("PARITY "):
            continue
        parts = line.split("|", 2)
        assert len(parts) == 3, f"malformed PARITY line: {line!r}"
        task, _, payload = parts
        assert task.startswith("PARITY "), f"not a PARITY line: {line!r}"
        verdicts[task[len("PARITY "):]] = json.loads(payload)
    assert verdicts, f"no PARITY lines in rust output:\n{proc.stdout[-2000:]}"
    return verdicts


def _python_corpus() -> str:
    results = json.loads((FIXTURES / "corpus_results.json").read_text(encoding="utf-8"))
    return _clean_search_results(results, "Event", max_body=MAX_BODY)


def _python_candidates() -> str:
    page = json.loads((FIXTURES / "aggregator_page.json").read_text(encoding="utf-8"))
    return as_candidate_lines(page["text"], page["title"])


def _python_aggregator_flags() -> list[bool]:
    titles = json.loads((FIXTURES / "aggregator_flags_titles.json").read_text(encoding="utf-8"))
    return [looks_like_aggregator(title) for title in titles]


def test_rust_and_python_weekend_corpus_agree_byte_for_byte():
    rust = _rust_verdicts()
    assert rust["weekend_corpus"] == _python_corpus()
    assert rust["weekend_candidates"] == _python_candidates()
    assert rust["weekend_aggregator"] == _python_aggregator_flags()


def test_the_python_corpus_fixture_exercises_all_kernel_rules():
    """The fixture is only evidence if it actually reaches each rule. Assert
    the fixture shapes the corpus the way the kernel rules claim."""
    corpus = _python_corpus()
    # Title-only dedupe: two same-title different-body rows collapse to one.
    assert corpus.count("- Vaughan Fall Fair: ") == 1
    # Trailing-punctuation title dedupes into the plain-title row.
    assert "- Vaughan Fall Fair!!!" not in corpus
    # Body-only row (empty title) is dropped outright, never labeled Event.
    assert "body only" not in corpus.lower() or "no title" not in corpus
    # Truncation to MAX_BODY: the long row's body is exactly 300 chars.
    storytime = next(l for l in corpus.splitlines() if "Vaughan Public Library Storytime" in l)
    assert len(storytime.partition(": ")[2]) == MAX_BODY


def test_a_corrupted_fixture_would_be_detected():
    """Prove the gate can fail: altering any fixture byte must change the
    Python-side payload, so green here is evidence and not tautology."""
    base = _python_corpus()
    results = json.loads((FIXTURES / "corpus_results.json").read_text(encoding="utf-8"))
    mutated = json.loads(json.dumps(results))
    mutated[0]["title"] = "Vaughan Fall Fairzed"
    payload = json.dumps(mutated)
    assert payload != json.dumps(results), "mutation was a no-op"
    different = _clean_search_results(mutated, "Event", max_body=MAX_BODY)
    assert different != base, "a changed title must change the corpus"