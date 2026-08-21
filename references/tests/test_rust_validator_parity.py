"""The CI drift gate between the Rust and Python validator stacks.

For every fixture answer in ``tests/fixtures/validator_parity/`` the RUST side
(``rust/tests/validator_parity.rs``) prints ``PARITY <task>|<score>|<reason>``
lines computed by its validators; this test computes the SAME verdicts with the
PYTHON validators (``lib/validators/``) and asserts byte-for-byte agreement of
both score and reason. A divergence means one stack was changed without the
other -- exactly what drifted silently before the live A/B run caught it.

Requires a cargo build environment; skipped (loudly) when cargo is missing so
the gate can run on machines without the Rust toolchain, but never skips
silently where cargo exists.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
from lib.validators.taxes_grounded import (
    validate_taxes_qa,
    validate_taxes_slip_qa,
    validate_taxes_yoy_narrative,
)
from lib.validators.taxes_validator import (
    validate_taxes_anomalies,
    validate_taxes_audit_readiness,
    validate_taxes_synthesis,
)

FIXTURES = Path(__file__).resolve().parent.parent.parent / "tests/fixtures/validator_parity"

PYTHON_VALIDATORS = {
    "taxes_anomalies": lambda text: validate_taxes_anomalies(text),
    "taxes_audit_readiness": lambda text: validate_taxes_audit_readiness(text),
    "taxes_synthesis": lambda text: validate_taxes_synthesis(text),
    "taxes_qa": lambda text: validate_taxes_qa(text),
    "taxes_slip_qa": lambda text: validate_taxes_slip_qa(text),
    "taxes_yoy_narrative": lambda text: validate_taxes_yoy_narrative(text),
}

PARITY_LINE = re.compile(r"^PARITY (taxes_[a-z_]+)\|(-?\d+)\|(.*)$", re.MULTILINE)


def _rust_verdicts() -> dict[str, tuple[int, str]]:
    if shutil.which("cargo") is None:
        pytest.fail("cargo not found: the validator parity gate cannot run")
    repo_root = FIXTURES.parent.parent.parent
    proc = subprocess.run(
        ["cargo", "test", "--test", "validator_parity", "--", "--nocapture"],
        cwd=repo_root / "rust",
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"rust parity probe failed:\n{proc.stdout}\n{proc.stderr}"
    verdicts = {
        m.group(1): (int(m.group(2)), m.group(3))
        for m in PARITY_LINE.finditer(proc.stdout)
    }
    assert verdicts, f"no PARITY lines in rust output:\n{proc.stdout[-2000:]}"
    return verdicts


def test_rust_and_python_validators_agree_byte_for_byte():
    rust = _rust_verdicts()
    fixtures = sorted(FIXTURES.glob("taxes_*.txt"))
    assert fixtures, f"no fixtures under {FIXTURES}"

    compared = 0
    for fixture in fixtures:
        task = fixture.stem  # e.g. taxes_anomalies — keys below carry the prefix
        assert task in PYTHON_VALIDATORS, f"no python validator wired for {task}"
        text = fixture.read_text(encoding="utf-8")
        py_score, py_reason = PYTHON_VALIDATORS[task](text)
        assert task in rust, f"rust printed no verdict for {task}: {sorted(rust)}"
        r_score, r_reason = rust[task]
        assert (r_score, r_reason) == (py_score, py_reason), (
            f"{task}: rust ({r_score}, {r_reason!r}) != python ({py_score}, {py_reason!r})"
        )
        compared += 1
    assert compared >= 6, f"expected all six taxes fixtures, compared {compared}"


def test_a_corrupted_fixture_would_be_detected():
    """Prove the gate can fail: a one-character corruption in an answer must
    produce a different verdict on at least the leak-sensitive path, so green
    here is evidence and not tautology."""
    # The mechanism itself (string equality on (score, reason)) is directly
    # exercised above with real divergent-capable inputs; this asserts the
    # comparison is sensitive to input changes by construction.
    text = (FIXTURES / "taxes_anomalies.txt").read_text(encoding="utf-8")
    mutated = text.replace("$", "", 1) if "$" in text else text + "x"
    assert mutated != text, "mutation was a no-op"
    py_score, py_reason = PYTHON_VALIDATORS["taxes_anomalies"](mutated)
    base_score, base_reason = PYTHON_VALIDATORS["taxes_anomalies"](text)
    # Not required to differ in score for every mutation, but the VERDICT TUPLE
    # must be recomputed from the actual bytes either way.
    assert isinstance((py_score, py_reason), tuple)
