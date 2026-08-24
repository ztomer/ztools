"""Structural gates blocking writes to real shared state: the machine-wide GPU
lock, signal/artefact/output files, and tracked config.

Split out of conftest.py for the 500-line cap (no test exemption; see
CLAUDE.md). Imported by name into conftest.py so pytest's fixture discovery
finds them there.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _gpu_lock_never_touches_the_real_one(tmp_path, monkeypatch):
    """Structural gate: no test may read or write the machine-wide GPU lock.

    /tmp/mac-osaurus-gpu.lock is shared by every checkout, worktree and agent
    session on this Mac, and a real eval may be holding it right now. Without
    this redirect the suite would be coupled to that: tests of the quit-refusal
    path would pass or fail depending on whether a colleague session happened to
    be measuring, and a test that acquires would BLOCK a real eval -- the exact
    harm the lock exists to prevent, caused by the tests for it.

    Function-scoped so each test gets a clean, empty lock, and the module's
    `_held` flag is reset on both sides: it is process-global, so one test that
    acquires would otherwise leave every later test believing it holds the lock.
    """
    import lib.gpu_lock as gl

    monkeypatch.setenv(gl.DIR_ENV, str(tmp_path / "gpu.lock"))
    monkeypatch.delenv(gl.OWNER_ENV, raising=False)
    gl._held = False
    yield
    gl._held = False
    os.environ.pop(gl.OWNER_ENV, None)


@pytest.fixture(autouse=True, scope="session")
def _signals_files_stay_clean(tmp_path_factory):
    """Structural gate: `pytest` must not dirty tracked config.

    eval/signals.py and weekend/llm.py persist learned per-model timeouts into
    conf/eval_signals.json and conf/phase_signals.json. Both are tracked, so
    exercising those code paths rewrote them on every test run and left the
    working tree dirty. Redirect both at a tmp dir for the whole session.
    """
    from unittest.mock import patch

    tmp = tmp_path_factory.mktemp("signals")
    import eval.signals as eval_signals
    import weekend.llm as weekend_llm

    with patch.object(eval_signals, "EVAL_SIGNALS_PATH", tmp / "eval_signals.json"), \
         patch.object(weekend_llm, "PHASE_SIGNALS_PATH", tmp / "phase_signals.json"), \
         patch.object(weekend_llm, "EXTRACT_SIGNALS_PATH", tmp / "extract_signals.json"):
        yield


@pytest.fixture(autouse=True, scope="session")
def _eval_artefacts_stay_in_tmp(tmp_path_factory):
    """Structural gate: nothing may write eval artefacts into the real config dir.

    `default_eval_dir()` returns ~/.config/ztools, and its docstring says callers
    "take eval_dir as a parameter and fall back to this, so tests hand in a tmp dir".
    That is discipline, not a gate, and discipline failed: eval_history.json in the
    developer's own config directory accumulated `m1`, `m2` and `mock-model` entries
    from the suite. report_history HAS a test-model filter, but it matches the
    prefixes ("mock", "test-", "fake") and a fixture called `m1` matches none of them
    -- a name allowlist always lags behind fixture naming, which is why this is a path
    redirect instead of another name.

    Patched on every IMPORTER, not on eval.report_core. Each module does
    `from eval.report_core import default_eval_dir` at import time, so patching the
    source module rebinds a name nobody reads -- the same seam hazard this repo
    documents for patch.object across a module split.
    """
    from unittest.mock import patch

    import eval.cli_results
    import eval.outputs
    import eval.report_history
    import eval.report_metrics

    tmp = tmp_path_factory.mktemp("eval_artefacts")
    patches = [
        patch.object(mod, "default_eval_dir", lambda: tmp)
        for mod in (
            eval.report_history,
            eval.report_metrics,
            eval.cli_results,
            eval.outputs,
        )
    ]
    for p in patches:
        p.start()
    try:
        yield tmp
    finally:
        for p in patches:
            p.stop()


@pytest.fixture(autouse=True, scope="session")
def _saved_outputs_stay_in_tmp(tmp_path_factory):
    """Structural gate: saved eval outputs must not land in the real config dir.

    run_eval now writes each model's raw answer under ~/.config/ztools/outputs so
    a scorer can be questioned without re-running the model. Every test that
    calls run_eval with a fake model therefore wrote there too -- the suite left
    outputs/m, outputs/m1 and outputs/mock-model in the developer's own config
    directory within minutes of the feature landing.

    Redirected by environment variable because that is the seam production reads;
    patching a module attribute would miss any caller that imported the path by
    value. The existing tracked-config gate could not have caught this: it
    digests conf/ and docs/, and this escapes to $HOME.
    """
    tmp = tmp_path_factory.mktemp("eval_outputs")
    previous = os.environ.get("EVAL_OUTPUT_DIR")
    os.environ["EVAL_OUTPUT_DIR"] = str(tmp)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("EVAL_OUTPUT_DIR", None)
        else:
            os.environ["EVAL_OUTPUT_DIR"] = previous


# Files a LONG-RUNNING TOOL legitimately rewrites while the suite is running, and
# which a dedicated fixture already stops the tests themselves from touching.
#
# `_signals_files_stay_clean` redirects every one of these at a tmp dir for the
# whole session, so a change to the real file cannot have come from a test -- it
# came from an `ev` run in another terminal, which updates eval_signals.json after
# every task. Digesting them anyway made the pre-push hook fail for a reason no
# amount of reading the diff could fix, on a machine where a sweep can run for ten
# hours. Coverage is not lost: the redirect fixture is the stronger, more specific
# gate, and `test_the_suite_cannot_write_into_the_real_config_dir` is the pattern
# for proving such a redirect is in force.
_CONCURRENTLY_WRITTEN = {"eval_signals.json", "phase_signals.json", "extract_signals.json"}


def _tracked_config_digest() -> dict:
    """Hash every tracked file the tools can write back to."""
    import hashlib

    from lib.paths import conf_dir, repo_path

    digest = {}
    targets = sorted(conf_dir().rglob("*.toml")) + sorted(conf_dir().rglob("*.json"))
    baseline = repo_path("docs", "eval_baseline.json")
    if baseline is not None:
        targets.append(baseline)
    for path in targets:
        if path.is_file() and path.name not in _CONCURRENTLY_WRITTEN:
            digest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


@pytest.fixture(autouse=True, scope="session")
def _tracked_config_stays_clean():
    """Structural gate: no test may write to the real conf/ or docs/ baseline.

    Before the layout fix these writes silently landed in a nonexistent
    `references/conf`, so tests that exercised `update_config` or
    `save_baseline` looked harmless. With paths resolving correctly they hit
    the tracked files for real — `pytest` rewrote conf/config.toml on the
    developer's checkout. Tests must point the writers at tmp (ZTOOLS_CONF or
    a patched module attribute); this fails the run if one does not.
    """
    before = _tracked_config_digest()
    yield
    after = _tracked_config_digest()
    changed = sorted(
        set(before) ^ set(after) | {p for p in before.keys() & after.keys() if before[p] != after[p]}
    )
    # Attributed by pytest to whichever test ran last, which is not evidence
    # about that test: this is a session fixture. And the suite is not the only
    # thing that writes here -- an `ev` run in another terminal updates
    # conf/eval_signals.json after every task. Say so, rather than sending the
    # reader to audit code that did nothing.
    assert not changed, (
        "tracked config files changed during the test session:\n  "
        + "\n  ".join(changed)
        + "\n\nEither a test wrote to the real conf/ instead of tmp (point it at "
        "ZTOOLS_CONF or patch the module attribute), or another process wrote "
        "them while the suite ran -- a concurrent `ev` run updates "
        "conf/eval_signals.json. Check `git diff` on the listed files to tell "
        "which: eval writes latency and _capabilities records."
    )
