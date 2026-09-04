"""
Pytest configuration: shared fixtures for all tests.

Shim: split into conftest_fixtures/ (legacy, gates_network, gates_state,
gates_determinism) for the 500-line cap -- no test exemption; see CLAUDE.md.
Every fixture is IMPORTED BY NAME below rather than left in its submodule,
because pytest discovers fixtures (autouse ones especially) by scanning
conftest.py's own module namespace: a fixture merely defined in an importable
module is invisible to pytest until something binds it into conftest.py's
namespace, which is exactly what these imports do.

The bootstrap below (env vars, then `import lib.mlx_lib`) stays HERE rather
than moving to a submodule: it has to run at conftest.py's own module-import
time, in this order, and moving it behind another import would only relocate
the same ordering constraint, not remove it.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Local runs must match what the documented quality gate enforces, so a plain
# `pytest` cannot silently differ from CI. Without these, any test whose mock
# misses one LLM boundary falls through to REAL infrastructure: an MLX fallback
# scans ~/MLXModels (present on this machine) and loads a multi-GB model, and
# nine weekend_llm / osaurus_server / gpu_lock tests then hang for minutes or
# fail on whatever answers port 1337. Opt back in per-session with
# ZTOOLS_TEST_ALLOW_REAL_MLX=1 / ZTOOLS_TEST_ALLOW_REAL_LLM=1 -- the same
# explicit-opt-out shape as the real_cookie_discovery marker.
#
# MUST stay ABOVE every project import below: `from lib.testing import ...`
# transitively imports the whole `lib` package, and MLX_MODELS_DIR is read at
# module-import time -- an env var set after that import is silently ignored.
if "ZTOOLS_TEST_ALLOW_REAL_MLX" not in os.environ:
    os.environ.setdefault(
        "MLX_MODELS_DIR", "/tmp/nonexistent-mlx-models-dir-for-testing"
    )
if "ZTOOLS_TEST_ALLOW_REAL_LLM" not in os.environ:
    os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:1")

# Capture references to lib.mlx_lib functions at conftest load time.
# This MUST happen after the env block above and before any mock patches them.
import lib.mlx_lib  # noqa: E402
import lib.mlx_vlm  # noqa: E402
from lib.testing import MockLLM  # noqa: E402

_REAL_MLX_FUNCTIONS = {
    "call": lib.mlx_lib.call,
    "call_mlx": lib.mlx_lib.call_mlx,
    "call_mlx_vlm": lib.mlx_vlm.call_mlx_vlm,
    "probe_mlx_vlm_loadable": lib.mlx_vlm.probe_mlx_vlm_loadable,
    "process_mlx_content": lib.mlx_lib.process_mlx_content,
    "find_mlx_model": lib.mlx_lib.find_mlx_model,
    "find_text_mlx_model": lib.mlx_lib.find_text_mlx_model,
}


@pytest.fixture(autouse=True)
def reset_global_sessions():
    """Reset persistent session variables in client and osaurus_lib before and after each test."""
    import lib.llm.client
    import lib.osaurus_lib

    lib.llm.client.reset_session()
    lib.osaurus_lib.reset_session()
    yield
    lib.llm.client.reset_session()
    lib.osaurus_lib.reset_session()


@pytest.fixture
def real_mlx_functions():
    """Return real (unmocked) lib.mlx_lib functions.

    Captures references at conftest load time, before any mock patches.
    """
    return _REAL_MLX_FUNCTIONS


@pytest.fixture
def mock_llm():
    """Fixture that patches all LLM functions with a MockLLM provider.

    Usage:
        def test_something(mock_llm):
            import eval.run as er
            from eval import run_transport
            from unittest.mock import patch
            with patch.object(run_transport, "call", mock_llm.call):
                result = er.run_eval(...)
    """
    mock = MockLLM()
    mock.patch_all()
    yield mock
    mock.unpatch()


@pytest.fixture
def mock_llm_osaurus():
    """Same as mock_llm but only patches lib.osaurus_lib."""
    mock = MockLLM()
    mock.patch_osaurus()
    yield mock
    mock.unpatch()


# Every remaining fixture and structural gate lives in conftest_fixtures/ and is
# imported BY NAME here so pytest's fixture scan finds it on this module.
from tests.conftest_fixtures.gates_determinism import (  # noqa: E402,F401
    bounded_restart_ready_budget,
    deterministic_machine_contention,
)
from tests.conftest_fixtures.gates_network import (  # noqa: E402,F401
    no_real_browsers_or_cookies,
    no_real_llm_server,
    no_real_server_restart,
)
from tests.conftest_fixtures.gates_state import (  # noqa: E402,F401
    _eval_artefacts_stay_in_tmp,
    _gpu_lock_never_touches_the_real_one,
    _saved_outputs_stay_in_tmp,
    _signals_files_stay_clean,
    _tracked_config_stays_clean,
)
from tests.conftest_fixtures.legacy import (  # noqa: E402,F401
    mock_llm_response,
    mock_osaurus_server,
    sample_events_data,
    sample_tweets,
    sample_venues_data,
)
