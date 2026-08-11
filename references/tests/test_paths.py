"""Layout resolution for shipped resources (`lib.paths`).

Regression guard for the checkout layout: `lib/` lives under `references/`
while `conf/`, `eval_tasks/` and `docs/` sit at the repo root, so a
package-relative `__file__.parent.parent / "conf"` resolves to a nonexistent
`references/conf` and every config load falls back to defaults (or raises).
"""

import importlib
import re
from pathlib import Path

import pytest
from lib import paths
from lib.paths import conf_dir, conf_path, eval_tasks_dir, eval_tasks_path, repo_path, repo_root

REFERENCES_DIR = Path(__file__).resolve().parent.parent


def test_conf_dir_exists_in_current_layout():
    """The resolved conf dir must exist wherever the package is imported from."""
    resolved = conf_dir()
    assert resolved.is_dir(), f"conf dir does not exist: {resolved}"


def test_config_toml_is_reachable():
    """The file whose absence crashed `wk`/`ev` with exit 1."""
    assert conf_path("config.toml").is_file()


def test_model_configs_and_eval_inputs_are_reachable():
    """The two call sites that failed *silently* rather than raising."""
    assert conf_path("eval_inputs.toml").is_file()
    assert conf_path("models").is_dir()
    assert any(conf_path("models").glob("*.toml"))


def test_env_override_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path))
    assert conf_dir() == tmp_path
    assert conf_path("config.toml") == tmp_path / "config.toml"


def test_env_override_to_missing_dir_raises(monkeypatch, tmp_path):
    """An explicit instruction must fail loudly, not degrade to defaults."""
    monkeypatch.setenv("ZTOOLS_CONF", str(tmp_path / "no-such-dir"))
    with pytest.raises(FileNotFoundError, match="ZTOOLS_CONF"):
        conf_dir()


def test_env_override_ignored_when_empty(monkeypatch):
    monkeypatch.setenv("ZTOOLS_CONF", "")
    assert conf_dir().is_dir()


def test_candidates_cover_installed_and_checkout_layouts():
    """Both layouts must be searched, not just the one this checkout uses.

    Installed wheels flatten `lib/` and `conf/` into site-packages as
    siblings; the source tree nests `lib/` one level deeper.
    """
    lib_dir = Path(paths.__file__).resolve().parent
    assert lib_dir.parent / "conf" in paths._CANDIDATES  # installed (flat)
    assert lib_dir.parent.parent / "conf" in paths._CANDIDATES  # checkout


def test_empty_decoy_conf_dir_does_not_win(tmp_path, monkeypatch):
    """A stray empty `references/conf/` must not shadow the real config.

    Signal writers used to `mkdir` exactly that directory on the first real
    run, which would resurrect the original FileNotFoundError.
    """
    decoy = tmp_path / "decoy_conf"
    decoy.mkdir()
    real = tmp_path / "real_conf"
    real.mkdir()
    (real / "config.toml").write_text("[timeouts]\njson = 1\n")

    monkeypatch.delenv("ZTOOLS_CONF", raising=False)
    monkeypatch.setattr(paths, "_package_dir", lambda name: None)
    monkeypatch.setattr(paths, "_CANDIDATES", (decoy, real))

    assert conf_dir() == real


def test_package_dir_resolution_beats_candidate_scan(tmp_path, monkeypatch):
    """The import system is authoritative: it is correct in both layouts."""
    packaged = tmp_path / "packaged_conf"
    packaged.mkdir()
    (packaged / "config.toml").write_text("")
    monkeypatch.delenv("ZTOOLS_CONF", raising=False)
    monkeypatch.setattr(paths, "_package_dir", lambda name: packaged)
    monkeypatch.setattr(paths, "_CANDIDATES", (tmp_path / "unused",))
    assert conf_dir() == packaged


def test_conf_dir_falls_back_to_installed_path_when_nothing_exists(monkeypatch):
    """A missing conf dir must still yield a nameable path for error messages."""
    monkeypatch.delenv("ZTOOLS_CONF", raising=False)
    monkeypatch.setattr(paths, "_package_dir", lambda name: None)
    monkeypatch.setattr(paths, "_CANDIDATES", (Path("/nonexistent/a"), Path("/nonexistent/b")))
    assert conf_dir() == Path("/nonexistent/a")


def test_eval_tasks_dir_resolves_to_the_real_rubric_tree():
    assert eval_tasks_dir().is_dir()
    assert eval_tasks_path("data", "taxes").is_dir()


def test_repo_root_is_the_checkout_root():
    root = repo_root()
    assert root is not None, "running from a checkout, so repo_root must resolve"
    assert (root / "pyproject.toml").is_file()
    assert (root / "conf").is_dir()
    assert repo_path("docs", "eval_baseline.json").is_file()


@pytest.mark.parametrize(
    "module_name",
    ["lib.config_core", "lib.config_getters", "lib.config_tasks"],
)
def test_config_modules_resolve_through_the_shared_helper(module_name):
    module = importlib.import_module(module_name)
    assert module.conf_path is conf_path


def test_no_module_rederives_a_shipped_resource_path():
    """Class-level guard: the whole tree, not just the modules fixed by hand.

    Any new `Path(__file__)...  / "conf"` (or docs/eval_tasks) is the same bug
    again, so it fails here rather than in a user's terminal.
    """
    pattern = re.compile(r'parent\s*/\s*"(conf|docs|eval_tasks)"')
    offenders = []
    for py in REFERENCES_DIR.rglob("*.py"):
        if py.name == "paths.py" or "tests" in py.parts:
            continue
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{py.relative_to(REFERENCES_DIR)}:{lineno}: {line.strip()}")
    assert not offenders, "derive shipped-resource paths via lib.paths:\n" + "\n".join(offenders)


def test_init_config_loads_the_real_file():
    """End-to-end: the exact call that raised FileNotFoundError from a checkout."""
    from lib.config_core import init_config, reset_config

    try:
        assert init_config() is True
    finally:
        reset_config()


def test_package_dir_returns_none_for_a_missing_package():
    """The import-system lookup must fail soft, not raise."""
    assert paths._package_dir("no_such_package_anywhere") is None


def test_package_dir_returns_none_when_the_location_is_not_a_dir(monkeypatch):
    class _Spec:
        submodule_search_locations = ["/definitely/not/a/directory"]

    monkeypatch.setattr(paths.importlib.util, "find_spec", lambda name: _Spec())
    assert paths._package_dir("anything") is None


def test_package_dir_survives_a_broken_import_system(monkeypatch):
    def boom(name):
        raise ValueError("__spec__ is not set")

    monkeypatch.setattr(paths.importlib.util, "find_spec", boom)
    assert paths._package_dir("anything") is None


def test_eval_tasks_dir_falls_back_to_the_candidate_scan(tmp_path, monkeypatch):
    """With the package unreachable, the layout candidates are tried."""
    monkeypatch.setattr(paths, "_package_dir", lambda name: None)
    resolved = eval_tasks_dir()
    assert resolved.name == "eval_tasks"
    assert resolved.is_dir()


def test_eval_tasks_dir_last_resort_names_the_installed_path(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "_package_dir", lambda name: None)
    monkeypatch.setattr(paths, "_LIB_DIR", tmp_path / "nowhere" / "lib")
    resolved = eval_tasks_dir()
    assert resolved == tmp_path / "nowhere" / "eval_tasks"


def test_repo_path_is_none_outside_a_checkout(monkeypatch, tmp_path):
    """An installed layout has no repo root, and callers must see that."""
    monkeypatch.setattr(paths, "_LIB_DIR", tmp_path / "site-packages" / "lib")
    assert repo_root() is None
    assert repo_path("docs", "eval_baseline.json") is None
