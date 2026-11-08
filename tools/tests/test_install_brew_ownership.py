"""install.sh never writes into the Homebrew prefix bin (queue item ZT.BREWLINK).

ONE OWNER PER PATH. `$(brew --prefix)/bin/ztools` and its alias links belong to
the tap formula (`brew link` creates them as links into the Cellar). install.sh
used to copy a REAL file there, so the next `brew reinstall ztomer/tap/ztools`
failed at `brew link` ("Target ... already exists") until someone ran
`brew link --overwrite`. install.sh now installs into ~/.local/bin by default and
REFUSES the brew prefix bin, before it builds anything.

Hermetic: brew and cargo are stubs (a fake prefix under tmp_path; `cargo build`
is a no-op and `cargo metadata --no-deps` names a fake target dir holding a fake
release binary). Only the platform gate's dependency-graph read reaches the real
cargo. `brew link` is simulated the way brew decides a conflict: a target that
exists and is not already a link into the Cellar.
"""

import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
INSTALL = REPO / "install.sh"
TIMEOUT = 60
REAL_CARGO = shutil.which("cargo")
REAL_JQ = shutil.which("jq")
PASSTHROUGH = ("CARGO_HOME", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN", "CARGO_NET_OFFLINE")

#: What the tap formula links into the prefix bin (Formula/ztools.rb).
FORMULA_LINKS = (
    "ztools",
    "weekend",
    "weekend-plan",
    "twitter",
    "twitter-summarize",
    "oeval",
    "model-eval",
    "rename_images",
    "image-renamer",
)


def _setup(tmp_path):
    prefix = tmp_path / "brew"
    (prefix / "bin").mkdir(parents=True)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    target = tmp_path / "target"
    (target / "release").mkdir(parents=True)
    fake = target / "release" / "ztools"
    fake.write_text("#!/bin/sh\necho fake-ztools\n")
    fake.chmod(0o755)
    (stubs / "brew").write_text(
        f'#!/bin/sh\n[ "$1" = --prefix ] && echo "{prefix}" && exit 0\nexit 1\n'
    )
    (stubs / "cargo").write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  build) exit 0 ;;\n"
        "  metadata)\n"
        '    case " $* " in *" --no-deps "*)\n'
        f'      echo \'{{"target_directory":"{target}"}}\'; exit 0 ;;\n'
        "    esac\n"
        f'    exec "{REAL_CARGO}" "$@" ;;\n'
        "esac\nexit 1\n"
    )
    for s in ("brew", "cargo"):
        (stubs / s).chmod(0o755)
    (stubs / "jq").symlink_to(REAL_JQ)
    return prefix, stubs


def _run(tmp_path, stubs, **extra):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {
        "PATH": f"{stubs}:{Path(REAL_CARGO).parent}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "CARGO_HOME": os.environ.get("CARGO_HOME", str(Path.home() / ".cargo")),
        "RUSTUP_HOME": os.environ.get("RUSTUP_HOME", str(Path.home() / ".rustup")),
        "NO_COLOR": "1",
        "GOH_DIR": os.environ.get("GOH_DIR", str(Path.home() / "Projects" / "gates_of_heck")),
        "RUSTC_WRAPPER": "",
    }
    env.update({k: os.environ[k] for k in PASSTHROUGH if k in os.environ})
    env.update(extra)
    r = subprocess.run(
        ["/bin/bash", str(INSTALL)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
        check=False,
        timeout=TIMEOUT,
        stdin=subprocess.DEVNULL,
    )
    return r, home


def _brew_link_conflicts(prefix):
    """The targets `brew link` would refuse: present and not a link into the Cellar."""
    out = []
    for name in FORMULA_LINKS:
        p = prefix / "bin" / name
        if p.is_symlink() and "Cellar/ztools/" in os.readlink(p):
            continue
        if p.exists() or p.is_symlink():
            out.append(name)
    return out


def test_default_install_leaves_the_brew_bin_to_brew(tmp_path):
    prefix, stubs = _setup(tmp_path)
    r, home = _run(tmp_path, stubs)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _brew_link_conflicts(prefix) == [], "install.sh wrote into the brew prefix bin"
    assert (home / ".local" / "bin" / "ztools").is_file()
    assert os.readlink(home / ".local" / "bin" / "weekend") == "ztools"


def test_explicit_brew_bin_is_refused_before_any_write(tmp_path):
    prefix, stubs = _setup(tmp_path)
    r, _ = _run(tmp_path, stubs, ZTOOLS_INSTALL_DIR=str(prefix / "bin"))
    assert r.returncode != 0
    assert "Homebrew" in r.stdout + r.stderr
    assert list((prefix / "bin").iterdir()) == []
