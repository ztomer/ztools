"""install.sh's platform gate: 64-bit only, macOS Apple silicon only.

House rule (operator standing rules): every unsupported OS/arch combination is a
HARD FAILURE in the build/install script, never an `arch` `case` whose `*)`
warns and builds anyway -- that silently ships a binary nobody tested on that
platform. "Drop Intel" is macOS-SPECIFIC: Linux x86_64 and aarch64 stay
supported, so an x86_64 code path may never be deleted without checking whether
it still serves Linux.

install.sh had no gate at all: it read `uname` nowhere, called `brew --prefix`
unconditionally and built whatever it happened to be standing on. On an Intel
Mac that installed a binary nobody has ever tested; on Linux with no Homebrew it
died with "brew: command not found", which names a missing package rather than
the unsupported platform that caused it.

BOTH DIRECTIONS ON ONE MACHINE. The gate derives its verdict from ZTOOLS_OS and
ZTOOLS_ARCH and returns at ZTOOLS_PLATFORM_CHECK_ONLY=1, which fires AFTER the
gate and BEFORE the dependency check, the build and the install -- so the accept
path is testable here on this Apple silicon Mac, and no test in this file can
install a binary as a side effect of checking the gate. `test_the_gate_starts_no
_build` proves that rather than assuming it.

Modelled on the sibling gates (app_updates and sys_updater each drive their
build.sh through the same shape of seam).
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent  # tools/tests/ -> repo root
INSTALL = REPO / "install.sh"

#: The gate settles in milliseconds. Bounded, because a lock/gate test that can
#: block indefinitely is not a test -- it is a hang with no indication of where.
TIMEOUT = 30

#: Everything install.sh needs a real toolchain for. Stubbed at the FRONT of PATH
#: so a build or a `brew --prefix` is CAUGHT rather than performed.
TOOL_COMMANDS = ("cargo", "jq", "brew")

#: install.sh's accept set, quoted verbatim from the message it prints on a
#: reject. Asserted on every rejection: a gate that stops naming what IS
#: supported is a gate nobody can act on the message of.
SUPPORTED = "macOS/arm64, Linux/x86_64, Linux/aarch64"


def _goh() -> str:
    """The house TUI checkout, as the script itself resolves it."""
    return os.environ.get("GOH_DIR", str(Path.home() / "Projects" / "gates_of_heck"))


def _run(os_name, arch, **overrides):
    """Run the REAL install.sh through the OS/ARCH seam, stopping after the gate.

    The env is built from scratch rather than inherited, so a ZTOOLS_OS or
    ZTOOLS_ARCH in the caller's shell cannot mask the combination under test --
    an inherited override that silently won is a test that proves nothing.
    """
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(Path.home()),
        "NO_COLOR": "1",
        "GOH_DIR": _goh(),
        "ZTOOLS_OS": os_name,
        "ZTOOLS_ARCH": arch,
        "ZTOOLS_PLATFORM_CHECK_ONLY": "1",
    }
    env.update(overrides)
    return subprocess.run(
        ["/bin/bash", str(INSTALL)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO),
        # Not an exception path: a reject IS a result here, and the tests assert
        # on the code and the message.
        check=False,
        timeout=TIMEOUT,
    )


def _out(result):
    """stdout and stderr together.

    Which stream carries what depends on whether the house TUI was found: its
    `ok` prints to stdout, the plain-text fallback `ok` in install.sh prints to
    stderr. A test that pinned one stream would be asserting which of the two
    degraded modes the machine happens to be in.
    """
    return result.stdout + result.stderr


# ── accept ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "os_name,arch,normalised",
    [
        ("Darwin", "arm64", "macOS/arm64"),
        # Same silicon, two spellings. macOS says arm64 and Linux says aarch64,
        # so the accept set is three PAIRS after normalisation rather than a
        # list of aliases that has to grow every time a platform says one more.
        ("Darwin", "aarch64", "macOS/arm64"),
        ("Linux", "x86_64", "Linux/x86_64"),
        ("Linux", "amd64", "Linux/x86_64"),
        ("Linux", "aarch64", "Linux/aarch64"),
        ("Linux", "arm64", "Linux/aarch64"),
    ],
)
def test_supported_platforms_pass_the_gate(os_name, arch, normalised):
    r = _run(os_name, arch)
    assert r.returncode == 0, f"{os_name}/{arch} must pass: {_out(r)}"
    assert f"gate ok: {normalised}" in _out(r)


# ── reject: macOS Intel ───────────────────────────────────────────────────────


@pytest.mark.parametrize("arch", ["x86_64", "amd64"])
def test_macos_intel_is_rejected(arch):
    """The specific loss, so the message can say WHY rather than only what."""
    r = _run("Darwin", arch)
    assert r.returncode == 1, f"macOS/{arch} must hard-fail, got {r.returncode}"
    out = _out(r)
    # The combination found, NORMALISED -- so the operator sees the pair the
    # policy is written in, not just the raw uname spelling.
    assert "macOS/x86_64" in out
    # ...and what uname actually reported, so an alias cannot hide the real host.
    assert f"uname reported Darwin/{arch}" in out
    assert "Intel Macs are dropped" in out
    # Linux x86_64 must NOT be collateral: "drop Intel" is macOS-specific, and a
    # message implying otherwise would push someone to delete the Linux path.
    assert "Linux x86_64 and Linux aarch64 remain supported" in out
    assert SUPPORTED in out


# ── reject: 32-bit ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("arch", ["i386", "i686", "armv7l", "armhf"])
@pytest.mark.parametrize("os_name", ["Darwin", "Linux"])
def test_32bit_is_rejected_on_every_os(os_name, arch):
    """64-bit first and on EVERY OS.

    Ordering is the point: if the macOS-Intel branch ran first, a 32-bit Mac
    would be told Intel Macs are dropped -- true, and the wrong diagnosis for
    `uname -m` reporting i386, which is not an Intel Mac at all.
    """
    r = _run(os_name, arch)
    assert r.returncode == 1, f"{os_name}/{arch} must hard-fail"
    out = _out(r)
    assert f"uname reported {os_name}/{arch}" in out
    assert "64-bit only" in out
    assert "32-bit is never supported" in out
    assert SUPPORTED in out


# ── reject: unknown OS ────────────────────────────────────────────────────────


@pytest.mark.parametrize("os_name", ["SunOS", "FreeBSD", "OpenBSD", "Windows_NT"])
def test_unknown_os_is_rejected(os_name):
    r = _run(os_name, "x86_64")
    assert r.returncode == 1, f"{os_name} must hard-fail, got {r.returncode}"
    out = _out(r)
    assert f"unsupported platform: {os_name}/x86_64" in out
    assert "Unsupported OS" in out
    assert SUPPORTED in out


# ── the gate is early, and it is a gate and not a warning ─────────────────────


def _stub_bin(tmp_path):
    """A PATH whose cargo/jq/brew each record that they were reached, then fail.

    `install.sh` cannot be run for real to test it: it installs binaries into the
    Homebrew prefix. Stubbing the three commands it needs is what makes "nothing
    was built and nothing was installed" an assertion instead of a hope.
    """
    sentinel = tmp_path / "reached"
    binroot = tmp_path / "stubbin"
    binroot.mkdir()
    for name in TOOL_COMMANDS:
        path = binroot / name
        path.write_text(f'#!/bin/sh\necho {name} >> "{sentinel}"\nexit 1\n')
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return binroot, sentinel


def test_the_gate_starts_no_build(tmp_path):
    """The accept path must not have touched a toolchain.

    Proven two independent ways, because they fail differently: the sentinel
    catches a command being RUN, and the install dir catches anything landing on
    disk. Either alone is a single point of failure -- a build that never invoked
    cargo through PATH, or an install whose commands all resolved elsewhere.
    """
    binroot, sentinel = _stub_bin(tmp_path)
    install_dir = tmp_path / "bin"
    r = _run(
        "Darwin",
        "arm64",
        PATH=f"{binroot}:/usr/bin:/bin:/usr/sbin:/sbin",
        ZTOOLS_INSTALL_DIR=str(install_dir),
    )
    assert r.returncode == 0, _out(r)
    assert not sentinel.exists(), (
        "install.sh reached a toolchain before the check-only exit: " + sentinel.read_text()
    )
    assert not install_dir.exists(), "install.sh created the install dir"


def test_a_reject_starts_no_build_either(tmp_path):
    """The reject path must fail BEFORE the build, not after it.

    This is the direction that used to ship a binary nobody tested: a warn-and-
    build fallback cannot be caught by anything the script does afterwards.
    """
    binroot, sentinel = _stub_bin(tmp_path)
    r = _run(
        "Darwin",
        "x86_64",
        PATH=f"{binroot}:/usr/bin:/bin:/usr/sbin:/sbin",
        ZTOOLS_INSTALL_DIR=str(tmp_path / "bin"),
    )
    assert r.returncode == 1
    assert not sentinel.exists(), "it built something on an unsupported platform"
    assert not (tmp_path / "bin").exists()


def test_missing_dependencies_are_named(tmp_path):
    """ "brew: command not found" is not a diagnosis.

    An empty PATH plus an absolute /bin/bash means none of cargo, jq or brew can
    possibly resolve, so this reaches require_commands and nothing else -- no
    build, no install. The precondition is asserted rather than assumed: macOS
    ships jq in /usr/bin, and a PATH that happened to contain cargo would have
    this test invoke a real release build.
    """
    empty = tmp_path / "empty-path"
    empty.mkdir()
    for name in TOOL_COMMANDS:
        found = subprocess.run(
            ["/bin/bash", "-c", f"command -v {name}"],
            capture_output=True,
            text=True,
            env={"PATH": str(empty)},
            check=False,
            timeout=TIMEOUT,
        )
        assert found.returncode != 0, (
            f"{name} resolves under the empty PATH; this test would build for real"
        )

    r = _run(
        "Linux",
        "x86_64",
        PATH=str(empty),
        ZTOOLS_PLATFORM_CHECK_ONLY="",
        GOH_DIR=str(tmp_path / "nope"),
    )
    assert r.returncode == 1, _out(r)
    out = _out(r).lower()
    assert "missing required command" in out
    assert "cargo" in out
    assert "jq" in out
    # The point of the fix: no 127, and nothing blaming a package that is only
    # missing because the platform was wrong.
    assert "command not found" not in out
    assert "installing binaries" not in out.lower()


def test_macos_also_names_brew(tmp_path):
    """brew is what install.sh's own default BIN_DIR needs, so it is required --
    but only on the platform whose Homebrew prefix it goes into."""
    empty = tmp_path / "empty-path"
    empty.mkdir()
    r = _run(
        "Darwin",
        "arm64",
        PATH=str(empty),
        ZTOOLS_PLATFORM_CHECK_ONLY="",
        GOH_DIR=str(tmp_path / "nope"),
    )
    assert r.returncode == 1, _out(r)
    assert "brew" in _out(r)


def test_the_gate_hard_fails_without_the_house_tui(tmp_path):
    """A missing tui/lib.sh must not turn `die` into "die: command not found".

    tui/lib.sh was once deleted while scripts still sourced it, and `die` became
    a 127 with no message -- so install.sh's helpers are DEFINED before the
    source, not merely checked for. Under `set -e` a `return` from a sourced file
    also returns from the SOURCE and the caller carries on, which is why the
    fallback is a definition.
    """
    r = _run("Darwin", "x86_64", GOH_DIR=str(tmp_path / "no-tui-here"))
    assert r.returncode == 1, _out(r)
    out = _out(r)
    assert "macOS/x86_64" in out
    assert "command not found" not in out
    assert "no tui/lib.sh" in out


# ── the fix is not silently reverted ──────────────────────────────────────────


def test_the_seam_is_documented_in_the_header():
    """The override is the escape hatch a developer needs, so it is documented --
    in the header, where someone reading the script looks first."""
    lines = INSTALL.read_text().splitlines()
    end = next((i for i, line in enumerate(lines) if not line.startswith("#")), 0)
    header = "\n".join(lines[:end])
    for var in ("ZTOOLS_OS", "ZTOOLS_ARCH", "ZTOOLS_PLATFORM_CHECK_ONLY"):
        assert var in header, f"{var} is not documented in install.sh's header"
    assert "uname" in header


def test_the_retired_target_dir_fallback_is_gone():
    """The second defect: `cargo metadata ... || echo "$HOME/.cache/cargo-target"`.

    Unreachable, because `set -e` cannot see a failure inside `$( ... || ... )` --
    and WRONG, because ~/.cargo/config.toml sets a machine-wide `build-dir`
    namespaced per workspace by path hash, so nothing is built under that path
    at all. A failure therefore pointed at a directory that cannot hold the
    artifact. The resolution is now loud and named.
    """
    source = INSTALL.read_text()
    # The executable lines only. The path is quoted in the comment explaining WHY
    # it was retired, and a test that greps the whole file cannot tell the
    # narration from the code -- it would fail on the fix that documents the
    # defect, and pass on any file that merely mentioned the old path.
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert ".cache/cargo-target" not in code
    # `-e` as well as `-r`: without it a missing key yields the string "null",
    # which is a path that does not exist and looks like an answer.
    assert "jq -er" in code
