//! ztools' runtime dependency table, with its pinned versions.
//!
//! The ONE place every external binary, browser and Python module ztools
//! shells out to or loads is declared, with its install command. The shape that probes and
//! installs is [`crate::ztools::selfheal`]; this file is only the table and
//! the real [`Host`].
//!
//! Homebrew cannot pin a formula to an exact version, so a brew row's `pin` is
//! the version ztools was verified against and `brew install` takes the
//! current bottle; the pip and Camoufox rows install exactly their pin.

use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::ztools::selfheal::{Dependency, Health, Host, Probe, ensure, switched_off};

/// The switch that turns installing off (`0`/`off`/`no`/`false`).
pub const SWITCH: &str = "ZTOOLS_SELF_HEAL";

/// The Camoufox browser build the collector is verified against.
pub const CAMOUFOX_BROWSER_PIN: &str = "156.0.1-beta.39";

/// Every runtime dependency, prerequisites before the rows that need them.
pub const TABLE: &[Dependency] = &[
    Dependency {
        name: "brew",
        pin: "5",
        needed_for: "installing every other row",
        probe: Probe::Command("brew"),
        install: &[],
        requires: &[],
        manual: "Homebrew installs itself interactively; run the installer from https://brew.sh once",
    },
    Dependency {
        name: "python3",
        pin: "3.15",
        needed_for: "the Camoufox installer",
        probe: Probe::Command("python3"),
        install: &["brew", "install", "python@3.15"],
        requires: &["brew"],
        manual: "",
    },
    Dependency {
        name: "camoufox-python",
        pin: "0.5.5",
        needed_for: "fetching the Camoufox browser",
        probe: Probe::PythonModule("camoufox"),
        install: &[
            "python3",
            "-m",
            "pip",
            "install",
            "--user",
            "--break-system-packages",
            "camoufox==0.5.5",
        ],
        requires: &["python3"],
        manual: "",
    },
    Dependency {
        name: "camoufox-browser",
        pin: CAMOUFOX_BROWSER_PIN,
        needed_for: "the twitter timeline collector",
        probe: Probe::Located("camoufox-browser"),
        install: &[
            "python3",
            "-m",
            "camoufox",
            "fetch",
            "official/156.0.1-beta.39",
        ],
        requires: &["camoufox-python"],
        manual: "",
    },
    Dependency {
        name: "bat",
        pin: "0.26.1",
        needed_for: "rendering markdown in a terminal",
        probe: Probe::Command("bat"),
        install: &["brew", "install", "bat"],
        requires: &["brew"],
        manual: "",
    },
    Dependency {
        name: "fm",
        pin: "macOS 27",
        needed_for: "the on-device model fallback",
        probe: Probe::Command("/usr/bin/fm"),
        install: &[],
        requires: &[],
        manual: "it ships with macOS 27 and nothing else installs it; the fallback is off without it",
    },
];

/// The real machine: `PATH`, `python3`, the Camoufox cache, stderr.
pub struct SystemHost;

impl Host for SystemHost {
    fn probe(&self, dep: &Dependency) -> Option<String> {
        match dep.probe {
            Probe::Command(cmd) => on_path(cmd).map(|p| p.display().to_string()),
            Probe::PythonModule(module) => Command::new("python3")
                .args(["-c", &format!("import {module}")])
                .stdin(Stdio::null())
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|_| format!("python module {module}")),
            Probe::Located(_) => crate::ztools::twitter::browser_bin::resolve()
                .ok()
                .map(|p| p.display().to_string()),
        }
    }

    fn install(&self, argv: &[&str]) -> Result<(), String> {
        if switched_off(std::env::var(SWITCH).ok().as_deref()) {
            return Err(format!("self-heal is off ({SWITCH})"));
        }
        let (program, rest) = argv.split_first().ok_or("empty install command")?;
        let out = Command::new(program)
            .args(rest)
            .stdin(Stdio::null())
            .output()
            .map_err(|e| format!("cannot run {program}: {e}"))?;
        if out.status.success() {
            return Ok(());
        }
        let stderr = String::from_utf8_lossy(&out.stderr);
        let last = stderr.lines().rev().find(|l| !l.trim().is_empty());
        Err(match last {
            Some(line) => format!("{}: {}", out.status, line.trim()),
            None => out.status.to_string(),
        })
    }

    fn say(&self, line: &str) {
        eprintln!("{line}");
    }
}

/// An executable named `cmd`: a path with a `/` is checked as is, a bare name
/// is resolved by the shell's own `command -v` (the lookup every spawn makes,
/// so the probe and the use cannot disagree about `PATH`).
#[must_use]
pub fn on_path(cmd: &str) -> Option<PathBuf> {
    use std::os::unix::fs::PermissionsExt;
    let resolved = if cmd.contains('/') {
        PathBuf::from(cmd)
    } else {
        let out = Command::new("/bin/sh")
            .args(["-c", "command -v \"$1\"", "sh", cmd])
            .stdin(Stdio::null())
            .output()
            .ok()?;
        PathBuf::from(String::from_utf8_lossy(&out.stdout).trim())
    };
    std::fs::metadata(&resolved)
        .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .then_some(resolved)
}

/// Ensure one row on the real machine.
#[must_use]
pub fn ensure_system(name: &str) -> Health {
    ensure(TABLE, name, &SystemHost)
}

/// `ztools deps`: ensure every row, print the table.
///
/// # Errors
///
/// When any row stays unavailable — the table says which, and why.
pub fn cli() -> anyhow::Result<()> {
    let rows = crate::ztools::selfheal::ensure_all(TABLE, &SystemHost);
    print!("{}", crate::ztools::selfheal::render(&rows));
    let down: Vec<&str> = rows
        .iter()
        .filter(|(_, h)| h.location().is_none())
        .map(|(d, _)| d.name)
        .collect();
    if down.is_empty() {
        Ok(())
    } else {
        anyhow::bail!("unavailable: {}", down.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ztools::selfheal::lookup;

    #[test]
    fn every_prerequisite_names_a_row_declared_before_it() {
        for (i, dep) in TABLE.iter().enumerate() {
            for req in dep.requires {
                let at = TABLE.iter().position(|d| d.name == *req);
                assert!(at.is_some_and(|a| a < i), "{} needs {req}", dep.name);
            }
            assert!(
                !dep.install.is_empty() || !dep.manual.is_empty(),
                "{}: a row with no install command must say why",
                dep.name
            );
        }
    }

    #[test]
    fn the_browser_row_installs_exactly_its_pin() {
        let row = lookup(TABLE, "camoufox-browser").unwrap();
        assert!(row.install.last().unwrap().ends_with(CAMOUFOX_BROWSER_PIN));
        let pkg = lookup(TABLE, "camoufox-python").unwrap();
        assert!(pkg.install.last().unwrap().ends_with(pkg.pin));
    }

    #[test]
    fn the_system_probe_sees_a_real_command_and_not_a_fake_one() {
        let sh = Dependency {
            probe: Probe::Command("sh"),
            ..TABLE[0]
        };
        assert!(SystemHost.probe(&sh).is_some());
        let none = Dependency {
            probe: Probe::Command("/nonexistent/ztools-no-such-tool"),
            ..TABLE[0]
        };
        assert!(SystemHost.probe(&none).is_none());
        let module = Dependency {
            probe: Probe::PythonModule("ztools_no_such_module_xyz"),
            ..TABLE[0]
        };
        assert!(SystemHost.probe(&module).is_none());
    }

    #[test]
    fn the_system_installer_reports_failure_with_its_reason() {
        let err = SystemHost.install(&["/bin/sh", "-c", "echo boom >&2; exit 3"]);
        let err = err.unwrap_err();
        assert!(err.contains("boom"), "{err}");
        assert!(SystemHost.install(&["/bin/sh", "-c", "exit 0"]).is_ok());
        assert!(
            SystemHost
                .install(&["/nonexistent/ztools-installer"])
                .is_err()
        );
        assert!(SystemHost.install(&[]).is_err());
    }
}
