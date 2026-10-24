//! Self-healing runtime dependencies (owner ask O1: "if a dependency is not
//! available it should be installed").
//!
//! The SHAPE, kept identical in `ztools`, `app_updates` and `sys_updater`:
//!
//! 1. Each tool declares every runtime dependency it shells out to or loads
//!    in ONE table of [`Dependency`] rows: a name, the pinned version, what it
//!    is needed for, how to probe for it, and the install command.
//! 2. At use time a miss is detected by PROBING ([`Host::probe`]), never by
//!    assuming: a dependency is present only when the probe finds it.
//! 3. A miss is installed with a one-line stated reason (`→ … is missing —
//!    installing …`), then probed AGAIN: an install command that exits 0 but
//!    leaves nothing behind is not a success.
//! 4. Anything that cannot be installed — no install command, a prerequisite
//!    that is itself unavailable, an installer that fails, or self-heal turned
//!    off — comes back as [`Health::Unavailable`] with the reason, and the
//!    caller degrades on it (engineering rule #13). Nothing here panics.
//!
//! The installer and the probe sit behind [`Host`], so the whole
//! miss → install → use path runs against a fake in tests and no test ever
//! runs a real installer.

use std::fmt::Write as _;

/// How a dependency's presence is established.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    /// An executable on `PATH` (or an absolute path to one).
    Command(&'static str),
    /// A Python module `python3 -c "import <name>"` loads.
    PythonModule(&'static str),
    /// A tool-specific locator; the [`Host`] knows how to resolve the name.
    Located(&'static str),
}

/// One runtime dependency, as the tool's table declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dependency {
    /// The name the table, the messages and `requires` use.
    pub name: &'static str,
    /// The version installed on a miss (or, for a package manager that cannot
    /// pin, the version this tool was verified against).
    pub pin: &'static str,
    /// What breaks without it, stated in the install line.
    pub needed_for: &'static str,
    /// How a miss is detected.
    pub probe: Probe,
    /// The install command's argv. Empty: this tool cannot install it, and
    /// [`Dependency::manual`] says why and what to do.
    pub install: &'static [&'static str],
    /// Other rows (by name) that must be present before `install` can run.
    pub requires: &'static [&'static str],
    /// For a row with no install command: the stated reason and the manual step.
    pub manual: &'static str,
}

/// What [`ensure`] established about one dependency.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Health {
    /// Found by the probe; the payload is where (or what) it found.
    Present(String),
    /// Missing, installed, and found by the second probe.
    Installed(String),
    /// Missing and not installable; the payload is the stated reason.
    Unavailable(String),
}

impl Health {
    /// Where the dependency is, when it can be used.
    #[must_use]
    pub fn location(&self) -> Option<&str> {
        match self {
            Self::Present(at) | Self::Installed(at) => Some(at),
            Self::Unavailable(_) => None,
        }
    }

    /// The glyph and word a status table shows for this state.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Present(_) => "✓ present",
            Self::Installed(_) => "✓ installed",
            Self::Unavailable(_) => "⚠ unavailable",
        }
    }
}

/// The impure edge: probing, installing, and telling the user.
pub trait Host {
    /// `Some(location)` when `dep` is present, by `dep.probe`.
    fn probe(&self, dep: &Dependency) -> Option<String>;
    /// Run one install command.
    ///
    /// # Errors
    ///
    /// The command could not start, exited non-zero, or self-heal is off —
    /// the message is the stated reason.
    fn install(&self, argv: &[&str]) -> Result<(), String>;
    /// One line of house-TUI output.
    fn say(&self, line: &str);
}

/// The row named `name`.
#[must_use]
pub fn lookup<'t>(table: &'t [Dependency], name: &str) -> Option<&'t Dependency> {
    table.iter().find(|d| d.name == name)
}

/// Make `name` usable: probe, and on a miss install its prerequisites and
/// then it, with a stated reason; never panic.
pub fn ensure(table: &[Dependency], name: &str, host: &dyn Host) -> Health {
    ensure_at(table, name, host, 0)
}

/// Every row of `table`, ensured in table order.
pub fn ensure_all<'t>(table: &'t [Dependency], host: &dyn Host) -> Vec<(&'t Dependency, Health)> {
    table
        .iter()
        .map(|d| (d, ensure(table, d.name, host)))
        .collect()
}

fn ensure_at(table: &[Dependency], name: &str, host: &dyn Host, depth: usize) -> Health {
    let Some(dep) = lookup(table, name) else {
        return Health::Unavailable(format!("{name} is not in the dependency table"));
    };
    if let Some(at) = host.probe(dep) {
        return Health::Present(at);
    }
    let unavailable = |reason: String| {
        host.say(&format!("⚠ {reason}"));
        Health::Unavailable(reason)
    };
    if dep.install.is_empty() {
        return unavailable(format!(
            "{} is not installed and cannot be installed automatically — {}",
            dep.name, dep.manual
        ));
    }
    if depth > table.len() {
        return unavailable(format!(
            "{} cannot be installed — its prerequisites form a cycle",
            dep.name
        ));
    }
    for req in dep.requires {
        if let Health::Unavailable(why) = ensure_at(table, req, host, depth + 1) {
            return unavailable(format!(
                "{} cannot be installed — it needs {req}, which is unavailable ({why})",
                dep.name
            ));
        }
    }
    host.say(&format!(
        "→ {} {} is missing — installing it, needed for {}: {}",
        dep.name,
        dep.pin,
        dep.needed_for,
        dep.install.join(" ")
    ));
    if let Err(why) = host.install(dep.install) {
        return unavailable(format!(
            "{} {} could not be installed ({why}) — {} is degraded",
            dep.name, dep.pin, dep.needed_for
        ));
    }
    host.probe(dep).map_or_else(
        || {
            unavailable(format!(
                "{} {} was installed but the probe still cannot find it — {} is degraded",
                dep.name, dep.pin, dep.needed_for
            ))
        },
        |at| {
            host.say(&format!("✓ installed {} {} — {at}", dep.name, dep.pin));
            Health::Installed(at)
        },
    )
}

/// The `deps` report: one line per row, its state, and where or why.
#[must_use]
pub fn render(rows: &[(&Dependency, Health)]) -> String {
    let width = rows.iter().map(|(d, _)| d.name.len()).max().unwrap_or(0);
    let mut out = String::new();
    for (dep, health) in rows {
        let detail = match health {
            Health::Present(at) | Health::Installed(at) => at.as_str(),
            Health::Unavailable(why) => why.as_str(),
        };
        let _ = writeln!(
            out,
            "{:<width$}  {:<13}  pin {:<18}  {detail}",
            dep.name,
            health.label(),
            dep.pin
        );
    }
    out
}

/// Whether the `<TOOL>_SELF_HEAL` switch `value` turns installing off.
/// Off only for an explicit `0`, `off`, `no` or `false`: the default is on.
#[must_use]
pub fn switched_off(value: Option<&str>) -> bool {
    value.is_some_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "no" | "false"
        )
    })
}

#[cfg(test)]
#[path = "selfheal_tests.rs"]
mod tests;
