//! Which Python interpreter the remaining shell-outs run.
//!
//! The class this closes: **an interpreter resolved from the ambient `PATH`**.
//! Three call sites used a bare `Command::new("python3")`. That is correct in a
//! login shell and wrong everywhere else — the menubar app's GUI environment
//! has `PATH=/usr/bin:/bin:/usr/sbin:/sbin`, so `python3` resolves to
//! `/usr/bin/python3`, which has none of the deps. The twitter refresh died on
//! `ModuleNotFoundError: No module named 'requests'`; the weekend planner's
//! search helper swallowed the same failure and reported "no candidates found",
//! which was not true — nothing had searched.
//!
//! So resolution is explicit, ordered and **probed**: a candidate is only
//! accepted after it has been asked to import the exact modules the caller
//! needs. A candidate that cannot is skipped with its reason recorded, and if
//! none survive the caller gets a hard error naming every path tried — never a
//! silent fall-through to an interpreter nobody verified.
//!
//! `resolve_with` is the seam: both the accept and the reject direction are
//! testable on one machine without depending on what this machine has
//! installed.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Why no interpreter could be used, with everything that was tried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyEnvError {
    /// The modules the caller asked for.
    pub required: Vec<String>,
    /// Each candidate that was considered and why it lost.
    pub rejected: Vec<(String, String)>,
}

impl std::fmt::Display for PyEnvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no Python interpreter has {}", self.required.join(", "))?;
        for (path, why) in &self.rejected {
            write!(f, "\n  ✗ {path}: {why}")?;
        }
        write!(
            f,
            "\n  → set ZTOOLS_PYTHON to an interpreter that has them, or install them into one above"
        )
    }
}

impl std::error::Error for PyEnvError {}

/// The interpreters to try, best first.
///
/// `ZTOOLS_PYTHON` is first so a machine with an unusual layout has one lever
/// that needs no code change. Absolute paths come before the `PATH` lookup
/// precisely because the `PATH` we inherit is the thing that cannot be trusted;
/// bare `python3` stays last so a login shell that has everything still works.
pub fn candidates() -> Vec<String> {
    let mut out = Vec::new();
    if let Ok(explicit) = std::env::var("ZTOOLS_PYTHON") {
        if !explicit.trim().is_empty() {
            out.push(explicit);
        }
    }
    if let Some(home) = dirs::home_dir() {
        out.push(
            home.join("Projects/ztools/.venv/bin/python")
                .display()
                .to_string(),
        );
    }
    out.push("/opt/homebrew/bin/python3".to_string());
    out.push("/usr/local/bin/python3".to_string());
    out.push("python3".to_string());
    out
}

/// Ask one interpreter to import the required modules. `Ok(())` means it can.
///
/// A real import, not a version string or a path test: "the binary exists" and
/// "the binary can run this pipeline" are different facts, and only the second
/// one is worth anything here.
fn probe_imports(program: &str, required: &[&str]) -> Result<(), String> {
    let stmt = format!("import {}", required.join(", "));
    let mut cmd = std::process::Command::new(program);
    cmd.args(["-c", &stmt]);
    apply_pythonpath(&mut cmd);
    match cmd.output() {
        Err(e) => Err(format!("cannot run: {e}")),
        Ok(out) if out.status.success() => Ok(()),
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            let last = stderr
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("import failed")
                .trim()
                .to_string();
            Err(last)
        }
    }
}

/// Resolve against an explicit candidate list and probe. The production entry
/// points call this with [`candidates`] and [`probe_imports`]; tests call it
/// with their own so a machine's real installs never decide a test's outcome.
pub fn resolve_with<P>(
    required: &[&str],
    candidates: &[String],
    mut probe: P,
) -> Result<String, PyEnvError>
where
    P: FnMut(&str, &[&str]) -> Result<(), String>,
{
    let mut rejected = Vec::new();
    for candidate in candidates {
        match probe(candidate, required) {
            Ok(()) => return Ok(candidate.clone()),
            Err(why) => rejected.push((candidate.clone(), why)),
        }
    }
    Err(PyEnvError {
        required: required.iter().map(|s| s.to_string()).collect(),
        rejected,
    })
}

/// Memo of resolutions, keyed by the module set asked for. Probing spawns a
/// process per candidate; a pipeline that resolves the same set twice should
/// not pay for it twice.
fn cache() -> &'static Mutex<HashMap<String, Result<String, PyEnvError>>> {
    static CACHE: OnceLock<Mutex<HashMap<String, Result<String, PyEnvError>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The interpreter to run for a pipeline needing `required`, or a stated error.
pub fn resolve(required: &[&str]) -> Result<String, PyEnvError> {
    let key = required.join(",");
    if let Some(hit) = cache().lock().unwrap().get(&key) {
        return hit.clone();
    }
    let found = resolve_with(required, &candidates(), probe_imports);
    cache().lock().unwrap().insert(key, found.clone());
    found
}

/// The `references/` tree the shipped Python modules live in.
///
/// Was duplicated in `twitter::browser::setup_python_env`; it belongs here
/// because an interpreter and the `PYTHONPATH` it needs are one decision — the
/// probe above has to see the same module search path the real run will, or it
/// would verify an interpreter that then fails on the shipped modules.
pub fn reference_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        if let Some(p) = std::path::Path::new(&manifest_dir)
            .parent()
            .map(|p| p.join("references"))
        {
            if p.exists() {
                paths.push(p);
            }
        }
    }
    if let Some(home) = dirs::home_dir() {
        let p = home.join("Projects/ztools/references");
        if p.exists() && !paths.contains(&p) {
            paths.push(p);
        }
    }
    paths
}

/// Put `references/` on a command's `PYTHONPATH`, preserving any inherited one.
pub fn apply_pythonpath(cmd: &mut std::process::Command) {
    let paths = reference_paths();
    if paths.is_empty() {
        return;
    }
    let combined = paths
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join(":");
    match std::env::var("PYTHONPATH") {
        Ok(existing) if !existing.is_empty() => {
            cmd.env("PYTHONPATH", format!("{combined}:{existing}"));
        }
        _ => {
            cmd.env("PYTHONPATH", combined);
        }
    }
}

/// A `Command` for `required`, already pointed at a probed interpreter and
/// carrying the `references/` search path. The one way these subsystems should
/// start Python.
pub fn command(required: &[&str]) -> Result<std::process::Command, PyEnvError> {
    let program = resolve(required)?;
    let mut cmd = std::process::Command::new(program);
    apply_pythonpath(&mut cmd);
    Ok(cmd)
}

#[cfg(test)]
#[path = "pyenv_tests.rs"]
mod tests;
