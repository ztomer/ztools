//! Which Camoufox browser binary a Rust-driven launch uses.
//!
//! The class this closes: **a binary resolved from one hardcoded layout**.
//! `camoufox-rs` examples only check `~/.cache/camoufox/camoufox`, which misses
//! the macOS pip layout (`~/Library/Caches/camoufox/browsers/official/*/`)
//! entirely — on this fleet's Macs that is the ONLY layout that exists, so the
//! default resolution fails everywhere it matters. The Python side never had
//! this problem because the `camoufox` pip package knows its own install dir.
//!
//! So resolution is explicit, ordered and **probed**: `CAMOUFOX_BIN` first (one
//! lever, no code change), then the macOS pip cache glob (newest version dir
//! wins), then the Linux `~/.cache` file. A candidate is only accepted after
//! it exists on disk; if none survive the caller gets a hard error naming
//! every path tried — never a silent fall-through to a binary nobody verified.
//!
//! `resolve_with` is the seam: both the accept and the reject direction are
//! testable on one machine with fixture directories.

use std::path::{Path, PathBuf};

/// Why no browser binary could be used, with everything that was tried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryError {
    /// Each candidate that was considered and why it lost.
    pub rejected: Vec<(String, String)>,
}

impl std::fmt::Display for BinaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no Camoufox browser binary found")?;
        for (path, why) in &self.rejected {
            write!(f, "\n  ✗ {path}: {why}")?;
        }
        write!(
            f,
            "\n  → install it (`pip install camoufox && python3 -m camoufox fetch`) or set CAMOUFOX_BIN"
        )
    }
}

impl std::error::Error for BinaryError {}

/// The macOS pip-install layout: one version dir per fetched release.
fn macos_pip_candidates(home: &Path) -> Vec<PathBuf> {
    let official = home.join("Library/Caches/camoufox/browsers/official");
    let Ok(entries) = std::fs::read_dir(&official) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path().join("Camoufox.app/Contents/MacOS/camoufox"))
        .filter(|p| p.is_file())
        .collect();
    // Version dirs sort newest-last (`152.0.4-beta.29-…` < `152.0.4-beta.30-…`).
    out.sort();
    out
}

/// Resolve against explicit inputs so tests never touch the real home.
///
/// `env_override` is the value of `CAMOUFOX_BIN` (if set); `home` stands in
/// for the user home. Returns the first candidate that exists on disk.
///
/// # Errors
///
/// When no candidate exists on disk — the error names every path tried.
pub fn resolve_with(home: &Path, env_override: Option<&str>) -> Result<PathBuf, BinaryError> {
    let mut rejected: Vec<(String, String)> = Vec::new();

    if let Some(raw) = env_override {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let p = PathBuf::from(trimmed);
            if p.is_file() {
                return Ok(p);
            }
            rejected.push((trimmed.to_owned(), "CAMOUFOX_BIN does not exist".to_owned()));
        }
    }

    let mut pip = macos_pip_candidates(home);
    // Newest version dir wins; older ones stay as fallbacks.
    pip.reverse();
    for p in pip {
        if p.is_file() {
            return Ok(p);
        }
        rejected.push((p.display().to_string(), "not a file".to_owned()));
    }

    for p in [
        home.join(".cache/camoufox/camoufox"),
        PathBuf::from("/root/.cache/camoufox/camoufox"),
    ] {
        if p.is_file() {
            return Ok(p);
        }
        rejected.push((p.display().to_string(), "not found".to_owned()));
    }

    Err(BinaryError { rejected })
}

/// Resolve the Camoufox binary against the real environment.
///
/// Narrow wrapper over [`resolve_with`] so production reads `HOME` and
/// `CAMOUFOX_BIN` in exactly one place.
///
/// # Errors
///
/// When no candidate exists on disk — the error names every path tried.
pub fn resolve() -> Result<PathBuf, BinaryError> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/root"));
    let env_override = std::env::var("CAMOUFOX_BIN").ok();
    resolve_with(&home, env_override.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_home(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ztools-bin-test-{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn env_override_wins_when_it_exists() {
        let home = fixture_home("override");
        let bin = home.join("my-camoufox");
        std::fs::write(&bin, b"x").unwrap();
        let got = resolve_with(&home, Some(bin.to_str().unwrap())).unwrap();
        assert_eq!(got, bin);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn missing_override_is_reported_not_skipped_silently() {
        let home = fixture_home("missing-override");
        let err = resolve_with(&home, Some("/nonexistent/ztools_no_browser")).unwrap_err();
        assert!(
            err.rejected
                .iter()
                .any(|(p, _)| p.contains("ztools_no_browser")),
            "override must appear in rejected: {err}"
        );
        assert!(err.to_string().contains("CAMOUFOX_BIN"), "{err}");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn macos_pip_layout_resolves_newest_first() {
        let home = fixture_home("pip");
        let old = home.join(
            "Library/Caches/camoufox/browsers/official/152.0.4-beta.29-620d3328/Camoufox.app/Contents/MacOS/camoufox",
        );
        let new = home.join(
            "Library/Caches/camoufox/browsers/official/152.0.4-beta.30-3b43e766/Camoufox.app/Contents/MacOS/camoufox",
        );
        for p in [&old, &new] {
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"x").unwrap();
        }
        let got = resolve_with(&home, None).unwrap();
        assert_eq!(got, new, "newest version dir must win");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn empty_machine_errors_naming_everything_tried() {
        let home = fixture_home("empty");
        let err = resolve_with(&home, None).unwrap_err();
        let text = err.to_string();
        assert!(text.contains(".cache/camoufox/camoufox"), "{text}");
        assert!(text.contains("camoufox fetch"), "{text}");
        let _ = std::fs::remove_dir_all(&home);
    }
}
