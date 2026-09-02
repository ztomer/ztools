//! Native OCR extraction engine for image renaming.
//!
//! Provides trait-based OCR extraction supporting Tesseract CLI / pure native engines
//! and deterministic test doubles.

use std::path::{Path, PathBuf};
use std::process::Command;

pub const DEFAULT_TESSERACT_PATH: &str = "/opt/homebrew/bin/tesseract";

pub trait OcrEngine: Send + Sync {
    fn is_available(&self) -> bool;
    fn extract_text(&self, image_path: &Path) -> Option<String>;
    fn extract_first_line(&self, image_path: &Path) -> Option<String> {
        self.extract_text(image_path).and_then(|text| {
            text.lines()
                .map(|l| l.trim())
                .find(|l| !l.is_empty())
                .map(|s| s.to_string())
        })
    }
}

/// Tesseract CLI OCR engine.
pub struct TesseractEngine {
    binary_path: PathBuf,
}

impl TesseractEngine {
    pub fn new(path: Option<PathBuf>) -> Self {
        let binary_path = path.unwrap_or_else(|| {
            if Path::new(DEFAULT_TESSERACT_PATH).exists() {
                PathBuf::from(DEFAULT_TESSERACT_PATH)
            } else {
                PathBuf::from("tesseract")
            }
        });
        Self { binary_path }
    }
}

impl Default for TesseractEngine {
    fn default() -> Self {
        Self::new(None)
    }
}

impl OcrEngine for TesseractEngine {
    fn is_available(&self) -> bool {
        if self.binary_path.is_absolute() {
            return self.binary_path.exists();
        }
        Command::new(&self.binary_path)
            .arg("--version")
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    }

    fn extract_text(&self, image_path: &Path) -> Option<String> {
        if !image_path.exists() {
            return None;
        }
        let output = Command::new(&self.binary_path)
            .args([image_path.to_str()?, "stdout", "--oem", "1", "-l", "eng"])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }
}

/// Global convenience function checking if default OCR engine is available.
pub fn ocr_available() -> bool {
    TesseractEngine::default().is_available()
}

/// Extract first readable line using the default OCR engine.
pub fn extract_first_line(image_path: &Path) -> Option<String> {
    TesseractEngine::default().extract_first_line(image_path)
}

/// Extract full text using the default OCR engine.
pub fn extract_full_text(image_path: &Path) -> Option<String> {
    TesseractEngine::default().extract_text(image_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEngine {
        canned: Option<String>,
    }

    impl OcrEngine for MockEngine {
        fn is_available(&self) -> bool {
            true
        }
        fn extract_text(&self, _image_path: &Path) -> Option<String> {
            self.canned.clone()
        }
    }

    #[test]
    fn test_extract_first_line_skips_empty_lines() {
        let engine = MockEngine {
            canned: Some("\n   \n  Invoice #10423  \nDate: 2026-08-01\n".to_string()),
        };
        assert!(engine.is_available());
        let first = engine.extract_first_line(Path::new("dummy.png"));
        assert_eq!(first.as_deref(), Some("Invoice #10423"));
    }

    #[test]
    fn test_extract_first_line_none_when_empty() {
        let engine = MockEngine {
            canned: Some("\n   \n  \n".to_string()),
        };
        let first = engine.extract_first_line(Path::new("dummy.png"));
        assert_eq!(first, None);
    }

    #[test]
    fn test_is_available_absolute_path_checks_existence() {
        let missing =
            TesseractEngine::new(Some(PathBuf::from("/nonexistent/ztools_missing_tesseract")));
        assert!(!missing.is_available());

        let present_file = tempfile::NamedTempFile::new().unwrap();
        // An existing absolute binary_path satisfies the availability check.
        let present = TesseractEngine::new(Some(present_file.path().to_path_buf()));
        assert!(present.is_available());
    }

    #[test]
    fn test_is_available_relative_path_requires_runnable_binary() {
        // A PATH lookup that cannot spawn must report unavailable, not panic.
        let bogus = TesseractEngine::new(Some(PathBuf::from(
            "ztools_definitely_not_a_real_binary_4f8a",
        )));
        assert!(!bogus.is_available());
    }

    #[test]
    fn test_extract_text_returns_none_for_missing_image() {
        let engine = TesseractEngine::new(Some(PathBuf::from("/bin/echo")));
        assert!(engine
            .extract_text(Path::new("/nonexistent/ztools_img.png"))
            .is_none());
    }

    #[test]
    fn test_extract_text_none_when_command_fails() {
        let img = tempfile::NamedTempFile::new().unwrap();
        // /usr/bin/false exits nonzero: mirrors a tesseract failure on bad input.
        let engine = TesseractEngine::new(Some(PathBuf::from("/usr/bin/false")));
        assert!(engine.extract_text(img.path()).is_none());
    }

    #[test]
    fn test_extract_text_returns_stdout_on_success() {
        let img = tempfile::NamedTempFile::new().unwrap();
        // /bin/echo exits 0 with the argv on stdout: exercises the decode+trim
        // plumbing without an OCR engine.
        let engine = TesseractEngine::new(Some(PathBuf::from("/bin/echo")));
        let text = engine.extract_text(img.path()).unwrap();
        assert_eq!(
            text,
            format!("{} stdout --oem 1 -l eng", img.path().display())
        );
    }

    #[test]
    fn test_extract_text_none_when_stdout_empty() {
        let img = tempfile::NamedTempFile::new().unwrap();
        // Exit 0 but no output: OCR found nothing readable.
        let engine = TesseractEngine::new(Some(PathBuf::from("/usr/bin/true")));
        assert!(engine.extract_text(img.path()).is_none());
    }

    #[test]
    fn test_global_helpers_none_without_image() {
        assert!(extract_first_line(Path::new("/nonexistent/ztools_img.png")).is_none());
        assert!(extract_full_text(Path::new("/nonexistent/ztools_img.png")).is_none());
    }

    #[test]
    fn test_ocr_available_matches_default_engine_availability() {
        if Path::new(DEFAULT_TESSERACT_PATH).exists() {
            assert!(ocr_available());
        }
        // Either way the wrapper must agree with constructing the default engine.
        assert_eq!(ocr_available(), TesseractEngine::default().is_available());
    }
}
