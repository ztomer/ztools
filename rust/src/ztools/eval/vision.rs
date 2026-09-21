//! Did the model actually SEE the images, or is it describing nothing?
//!
//! Port of `eval/vision_fixtures.py` + `lib/validators/vision_validator.py`.
//! The failure this catches does not look like a failure: `rn`'s vision path
//! once sent images in a shape osaurus silently ignores, so the model answered
//! from the text alone with confident, well-formed, entirely invented
//! descriptions — "large brown dog" for a red circle — and a shape-only
//! validator scored it 100. The score here is keyword recall against KNOWN
//! image contents, and the fixtures are chosen so a blind model cannot pass.
//!
//! The fixtures are DATA (`conf/eval_vision.toml`): shape, geometry, colour and
//! the words that prove the model saw it. This module only rasterises them
//! (a flat-fill renderer is enough — the task measures sight, not
//! anti-aliasing) and grades answers against them.

use crate::units::signed;
use anyhow::{Context, Result};
use base64::Engine as _;
use serde::Deserialize;
use std::collections::HashSet;
use std::path::Path;

pub const MAX_SCORE: i64 = 100;

/// One synthetic image and the words that prove it was seen.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct VisionFixture {
    pub name: String,
    pub background: [u8; 3],
    pub shape: String,
    #[serde(default)]
    pub r#box: Option<[i64; 4]>,
    #[serde(default)]
    pub points: Option<Vec<[i64; 2]>>,
    pub colour: [u8; 3],
    pub accept: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionSpec {
    pub size: u32,
    #[serde(rename = "fixture")]
    pub fixtures: Vec<VisionFixture>,
}

/// Load the fixture spec.
///
/// # Errors
///
/// When the file is missing or does not parse: a vision task with no ground
/// truth would have to invent a score, which is the failure this exists for.
pub fn load_vision_spec(path: &Path) -> Result<VisionSpec> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("vision fixtures at {}", path.display()))?;
    let spec: VisionSpec = toml::from_str(&text).context("conf/eval_vision.toml parses")?;
    anyhow::ensure!(!spec.fixtures.is_empty(), "vision spec lists no fixtures");
    Ok(spec)
}

/// The shipped spec next to the checkout's `conf/`.
///
/// # Errors
///
/// As [`load_vision_spec`].
pub fn shipped_vision_spec() -> Result<VisionSpec> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(|p| p.join("conf/eval_vision.toml"))
        .context("checkout layout")?;
    load_vision_spec(&path)
}

/// Point-in-polygon by the even-odd rule, on pixel centres.
fn inside_polygon(points: &[[i64; 2]], x: f64, y: f64) -> bool {
    let mut inside = false;
    let n = points.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi, xj, yj) = (
            signed(points[i][0]),
            signed(points[i][1]),
            signed(points[j][0]),
            signed(points[j][1]),
        );
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Whether the pixel at `(x, y)` is inside the fixture's shape.
fn covers(fixture: &VisionFixture, x: u32, y: u32) -> bool {
    let (px, py) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
    match (fixture.shape.as_str(), fixture.r#box, &fixture.points) {
        ("rectangle", Some([x0, y0, x1, y1]), _) => {
            let (x0, y0, x1, y1) = (signed(x0), signed(y0), signed(x1), signed(y1));
            px >= x0 && px <= x1 && py >= y0 && py <= y1
        }
        ("ellipse", Some([x0, y0, x1, y1]), _) => {
            let (x0, y0, x1, y1) = (signed(x0), signed(y0), signed(x1), signed(y1));
            let (cx, cy) = (f64::midpoint(x0, x1), f64::midpoint(y0, y1));
            let (rx, ry) = ((x1 - x0) / 2.0, (y1 - y0) / 2.0);
            let (dx, dy) = ((px - cx) / rx, (py - cy) / ry);
            dx.mul_add(dx, dy * dy) <= 1.0
        }
        ("polygon", _, Some(points)) if points.len() >= 3 => inside_polygon(points, px, py),
        _ => false,
    }
}

/// Draw one fixture as PNG bytes. Deterministic: same spec, same pixels.
///
/// # Errors
///
/// Only if the PNG encoder rejects the buffer, which a fixed-size RGB canvas
/// cannot make it do.
pub fn render(fixture: &VisionFixture, size: u32) -> Result<Vec<u8>> {
    let mut rgb = Vec::with_capacity((size * size * 3) as usize);
    for y in 0..size {
        for x in 0..size {
            let c = if covers(fixture, x, y) {
                fixture.colour
            } else {
                fixture.background
            };
            rgb.extend_from_slice(&c);
        }
    }
    let mut out = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut out, size, size);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().context("png header")?;
        writer.write_image_data(&rgb).context("png data")?;
    }
    Ok(out)
}

/// The fixture as an `OpenAI` `image_url` payload: a `data:` URI.
///
/// # Errors
///
/// As [`render`].
pub fn data_uri(fixture: &VisionFixture, size: u32) -> Result<String> {
    let bytes = render(fixture, size)?;
    Ok(format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    ))
}

/// Every fixture image as a data URI, in spec order — all in ONE message on
/// purpose: one image is a coin flip a blind model passes one time in three.
///
/// # Errors
///
/// As [`render`].
pub fn fixture_images(spec: &VisionSpec) -> Result<Vec<String>> {
    spec.fixtures
        .iter()
        .map(|f| data_uri(f, spec.size))
        .collect()
}

fn words(text: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut cur = String::new();
    for ch in text.to_lowercase().chars() {
        if ch.is_ascii_lowercase() {
            cur.push(ch);
        } else if !cur.is_empty() {
            out.insert(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.insert(cur);
    }
    out
}

/// `(fixture name, was it recognised)` for each fixture, in order.
#[must_use]
pub fn matched_fixtures(text: &str, fixtures: &[VisionFixture]) -> Vec<(String, bool)> {
    let seen = words(text);
    fixtures
        .iter()
        .map(|f| {
            let hit = f.accept.iter().any(|w| seen.contains(&w.to_lowercase()));
            (f.name.clone(), hit)
        })
        .collect()
}

/// Score how many of the known images the description actually accounts for.
#[must_use]
pub fn validate_image_description(text: &str, fixtures: &[VisionFixture]) -> (i64, String) {
    if text.trim().is_empty() {
        return (0, "empty response".to_string());
    }
    if fixtures.is_empty() {
        // No ground truth means nothing can be judged; a score here would be
        // invented, which is the whole failure mode this file exists for.
        return (0, "no fixtures to check against".to_string());
    }
    let results = matched_fixtures(text, fixtures);
    let hits = results.iter().filter(|(_, ok)| *ok).count();
    let score = super::scoring_math::pct_round(hits, results.len());
    if hits == results.len() {
        return (score, String::new());
    }
    let missed: Vec<&str> = results
        .iter()
        .filter(|(_, ok)| !ok)
        .map(|(name, _)| name.as_str())
        .collect();
    let mut detail = format!(
        "described {hits}/{} images; missed {}",
        results.len(),
        missed.join(", ")
    );
    if hits == 0 {
        // The signature of a model that received no image at all, as opposed
        // to one that saw them and described them poorly.
        detail.push_str(" (no image content recognised at all -- is the payload reaching it?)");
    }
    (score, detail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> VisionSpec {
        shipped_vision_spec().expect("shipped spec loads")
    }

    #[test]
    fn shipped_spec_has_three_unrelated_fixtures() {
        let s = spec();
        assert_eq!(s.fixtures.len(), 3);
        for (i, a) in s.fixtures.iter().enumerate() {
            for b in &s.fixtures[i + 1..] {
                let wa: HashSet<&String> = a.accept.iter().collect();
                assert!(
                    b.accept.iter().all(|w| !wa.contains(w)),
                    "{} and {} share an accept word",
                    a.name,
                    b.name
                );
            }
        }
    }

    #[test]
    fn renders_a_valid_png_of_the_spec_size_with_the_shape_colour_in_the_middle() {
        let s = spec();
        for f in &s.fixtures {
            let bytes = render(f, s.size).unwrap();
            assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "{} is not a PNG", f.name);
            // Decode and probe: centre pixel is the shape, corner is background.
            let decoder = png::Decoder::new(std::io::Cursor::new(&bytes));
            let mut reader = decoder.read_info().unwrap();
            let mut buf = vec![0; reader.output_buffer_size().unwrap()];
            let info = reader.next_frame(&mut buf).unwrap();
            assert_eq!((info.width, info.height), (s.size, s.size));
            let px = |x: u32, y: u32| -> [u8; 3] {
                let i = ((y * s.size + x) * 3) as usize;
                [buf[i], buf[i + 1], buf[i + 2]]
            };
            assert_eq!(px(s.size / 2, s.size / 2), f.colour, "{} centre", f.name);
            assert_eq!(px(0, 0), f.background, "{} corner", f.name);
        }
    }

    #[test]
    fn triangle_covers_its_centroid_and_not_the_top_corners() {
        let s = spec();
        let tri = s.fixtures.iter().find(|f| f.shape == "polygon").unwrap();
        assert!(covers(tri, 256, 300));
        assert!(!covers(tri, 110, 120));
        assert!(!covers(tri, 400, 120));
    }

    #[test]
    fn data_uri_is_png_base64() {
        let s = spec();
        let uri = data_uri(&s.fixtures[0], s.size).unwrap();
        assert!(
            uri.starts_with("data:image/png;base64,iVBOR"),
            "{}",
            &uri[..40]
        );
        assert_eq!(fixture_images(&s).unwrap().len(), 3);
    }

    #[test]
    fn full_recall_scores_100_with_no_detail() {
        let s = spec();
        let (score, detail) = validate_image_description(
            "A red circle, a green triangle and a blue square.",
            &s.fixtures,
        );
        assert_eq!((score, detail.as_str()), (100, ""));
    }

    #[test]
    fn synonyms_count_as_sight() {
        let s = spec();
        let (score, _) =
            validate_image_description("a crimson dot, a green cone, and a navy box", &s.fixtures);
        assert_eq!(score, 100);
    }

    #[test]
    fn partial_and_blind_answers_are_named() {
        let s = spec();
        assert_eq!(
            validate_image_description("I see a red circle.", &s.fixtures),
            (
                33,
                "described 1/3 images; missed green_triangle, blue_square".to_string()
            )
        );
        let (score, detail) = validate_image_description("A large brown dog.", &s.fixtures);
        assert_eq!(score, 0);
        assert!(detail.contains("is the payload reaching it?"), "{detail}");
        assert_eq!(
            validate_image_description("   ", &s.fixtures),
            (0, "empty response".to_string())
        );
        assert_eq!(
            validate_image_description("anything", &[]),
            (0, "no fixtures to check against".to_string())
        );
    }
}
