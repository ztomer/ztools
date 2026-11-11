//! Per-model similarity thresholds, each measured, never assumed.
//!
//! `potion-base-4M` (ZT.EMBEDMODEL, 2026-11-11, Osaurus 1.0.1). Sample: 243
//! tweets quoted in the 19 accepted summaries under `~/Documents/twitter_summaries`
//! (no new fetch). Among the 236 Latin-script ones (27,730 pairs) the cosine
//! median is 0.15 and p99.9 is 0.59; the only four pairs at or above 0.64 are
//! each one story told by two accounts (0.78 "AI creates the illusion of
//! knowledge", 0.67 Dave Chappelle at MSG, 0.66 the Chinese-spy story, 0.65
//! firmware locks). The next pairs, 0.625 and 0.623, are related but not the
//! same event, and below 0.60 they are unrelated. The old 0.85 merged none of
//! the Latin pairs: it was calibrated for no model this server serves.
//!
//! The same sample showed the model cannot read Hebrew: its English vocabulary
//! maps every Hebrew tweet to nearly one vector (unrelated Hebrew tweets at
//! cosine 0.96-0.99), so at any threshold they would all merge. Text whose
//! letters are mostly non-Latin is therefore kept out of the comparison.
//!
//! `tests/fixtures/embeddings/potion_base_4m.json` holds a committed
//! sample with its recorded vectors; the calibration test clusters it at this
//! threshold and fails at 0.0, at 1.0 and at the old 0.85. To regenerate it,
//! POST its texts to `/v1/embeddings` with model `potion-base-4M` (under the
//! GPU lock) and record each vector rounded to six decimals.

/// One embedding model's measured clustering parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    pub model: &'static str,
    pub threshold: f32,
    /// The model's vocabulary is Latin-script only.
    pub latin_only: bool,
}

const CALIBRATED: &[Calibration] = &[Calibration {
    model: "potion-base-4M",
    threshold: 0.64,
    latin_only: true,
}];

/// The calibration for `model`, matched without an organisation prefix
/// (`minishlab/potion-base-4M` is `potion-base-4M`) and without case.
#[must_use]
pub fn calibration_for(model: &str) -> Option<Calibration> {
    let bare = model.rsplit('/').next().unwrap_or(model);
    CALIBRATED
        .iter()
        .find(|c| c.model.eq_ignore_ascii_case(bare))
        .copied()
}

impl Calibration {
    /// Whether the model can represent `text`: always, unless it is
    /// Latin-only and fewer than half of the text's letters are ASCII.
    #[must_use]
    pub fn can_read(&self, text: &str) -> bool {
        if !self.latin_only {
            return true;
        }
        let (latin, letters) = text
            .chars()
            .filter(|c| c.is_alphabetic())
            .fold((0usize, 0usize), |(l, n), c| {
                (l + usize::from(c.is_ascii()), n + 1)
            });
        letters > 0 && latin * 2 >= letters
    }
}
