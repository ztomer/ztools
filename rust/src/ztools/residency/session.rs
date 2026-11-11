//! A run that uses models one after another: each is held while it is used,
//! released before the next is loaded, and the last released at the end.
//!
//! For a caller that does not know up front which model it needs, or whether
//! it needs one at all (the renamer asks a text model or a vision model per
//! image). Loading is lazy, so a run that never asks loads nothing.

use anyhow::Result;

use super::{Held, Timing, load, same_model};

/// The models a run holds, one at a time.
pub struct Session {
    base_url: String,
    timing: Timing,
    held: Option<Held>,
}

impl Session {
    /// A session against `base_url` that has loaded nothing.
    #[must_use]
    pub fn new(base_url: &str, timing: Timing) -> Self {
        Self {
            base_url: base_url.to_string(),
            timing,
            held: None,
        }
    }

    /// Make `model` the held model: a no-op when it already is, otherwise the
    /// held one is released (and waited out) and `model` loaded.
    ///
    /// # Errors
    ///
    /// From the release of the previous model and from the load.
    pub fn hold(&mut self, model: &str) -> Result<()> {
        if self
            .held
            .as_ref()
            .is_some_and(|h| same_model(h.model(), model))
        {
            return Ok(());
        }
        if let Some(previous) = self.held.take() {
            previous.release()?;
        }
        self.held = Some(load(&self.base_url, model, self.timing)?);
        Ok(())
    }

    /// Release whatever is held. Dropping the session does the same, and
    /// can only print a failure.
    ///
    /// # Errors
    ///
    /// From the release.
    pub fn finish(mut self) -> Result<()> {
        self.held.take().map_or(Ok(()), Held::release)
    }
}
