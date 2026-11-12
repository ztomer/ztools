//! Where one chain candidate is asked, and the release after it.

use anyhow::Result;

use super::{chain, repair};

/// Where one candidate is asked: the on-device model by `fm` when the policy
/// routes it there, every other model by the server.
pub(super) struct Route<'a> {
    pub(super) policy: &'a chain::FallbackPolicy,
    pub(super) base_url: &'a str,
    pub(super) timeout_secs: u64,
    pub(super) config: &'a crate::config::ZtoolsConfig,
    pub(super) timing: crate::ztools::residency::Timing,
}

impl Route<'_> {
    /// The candidate's raw answer, whether it was cut at the length limit,
    /// and the release after it: a model that outstays a FINISHED answer
    /// keeps the answer (ZT.CHAINRELEASE).
    pub(super) fn ask(
        &self,
        candidate: &str,
        prompt: &str,
        processed: usize,
    ) -> Result<(String, bool, crate::ztools::residency::Released)> {
        if let Some(fm) = self.policy.on_device_for(candidate) {
            eprintln!("· Summarizing {processed} tweets with {candidate} on this Mac...");
            let raw = crate::ztools::on_device::respond(fm, prompt, self.timeout_secs)?;
            return Ok((raw, false, crate::ztools::residency::Released::clean()));
        }
        eprintln!(
            "· Summarizing {processed} tweets with {candidate} on {}...",
            self.base_url
        );
        // Loaded explicitly, released on every exit path (ZT.OSAURUSLOAD).
        let ran = crate::ztools::residency::with_model_kept(
            self.base_url,
            candidate,
            self.timing,
            |_| {
                repair::ask(
                    self.base_url,
                    candidate,
                    prompt,
                    self.timeout_secs,
                    self.config,
                )
            },
        )?;
        match ran.out {
            Ok((raw, cut)) => Ok((raw, cut, ran.released)),
            Err(e) => {
                ran.released
                    .report(&format!("{candidate}: the call's own error is reported"));
                Err(e)
            }
        }
    }
}
