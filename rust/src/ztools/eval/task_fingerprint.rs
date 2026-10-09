//! A task's IDENTITY: the fingerprint that says which task a score belongs to.
//!
//! `eval_history.json` and `conf/eval_signals.json` key every observation by
//! task NAME, and a name is not a task. On 2026-10-08 the `file_summary` rows
//! became repo-relative and started carrying each listed file's own contents
//! (M1), so the name `file_summary` in those two files has described two
//! different questions since then, and which one it describes depends only on
//! when the row was written. Averaging them contradicts the prompt the model
//! was actually sent: a trend line through both measures neither, and a median
//! over them is a number about nothing.
//!
//! [`task_fingerprint`] is the fix. It is a short digest of what a task ASKS
//! and how it is SCORED; every stored observation carries it, and every
//! aggregate averages only the observations whose digest is the current task's.
//!
//! WHAT THE FINGERPRINT COVERS — the task's identity and nothing else:
//!
//! * `name`, because the stores are keyed by it: two rows sharing a name are
//!   one row only when the task is the same, and the roster's `json` /
//!   `detailed_json` aliases prove the name alone was never identity (they are
//!   `weekend_transient` / `weekend_fixed` under a second name);
//! * every message verbatim, in order — role, content, and the `data:` image
//!   URIs that ride along. A task is the text it sends;
//! * every check, in order, with each variant's arguments. A threshold that
//!   moves is a different task, because the same answer then scores differently;
//! * `parse_json`, which decides whether the validator is handed parsed JSON or
//!   the raw text.
//!
//! WHAT IT DELIBERATELY DOES NOT COVER:
//!
//! * Anything read at SCORE time rather than at LOAD time: the taxes rubric
//!   snapshots, and `conf/eval_vision.toml`'s expected descriptions for
//!   `image_real`. Those are the validator's own data, not the task's, and a
//!   change there alters the grading without altering what was asked. (The
//!   task's IMAGES do count: the loader draws them from that same spec, so they
//!   are already part of `messages`.)
//! * The shape around the prompt: `max_tokens`, the `[timeouts]` table in
//!   `conf/config.toml`, temperature, the transport, the model. Those change a
//!   measurement without changing the question, which is why they are recorded
//!   beside a score rather than folded into its identity.
//! * Time, machine, process, working directory, iteration order. The digest is
//!   taken over a serialisation whose field order is the declaration order
//!   below and whose sequences are vectors, so it is a pure function of the task
//!   and identical on every machine and in every run. std's `DefaultHasher` is
//!   `SipHash` over constants rustc makes no promise about across releases; a
//!   digest that could move when the toolchain moves would silently supersede
//!   every stored entry on the next `rustup update`.
//!
//! A CONFIG CHANGE THAT ALTERS A PROMPT COUNTS, through the prompt. The roster
//! renders `conf/eval_inputs.toml [test_inputs].filename` and the file-summary
//! rows out of the checkout at load time, so their bytes ARE `messages` and any
//! edit moves the digest. The consequence is deliberate and worth stating
//! plainly: a task whose prompt embeds file contents restarts its history
//! whenever those contents change, because a score describes the contents it
//! was taken against — which is exactly the pre-2026-10-08 defect, seen from
//! the other side.

use std::collections::BTreeMap;
use std::sync::{PoisonError, RwLock};

use serde::Serialize;

use super::task_loader::{ChatMessage, EvalTask};

/// The digest's payload, in the order the digest is taken over it.
///
/// DECLARED rather than derived from `EvalTask` on purpose: a derived
/// `Serialize` would follow whatever field order that struct happens to
/// declare, so adding or reordering a field there would silently re-fingerprint
/// every task in the store. Field order here is the wire format of a task's
/// identity, and `version` means a deliberate payload change moves every
/// digest — a clean break, not a correlated one.
#[derive(Serialize)]
struct Identity<'a> {
    version: u8,
    name: &'a str,
    messages: Vec<MessageIdentity<'a>>,
    checks: &'a [super::task_loader::Check],
    parse_json: bool,
}

/// One message's identity. Deliberately NOT `ChatMessage`'s own `Serialize`:
/// that one is the WIRE form, where images are folded into content parts, and
/// a transport change to the wire shape would then move a task's identity
/// without the task changing.
#[derive(Serialize)]
struct MessageIdentity<'a> {
    role: &'a str,
    content: &'a str,
    images: &'a [String],
}

impl<'a> MessageIdentity<'a> {
    fn of(message: &'a ChatMessage) -> Self {
        Self {
            role: &message.role,
            content: &message.content,
            images: &message.images,
        }
    }
}

/// FNV-1a, 64 bits. Four lines, no dependency, and fixed: the alternative in
/// std is keyed by constants that are explicitly not promised across releases.
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[must_use]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= u64::from(bytes[i]);
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}

/// The identity of `task`, as 16 hex chars of FNV-1a over the canonical JSON of
/// [`Identity`].
///
/// Stable across runs and machines, and different for any task that asks or
/// scores differently. The module comment states what that covers and what it
/// deliberately does not.
///
/// # Panics
///
/// Never in practice: the payload is strings and vectors. It is written as an
/// `expect` rather than an `unwrap_or_default` so that a failure to serialise
/// cannot silently file every task under the digest of an EMPTY payload — one
/// shared fingerprint, which would make every task in the store look current.
#[must_use]
pub fn task_fingerprint(task: &EvalTask) -> String {
    let identity = Identity {
        version: 1,
        name: &task.name,
        messages: task.messages.iter().map(MessageIdentity::of).collect(),
        checks: &task.checks,
        parse_json: task.parse_json,
    };
    let payload = serde_json::to_vec(&identity)
        .expect("a task identity is strings and vectors, which always serialise");
    format!("{:016x}", fnv1a(&payload))
}

/// Task name → the fingerprint of the task this process is running under that
/// name.
pub type TaskIdentities = BTreeMap<String, String>;

/// The fingerprints of the tasks THIS process loaded.
///
/// The writers reach the stores by task NAME only: `save_historical_results`
/// takes a `ModelRun` of `TaskOutcome`s, which carry a name, and the runner
/// calls `record_signal(signals, model, &task.name, ...)`. Neither signature
/// can be widened from here — both are called from the CLI dispatch — so at the
/// moment a row is written the digest has to be resolvable from the name. The
/// loader therefore registers every task it hands out, and the writers look the
/// name up.
///
/// Registration is ADDITIVE and idempotent for one roster: loading the same
/// roster twice writes the same values, and two rosters in one process let the
/// later one win per name. A name nobody registered yields `None` — UNKNOWN,
/// never assumed current — and an aggregate that cannot resolve a task's
/// current identity sets that task's rows aside rather than averaging them.
static IDENTITIES: std::sync::OnceLock<RwLock<TaskIdentities>> = std::sync::OnceLock::new();

fn identities() -> &'static RwLock<TaskIdentities> {
    IDENTITIES.get_or_init(|| RwLock::new(TaskIdentities::new()))
}

/// Register the tasks this process is about to run, so every row written for
/// them carries their digest. Called by the loader; a task nobody registers
/// writes rows no aggregate can average.
pub fn remember_current_tasks(tasks: &[EvalTask]) {
    // A poisoned lock is recovered rather than propagated: the registry holds
    // plain strings and a panic elsewhere must not turn every later row into
    // an unfingerprintable one.
    let mut current = identities().write().unwrap_or_else(PoisonError::into_inner);
    for task in tasks {
        current.insert(task.name.clone(), task_fingerprint(task));
    }
}

/// The fingerprint of the task `task_name` names in this process, if any.
#[must_use]
pub fn current_task_fingerprint(task_name: &str) -> Option<String> {
    identities()
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .get(task_name)
        .cloned()
}

/// Every fingerprint this process knows, for the aggregates that read them.
#[must_use]
pub fn current_task_identities() -> TaskIdentities {
    identities()
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// How one stored observation relates to the task as it stands now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Standing {
    /// Taken against the task as it stands now: it may be averaged.
    Current,
    /// Taken against a DIFFERENT task under the same name — the task was
    /// replaced between the two runs.
    Superseded,
    /// Nothing proves it describes the task as it is now: either the entry
    /// carries no fingerprint (every entry on disk before 2026-10-08, and any
    /// written by a process that could not resolve one) or this process has no
    /// current fingerprint for the name. UNKNOWN is never read as CURRENT.
    Unrecorded,
}

impl Standing {
    /// Whether an aggregate may average an observation with this standing.
    #[must_use]
    pub const fn counts(self) -> bool {
        matches!(self, Self::Current)
    }
}

/// The standing of a stored fingerprint against the current identities.
#[must_use]
pub fn standing_of(stored: Option<&str>, current: &TaskIdentities, task: &str) -> Standing {
    match (stored, current.get(task)) {
        (Some(stored), Some(now)) if stored == now => Standing::Current,
        (Some(_), Some(_)) => Standing::Superseded,
        // Absent fingerprint, or a task this process does not know: both are
        // "cannot prove this is the same task", which is the same verdict.
        (None, _) | (Some(_), None) => Standing::Unrecorded,
    }
}

#[cfg(test)]
#[path = "task_fingerprint_tests.rs"]
mod tests;
