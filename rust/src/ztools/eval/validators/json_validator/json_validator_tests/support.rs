//! Shared fixtures for json_validator's test modules.

use serde_json::{json, Value};

pub(super) fn detailed_items() -> Vec<Value> {
    [
        ("Kappa Zeta", "12 Alpha St", "$5"),
        ("Lambda Mu", "34 Beta Ave", "$6"),
        ("Nu Xi", "56 Gamma Rd", "$7"),
        ("Omicron Pi", "78 Delta St", "$8"),
        ("Rho Sigma", "90 Epsilon Ave", "$9"),
        ("Tau Upsilon", "24 Hotel St", "$10"),
        ("Phi Chi", "36 India Ave", "$11"),
        ("Psi Omega", "48 Juliet Rd", "$12"),
    ]
    .iter()
    .map(|(name, loc, price)| json!({"name": name, "location": loc, "price": price}))
    .collect()
}

// Marker-pair sources over detailed_items(). Every item carries two
// distinctive name tokens plus neutral location/price text, so listing
// marker pairs in the source controls exactly how many items ground:
// FULL grounds 8/8 (some via the primary-name containment fallback),
// MED grounds 5/8 = 0.625, LOW grounds 2/8 = 0.25, NONE grounds 0.
pub(super) const SRC_FULL: &str =
    "kappa zeta lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega guide";
pub(super) const SRC_MED: &str = "kappa zeta lambda mu nu xi omicron pi rho sigma guide";
pub(super) const SRC_LOW: &str = "kappa zeta lambda mu quiet river stones";
pub(super) const SRC_NONE: &str = "quiet river stones only";
