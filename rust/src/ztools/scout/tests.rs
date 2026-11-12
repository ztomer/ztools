//! The scout on feeds recorded 2026-11-12 with plain unauthenticated GETs
//! (`ztools models scout --record tests/fixtures/scout`), and the same fixture
//! with one feed made to refuse.

use super::*;

const FEEDS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tests/fixtures/scout/feeds.jsonl"
));
const LOCAL: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tests/fixtures/scout/local.json"
));
const CRACK: &str = "dealignai/Qwen3.8-27B-JANG_6D-CRACK";
const RAPTOR5: &str = "OsaurusAI/Raptor-v0.5-8B-A1B-JANG_6M";
const RAPTOR6: &str = "OsaurusAI/Raptor-0.6-4B-JANG_6M";
const GIB: u64 = 1 << 30;

fn fixture() -> Fixture {
    Fixture::parse(FEEDS).expect("recorded feeds")
}

fn local() -> Local {
    serde_json::from_str(LOCAL).expect("recorded local facts")
}

fn incumbent<'a>(p: &'a Proposal, slot: &str) -> &'a Incumbent {
    p.incumbents
        .iter()
        .find(|i| i.slots.iter().any(|s| s == slot))
        .expect("slot has an incumbent")
}

fn cand<'a>(p: &'a Proposal, repo: &str) -> &'a Candidate {
    p.incumbents
        .iter()
        .flat_map(|i| &i.candidates)
        .find(|c| c.repo == repo)
        .unwrap_or_else(|| panic!("{repo} is not nominated"))
}

fn failed(c: Option<&Check>) -> bool {
    matches!(c, Some(Check::Fail(_)))
}

#[test]
fn the_6d_crack_is_the_think_incumbent_and_its_dealignai_siblings_carry_fit_and_disk() {
    let p = scout(&fixture(), &local());
    let think = incumbent(&p, "think");
    assert_eq!(think.repo.as_deref(), Some(CRACK));
    assert_eq!(think.family.as_deref(), Ok("Qwen/Qwen3.8-27B"));
    assert_eq!(think.slots, ["filename", "think", "vlm"]);
    let siblings: Vec<&Candidate> = think
        .candidates
        .iter()
        .filter(|c| c.repo.starts_with("dealignai/Qwen3.8-27B-"))
        .collect();
    let names: Vec<&str> = siblings.iter().map(|c| c.repo.as_str()).collect();
    assert_eq!(
        names,
        [
            "dealignai/Qwen3.8-27B-JANG_2D-CRACK",
            "dealignai/Qwen3.8-27B-JANG_4D-CRACK",
            "dealignai/Qwen3.8-27B-MXFP8-CRACK",
            "dealignai/Qwen3.8-27B-UNCENSORED-GGUF",
        ]
    );
    for c in &siblings[..3] {
        assert!(c.passes(), "{} should pass: {c:?}", c.repo);
        assert!(
            c.fit
                .as_ref()
                .is_some_and(|f| f.text().contains("GiB vs 40.8 GiB"))
        );
        assert!(
            c.disk
                .as_ref()
                .is_some_and(|d| d.text().contains("of 79.1 GiB free"))
        );
    }
    assert!(
        !think.candidates.iter().any(|c| c.repo == CRACK),
        "the incumbent is not its own candidate"
    );
}

#[test]
fn raptor_0_6_is_flagged_against_the_retired_raptor_v0_5_and_is_the_one_it_would_stage() {
    let p = scout(&fixture(), &local());
    let json = incumbent(&p, "json");
    assert_eq!(json.repo.as_deref(), Some(RAPTOR5));
    assert!(json.retired);
    let first = &json.candidates[0];
    assert_eq!(first.repo, RAPTOR6);
    assert_eq!(first.nominated, format!("successor of {RAPTOR5}"));
    assert!(first.passes());
    assert_eq!(p.stage.as_deref(), Some(RAPTOR6));
    assert!(!incumbent(&p, "think").retired);
    let lines = render(&p).join("\n");
    assert!(lines.contains(&format!("⚠ {RAPTOR5} is retired from the Osaurus catalog")));
    assert!(lines.contains("✓ feeds: 85 of 85 answered"), "{lines}");
}

#[test]
fn a_mirror_of_an_allowlisted_repo_is_dropped_by_provenance() {
    let p = scout(&fixture(), &local());
    let mirror = cand(&p, "Openintelligent123/Qwen3.8-27B-JANG_6D-CRACK");
    assert!(mirror.provenance.text().starts_with("mirror"), "{mirror:?}");
    assert!(mirror.format.is_none() && !mirror.passes());
}

#[test]
fn formats_osaurus_does_not_load_are_dropped_before_any_fit() {
    let p = scout(&fixture(), &local());
    for (repo, why) in [
        ("dealignai/Qwen3.8-27B-UNCENSORED-GGUF", "GGUF"),
        ("mlx-community/Qwen3.8-27B-nvfp4", "nvfp4 quantisation"),
        (
            "mlx-community/Qwen3.8-27B-MTP-4bit",
            "model_type qwen3_5_mtp",
        ),
    ] {
        let c = cand(&p, repo);
        assert!(failed(c.format.as_ref()), "{repo}: {c:?}");
        assert!(
            c.format.as_ref().is_some_and(|f| f.text().starts_with(why)),
            "{repo}: {c:?}"
        );
        assert!(c.fit.is_none());
    }
    let jang = cand(&p, "dealignai/Qwen3.8-27B-MXFP8-CRACK");
    assert_eq!(
        jang.format.as_ref().map(Check::text),
        Some("JANG mxfp8 qwen3_5")
    );
}

#[test]
fn a_model_over_the_memory_line_is_rejected_and_a_smaller_budget_rejects_more() {
    let p = scout(&fixture(), &local());
    let bf16 = cand(&p, "mlx-community/Qwen3.8-27B-bf16");
    assert!(failed(bf16.fit.as_ref()), "{bf16:?}");
    let mut f = fixture();
    f.set(
        feeds::HEALTH,
        serde_json::json!({"gpu_budget_bytes": 24 * GIB}),
    );
    let p = scout(&f, &local());
    assert!(failed(
        cand(&p, "dealignai/Qwen3.8-27B-JANG_4D-CRACK").fit.as_ref()
    ));
    assert!(cand(&p, "dealignai/Qwen3.8-27B-JANG_2D-CRACK").passes());
}

#[test]
fn at_the_designs_35_gib_free_a_26_gib_sibling_cannot_be_staged() {
    let mut l = local();
    l.free_bytes = Some(35 * GIB);
    let p = scout(&fixture(), &l);
    assert!(failed(
        cand(&p, "dealignai/Qwen3.8-27B-MXFP8-CRACK").disk.as_ref()
    ));
    assert!(cand(&p, "dealignai/Qwen3.8-27B-JANG_2D-CRACK").passes());
    l.staged = 1;
    let p = scout(&fixture(), &l);
    assert!(failed(cand(&p, RAPTOR6).disk.as_ref()));
    assert_eq!(p.stage, None, "one staged candidate at most");
}

#[test]
fn a_refusing_feed_reads_unverified_with_its_denominator() {
    let mut f = fixture();
    let sibling = "dealignai/Qwen3.8-27B-JANG_4D-CRACK";
    f.refuse(&feeds::tree_key(sibling), "HTTP 429 Too Many Requests");
    let p = scout(&f, &local());
    let c = cand(&p, sibling);
    assert!(matches!(c.format, Some(Check::Unverified(_))), "{c:?}");
    assert!(!c.passes());
    assert_eq!(p.refused.len(), 1);
    let lines = render(&p).join("\n");
    assert!(
        lines.contains("⚠ feeds: 83 of 84 answered; 1 unverified"),
        "{lines}"
    );
    assert!(lines.contains("HTTP 429"));

    let mut f = fixture();
    f.refuse(feeds::HEALTH, "no answer");
    let p = scout(&f, &local());
    assert!(matches!(cand(&p, RAPTOR6).fit, Some(Check::Unverified(_))));
    assert_eq!(p.stage, None, "an unverified fit stages nothing");

    let mut f = fixture();
    f.refuse(&feeds::model_key(CRACK), "HTTP 503");
    let p = scout(&f, &local());
    let think = incumbent(&p, "think");
    assert!(think.family.is_err() && think.candidates.is_empty());
    assert!(
        render(&p)
            .join("\n")
            .contains("family unverified: HTTP 503")
    );
}

#[test]
fn an_unread_catalog_says_so_and_judges_types_unverified() {
    let mut f = fixture();
    f.refuse(&feeds::catalog_key("1.0.1"), "HTTP 404");
    let p = scout(&f, &local());
    assert!(
        p.notes
            .iter()
            .any(|n| n.contains("catalog unverified (HTTP 404)"))
    );
    assert!(!incumbent(&p, "json").retired);
    assert!(
        incumbent(&p, "json")
            .candidates
            .iter()
            .all(|c| !c.nominated.starts_with("successor"))
    );
    let mtp = cand(&p, "mlx-community/Qwen3.8-27B-MTP-4bit");
    assert!(matches!(mtp.format, Some(Check::Unverified(_))), "{mtp:?}");
    let mut l = local();
    l.osaurus_version = None;
    let p = scout(&fixture(), &l);
    assert!(
        p.notes
            .iter()
            .any(|n| n.contains("Osaurus version unknown"))
    );
}

#[test]
fn a_slot_naming_an_uninstalled_model_asks_nothing_for_it() {
    let mut l = local();
    l.slots.insert("think".into(), "nope-7b".into());
    let p = scout(&fixture(), &l);
    let think = incumbent(&p, "think");
    assert!(think.repo.is_none());
    assert!(
        render(&p)
            .iter()
            .any(|x| x == "✗ think: nope-7b is not installed (models refcheck)")
    );
}

#[test]
fn every_recorded_body_is_already_reduced_so_the_fixture_is_what_live_reads() {
    let text = FEEDS;
    let mut n = 0;
    for line in text.lines() {
        let v: Value = serde_json::from_str(line).expect("json line");
        let key = v["key"].as_str().expect("key");
        if let Some(body) = v.get("body") {
            assert_eq!(&feeds::reduce(key, body), body, "{key} is not reduced");
            n += 1;
        }
    }
    assert_eq!(n, 85);
    assert!(!text.contains("\"oid\"") && !text.contains("\"_id\""));
}
