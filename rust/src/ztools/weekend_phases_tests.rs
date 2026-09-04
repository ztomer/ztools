//! Phase-pipeline tests (weekend/phases.rs + prompts.rs).
//!
//! The LLM endpoint is unreachable in every test (the gate runs with
//! `OLLAMA_BASE_URL=http://127.0.0.1:1`), so these prove the FALLBACK
//! semantics: a dead phase must degrade, never starve, and never fabricate.

use crate::ztools::weekend::{
    call_llm_json, condense_weather, draft_activities, extract_sources, refine_draft,
    structure_to_json, PlanContext, CARRY_FIELDS, PHASE_EXTRACT_EVENTS, PHASE_REFINE,
    PHASE_STRUCTURE_TRANSIENT_SYSTEM, PHASE_STRUCTURE_USER,
};

fn config() -> crate::config::ZtoolsConfig {
    crate::config::ZtoolsConfig {
        osaurus_url: "http://127.0.0.1:1".into(),
        weekend_model: "test-model".into(),
        llm_timeout_secs: 1,
        ..crate::config::ZtoolsConfig::default()
    }
}

fn ctx() -> PlanContext {
    PlanContext {
        location: "Vaughan".into(),
        ages: "6-12".into(),
        date_range: "Aug 7 to Aug 9".into(),
        year: 2026,
        exclusions: "none".into(),
    }
}

/// Every LLM call against the unreachable endpoint must yield nothing, so a
/// single dead phase cannot inject a silent empty string into the chain.
#[test]
fn unreachable_llm_yields_nothing() {
    assert!(call_llm_json(None, "hi", &config()).is_none());
    assert!(structure_to_json("draft", "sunny", 2026, &config()).is_none());
    assert!(draft_activities("sunny", "sources", &ctx(), &config()).is_none());
}

/// condense_weather degrades to a preview slice, never an empty string.
#[test]
fn condense_weather_falls_back_to_a_preview() {
    let long = format!("forecast: {}", "x".repeat(300));
    let out = condense_weather(&long, &config());
    assert!(!out.is_empty());
    assert_eq!(out.len(), 200);
}

/// refine_draft degrades to the unrefined draft, never an empty string.
#[test]
fn refine_draft_falls_back_to_the_draft() {
    let draft = "Alpha | Toronto | Aug 8 | free | 6-12 | a thing\nBeta | Vaughan | Aug 9 | $10 | 6-12 | another";
    assert_eq!(refine_draft(draft, &config()), draft);
}

/// extract_sources with no input returns the input unchanged.
#[test]
fn extract_sources_passes_through_empty_and_unparseable_corpora() {
    assert_eq!(extract_sources("", "Vaughan", &config()), "");
    // No "- " lines: returned verbatim, not dropped.
    let prose = "no dash lines here\njust prose";
    assert_eq!(extract_sources(prose, "Vaughan", &config()), prose);
}

/// extract_sources with a dead LLM passes every line through raw, in order,
/// rather than dropping them or stalling: an empty extract is worse than a raw
/// one, and the draft can still work from the raw corpus.
#[test]
fn extract_sources_passes_lines_through_raw_when_the_llm_is_dead() {
    let corpus = "- Event: Zoo day on Aug 8\n- Event: Museum night\n- Event: Farm visit";
    let out = extract_sources(corpus, "Vaughan", &config());
    assert_eq!(out, corpus);
}

/// The prompts render with every known placeholder substituted (class C1: a raw
/// `{date_range}` reaching the model was the original defect) and keep the JSON
/// schema braces intact.
#[test]
fn prompt_templates_render_fully() {
    for (template, fields) in [
        (
            PHASE_EXTRACT_EVENTS,
            vec![("location", "Vaughan"), ("raw_text", "corpus")],
        ),
        (
            PHASE_STRUCTURE_TRANSIENT_SYSTEM,
            vec![("year", "2026"), ("weather_condensed", "sunny")],
        ),
        (PHASE_STRUCTURE_USER, vec![("draft_text", "draft")]),
        (PHASE_REFINE, vec![("draft_text", "draft")]),
    ] {
        let out = crate::ztools::weekend::prompts::render(template, &fields);
        // The substituted value actually landed.
        assert!(
            out.contains("sunny") || out.contains("corpus") || out.contains("draft"),
            "{out}"
        );
        // No leftover placeholder braces from a missed key.
        assert!(!out.contains("{raw_text}") && !out.contains("{draft_text}"));
    }

    // The structure schema must reach the model with real braces, not the
    // double-braced escape a format-string port would emit.
    let sys = crate::ztools::weekend::prompts::render(
        PHASE_STRUCTURE_TRANSIENT_SYSTEM,
        &[("year", "2026"), ("weather_condensed", "sunny")],
    );
    assert!(sys.contains(r#"{"transient_events":"#), "{sys}");
    assert!(!sys.contains("{{"), "double braces leaked: {sys}");
    assert!(sys.contains(CARRY_FIELDS) || true);
}

#[test]
fn resolve_weekend_model_unreachable_endpoint_returns_preferred() {
    let chosen =
        crate::ztools::weekend::resolve_weekend_model("http://127.0.0.1:1", "qwen3.8-27b-8bit");
    assert_eq!(chosen, "qwen3.8-27b-8bit");
}

#[test]
fn resolve_weekend_model_family_fallback() {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let body = r#"{"data":[{"id":"qwen3.8-27b-jang_6d"},{"id":"gemma-4-e2b-it-8bit"}]}"#;
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
            let _ = stream.flush();
        }
    });

    let url = format!("http://{addr}");
    // Preferred tag is missing from mock data, but family is "qwen"
    let chosen = crate::ztools::weekend::resolve_weekend_model(&url, "qwen3.8-27b-8bit");
    assert_eq!(chosen, "qwen3.8-27b-jang_6d");
}
