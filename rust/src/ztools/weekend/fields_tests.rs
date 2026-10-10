//! W2: a field holds a value of its own kind, copied from the source, or is
//! unknown. The 2026-10-10 rows these come from are verbatim.

use super::*;

/// A price is an amount or a word that means one; "Unlimited" (lifted from
/// "Unlimited midway rides") is not.
#[test]
fn a_price_must_read_as_a_price() {
    for (raw, kept) in [
        ("Free", "Free"),
        ("FREE ENTRY & $5 TEQUILA", "FREE ENTRY & $5 TEQUILA"),
        ("$18-24", "$18-24"),
        ("By donation", "By donation"),
        ("12 CAD", "12 CAD"),
        ("Unlimited", "unknown"),
        ("Unknown", "unknown"),
        ("", "unknown"),
        ("Family-friendly", "unknown"),
    ] {
        assert_eq!(price_value(raw), kept, "{raw:?}");
    }
}

/// A duration is a time or a length; "Unlimited" and a description pushed
/// into the column are not.
#[test]
fn a_duration_must_read_as_a_duration() {
    for (raw, kept) in [
        ("from 11:00 AM onwards", "from 11:00 AM onwards"),
        ("11:00 am to 5:00 pm", "11:00 am to 5:00 pm"),
        ("10:00 AM", "10:00 AM"),
        ("2-3 hours", "2-3 hours"),
        ("All day", "All day"),
        ("Unlimited", ""),
        ("unknown", ""),
        ("Saturday Scent Social with candle or perfume crafting.", ""),
    ] {
        assert_eq!(duration_value(raw), kept, "{raw:?}");
    }
}

fn priced(name: &str, price: &str) -> WeekendEvent {
    WeekendEvent {
        name: name.into(),
        location: "Toronto".into(),
        price: price.into(),
        target_ages: String::new(),
        day: String::new(),
        dates: String::new(),
        description: String::new(),
        is_transient: true,
        score: 0.0,
        start_date: String::new(),
        end_date: String::new(),
        weather: String::new(),
        duration: String::new(),
        for_families: String::new(),
    }
}

/// A dollar amount must be one the sources state. "$10,000+" for Pumpkins
/// After Dark came from "10,000+ glowing pumpkins": the number is in the
/// corpus, the price is not.
#[test]
fn a_dollar_amount_the_corpus_never_states_is_cleared() {
    let corpus = "Pumpkins After Dark: 10,000+ glowing pumpkins. \
                  BARBIE NIGHT @ FICTION | FRI OCT 9 | FREE ENTRY & $5 TEQUILA";
    let rows = vec![
        priced("Pumpkins After Dark", "$10,000+"),
        priced("BARBIE NIGHT @ FICTION", "FREE ENTRY & $5 TEQUILA"),
        priced("Fall Fair", "Free"),
    ];
    let (rows, notes) = clear_unsourced_prices(rows, corpus);
    let prices: Vec<&str> = rows.iter().map(|r| r.price.as_str()).collect();
    assert_eq!(prices, vec!["unknown", "FREE ENTRY & $5 TEQUILA", "Free"]);
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(notes[0].contains("$10,000+"), "{notes:?}");
    // No corpus, nothing to judge against: nothing changes.
    let (rows, notes) = clear_unsourced_prices(rows, "");
    assert_eq!(rows.len(), 3);
    assert_empty!(notes);
}

/// The model's answer goes through both shape checks at parse.
#[test]
fn parse_keeps_only_values_of_the_fields_kind() {
    let content = serde_json::json!({"transient_events": [
        {"name": "Pumpkinfest", "price": "Unlimited", "duration": "Unlimited"}
    ]})
    .to_string();
    let resp = serde_json::json!({"choices": [{"message": {"content": content}}]});
    let events = crate::ztools::weekend::parse_llm_events(&resp).unwrap();
    assert_eq!(events[0].price, "unknown");
    assert_eq!(events[0].duration, "");
}
