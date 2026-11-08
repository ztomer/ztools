use super::*;

fn terms() -> Vec<String> {
    vec!["Following".to_string(), "Siguiendo".to_string()]
}

fn page(tabs: &[(&str, bool)]) -> TabState {
    TabState {
        url: "https://x.com/home".to_string(),
        title: "Home / X".to_string(),
        primary_column: true,
        tablist: !tabs.is_empty(),
        tabs: tabs
            .iter()
            .map(|(text, selected)| Tab {
                text: (*text).to_string(),
                selected: *selected,
            })
            .collect(),
    }
}

/// The class this module exists for: For You selected is NOT a switch.
/// (The old wait accepted "any tab selected" and so passed here.)
#[test]
fn for_you_selected_is_not_the_following_feed() {
    let p = page(&[("For you", true), ("Following", false)]);
    assert!(!p.following_selected(&terms()));
}

#[test]
fn following_selected_is_verified() {
    let p = page(&[("For you", false), ("Following", true)]);
    assert!(p.following_selected(&terms()));
}

#[test]
fn no_tabs_is_not_verified_and_says_so() {
    let p = page(&[]);
    assert!(!p.following_selected(&terms()));
    let d = p.describe(&terms());
    assert!(d.contains("no [role=tab] element"), "{d}");
    assert!(d.contains("tablist absent"), "{d}");
    assert!(d.contains("selected: no tab"), "{d}");
}

#[test]
fn a_second_tab_that_is_not_following_never_counts() {
    // A localized or renamed bar: the old positional fallback clicked tabs[1].
    let p = page(&[("For you", false), ("Live", true)]);
    assert!(!p.following_selected(&terms()));
    assert!(p.describe(&terms()).contains("matched a tab"));
}

#[test]
fn describe_names_the_selected_tab_and_the_match() {
    let p = page(&[("For you", true), ("Following", false)]);
    let d = p.describe(&terms());
    assert!(d.contains("selected: \"For you\""), "{d}");
    assert!(d.contains("\"Following\" matched"), "{d}");
    assert!(d.contains("https://x.com/home"), "{d}");
}

#[test]
fn localized_term_matches_in_term_order() {
    let p = page(&[("Para ti", false), ("Siguiendo", true)]);
    assert_eq!(p.following_index(&terms()), Some(1));
    assert!(p.following_selected(&terms()));
}

#[test]
fn unreadable_state_parses_to_an_empty_page() {
    let p = TabState::parse("not json");
    assert_eq!(p, TabState::default());
    let p = TabState::parse(
        r#"{"url":"u","title":"t","primary_column":true,"tablist":true,
            "tabs":[{"text":"Following","selected":true}]}"#,
    );
    assert!(p.following_selected(&terms()));
}

#[test]
fn click_js_carries_the_terms_and_no_positional_fallback() {
    let js = click_js(&terms());
    assert!(js.contains("\"Following\""));
    assert!(!js.contains("tabs[1]"));
}
