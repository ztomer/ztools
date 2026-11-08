//! What the timeline's tab bar shows, and whether the Following tab is the
//! one selected.
//!
//! The collector used to click a tab and then wait for "a tab is selected",
//! which For You always is: the wait passed at once whether or not the click
//! landed, and a run whose click came before the tab bar was ready scrolled
//! For You for minutes before failing with "the feed never switched" and no
//! word of what the page had shown (2026-11-01 noon, 2026-11-07 morning).
//! The switch is now VERIFIED: the Following tab itself must carry
//! `aria-selected="true"`, and when it never does the error says which tab
//! was selected and what the page held.
//!
//! The browser half is one expression ([`TAB_STATE_JS`]) returning JSON; every
//! decision about that JSON is here, pure, so it tests without a browser.

use serde::Deserialize;

/// Localized "Following" labels, in `browser.py` order, overridable via
/// `TWITTER_FOLLOWING_TAB` (comma-separated).
pub const FOLLOWING_TERMS: &[&str] = &[
    "Following",
    "Abonnements",
    "Siguiendo",
    "Sigo",
    "Gefolgt",
    "关注",
    "フォロー中",
];

#[must_use]
pub fn following_terms() -> Vec<String> {
    if let Ok(raw) = std::env::var("TWITTER_FOLLOWING_TAB") {
        let terms: Vec<String> = raw
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(str::to_owned)
            .collect();
        if !terms.is_empty() {
            return terms;
        }
    }
    FOLLOWING_TERMS.iter().map(ToString::to_string).collect()
}

/// The page's tab bar as one JSON string.
///
/// The url, the title, whether the primary column and a tablist are present,
/// and every `[role=tab]` with its text and selection. A bare expression, as
/// `Runtime.evaluate` wants.
pub const TAB_STATE_JS: &str = "JSON.stringify({\
     url: window.location.href, \
     title: document.title, \
     primary_column: document.querySelector('[data-testid=\"primaryColumn\"]') !== null, \
     tablist: document.querySelector('[role=\"tablist\"]') !== null, \
     tabs: Array.from(document.querySelectorAll('[role=\"tab\"]')).map(t => ({\
       text: (t.innerText || t.textContent || '').trim().slice(0, 40), \
       selected: t.getAttribute('aria-selected') === 'true'\
     }))\
   })";

/// One `[role=tab]` element as the page reported it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Tab {
    pub text: String,
    pub selected: bool,
}

/// The page as [`TAB_STATE_JS`] reported it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Default)]
pub struct TabState {
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub primary_column: bool,
    #[serde(default)]
    pub tablist: bool,
    #[serde(default)]
    pub tabs: Vec<Tab>,
}

impl TabState {
    /// Parse what [`TAB_STATE_JS`] returned. An unreadable answer is a state
    /// with nothing in it, which [`Self::describe`] says plainly.
    #[must_use]
    pub fn parse(raw: &str) -> Self {
        serde_json::from_str(raw).unwrap_or_default()
    }

    /// The index of the Following tab: the first tab whose text contains a
    /// localized term, in term order.
    #[must_use]
    pub fn following_index(&self, terms: &[String]) -> Option<usize> {
        terms.iter().find_map(|term| {
            self.tabs
                .iter()
                .position(|t| !term.is_empty() && t.text.contains(term.as_str()))
        })
    }

    /// Whether the Following tab is present AND the selected one. Nothing
    /// else counts: For You is always selected, so "a tab is selected" proves
    /// nothing about the switch.
    #[must_use]
    pub fn following_selected(&self, terms: &[String]) -> bool {
        self.following_index(terms)
            .is_some_and(|i| self.tabs[i].selected)
    }

    /// The page in one line, for the error a failed switch raises: which tab
    /// is selected, which tabs exist, and what matched.
    #[must_use]
    pub fn describe(&self, terms: &[String]) -> String {
        let selected = self
            .tabs
            .iter()
            .find(|t| t.selected)
            .map_or_else(|| "no tab".to_string(), |t| format!("{:?}", t.text));
        let tabs = if self.tabs.is_empty() {
            "no [role=tab] element".to_string()
        } else {
            format!(
                "tabs [{}]",
                self.tabs
                    .iter()
                    .map(|t| format!("{:?}", t.text))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let matched = self.following_index(terms).map_or_else(
            || format!("none of {terms:?} matched a tab"),
            |i| format!("{:?} matched", self.tabs[i].text),
        );
        format!(
            "selected: {selected}; {tabs}; {matched}; tablist {}, primaryColumn {}; page {:?} at {}",
            present(self.tablist),
            present(self.primary_column),
            self.title,
            self.url
        )
    }
}

const fn present(yes: bool) -> &'static str {
    if yes { "present" } else { "absent" }
}

/// The click expression for the Following tab.
///
/// It matches by the same terms [`TabState::following_index`] uses and
/// returns the clicked tab's text, or "" when no tab matched (never a
/// positional guess: a second tab that is not Following is the wrong feed).
#[must_use]
pub fn click_js(terms: &[String]) -> String {
    let terms_json = serde_json::to_string(terms).unwrap_or_else(|_| "[]".to_string());
    format!(
        "(function() {{ \
           const terms = {terms_json}; \
           const tabs = Array.from(document.querySelectorAll('[role=\"tab\"]')); \
           for (const term of terms) {{ \
             const hit = tabs.find(t => term && (t.innerText || t.textContent || '').includes(term)); \
             if (hit) {{ hit.click(); return (hit.innerText || hit.textContent || '').trim(); }} \
           }} \
           return ''; \
         }})()"
    )
}

#[cfg(test)]
#[path = "tabs_tests.rs"]
mod tests;
