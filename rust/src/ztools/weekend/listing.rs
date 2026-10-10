//! A2, listing-page titles: a page that LISTS events is neither a venue nor
//! an event. Split from `suitability.rs` (the house 500-line cap); its items
//! are re-exported there, where the gates are called from.

use std::sync::LazyLock;

use regex::Regex;

use super::WeekendEvent;

fn re(pattern: &str) -> Regex {
    // The patterns are literals in this file; a bad one fails every test.
    Regex::new(pattern).unwrap_or_else(|e| panic!("bad listing pattern {pattern}: {e}"))
}

/// Phrases only a listing or aggregator page title carries.
const LISTING_PHRASES: &[&str] = &[
    "things to do",
    "events this weekend",
    "events & activities",
    "events and activities",
    "what's on",
    "what’s on",
    "events calendar",
    "event calendar",
    "calendar of events",
    "upcoming events",
    "events near",
    "events in ",
    "best things",
    "top 10",
    "guide to",
];

/// "- Oct 2026", "(October 2026)": a page dated by month, which a venue never is.
static MONTH_YEAR: LazyLock<Regex> = LazyLock::new(|| {
    re(r"(?i)\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+20\d\d\b")
});

/// A2: is this the title of a listing page rather than a venue or an event?
///
/// The phrases here, plus every marker the follow-up step uses to decide a
/// page is an aggregator worth reading (`followup::looks_like_aggregator`):
/// one question, "does this title belong to a page that lists things", must
/// not get two answers in one pipeline.
#[must_use]
pub fn is_listing_page_title(text: &str) -> bool {
    let lower = text.to_lowercase();
    LISTING_PHRASES.iter().any(|p| lower.contains(p))
        || MONTH_YEAR.is_match(text)
        || super::followup::looks_like_aggregator(text)
}

/// Words a listing page's title is made of and an event's name is not made
/// of ENTIRELY: listing vocabulary, PLURAL category nouns (a page lists
/// "festivals"; an event is "a festival"), the region words, seasons,
/// holidays, weekdays and months. Matched as whole words after the places are
/// removed.
const LISTING_WORDS: &[&str] = &[
    // listing vocabulary
    "events",
    "activities",
    "things",
    "to",
    "do",
    "fun",
    "family",
    "families",
    "kids",
    "kid",
    "children",
    "weekend",
    "weekends",
    "long",
    "this",
    "next",
    "best",
    "top",
    "near",
    "nearby",
    "around",
    "in",
    "at",
    "on",
    "and",
    "the",
    "of",
    "for",
    "with",
    "upcoming",
    "today",
    "tonight",
    "guide",
    "ideas",
    "list",
    "what",
    "whats",
    "happening",
    "free",
    "more",
    // plural categories
    "festivals",
    "fairs",
    "markets",
    "shows",
    "workshops",
    "concerts",
    "attractions",
    "places",
    "celebrations",
    "parties",
    "programs",
    "classes",
    "camps",
    "outings",
    "adventures",
    "categories",
    "dates",
    // a listing page's category and filter labels ("Arts & Culture", "This
    // Week", "All Categories"), which a followed page's nav yields as lines
    "all",
    "week",
    "month",
    "community",
    "arts",
    "culture",
    "music",
    "food",
    "drink",
    "sports",
    "fitness",
    "health",
    "wellness",
    "education",
    "shopping",
    "business",
    "networking",
    "seasonal",
    // region words beyond the configured places
    "ontario",
    "canada",
    "gta",
    "area",
    "region",
    "city",
    "downtown",
    // seasons and holidays
    "fall",
    "autumn",
    "spring",
    "summer",
    "winter",
    "season",
    "thanksgiving",
    "halloween",
    "christmas",
    "easter",
    "holiday",
    "holidays",
    // weekdays and months
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
];

/// A2: does this NAME name no event at all?
///
/// True when, once the configured `places` are taken out, every word left is
/// listing vocabulary, a number or a single letter: "Richmond Hill, ON",
/// "Toronto Events", "Best Thanksgiving Events and Fall Festivals Near
/// Toronto 2026". Verifiable from the name alone, so it is decided here and
/// not by the model; a name with ONE word of its own ("Woodbridge Fall
/// Fair", "Pumpkins After Dark", "Very Toronto") is left to the model's
/// judgement, which is where a site name belongs. An empty name is not a
/// listing title: it is no name, and the provenance gate's to judge.
#[must_use]
pub fn names_no_event(name: &str, places: &[String]) -> bool {
    let mut text: String = super::normalize_for_match(name)
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect();
    text = format!(
        " {} ",
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    );
    if text.trim().is_empty() {
        return false;
    }
    for place in places {
        let place = format!(" {} ", place.trim().to_lowercase());
        if place.trim().is_empty() {
            continue;
        }
        while text.contains(&place) {
            text = text.replace(&place, " ");
        }
    }
    text.split_whitespace().all(|w| {
        w.chars().count() < 2 || w.chars().all(|c| c.is_ascii_digit()) || LISTING_WORDS.contains(&w)
    })
}

/// A2: a row NAMED by a listing page is dropped; a listing-page VENUE is cleared.
///
/// A name is a listing page's by its phrases, or by naming nothing but places
/// and listing words ([`names_no_event`]). A row whose venue is a listing
/// page's title keeps its event and loses the venue, so the renderer prints
/// the missing-value sentinel instead of a page title. A venue that is a place
/// ("Richmond Hill, ON") is a venue.
#[must_use]
pub fn reject_listing_page_titles(
    events: Vec<WeekendEvent>,
    places: &[String],
) -> (Vec<WeekendEvent>, Vec<String>, usize) {
    let mut notes = Vec::new();
    let mut dropped = 0;
    let mut kept = Vec::new();
    for mut ev in events {
        if is_listing_page_title(&ev.name) || names_no_event(&ev.name, places) {
            notes.push(format!(
                "dropped '{}' — a listing page, not an event",
                ev.name
            ));
            dropped += 1;
            continue;
        }
        if is_listing_page_title(&ev.location) {
            notes.push(format!(
                "'{}': venue {:?} is a listing page's title — cleared",
                ev.name, ev.location
            ));
            ev.location.clear();
        }
        kept.push(ev);
    }
    (kept, notes, dropped)
}
