//! Weekend planner cache and list directory filter helper module.

use std::fs;

use super::weekend::{matches_exclusion, WeekendEvent};

/// Check if a title refers to a directory page, list article, social video, or round-up.
pub fn is_directory_or_list_page(title: &str) -> bool {
    let lower = title.to_lowercase();
    let keywords = [
        "tiktok",
        "youtube",
        "tripadvisor",
        "realtor",
        "hotelsbyday",
        "real estate",
        "mls®",
        "listing",
        "directory",
        "calendar of events",
        "things to do",
        "best kids parks",
        "best hikes",
        "day trips",
        "easy adventures",
        "townhouse",
        "hotel & resorts",
        "hotels",
        "top 10",
        "best museums",
        "winter hikes",
        "guide to",
        "allevents",
        "kathryn anywhere",
        "special events calendar",
        "best hiking trails",
        "toronto, ontario",
        "vaughan, ontario",
        "ontario, canada",
        "nearby attractions",
        "day use hotels",
        "you can hike through",
        "vaughan events & activities",
        "kids events in vaughan",
        "upcoming kids events",
    ];

    if keywords.iter().any(|kw| lower.contains(kw)) {
        return true;
    }

    // Filter out Non-ASCII characters (e.g. Cyrillic/Russian titles)
    if title.chars().any(|c| c as u32 > 0x024F) {
        return true;
    }

    // Filter out exact city names
    let trimmed = lower.trim();
    if [
        "toronto",
        "vaughan",
        "ontario",
        "canada",
        "toronto, ontario",
    ]
    .contains(&trimmed)
    {
        return true;
    }

    // Filter out street address patterns
    if lower.contains("road") && lower.contains("suite") || lower.contains("bedroom") {
        return true;
    }

    let mut words = lower.split_whitespace();
    if let Some(first) = words.next() {
        if first.chars().any(|c| c.is_ascii_digit())
            && (lower.contains("best") || lower.contains("things") || lower.contains("top"))
        {
            return true;
        }
    }

    false
}

/// Positive evidence matcher verifying an activity/event is located within the GTA region.
pub fn has_region_evidence(text: &str) -> bool {
    let lower = text.to_lowercase();
    let foreign_tokens = [
        "new york",
        "nyc",
        "san diego",
        "dublin",
        "acropolis",
        "athens",
        "chicago",
        "london",
        "paris",
        "los angeles",
        "miami",
        "astana",
        "berlin",
        "tokyo",
    ];

    if foreign_tokens.iter().any(|ft| lower.contains(ft)) {
        return false;
    }

    let gta_tokens = [
        "toronto",
        "vaughan",
        "markham",
        "richmond hill",
        "mississauga",
        "brampton",
        "scarborough",
        "north york",
        "etobicoke",
        "york",
        "woodbridge",
        "concord",
        "thornhill",
        "maple",
        "kleinburg",
        "aurora",
        "newmarket",
        "oakville",
        "burlington",
        "pickering",
        "ajax",
        "whitby",
        "oshawa",
        "milton",
        "caledon",
        "king city",
        "stouffville",
        "bolton",
        "georgetown",
        "hamilton",
        "guelph",
        "barrie",
        "gta",
        "halton",
        "yorkdale",
        "downsview",
        "ontario",
    ];

    gta_tokens.iter().any(|tok| lower.contains(tok))
}

/// Clean raw search titles to extract clean venue or event names.
pub fn clean_venue_or_event_title(raw_title: &str) -> Option<String> {
    let mut cleaned = raw_title.replace("**", "").trim().to_string();

    for delimiter in [" | ", " - ", " – ", " — "] {
        if let Some(pos) = cleaned.find(delimiter) {
            let suffix = cleaned[pos + delimiter.len()..].to_lowercase();
            if suffix.contains("tiktok")
                || suffix.contains("tripadvisor")
                || suffix.contains("youtube")
                || suffix.contains("yelp")
                || suffix.contains("narcity")
                || suffix.contains("realtor")
                || suffix.contains("hotelsbyday")
                || suffix.contains("medium")
                || suffix.contains("trip")
            {
                cleaned = cleaned[..pos].trim().to_string();
            }
        }
    }

    if is_directory_or_list_page(&cleaned) {
        return None;
    }

    if !has_region_evidence(&cleaned) {
        return None;
    }

    if cleaned.len() < 3 {
        return None;
    }

    Some(cleaned)
}

/// Load exclusions from the first configured file that yields any.
///
/// The paths come from config rather than the home directory: a hardcoded
/// `~/…` lookup made this read whichever machine was running, so the same test
/// took a different branch on a developer box than in CI -- and the coverage
/// number moved with it.
pub fn load_exclusions(config: &crate::config::ZtoolsConfig) -> Vec<String> {
    for p in config
        .weekend_exclusions_paths
        .iter()
        .map(|s| crate::manifest::expand_tilde(s))
    {
        if let Ok(content) = fs::read_to_string(p) {
            let mut list = Vec::new();
            let mut in_excl = false;
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("exclude_places = [") {
                    in_excl = true;
                    continue;
                }
                if in_excl {
                    if trimmed.starts_with(']') {
                        break;
                    }
                    let item = trimmed.trim_matches(|c| c == '"' || c == '\'' || c == ',');
                    if !item.is_empty() {
                        list.push(item.to_string());
                    }
                }
            }
            if !list.is_empty() {
                return list;
            }
        }
    }

    vec![
        "Canada's Wonderland".into(),
        "Toronto Zoo".into(),
        "ROM".into(),
        "Ripley's".into(),
        "LEGOLAND".into(),
        "CN Tower".into(),
        "Sky Zone Toronto".into(),
    ]
}

/// Load default cached activities returning clean, curated GTA family venues and events.
pub fn load_cached_activities(
    config: &crate::config::ZtoolsConfig,
) -> (Vec<WeekendEvent>, Vec<WeekendEvent>) {
    let exclusions = load_exclusions(config);

    let all_fixed = vec![
        WeekendEvent {
            name: "Kortright Centre for Conservation".into(),
            location: "Vaughan".into(),
            price: "$8-12".into(),
            target_ages: "All Ages".into(),
            day: "Sat-Sun".into(),
            dates: "Year-Round".into(),
            description: "800 acres of outdoor hiking trails, pond dipping, and interactive nature exhibits great for ages 6-13.".into(),
            is_transient: false, score: 0.0,
        },
        WeekendEvent {
            name: "Air Riderz Trampoline Park".into(),
            location: "Vaughan".into(),
            price: "$18-24".into(),
            target_ages: "6-13".into(),
            day: "Fri-Sun".into(),
            dates: "Year-Round".into(),
            description: "Indoor trampoline zone, 24ft climbing walls, dodgeball court, and ninja warrior obstacle course.".into(),
            is_transient: false, score: 0.0,
        },
        WeekendEvent {
            name: "Playdium Vaughan Arcade & VR".into(),
            location: "Vaughan".into(),
            price: "$15-30".into(),
            target_ages: "6-13".into(),
            day: "Fri-Sun".into(),
            dates: "Year-Round".into(),
            description: "40,000 sq ft venue featuring high-tech arcade games, virtual reality arenas, and indoor ropes courses.".into(),
            is_transient: false, score: 0.0,
        },
        WeekendEvent {
            name: "Mount Nemo Conservation Area".into(),
            location: "Halton / GTA".into(),
            price: "$7-10".into(),
            target_ages: "6-13".into(),
            day: "Fri-Sun".into(),
            dates: "Year-Round".into(),
            description: "Escarpment cliffside walking trails, cliffside lookout points, and limestone cave exploration.".into(),
            is_transient: false, score: 0.0,
        },
        WeekendEvent {
            name: "McMichael Canadian Art Collection Trails".into(),
            location: "Kleinburg / Vaughan".into(),
            price: "Free Trails / $15".into(),
            target_ages: "All Ages".into(),
            day: "Sat-Sun".into(),
            dates: "Year-Round".into(),
            description: "100-acre outdoor sculpture park, pine forest trails, and hands-on family art activities.".into(),
            is_transient: false, score: 0.0,
        },
    ];

    let all_transient = vec![
        WeekendEvent {
            name: "Vaughan Public Library Youth Science Workshop".into(),
            location: "Vaughan Library".into(),
            price: "Free".into(),
            target_ages: "6-13".into(),
            day: "Saturday".into(),
            dates: "Aug 08".into(),
            description: "Free hands-on STEM experiment and creative tech coding activity for kids ages 6-13.".into(),
            is_transient: true, score: 0.0,
        },
        WeekendEvent {
            name: "GTA Outdoor Nature Trail Discovery Walk".into(),
            location: "Kortright Conservation".into(),
            price: "Free with admission".into(),
            target_ages: "All Ages".into(),
            day: "Sunday".into(),
            dates: "Aug 09".into(),
            description: "Guided family nature walk with wildlife tracking, pond exploration, and bug identification.".into(),
            is_transient: true, score: 0.0,
        },
        WeekendEvent {
            name: "High Park Family Birding & Biodiversity Tour".into(),
            location: "Toronto / High Park".into(),
            price: "Free".into(),
            target_ages: "All Ages".into(),
            day: "Saturday".into(),
            dates: "Aug 08".into(),
            description: "Interactive woodland nature walk and birdwatching session tailored for young explorers.".into(),
            is_transient: true, score: 0.0,
        },
    ];

    let fixed = all_fixed
        .into_iter()
        .filter(|ev| {
            let combined = format!("{} {}", ev.name, ev.location);
            !exclusions
                .iter()
                .any(|excl| matches_exclusion(&combined, excl))
        })
        .collect();

    let transient = all_transient
        .into_iter()
        .filter(|ev| {
            let combined = format!("{} {}", ev.name, ev.location);
            !exclusions
                .iter()
                .any(|excl| matches_exclusion(&combined, excl))
        })
        .collect();

    (transient, fixed)
}
