//! Weekend prompt templates, ported verbatim from `weekend/prompts.py`.
//!
//! Templates use `{placeholder}` slots. `render` substitutes the KNOWN keys
//! and leaves everything else intact, so a typo'd placeholder (the class C1
//! failure mode -- a raw `{date_range}` reaching the model) stays VISIBLE in
//! the prompt instead of vanishing behind a format exception.

pub fn render(template: &str, fields: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (key, value) in fields {
        out = out.replace(&format!("{{{key}}}"), value);
    }
    out
}

pub const PHASE_WEATHER_CONDENSE: &str = "\
Given this weather forecast, summarize what to expect for the weekend in
1-2 sentences. Be specific about temperatures and conditions.

{weather_str}

Output only the summary, nothing else.";

/// The phase chain used to NARROW at every step: extract asked for dates, draft
/// asked only for "name, location, description", and refine only for
/// "name + description". So the dates were discarded two phases before the schema
/// that wanted them, and every date column rendered blank. Each phase now carries
/// the same fields through verbatim -- see class C2c.
pub const CARRY_FIELDS: &str = "Carry these fields through EXACTLY as they appear in the input,
never rewritten or dropped: DATES, PRICE, AGES, LOCATION. If the input does not
state one, write \"unknown\" for it -- never guess.

AGES means the age range the VENUE OR EVENT states for itself, copied from the
source. It is NOT the age range of the family this plan is for. Never fill it
with the family's ages -- if the source does not state an age range, \"unknown\"
is the correct answer.";

pub const PHASE_EXTRACT_EVENTS: &str = "\
Extract family-friendly event listings, near {location}, from the search
results below.

For each event output one line:
NAME | LOCATION | DATES | PRICE | AGES | short description

- DATES: copy the date text exactly as written, e.g. \"Saturday, August 1 -
  Monday, August 3\" or \"23rd August\" or \"February 14-16\". Include the year if
  the result shows one. Write \"unknown\" if the result gives no date.
- PRICE and AGES: copy verbatim if stated, else \"unknown\".
- WHERE: only include somewhere a family in {location} could drive to for a
  day out. The search engine returns results from all over the world -- a zoo in
  San Diego, a trampoline park in Dublin or Oswego is useless here however well
  it matches. If the result does not name a place in or near {location}, skip
  it. Do not \"adapt\" a foreign listing to the local area.
- Ignore ads and navigation text.
- DATES must be when the EVENT runs, not when the page was written or updated.
  If the only date on the page is a publication or \"last updated\" date, that is
  not an event date -- write \"unknown\".
- A result marked [THIS WEEKEND] already mentions a date inside the weekend
  being planned, so prefer those -- they are listed first. This is a hint about
  where to look, NOT a restriction: extract from the unmarked results too. Never
  invent an event to fill the list, and never move an event's dates to make it
  fit the weekend. Fewer real events beats more invented ones.

IS THIS AN ACTIVITY? Before listing anything, ask: \"is this an actual thing a
family can go and DO at a specific time and place, or is it a page that LISTS
things?\" A directory, guide, calendar, round-up, \"what's on\" page, \"things to do
in X\" article, blog archive or a venue's events INDEX is NOT an activity -- you
cannot attend a guide. Skip those results entirely, however well they match.
If the page names a SPECIFIC event, list that event, not the page.

Search results:
{raw_text}

One event per line, in the pipe-separated format above.";

pub const PHASE_DRAFT_TRANSIENT: &str = "\
You are an expert family activity planner. Suggest 10 specific weekend activities for
families with kids ages {age_range} in {location}. Focus on time-limited events
happening specifically on {date_range}.

IS THIS AN ACTIVITY? Before listing anything, ask: \"is this an actual thing a
family can go and DO at a specific time and place, or is it a page that LISTS
things?\" A directory, guide, calendar, round-up, \"what's on\" page, \"things to do
in X\" article, blog archive or a venue's events INDEX is NOT an activity -- you
cannot attend a guide. Skip those results entirely, however well they match.
If the page names a SPECIFIC event, list that event, not the page.

The year is {year}. Every date you output must be in {year}. An event dated in
any other year does not belong in this plan -- drop it rather than re-dating it.

Weather: {weather_condensed}

Available events:
{cleaned_sources}

Output one line per suggestion in this EXACT format:
NAME | LOCATION | DATES | PRICE | AGES | description (highlight themes/appeal)

{carry}DO NOT suggest any of these places -- the family has already ruled them out, and
a suggestion naming one is dropped after the fact, wasting a slot that could
have held a real option: {exclusions}";

pub const PHASE_REFINE: &str = "\
Here are activity suggestions:

{draft_text}

Merge any near-duplicates, keep the best 8, remove low-quality or irrelevant
ones, and sort by overall appeal.

Output the refined list in the SAME pipe-separated format you received:
NAME | LOCATION | DATES | PRICE | AGES | short description

Carry DATES, PRICE, AGES and LOCATION through unchanged from the input. Merging
two entries keeps the more specific value, never \"unknown\" over a real one.";

/// Class C4 (MANDATED-PLACEHOLDER) + C2b (DATE-DROPPED-AT-THE-LLM-BOUNDARY).
/// This prompt used to ORDER the model to emit "$20-30 per child or free" and
/// "2-3 hours" on every row and close with "Never leave any field empty" -- which
/// is what turned "unknown" into a fabricated constant the report then rendered as
/// fact. It also had no date field at all, so an event's real dates were
/// structurally impossible to carry. Both are fixed here: unknown is now an
/// explicit empty string, and start_date/end_date are first-class.
pub const PHASE_STRUCTURE_TRANSIENT_SYSTEM: &str = "\
Output JSON now. Use EXACT schema:
{\"transient_events\": [{\"name\": \"str\", \"location\": \"str\",
\"target_ages\": \"str\", \"price\": \"str\", \"start_date\": \"str\", \"end_date\": \"str\",
\"duration\": \"str\", \"weather\": \"str\", \"day\": \"str\"}]}

Rules for every field:
- Copy values from the source text. NEVER invent one.
- If the source does not state a value, output an empty string \"\" for it.
  An empty field is CORRECT and expected. Do not guess, do not use a typical
  or average value, and do not repeat a value from another row.
- start_date / end_date: ISO YYYY-MM-DD, from the DATES field of the input.
  The input dates are usually free text (e.g. \"Saturday, August 1 - Monday,
  August 3\") -- convert them to ISO. If the text gives no year, the year is
  {year}. If the input says \"unknown\", output \"\".
- target_ages: the age range the VENUE OR EVENT states for itself, copied
  from the input. NEVER the family's ages. If the input does not state one,
  output \"\".
- price: the actual price as written in the source, else \"\".

Weather: {weather_condensed}
Set weather from the activity type and the forecast above: \"outdoor\" for
outdoor activities (parks, zoo, sports), \"indoor\" for indoor venues (museums,
play centres, trampoline parks), \"both\" for flexible activities.

Output ONLY JSON.";

pub const PHASE_EXTRACT_VENUES: &str = "\
Extract family-friendly venues from the search results below. For each venue,
list its name, location, price if available, and what it offers for kids.
Ignore irrelevant search results, ads, and navigation text.

Search results:
{raw_text}

List each relevant venue with key details, one per line.";

pub const PHASE_DRAFT_FIXED: &str = "\
You are an expert family activity planner. Suggest 10 specific weekend activities for
families with kids ages {age_range} in {location}. Include a diverse mix of
year-round venues, outdoor seasonal places (parks, conservation areas, farms),
and indoor family spots. The year is {year}; the weekend is {date_range}.

Weather: {weather_condensed}
Prefer outdoor activities when clear/warm, and indoor venues when precipitation is expected.

Available venues:
{cleaned_sources}

Output one line per suggestion in this EXACT format:
NAME | LOCATION | DATES | PRICE | AGES | description (highlight features/exhibits)

{carry}DO NOT suggest any of these places -- the family has already ruled them out, and
a suggestion naming one is dropped after the fact, wasting a slot that could
have held a real option: {exclusions}";

pub const PHASE_STRUCTURE_FIXED_SYSTEM: &str = "\
Output JSON now. Use EXACT schema:
{\"fixed_activities\": [{\"name\": \"str\", \"location\": \"str\",
\"target_ages\": \"str\", \"price\": \"str\", \"weather\": \"str\"}]}

Rules for every field:
- Copy values from the source text. NEVER invent one.
- If the source does not state a value, output an empty string \"\" for it.
  An empty field is CORRECT and expected. Do not guess a typical price or age
  range, and do not repeat a value from another row.

Weather: {weather_condensed}
Set weather from the activity type and the forecast above: \"outdoor\" for
outdoor activities (parks, zoo, sports), \"indoor\" for indoor venues (museums,
play centres, trampoline parks), \"both\" for flexible activities.

Output ONLY JSON.";

pub const PHASE_STRUCTURE_USER: &str = "Convert these activities to the schema:

{draft_text}";

/// The placeholder keys a template may reference, used to prove no typo'd
/// placeholder survives rendering.
pub const KNOWN_KEYS: &[&str] = &[
    "location",
    "raw_text",
    "age_range",
    "date_range",
    "year",
    "weather_condensed",
    "cleaned_sources",
    "carry",
    "exclusions",
    "draft_text",
    "weather_str",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Every known placeholder in a template must be substituted by render, and
    /// no unknown placeholder may remain (class C1: a raw `{date_range}` reaching
    /// the model was the original defect).
    fn assert_renders_clean(template: &str, fields: &[(&str, &str)]) {
        let out = render(template, fields);
        for key in KNOWN_KEYS {
            assert!(
                !out.contains(&format!("{{{key}}}")),
                "placeholder {key} left unreplaced in:\n{out}"
            );
        }
    }

    #[test]
    fn extract_events_renders_clean() {
        assert_renders_clean(
            PHASE_EXTRACT_EVENTS,
            &[("location", "Vaughan/GTA"), ("raw_text", "corpus")],
        );
    }

    #[test]
    fn draft_transient_renders_clean() {
        assert_renders_clean(
            PHASE_DRAFT_TRANSIENT,
            &[
                ("age_range", "6-12"),
                ("location", "Vaughan/GTA"),
                ("date_range", "Aug 7 to Aug 9"),
                ("year", "2026"),
                ("weather_condensed", "sunny and warm"),
                ("cleaned_sources", "sources"),
                ("carry", CARRY_FIELDS),
                ("exclusions", "none"),
            ],
        );
    }

    #[test]
    fn refine_renders_clean() {
        assert_renders_clean(PHASE_REFINE, &[("draft_text", "draft")]);
    }

    #[test]
    fn structure_transient_renders_clean() {
        assert_renders_clean(
            PHASE_STRUCTURE_TRANSIENT_SYSTEM,
            &[("year", "2026"), ("weather_condensed", "sunny")],
        );
        assert_renders_clean(PHASE_STRUCTURE_USER, &[("draft_text", "draft")]);
    }

    #[test]
    fn test_weekend_prompts_match_shared_conf() {
        use std::path::Path;
        let manifest = env!("CARGO_MANIFEST_DIR");
        let conf_path = Path::new(manifest)
            .parent()
            .unwrap()
            .join("conf/prompts.toml");
        let content = std::fs::read_to_string(&conf_path).unwrap_or_else(|e| {
            panic!("conf/prompts.toml missing at {}: {e}", conf_path.display())
        });
        let val: toml::Value = toml::from_str(&content).expect("conf/prompts.toml must parse");
        let wk = val
            .get("weekend")
            .expect("conf/prompts.toml needs [weekend]");

        let get_inst = |k: &str| -> &str {
            wk.get(k)
                .and_then(|v| v.get("instructions"))
                .and_then(|i| i.as_str())
                .unwrap_or_else(|| panic!("missing [weekend.{k}].instructions"))
        };

        assert_eq!(PHASE_WEATHER_CONDENSE, get_inst("weather_condense"));
        assert_eq!(CARRY_FIELDS, get_inst("carry_fields"));
        assert_eq!(PHASE_EXTRACT_EVENTS, get_inst("extract_events"));
        assert_eq!(PHASE_EXTRACT_VENUES, get_inst("extract_venues"));
        assert_eq!(PHASE_DRAFT_TRANSIENT, get_inst("draft_transient"));
        assert_eq!(PHASE_DRAFT_FIXED, get_inst("draft_fixed"));
        assert_eq!(PHASE_REFINE, get_inst("refine"));
        assert_eq!(
            PHASE_STRUCTURE_TRANSIENT_SYSTEM,
            get_inst("structure_transient_system")
        );
        assert_eq!(
            PHASE_STRUCTURE_FIXED_SYSTEM,
            get_inst("structure_fixed_system")
        );
        assert_eq!(PHASE_STRUCTURE_USER, get_inst("structure_user"));
    }
}
