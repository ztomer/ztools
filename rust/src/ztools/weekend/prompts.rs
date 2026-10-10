//! Weekend prompt templates, ported verbatim from `weekend/prompts.py`.
//!
//! Templates use `{placeholder}` slots. `render` substitutes the KNOWN keys
//! and leaves everything else intact, so a typo'd placeholder (the class C1
//! failure mode -- a raw `{date_range}` reaching the model) stays VISIBLE in
//! the prompt instead of vanishing behind a format exception.

#[must_use]
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
/// asked only for "name, location, description", and refine only for "name +
/// description".
///
/// So the dates were discarded two phases before the schema that wanted them,
/// and every date column rendered blank. Each phase now carries the same fields
/// through verbatim -- see class C2c.
pub const CARRY_FIELDS: &str = "Carry these fields through EXACTLY as they appear in the input,
never rewritten or dropped: DATES, PRICE, AGES, LOCATION. If the input does not
state one, write \"unknown\" for it -- never guess.

AGES means the age range the VENUE OR EVENT states for itself, copied from the
source. It is NOT the age range of the family this plan is for. Never fill it
with the family's ages -- if the source does not state an age range, \"unknown\"
is the correct answer.";

/// The one rule for telling an activity from a page that lists activities,
/// bound into BOTH the extract and the draft prompts through `{activity_rule}`.
///
/// It used to be two pasted copies ending "Skip those results entirely,
/// however well they match", with the exception -- list the events a page
/// names -- trailing after the emphatic veto. On 2026-10-10 the extractor read
/// the veto and not the exception: eight Thanksgiving results naming the Sugar
/// Beach harvest market, the Erin Mills family festival and a pumpkin trail came
/// back as "I cannot extract ... they are all pages that list things", and the
/// plan had one event. Most results ARE listing pages; the events they name are
/// the supply. So the rule now leads with what to keep, and refuses only the
/// page's own title.
pub const ACTIVITY_RULE: &str =
    "IS THIS AN ACTIVITY? An activity is a thing a family can go and DO at a
specific time and place. A PAGE that lists things -- a directory, guide,
calendar, round-up, \"what's on\" page, \"things to do in X\" article, blog
archive or a venue's events index -- is not one, so never output a page's own
title as an activity.

But read what such a page SAYS. When its text names a specific event -- a
market, festival, fair, show, workshop -- that EVENT is an activity: output it,
with the venue and the dates the text gives for it. Most results here come from
listing pages, and the events they name are exactly what this plan is for.
Leave a result out only when its text names no specific event at all.";

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
  being planned, so prefer those -- search results so marked are listed first.
  A line from a followed page (\"- [page title] ...\") is marked where it stands:
  its event's name is often on a neighbouring line, so read the lines around
  it. This is a hint about where to look, NOT a restriction: extract from the
  unmarked results too. Never
  invent an event to fill the list, and never move an event's dates to make it
  fit the weekend. Fewer real events beats more invented ones.

{activity_rule}

Search results:
{raw_text}

Output ONLY event lines, one per line, in the pipe-separated format above. No
explanations, no notes about what you left out: when no result names an event,
output nothing.";

pub const PHASE_DRAFT_TRANSIENT: &str = "\
You are an expert family activity planner. Suggest 10 specific weekend activities for
families with kids ages {age_range} in {location}. Focus on time-limited events
happening specifically on {date_range}.

{activity_rule}

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

/// Refine judges which entries are events and which are the same event.
///
/// It never caps or ranks (the scorer ranks), and it must name every removal
/// with a reason, because `refine::merge_refined` restores any draft row its
/// answer neither keeps nor drops with one. It used to say "keep the best 8
/// ... and sort by overall appeal", and its answer replaced the draft.
pub const PHASE_REFINE: &str = "\
Here are activity suggestions:

{draft_text}

Merge entries that are the same event into one, and remove any entry that is
not a specific event a family can attend (a web page's title, a website's
name, a bare place name). Keep every other entry: do not cut the list to a
number, and do not reorder it.

Output the refined list in the SAME pipe-separated format you received:
NAME | LOCATION | DATES | PRICE | AGES | short description

Then, for EVERY input entry you did not output -- removed, or merged into
another -- output one line saying why, in this format:
DROPPED | NAME | reason

Carry DATES, PRICE, AGES and LOCATION through unchanged from the input. Merging
two entries keeps the more specific value, never \"unknown\" over a real one.";

/// How the structure phases label a row indoor/outdoor, bound into both
/// structure prompts through `{weather_rule}`.
///
/// The label is a fact about the ACTIVITY, so it is judged from the activity's
/// own text and may be unknown. It used to be "set weather from the activity
/// type and the forecast above", with three allowed values and no way to say
/// "the text does not tell": on 2026-10-10 a robotics workshop held in a hotel
/// came back "outdoor" under a clear forecast -- a placeholder the model was
/// made to fill, rendered as data (class C4). The forecast no longer reaches
/// either structure prompt at all, so it cannot decide the label.
pub const WEATHER_RULE: &str =
    "- weather: where the activity HAPPENS, judged from its name, venue and
  description: \"outdoor\" for activities in the open (parks, farms, fairs,
  open-air markets, trails), \"indoor\" for activities inside a building
  (museums, libraries, hotels, theatres, play centres), \"both\" when the text
  says it is both. When the text does not say where it happens, output \"\".
  The weather forecast does not decide this.";

/// Class C4 (MANDATED-PLACEHOLDER) + C2b (DATE-DROPPED-AT-THE-LLM-BOUNDARY).
///
/// This prompt used to ORDER the model to emit "$20-30 per child or free" and
/// "2-3 hours" on every row and close with "Never leave any field empty" -- which
/// is what turned "unknown" into a fabricated constant the report then rendered as
/// fact. It also had no date field at all, so an event's real dates were
/// structurally impossible to carry. Both are fixed here: unknown is now an
/// explicit empty string, and `start_date/end_date` are first-class.
pub const PHASE_STRUCTURE_TRANSIENT_SYSTEM: &str = "\
Output JSON now. Use EXACT schema:
{\"transient_events\": [{\"name\": \"str\", \"location\": \"str\",
\"target_ages\": \"str\", \"price\": \"str\", \"start_date\": \"str\", \"end_date\": \"str\",
\"duration\": \"str\", \"weather\": \"str\", \"day\": \"str\",
\"description\": \"str\"}]}

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
- description: the input's short description of the activity, copied, else \"\".
{weather_rule}

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
    "activity_rule",
    "weather_rule",
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
            &[
                ("location", "Vaughan/GTA"),
                ("raw_text", "corpus"),
                ("activity_rule", ACTIVITY_RULE),
            ],
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
                ("activity_rule", ACTIVITY_RULE),
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
            &[("year", "2026"), ("weather_rule", WEATHER_RULE)],
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

        for (key, text) in SHARED {
            assert_eq!(*text, get_inst(key), "[weekend.{key}] drifted");
        }
        // Both directions: a [weekend.*] entry no constant mirrors is a prompt
        // nobody sends, kept in step with nothing.
        let mut in_conf: Vec<&str> = wk
            .as_table()
            .expect("[weekend] is a table")
            .keys()
            .map(String::as_str)
            .collect();
        let mut mirrored: Vec<&str> = SHARED.iter().map(|(k, _)| *k).collect();
        in_conf.sort_unstable();
        mirrored.sort_unstable();
        assert_eq!(in_conf, mirrored, "[weekend.*] entries and SHARED disagree");
    }

    /// Every prompt this file holds, beside its `conf/prompts.toml` key.
    const SHARED: &[(&str, &str)] = &[
        ("weather_condense", PHASE_WEATHER_CONDENSE),
        ("carry_fields", CARRY_FIELDS),
        ("activity_rule", ACTIVITY_RULE),
        ("weather_rule", WEATHER_RULE),
        ("extract_events", PHASE_EXTRACT_EVENTS),
        ("draft_transient", PHASE_DRAFT_TRANSIENT),
        ("refine", PHASE_REFINE),
        (
            "structure_transient_system",
            PHASE_STRUCTURE_TRANSIENT_SYSTEM,
        ),
        ("structure_user", PHASE_STRUCTURE_USER),
    ];

    /// REACHABILITY: every `pub const` prompt here is SENT by some non-test
    /// code. `PHASE_EXTRACT_VENUES`, `PHASE_DRAFT_FIXED` and
    /// `PHASE_STRUCTURE_FIXED_SYSTEM` were each drift-gated against
    /// `conf/prompts.toml` and referenced by nothing else: three prompts kept
    /// in lockstep for a phase that did not exist. A `pub` const is invisible
    /// to `dead_code`, so the reference is looked for in the source itself.
    #[test]
    fn every_prompt_constant_is_sent_by_some_phase() {
        let src = env!("CARGO_MANIFEST_DIR");
        let root = std::path::Path::new(src).join("src");
        let mut corpus = String::new();
        let mut stack = vec![root];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).expect("src is readable").flatten() {
                let path = entry.path();
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "rs")
                    && !name.ends_with("_tests.rs")
                    && !path.ends_with("weekend/prompts.rs")
                {
                    corpus.push_str(&std::fs::read_to_string(&path).unwrap_or_default());
                }
            }
        }
        let decl = regex::Regex::new(r"(?m)^pub const ([A-Z_]+): &str").expect("static regex");
        let own = include_str!("prompts.rs");
        let names: Vec<&str> = decl
            .captures_iter(own)
            .filter_map(|c| c.get(1).map(|m| m.as_str()))
            .collect();
        assert!(
            names.len() >= SHARED.len(),
            "control: the scan found {names:?}"
        );
        let unsent: Vec<&&str> = names.iter().filter(|n| !corpus.contains(**n)).collect();
        assert!(
            unsent.is_empty(),
            "prompt constants no phase sends: {unsent:?}"
        );
    }
}
