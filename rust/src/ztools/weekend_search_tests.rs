use super::*;

const TITLED_HTML: &str = "<html><body><h2 class=\"result__title\">\
<a rel=\"nofollow\" class=\"result__a\" href=\"https://example.com/events/vaughan-fall-fair\">Vaughan Fall Fair</a>\
</h2><a class=\"result__snippet\" href=\"https://example.com/s\">\u{201c}Crush\u{201d} festival returns this weekend</a>\
<h2 class=\"result__title\"><a class=\"result__a\" href=\"https://example.com/aspenski\">Aspen ski school</a></h2>\
<a class=\"result__snippet\">Opens for the season</a></body></html>";

#[test]
fn parse_pairs_titles_with_snippets_in_position() {
    let results = search::parse_results_from_html(TITLED_HTML);
    assert_eq!(results.len(), 2, "{results:?}");
    assert_eq!(results[0].title, "Vaughan Fall Fair");
    assert_eq!(
        results[0].href,
        "https://example.com/events/vaughan-fall-fair"
    );
    assert_eq!(
        results[0].body,
        "\u{201c}Crush\u{201d} festival returns this weekend"
    );
    assert_eq!(results[1].title, "Aspen ski school");
    assert_eq!(results[1].body, "Opens for the season");
}

#[test]
fn parse_snippet_only_markup_yields_body_only_results() {
    let html = "<div><a class=\"result__snippet\">Vaughan Fall Fair returns this weekend</a></div>";
    let results = search::parse_results_from_html(html);
    assert_eq!(results.len(), 1);
    assert_empty!(&results[0].title);
    assert_eq!(results[0].body, "Vaughan Fall Fair returns this weekend");
}

#[test]
fn parse_unescapes_entities_in_titles_and_hrefs() {
    let html = "<a class=\"result__a\" href=\"https://x.com/?a=1&amp;b=2\">Tom &amp; Jerry &#x27;s Show</a>";
    let results = search::parse_results_from_html(html);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].title, "Tom & Jerry 's Show");
    assert_eq!(results[0].href, "https://x.com/?a=1&b=2");
}

#[test]
fn search_reports_a_challenge_instead_of_silent_empty() {
    let challenge =
        "<html><body>challenges.cloudflare.com just a moment verifying browser</body></html>";
    let urls = urls(
        serve_html(challenge),
        serve_html(challenge),
        serve_html(challenge),
    );
    let outcome = search::search_engines("Vaughan fall fair", &urls);
    assert!(
        outcome.results.is_empty(),
        "a WAF wall must yield no results, not parse the wall page: {outcome:?}"
    );
    assert_eq!(
        outcome.verdicts,
        [
            search::EngineVerdict::Blocked,
            search::EngineVerdict::Blocked,
            search::EngineVerdict::Blocked
        ],
        "{outcome:?}"
    );
    assert!(outcome.starved_by_bot_wall());
}

#[test]
fn a_walled_duckduckgo_falls_through_to_bing() {
    let challenge = "<html><body><div class=\"anomaly-modal\">checking</div></body></html>";
    let urls = urls(serve_html(challenge), serve_html(BING_HTML), dead());
    let outcome = search::search_engines("Vaughan fall fair", &urls);
    assert_eq!(outcome.results.len(), 2, "{outcome:?}");
    assert_eq!(outcome.results[0].title, "Vaughan Fall Fair 2026");
    assert_eq!(outcome.results[0].href, "https://example.com/fair?a=1&b=2");
    assert_eq!(
        outcome.results[0].body,
        "Rides, animals & a corn maze this weekend in Vaughan."
    );
    assert_eq!(
        outcome.verdicts,
        [
            search::EngineVerdict::Blocked,
            search::EngineVerdict::Answered(2),
            search::EngineVerdict::Skipped
        ]
    );
    assert!(!outcome.starved_by_bot_wall(), "Bing rescued the query");
}

#[test]
fn an_answering_duckduckgo_never_consults_bing() {
    let urls = urls(serve_html(TITLED_HTML), dead(), dead());
    let outcome = search::search_engines("Vaughan fall fair", &urls);
    assert_eq!(outcome.results.len(), 2);
    assert_eq!(
        outcome.verdicts,
        [
            search::EngineVerdict::Answered(2),
            search::EngineVerdict::Skipped,
            search::EngineVerdict::Skipped
        ]
    );
}

#[test]
fn unreachable_engines_are_recorded_as_such() {
    let outcome = search::search_engines("Vaughan fall fair", &urls(dead(), dead(), dead()));
    assert_empty!(&outcome.results);
    assert_eq!(
        outcome.verdicts,
        [
            search::EngineVerdict::Unreachable,
            search::EngineVerdict::Unreachable,
            search::EngineVerdict::Unreachable
        ]
    );
    assert!(!outcome.starved_by_bot_wall(), "unreachable is not a wall");
}

/// Real Bing markup shape (2026-09-19): `li.b_algo` > `h2 > a` + `p`, the
/// `h2` carrying an empty class and the first href a `ck/a` redirect whose
/// `u=a1<base64url>` payload is the destination.
const BING_HTML: &str = "<html><body><ol id=\"b_results\">\
<li class=\"b_algo\" data-id iid=\"SERP.1\"><div class=\"b_title\"><h2 class=\"\"><a target=\"_blank\" href=\"https://www.bing.com/ck/a?!&amp;&amp;p=30cd&amp;u=a1aHR0cHM6Ly9leGFtcGxlLmNvbS9mYWlyP2E9MSZiPTI&amp;ntb=1\" h=\"ID=1\">Vaughan <strong>Fall Fair</strong> 2026</a></h2></div>\
<div class=\"b_caption\"><p class=\"b_lineclamp2\"><span class=\"algoSlug_icon\">Sep 1</span>Rides, animals &amp; a corn maze this weekend in Vaughan.</p></div></li>\
<li class=\"b_algo\"><h2><a href=\"https://example.com/zoo\">Toronto Zoo</a></h2><p>Family programs.</p></li>\
<li class=\"b_ad\"><h2><a href=\"https://ads.example.com\">Sponsored</a></h2></li>\
</ol></body></html>";

/// The real page (2026-09-19): ten results AND `challenges.cloudflare.com`
/// in a script's domain list. The markers explain an empty page; they do
/// not overrule a parsed one.
#[test]
fn a_results_page_that_merely_mentions_a_challenge_host_is_an_answer() {
    let html = format!(
        "<html><script>var allow=[\"login.live.com\",\"challenges.cloudflare.com\"];</script>{}",
        BING_HTML.trim_start_matches("<html>")
    );
    let outcome = search::search_engines(
        "Vaughan fall fair",
        &urls(dead(), serve_html(&html), dead()),
    );
    assert_eq!(
        outcome.verdicts[1],
        search::EngineVerdict::Answered(2),
        "{outcome:?}"
    );
}

#[test]
fn bing_parser_reads_only_algo_blocks_and_strips_inline_markup() {
    let results = search_parse::parse_bing_results(BING_HTML);
    assert_eq!(results.len(), 2, "ads are not results: {results:?}");
    assert_eq!(results[1].title, "Toronto Zoo");
    assert_eq!(results[1].body, "Family programs.");
}

fn urls(duckduckgo: String, bing: String, brave: String) -> search::EngineUrls {
    search::EngineUrls {
        duckduckgo,
        bing,
        brave,
    }
}

fn dead() -> String {
    "http://127.0.0.1:1/".to_string()
}

/// Brave's live markup (2026-09-19): Svelte class hashes, the title in the
/// `title` attribute of the `search-snippet-title` div, body in `content`.
const BRAVE_HTML: &str = "<html><body><div id=\"results\">\
<div class=\"snippet svelte-jmfu5f\" data-pos=\"0\" data-type=\"web\"><div class=\"result-body\"><a href=\"https://www.vaughan.ca/events?a=1&amp;b=2\" class=\"svelte-14r20fy l1\"><div class=\"site-name-content\">City of Vaughan</div><div class=\"title search-snippet-title line-clamp-1 svelte-14r20fy\" title=\"Events | City of Vaughan\">Events | City of Vaughan</div></a><div class=\"generic-snippet\"><div class=\"content desktop-default-regular t-primary svelte-1cwdgg3\"><!--[-1--><!--]--> Community Events &amp; Fall Fair <b>Sept 20</b></div></div></div></div>\
<div class=\"snippet svelte-jmfu5f\" data-pos=\"1\" data-type=\"web\"><a href=\"https://example.com/zoo\"><div class=\"title search-snippet-title\" title=\"Toronto Zoo\">Toronto Zoo</div></a><div class=\"description desktop-default-regular svelte-1ajsqxo\">Family programs.</div></div>\
<div class=\"snippet\" data-type=\"news\"><a href=\"https://example.com/news\"><div class=\"title search-snippet-title\" title=\"Not a web hit\">x</div></a></div>\
</div></body></html>";

#[test]
fn brave_parser_reads_web_blocks_only_and_takes_the_title_attribute() {
    let results = search_parse::parse_brave_results(BRAVE_HTML);
    assert_eq!(
        results.len(),
        2,
        "news blocks are not web hits: {results:?}"
    );
    assert_eq!(results[0].title, "Events | City of Vaughan");
    assert_eq!(results[0].href, "https://www.vaughan.ca/events?a=1&b=2");
    assert_eq!(results[0].body, "Community Events & Fall Fair Sept 20");
    assert_eq!(results[1].body, "Family programs.");
}

#[test]
fn a_walled_bing_falls_through_to_brave() {
    let wall = "<html><body><div class=\"anomaly-modal\">checking</div></body></html>";
    let urls = urls(serve_html(wall), serve_html(wall), serve_html(BRAVE_HTML));
    let outcome = search::search_engines("Vaughan fall fair", &urls);
    assert_eq!(outcome.results.len(), 2, "{outcome:?}");
    assert_eq!(
        outcome.verdicts,
        [
            search::EngineVerdict::Blocked,
            search::EngineVerdict::Blocked,
            search::EngineVerdict::Answered(2)
        ]
    );
    assert!(!outcome.starved_by_bot_wall());
}

/// Bing's soft wall, verbatim from 2026-10-10: a well-formed results page for
/// "kids activities Vaughan October 2026" that is about something else.
const BING_OFF_TOPIC_HTML: &str = "<html><body><ol id=\"b_results\">\
<li class=\"b_algo\"><h2><a href=\"https://www.youtubekids.com/\">YouTube Kids</a></h2><p>YouTube Kids provides a more contained environment for kids to explore.</p></li>\
<li class=\"b_algo\"><h2><a href=\"https://pbskids.org/\">PBS KIDS</a></h2><p>Play free educational games and watch videos with your favorite PBS KIDS characters.</p></li>\
<li class=\"b_algo\"><h2><a href=\"https://www.cbc.ca/kids/\">Play Games, Watch Video, Explore | CBC Kids</a></h2><p>Games and videos for kids.</p></li>\
</ol></body></html>";

/// Brave's answer to the same query on the same day.
const BRAVE_ON_TOPIC_HTML: &str = "<html><body><div id=\"results\">\
<div class=\"snippet\" data-type=\"web\"><a href=\"https://childslife.ca/vaughan\"><div class=\"title search-snippet-title\" title=\"Vaughan Events &amp; Activities Guides for Kids &amp; Families | Child's Life\">x</div></a><div class=\"content\">Halloween Dance Party in Vaughan on October 31.</div></div>\
<div class=\"snippet\" data-type=\"web\"><a href=\"https://www.tripadvisor.ca/vaughan\"><div class=\"title search-snippet-title\" title=\"THE 10 BEST Things to Do in Vaughan with Kids (2026)\">x</div></a><div class=\"content\">Family attractions.</div></div>\
</div></body></html>";

/// A page that parses into results is not thereby an answer to the QUESTION.
/// An off-topic answer is a wall in disguise: recorded as walled (so the
/// learned order demotes the engine) and the next engine is asked -- where it
/// used to be "Answered(10)", Brave was never consulted, and the plan's
/// corpus became 88 lines about `YouTube`, condos and Grand Theft Auto.
#[test]
fn an_off_topic_answer_is_a_wall_and_the_next_engine_is_asked() {
    let urls = urls(
        dead(),
        serve_html(BING_OFF_TOPIC_HTML),
        serve_html(BRAVE_ON_TOPIC_HTML),
    );
    let outcome = search::search_engines("kids activities Vaughan October 2026", &urls);
    assert_eq!(
        outcome.verdicts,
        [
            search::EngineVerdict::Unreachable,
            search::EngineVerdict::Blocked,
            search::EngineVerdict::Answered(2)
        ],
        "{outcome:?}"
    );
    assert!(
        outcome.results.iter().all(|r| r.title.contains("Vaughan")),
        "only the on-topic answer reaches the corpus: {outcome:?}"
    );
}

/// When EVERY engine answers off-topic there is no answer, and the query is
/// starved by a wall -- never padded with the junk.
#[test]
fn off_topic_answers_from_every_engine_starve_the_query() {
    // Brave's soft wall on 2026-10-10: "GTA ..." answered with Grand Theft Auto.
    let brave_off_topic = "<html><body><div id=\"results\">\
<div class=\"snippet\" data-type=\"web\"><a href=\"https://www.rockstargames.com/gta-online\"><div class=\"title search-snippet-title\" title=\"GTA Online - Rockstar Games\">x</div></a><div class=\"content\">Experience GTA Online.</div></div>\
</div></body></html>";
    let urls = urls(
        dead(),
        serve_html(BING_OFF_TOPIC_HTML),
        serve_html(brave_off_topic),
    );
    let outcome = search::search_engines("kids activities Vaughan October 2026", &urls);
    assert_empty!(&outcome.results);
    assert_eq!(
        outcome.verdicts,
        [
            search::EngineVerdict::Unreachable,
            search::EngineVerdict::Blocked,
            search::EngineVerdict::Blocked
        ],
        "{outcome:?}"
    );
    assert!(outcome.starved_by_bot_wall(), "{outcome:?}");
}

fn serve_html(body: &str) -> String {
    use std::io::{Read, Write};

    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let body = body.to_string();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 1024];
        let _ = stream.read(&mut buf);
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let _ = stream.write_all(resp.as_bytes());
    });
    format!("http://{addr}")
}
