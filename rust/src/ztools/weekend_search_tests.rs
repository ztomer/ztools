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
    assert!(results[0].title.is_empty());
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
    let server = serve_html(challenge);
    let results = search::search_duckduckgo_html("kids events", &server);
    assert!(
        results.is_empty(),
        "a WAF wall must yield no results, not parse the wall page: {results:?}"
    );
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
