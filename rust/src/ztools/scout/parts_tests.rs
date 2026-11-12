//! The catalog parser, the local readers and the fixture format, apart.

use super::*;

const SWIFT: &str = r#"
    nonisolated fileprivate static let curatedSuggestedModels: [MLXModel] = [
        curated(
            id: "OsaurusAI/Raptor-0.6-4B-JANG_6M",
            description: "Raptor 0.6 4B.",
            modelType: "spark2_5",
            useCase: .general
        ),
        curated(
            id: "OsaurusAI/NoType",
            description: "no model type"
        ),
    ]
    nonisolated static let retiredOsaurusOrgIds: Set<String> = [
        "osaurusai/ornith-1.0-9b-mxfp8",
        // "osaurusai/commented-out",
        "osaurusai/raptor-v0.5-8b-a1b-jang_6m",
    ]
"#;

#[test]
fn the_catalog_source_yields_curated_types_and_the_retired_list() {
    let v = catalog::parse(SWIFT).expect("parses");
    assert_eq!(v["curated"][0]["id"], "OsaurusAI/Raptor-0.6-4B-JANG_6M");
    assert_eq!(v["curated"][0]["model_type"], "spark2_5");
    assert!(v["curated"][1]["model_type"].is_null());
    assert_eq!(
        v["retired"],
        serde_json::json!([
            "osaurusai/ornith-1.0-9b-mxfp8",
            "osaurusai/raptor-v0.5-8b-a1b-jang_6m"
        ])
    );
}

#[test]
fn a_catalog_source_that_moved_shape_refuses_rather_than_reading_empty() {
    let no_list = SWIFT.replace("retiredOsaurusOrgIds", "renamed");
    assert!(
        catalog::parse(&no_list)
            .unwrap_err()
            .contains("retiredOsaurusOrgIds")
    );
    let no_curated = SWIFT.replace("curated(", "entry(");
    assert!(catalog::parse(&no_curated).unwrap_err().contains("curated"));
}

#[test]
fn local_facts_are_read_from_df_the_plist_and_the_models_root() {
    let df = "Filesystem 1024-blocks Used Available Capacity Mounted on\n/dev/x 100 50 40 50% /\n";
    assert_eq!(local::parse_df(df), Some(40 * 1024));
    assert_eq!(local::parse_df("header only\n"), None);
    let plist = "<dict><key>CFBundleShortVersionString</key>\n\t<string>1.0.1</string></dict>";
    assert_eq!(local::plist_version(plist).as_deref(), Some("1.0.1"));
    assert_eq!(local::plist_version("<dict></dict>"), None);
    let dir = tempfile::tempdir().expect("tempdir");
    let model = dir.path().join("org/Model-4bit");
    std::fs::create_dir_all(&model).expect("mkdir");
    std::fs::write(model.join("config.json"), r#"{"model_type": "qwen3_5"}"#).expect("write");
    std::fs::create_dir_all(dir.path().join("org/empty")).expect("mkdir");
    assert_eq!(
        local::installed(dir.path()),
        [local::Installed {
            repo: "org/Model-4bit".into(),
            model_type: Some("qwen3_5".into())
        }]
    );
    assert!(local::free_bytes(dir.path()).is_some_and(|b| b > 0));
}

#[test]
fn a_fixture_line_is_a_body_or_a_refusal_and_anything_unrecorded_refuses() {
    let f = Fixture::parse(
        "{\"key\":\"a\",\"body\":[1]}\n\n{\"key\":\"b\",\"refused\":\"HTTP 429\"}\n",
    )
    .expect("parses");
    assert_eq!(f.get("a"), Ok(serde_json::json!([1])));
    assert_eq!(f.get("b"), Err("HTTP 429".into()));
    assert_eq!(f.get("c"), Err("not recorded".into()));
    assert!(f.body("a").is_some() && f.body("b").is_none());
    assert!(Fixture::parse("{\"key\":\"a\"}").is_err());
    assert!(Fixture::parse("{\"body\":1}").is_err());
    assert!(Fixture::parse("not json").is_err());
    assert!(Fixture::load(std::path::Path::new("/nonexistent/feeds.jsonl")).is_err());
}

#[test]
fn a_live_feed_that_cannot_connect_refuses_and_records_the_refusal() {
    let dir = tempfile::tempdir().expect("tempdir");
    let rec = dir.path().join("feeds.jsonl");
    let live = Live {
        hf: "http://127.0.0.1:9".into(),
        osaurus_url: "http://127.0.0.1:9".into(),
        record: Some(rec.clone()),
    };
    let key = feeds::model_key("org/x");
    assert!(live.get(&key).is_err());
    let recorded = Fixture::load(&rec).expect("recorded");
    assert!(recorded.get(&key).unwrap_err().starts_with("no answer"));
}

/// Serve `body` once on a local port and return the base URL.
fn serve_once(status: &str, body: &str) -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let reply = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    std::thread::spawn(move || {
        if let Ok((mut s, _)) = listener.accept() {
            let mut buf = [0u8; 4096];
            let _ = s.read(&mut buf);
            let _ = s.write_all(reply.as_bytes());
        }
    });
    format!("http://{addr}")
}

#[test]
fn a_live_answer_is_reduced_before_it_is_read_or_recorded() {
    let dir = tempfile::tempdir().expect("tempdir");
    let rec = dir.path().join("feeds.jsonl");
    let card = r#"{"id":"org/x","_id":"abc","sha":"def","tags":["mlx","base_model:org/base"],"downloads":3}"#;
    let live = Live {
        hf: serve_once("200 OK", card),
        osaurus_url: String::new(),
        record: Some(rec.clone()),
    };
    let key = feeds::model_key("org/x");
    let got = live.get(&key).expect("answers");
    assert_eq!(
        got,
        serde_json::json!({"id": "org/x", "downloads": 3, "tags": ["base_model:org/base"]})
    );
    assert_eq!(Fixture::load(&rec).expect("recorded").get(&key), Ok(got));
    let refusing = Live {
        hf: serve_once("429 Too Many Requests", "{}"),
        osaurus_url: String::new(),
        record: None,
    };
    assert_eq!(refusing.get(&key), Err("HTTP 429 Too Many Requests".into()));
}

#[test]
#[serial_test::serial]
fn the_scouts_health_read_is_made_under_the_gpu_lock() {
    let _env = crate::test_env::TestEnv::new();
    let fake = crate::ztools::residency::fake::FakeOsaurus::start("{}");
    let live = Live {
        hf: String::new(),
        osaurus_url: fake.url.clone(),
        record: None,
    };
    assert!(live.get(feeds::HEALTH).is_ok());
    assert!(!fake.calls().is_empty(), "the fake was asked");
    assert_eq!(fake.unlocked_requests(), Vec::<String>::new());
}
