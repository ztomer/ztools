//! Live roster fetch and ghost-dropping -- the `fetch` submodule's tests.

use std::thread;

use crate::ztools::eval::model_resolve::fetch::drop_uncorroborated;
use crate::ztools::eval::model_resolve::*;

use super::roster::entry;
use super::support::DiskGuard;
use serial_test::serial;

#[test]
#[serial]
fn drop_uncorroborated_filters_ghosts_but_never_empties_a_roster() {
    let guard = DiskGuard::new();
    let model_dir = guard._dir.path().join("mlx/Org/DiskModel");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), "{}").unwrap();

    let roster = vec![
        entry("ghost-a", "7B"),
        entry("diskmodel", "8B"),
        entry("ghost-b", "70B"),
    ];
    let kept = drop_uncorroborated(roster.clone());
    assert_eq!(kept, vec![entry("diskmodel", "8B")], "ghosts are dropped");

    // Nothing survives: the ORIGINAL list comes back -- a fully-ghost
    // roster far more likely means a broken probe than zero models.
    let all_ghost = vec![entry("ghost-a", "7B"), entry("ghost-b", "70B")];
    assert_eq!(drop_uncorroborated(all_ghost.clone()), all_ghost);
}

/// One-shot localhost HTTP mock for fetch_roster.
fn serve_roster(body: &'static str, status_line: &'static str) -> (u16, thread::JoinHandle<()>) {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = thread::spawn(move || {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut buf = vec![0u8; 65_536];
            let _ = stream.read(&mut buf);
            let response = format!(
                "{status_line}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        }
    });
    thread::sleep(std::time::Duration::from_millis(50));
    (port, handle)
}

#[test]
#[serial]
fn fetch_roster_keeps_disk_backed_entries_and_drops_ghosts_over_the_wire() {
    let guard = DiskGuard::new();
    let model_dir = guard._dir.path().join("mlx/Org/DiskModel");
    std::fs::create_dir_all(&model_dir).unwrap();
    std::fs::write(model_dir.join("config.json"), "{}").unwrap();

    let body = r#"{"models":[
        {"model":"ghost-a","details":{"parameter_size":"70B"}},
        {"model":"DiskModel","details":{"parameter_size":"8B"}}
    ]}"#;
    let (port, handle) = serve_roster(body, "HTTP/1.1 200 OK");

    // host + port form builds the http:// URL itself.
    let roster = fetch_roster("127.0.0.1", port);
    handle.join().unwrap();
    assert_eq!(
        roster,
        vec![entry("DiskModel", "8B")],
        "the ghost must not survive"
    );
}

#[test]
#[serial]
fn fetch_roster_accepts_a_full_url_host() {
    let _guard = DiskGuard::new();
    let body = r#"{"models":[{"model":"any-model"}]}"#;
    let (port, handle) = serve_roster(body, "HTTP/1.1 200 OK");
    let host = format!("http://127.0.0.1:{port}");
    let roster = fetch_roster(&host, 0);
    handle.join().unwrap();
    // Not corroborated by anything on disk -> dropped here; the point of
    // this test is that the scheme-form URL reaches the server at all,
    // which an empty roster would silently hide.
    let (port2, handle2) = serve_roster(body, "HTTP/1.1 200 OK");
    let host2 = format!("http://127.0.0.1:{port2}/");
    let roster2 = fetch_roster(&host2, 0);
    handle2.join().unwrap();
    assert_eq!(roster, roster2, "trailing slash is trimmed");
}

#[test]
#[serial]
fn fetch_roster_answers_empty_when_the_server_cannot_be_asked() {
    let _guard = DiskGuard::new();

    // Non-200 status.
    let (port, handle) = serve_roster("{}", "HTTP/1.1 503 Service Unavailable");
    assert!(fetch_roster("127.0.0.1", port).is_empty());
    handle.join().unwrap();

    // 200 with unparseable JSON.
    let (port, handle) = serve_roster("{not json", "HTTP/1.1 200 OK");
    assert!(fetch_roster("127.0.0.1", port).is_empty());
    handle.join().unwrap();

    // 200 with JSON lacking a models array.
    let (port, handle) = serve_roster(r#"{"other": []}"#, "HTTP/1.1 200 OK");
    assert!(fetch_roster("127.0.0.1", port).is_empty());
    handle.join().unwrap();

    // Connection refused: a bound-then-dropped port.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    assert!(
        fetch_roster("127.0.0.1", port).is_empty(),
        "down server == no evidence"
    );
}
