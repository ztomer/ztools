//! The forecast endpoint, as configuration, asserted on the wire.
//!
//! Split out of `cli_dispatch_ztools.rs` for the file-length cap and kept with
//! its own subject: everything here is about proving that a CONFIG key changes
//! where a request goes, which is a claim about the wire and not about the file
//! the test wrote.
//!
//! The stub records every request head it accepts, and the recording is the
//! point: the request line carries the path and the query, the `Host` header
//! carries the origin, and NEITHER can be checked by re-reading the config.
//! A key that parses, defaults, and is then ignored looks exactly like a working
//! override until you watch where the request goes.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

use super::{
    LLM_EVENTS, await_stub, fresh, stdout_of, stub_server, weekend_config, write_config, ztool,
};

/// Read a request HEAD, looping until the blank line that ends it: one `read`
/// is a guess about TCP segmentation, and an assertion made against half a
/// request line is a flake with a schedule.
fn read_head(stream: &mut std::net::TcpStream) -> String {
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
        .ok();
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
    }
    String::from_utf8_lossy(&buf)
        .lines()
        .take_while(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase()
}

/// The `daily` block for EXACTLY the window the client asked for, echoing the
/// two dates out of the request line so the stub never has to know what weekend
/// it is.
///
/// The temperatures are ones no GTA forecast produces (-3.5 and 21.5 across a
/// single weekend), which is the point: a plan carrying one of them can only
/// have got it from this stub, and the second day carries rain so the parsed
/// condition is exercised too.
fn forecast_json(request_line: &str) -> String {
    let date = |key: &str| {
        request_line
            .split(key)
            .nth(1)
            .and_then(|rest| rest.split('&').next())
            .unwrap_or("2026-01-02")
            .to_string()
    };
    format!(
        r#"{{"daily":{{"time":["{}","{}"],"temperature_2m_max":[-3.5,21.5],"precipitation_sum":[0,4.0]}}}}"#,
        date("start_date="),
        date("end_date=")
    )
}

/// A stub that answers a forecast request with a `daily` block, recording every
/// request head it saw.
///
/// Returns the port and the shared record, so the assertion runs after the
/// process has exited.
fn forecast_stub() -> (u16, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&seen);
    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let recorder = Arc::clone(&recorder);
            thread::spawn(move || {
                let mut stream = stream;
                let head = read_head(&mut stream);
                let request_line = head.lines().next().unwrap_or("").to_string();
                recorder.lock().unwrap().push(head);
                let reply = forecast_json(&request_line);
                let http = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{reply}",
                    reply.len()
                );
                let _ = stream.write_all(http.as_bytes());
                let _ = stream.flush();
            });
        }
    });
    await_stub(port);
    (port, seen)
}

/// The forecast endpoint is CONFIGURATION, and this is the assertion that was
/// impossible while it was a literal inside the client.
///
/// Two separate claims, both about the WIRE rather than about the config file:
/// the request the process actually made carries the configured host and
/// Open-Meteo's path with the weekend window in it, and the forecast the plan
/// renders is the stub's answer.
#[test]
fn the_forecast_request_carries_the_configured_host_and_the_plan_reads_its_answer() {
    let home = fresh("weekend-endpoint");
    // Two stubs on DIFFERENT ports: the model and search traffic goes to one,
    // the forecast to the other. Sharing a port would make the origin
    // unfalsifiable — a planner that built its forecast URL from the wrong config
    // field would reach the right server anyway and the test would pass.
    let model_port = stub_server(LLM_EVENTS);
    let (weather_port, seen) = forecast_stub();
    write_config(&home, &weekend_config(model_port, weather_port, &home));

    let md_out = home.join("weekend.md");
    let out = ztool(&home)
        .arg("weekend-plan")
        .arg("--md-out")
        .arg(&md_out)
        .output()
        .unwrap();
    stdout_of(&out);

    let heads = seen.lock().unwrap().clone();
    let request = heads
        .iter()
        .find(|h| h.starts_with("get /v1/forecast"))
        .unwrap_or_else(|| panic!("the planner made no forecast request at all; saw: {heads:#?}"));
    for fragment in [
        // The ORIGIN, which is the override. It is the model's port on neither
        // line, so a forecast URL built from `osaurus_url` cannot satisfy it.
        &format!("host: 127.0.0.1:{weather_port}"),
        // The endpoint contract, which is NOT overridable: an override
        // substitutes the host, not the API.
        "get /v1/forecast?",
        "latitude=43.8361",
        "longitude=-79.4982",
        "daily=temperature_2m_max,precipitation_sum",
        "timezone=america/new_york",
    ] {
        assert!(
            request.contains(fragment),
            "{fragment:?} missing from {request}"
        );
    }
    let date = |key: &str| {
        request
            .split(key)
            .nth(1)
            .and_then(|rest| rest.split(['&', ' ']).next())
            .unwrap_or_else(|| panic!("no {key} in {request}"))
            .to_string()
    };
    let (start, end) = (date("start_date="), date("end_date="));
    assert!(
        start < end,
        "the window must reach the endpoint in order, got {start}..{end}"
    );

    let doc = std::fs::read_to_string(&md_out).unwrap();
    assert!(
        doc.contains("-3.5°C") && doc.contains("21.5°C"),
        "the plan must carry the STUB's forecast: {doc}"
    );
}
