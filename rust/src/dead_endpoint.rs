//! A loopback endpoint that is guaranteed dead for as long as the test holds it.
//!
//! The tests that prove "an unreachable server is an error" used to dial a
//! literal port (`http://127.0.0.1:59999`). That port is inside the ephemeral
//! range (macOS 49152-65535, Linux 32768-60999), so any outgoing connection
//! anywhere on the machine could be handed it, and a busy machine eventually
//! is: the "dead" endpoint answered and `test_call_osaurus_invalid_host` failed
//! the push gate (EST.DEADPORT, 2026-11-15).
//!
//! [`DeadEndpoint`] holds a loopback port with a CONNECTED pair and no listener:
//! it binds `127.0.0.1:0` (without `SO_REUSEADDR`), listens, connects to
//! itself, accepts, and closes the listener. While the pair is held:
//! - a connect to the port is REFUSED at once: no socket listens there, and the
//!   only sockets on it belong to one four-tuple the SYN does not match;
//! - the port is in use, so the ephemeral allocator and `bind(0)` never hand it
//!   out, and a plain bind of the address fails with `EADDRINUSE`.
//!
//! Why not the obvious "bind and never listen" (probed 2026-11-15, macOS 27):
//! a SYN to a bound, CLOSED socket is DROPPED, not reset, so every connect
//! waits for its timeout instead of failing. The pair is what gets the refusal.
//! Limit, stated rather than hidden: on macOS (not Linux) a server that sets
//! `SO_REUSEADDR` and asks for this exact port can still take it while the
//! pair is held; nothing picks it by chance, which is the failure fixed here.
//!
//! std cannot bind a listener without `SO_REUSEADDR` (and the accepted socket
//! inherits it, which on Linux would let a reusing bind through), which is why
//! this uses `socket2`, a dev-dependency already in the graph through hyper at
//! the same version. `no_source_dials_a_literal_takeable_loopback_port` is the
//! gate that keeps a literal ephemeral port from coming back.

use socket2::{Domain, Socket, Type};
use std::net::{Ipv4Addr, SocketAddr, TcpStream};

/// A loopback port held by a connected pair with no listener, and its URL.
///
/// Keep the value alive for the whole test: dropping it frees the port.
pub struct DeadEndpoint {
    _held: (TcpStream, Socket),
    addr: SocketAddr,
    url: String,
}

impl DeadEndpoint {
    /// Take a fresh port on 127.0.0.1 and hold it with no listener.
    ///
    /// # Panics
    ///
    /// When the OS refuses a loopback socket, bind, listen, connect or accept:
    /// the test cannot prove anything about a dead endpoint without one.
    #[must_use]
    pub fn hold() -> Self {
        let listener = Socket::new(Domain::IPV4, Type::STREAM, None).expect("a TCP socket");
        listener
            .bind(&SocketAddr::from((Ipv4Addr::LOCALHOST, 0)).into())
            .expect("bind 127.0.0.1:0");
        listener.listen(1).expect("listen");
        let addr = listener
            .local_addr()
            .expect("the bound address")
            .as_socket()
            .expect("an inet address");
        let client = TcpStream::connect(addr).expect("connect to the holder");
        let (server, _) = listener.accept().expect("accept the holder");
        drop(listener);
        Self {
            _held: (client, server),
            addr,
            url: format!("http://{addr}"),
        }
    }

    /// `http://127.0.0.1:<port>`, refused for as long as `self` lives.
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    /// The held address.
    #[must_use]
    pub const fn addr(&self) -> SocketAddr {
        self.addr
    }
}

/// The first loopback URL literal in `text` whose port another process could hold.
///
/// It matches `"http(s)://<loopback>:<port>` with the port inside either OS's
/// ephemeral range (>= 32768). Port 0 and the privileged ports are not
/// takeable by an ordinary process.
///
/// # Panics
///
/// Never in practice: the only `expect` is on compiling a constant regex.
#[must_use]
pub fn source_has_takeable_port(text: &str) -> Option<String> {
    let re = regex::Regex::new(r#""https?://(?:127\.0\.0\.1|localhost|\[::1\]):(\d+)"#)
        .expect("static regex");
    re.captures_iter(text).find_map(|c| {
        let port: u32 = c[1].parse().ok()?;
        (port >= 32_768).then(|| c[0].to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::{DeadEndpoint, source_has_takeable_port};
    use socket2::{Domain, Socket, Type};
    use std::io::ErrorKind;
    use std::net::{TcpListener, TcpStream};
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    #[test]
    fn a_dead_endpoint_refuses_a_connect() {
        let dead = DeadEndpoint::hold();
        let err = TcpStream::connect_timeout(&dead.addr(), Duration::from_secs(2))
            .expect_err("nothing listens on a dead endpoint");
        assert_eq!(err.kind(), ErrorKind::ConnectionRefused, "{err:?}");
        assert_eq!(dead.url(), format!("http://{}", dead.addr()));
    }

    #[test]
    fn no_plain_bind_can_take_a_dead_endpoints_port() {
        let dead = DeadEndpoint::hold();
        let rival = Socket::new(Domain::IPV4, Type::STREAM, None).expect("a TCP socket");
        let err = rival
            .bind(&dead.addr().into())
            .expect_err("the held port is in use");
        assert_eq!(err.kind(), ErrorKind::AddrInUse, "{err:?}");
        if cfg!(target_os = "linux") {
            // Linux refuses even a reusing bind: the holder never set SO_REUSEADDR.
            assert!(
                TcpListener::bind(dead.addr()).is_err(),
                "a reusing listener took it"
            );
        }
        let err = TcpStream::connect_timeout(&dead.addr(), Duration::from_secs(2))
            .expect_err("still dead");
        assert_eq!(err.kind(), ErrorKind::ConnectionRefused, "{err:?}");
    }

    #[test]
    fn the_gate_sees_a_takeable_port_and_passes_safe_ones() {
        let lit = |port: u32| format!("let u = \"http://127.0.0.1:{port}\";");
        assert!(source_has_takeable_port(&lit(59_999)).is_some());
        assert!(source_has_takeable_port(&lit(32_768)).is_some());
        assert!(source_has_takeable_port(&format!("\"http://localhost:{}\"", 54_055)).is_some());
        assert_eq!(source_has_takeable_port(&lit(1)), None);
        assert_eq!(source_has_takeable_port(&lit(1337)), None);
        assert_eq!(source_has_takeable_port(&lit(0)), None);
        // Prose quoting an error message is not an endpoint a test dials.
        assert_eq!(
            source_has_takeable_port(&format!("/// 127.0.0.1:{}, Os", 54_055)),
            None
        );
    }

    fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("readable source dir") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                rust_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    #[test]
    fn no_source_dials_a_literal_takeable_loopback_port() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        rust_files(&root.join("src"), &mut files);
        rust_files(&root.join("tests"), &mut files);
        assert!(files.len() > 50, "the walk found the tree: {}", files.len());
        let hits: Vec<String> = files
            .iter()
            .filter_map(|f| {
                let text = std::fs::read_to_string(f).expect("readable source");
                source_has_takeable_port(&text).map(|hit| format!("{}: {hit}", f.display()))
            })
            .collect();
        assert_empty!(
            hits,
            "use crate::dead_endpoint::DeadEndpoint, not a literal ephemeral port"
        );
    }
}
