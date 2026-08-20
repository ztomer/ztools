//! Native Rust ztools binary: a thin wrapper over the `ztools` library.
//!
//! Ported from `routines/src/main.rs` when the ztools modules moved into their
//! own crate.

fn main() {
    if let Err(e) = ztools::cli::run() {
        eprintln!("ztools: {e:#}");
        std::process::exit(1);
    }
}