//! Native Rust ztools binary: a thin wrapper over the `ztools` library.
//!
//! Ported from `routines/src/main.rs` when the ztools modules moved into their
//! own crate.

fn main() {
    if let Err(e) = ztools::cli::run() {
        eprintln!("ztools: {e:#}");
        // A command whose exit status is part of its contract says which.
        let code = e
            .downcast_ref::<ztools::cli::Exit>()
            .map_or(1, |exit| i32::from(exit.code));
        std::process::exit(code);
    }
}
