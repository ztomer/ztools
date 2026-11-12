#!/usr/bin/env bash
# Cargo's runner for this repo (.cargo/config.toml): cargo invokes it as `runner <exe> <args>` for
# every binary it RUNS -- `cargo test` binaries and `cargo run` alike.
#
# It gives TEST binaries a HOME under rust/target/test-home and leaves every other run alone.
# The sandbox used to be a `[env] HOME` entry in the config, and cargo applies `[env]` to
# `cargo run` too: `cargo run -- models scout` read a machine with no installed models and no free
# disk (ZT.CARGORUNHOME, 2026-11-12). A test seam must not reach the real tool.
#
# What is a test binary: cargo keeps every test, bench and integration-test executable in a
# `deps/` directory (target/<profile>/deps/name-hash; also under --target and llvm-cov's target
# dir) or, with the newer build-dir layout, in an `out/` directory
# (<build-dir>/<hash>/<profile>/build/<pkg>/<hash>/out/name-hash -- the layout this machine's
# cargo uses, measured 2026-11-12); `cargo run` executes target/<profile>/<bin> or
# target/<profile>/examples/<name>. The names are an allowlist of test locations, so a layout
# cargo adds later fails LOUD (test_home_is_sandboxed and the scratch-crate check go red) rather
# than quietly sandboxing a `cargo run`.
# `rust/tests/test_home_is_sandboxed.rs` fails if a test process does not see the sandbox, and
# `tools/tests/test_cargo_run_home.py` fails if a `cargo run` does.
set -euo pipefail
exe="$1"
case "$(basename "$(dirname "$exe")")" in
deps | out)
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  HOME="$root/rust/target/test-home"
  export HOME
  ;;
esac
exec "$@"
