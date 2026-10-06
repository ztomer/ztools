# Changelog

All notable changes to camoufox-rs are documented here.

## LOCAL CARRIED FIXES — 2026-10-04 (ztools)

**These are local patches to a vendored third-party crate, not upstream
camoufox-rs work.** They are not in this crate's upstream history and will be
lost on the next re-vendor unless carried forward deliberately. Nothing here
changes the public API's *behaviour*; every entry either corrects a claim that
contradicted the shipped code, or fixes a build/test that was broken.

1. **Readiness stream: docs corrected to stdout (the code was right).**
   `src/process/{mod.rs,readiness.rs}` and `README.md` claimed the Juggler
   sentinel (`"Juggler listening to the pipe"`) arrives on *stderr*. The patched
   binary emits it on **stdout** — stdout carried 31 bytes (exactly the sentinel
   line) while stderr carried 840 bytes with no Juggler line at all. The prose
   also named a sentinel string (`"Juggler pipe initialized"`) that occurs
   nowhere in `Camoufox.app/Contents/Resources/omni.ja`. Docs now match the
   binary; the read was deliberately **not** moved to stderr, which would break
   every launch (the ztools twitter collector depends on it).
2. **`ProcessError` payload fields renamed `stderr` → `captured`.** They never
   held stderr; they hold the readiness stream's output (stdout). The `Display`
   label `"stderr:\n"` became `"stdout:\n"`. Renaming removes the lie from the
   API surface instead of leaving it in the identifiers.
3. **Seven readiness tests un-red.** They piped stderr and nulled stdout,
   because they were written from the wrong docs, so they failed with
   `Io("child stdout is not piped")`. Helpers now pipe stdout, and the helpers
   and tests named `..._stderr...` were renamed (`spawn_echo_stdout`,
   `empty_stdout_returns_exited_before_ready`, `returns_error_when_stdout_not_piped`, …).
4. **New two-sided regression tests** `sentinel_on_stdout_is_ready` and
   `sentinel_on_stderr_alone_is_not_ready` in `src/process/readiness.rs`, so the
   stream choice is pinned from both sides and cannot be "corrected" back to
   match PROTOCOL.md. The two `returns_error_when_stdout_*` tests were also
   tightened to pipe stderr, so they now fail too if the read moves (they
   previously passed under either stream).
5. **`cargo check --all-targets` now compiles with default features.** Three
   tests in `src/api/main_frame.rs` used `crate::cli::*` inside `#[cfg(test)]`
   while `pub mod cli` is gated on the `cli` feature — 4× `E0433`, exit 101. They
   asserted the serde shape of `DaemonRequest`/`DaemonResponse` and the behaviour
   of `print_response`, so they moved to `src/cli/ipc.rs` and `src/cli/output.rs`
   test modules, where those types actually live. No test was deleted or disabled.
6. **`Cargo.lock` committed** (560 lines). The crate had none, so its dependency
   versions floated between builds; all deps currently resolve to newest.
7. **`docs/PROTOCOL.md` annotated, not rewritten.** It stays the upstream spec;
   a marked note at the top (and at § 3) records the stderr/stdout divergence so
   the spec is not mistaken for evidence about this build.
8. **157 `clippy` findings fixed by machine sweep** (365 → 208 unique findings,
   668 → 291 diagnostic records, at `all` + `pedantic` + `nursery`,
   `--all-targets --all-features`). The 1 `clippy::all` finding
   (`for_kv_map`, `src/protocol/client.rs`) went with it, so `cargo clippy
   --all-targets -- -D warnings` now **exits 0 in both feature modes** — this
   crate can meet `clippy::all`, which it could not before.
9. **`clippy::must_use_candidate` was EXCLUDED from that sweep, on measurement.**
   Its machine-applicable fix is `#[must_use]` on 54 `pub` fns. Applying it took
   `clippy::all` on this crate from 1 finding to **9** — the 9 are
   `unused_must_use` at call sites that legitimately discard the value (the
   `sendMayFail` pattern) — and it pushes warnings onto every future caller of a
   third-party API, including `ztools`, whose gate runs `-D warnings`. The sweep
   therefore ran with `-A clippy::must_use_candidate` and the 52 findings stand.
   Re-apply deliberately, as an API change reviewed on its own, with `let _ =`
   at the discarding sites. **A re-vendor that runs a bare `clippy --fix` will
   silently take the other branch.**
10. **All 10 `#[allow]` / `#[cfg_attr]` sites removed.** Six were **stale** —
    the lint could not fire (`tests/fixtures/mod.rs`'s four `start()` fns are
    called from `tests/integration.rs`; `CAMOU_CONFIG_CHUNK_SIZE_WINDOWS` is
    read in a `cfg!(windows)` branch that is compiled on every target;
    `FixtureServer`/`TestHarness` aside). Two were genuinely dead and were
    **deleted** (`FixtureServer.iframe_url`, `StatusServer.base_url` — assigned,
    never read; `TestHarness.closed` — a second `Arc` handle on a flag
    `MockTransport` already owns). One was a **feature gate in disguise**:
    `MainFrame::execution_context_handle`'s only reader is `cli::instance`, so
    the method is now `#[cfg(feature = "cli")]` rather than
    `pub(crate)` + `cfg_attr(allow(dead_code))`. The last was
    `#[allow(clippy::type_complexity)]` on `BrowserContext::new_main_frame`,
    where the lint fires on one *local binding*, not the signature — hence the
    function-level attribute. Fixed by naming the type (`SkippedAttach` /
    `SkippedAttaches` in `src/api/context.rs`), which is what the lint asks for.
11. **`[lints.clippy] all = { level = "warn", priority = -1 }` added to this
    crate's `Cargo.toml`** (upstream had no lint table). It states the policy
    this crate meets; `pedantic`/`nursery` are deliberately not declared, and
    their 208 remaining findings are held as shrink-only ceilings in
    `tools/vendor_lint_baseline.json`, enforced by
    `tools/vendor_lint_ratchet.py`. This table cannot soften a `-D warnings`
    gate — measured on a scratch crate, a command-line group beats it
    (`-D clippy::ptr_arg implied by -D warnings`, exit 101).

**Consequence for a re-vendor:** items 8–11 are lost, and the crate goes back to
365 findings with 10 suppressions and no lint table. The parent gate will not
notice — `.gatesrc` sets `GOH_EXCLUDE='vendor/'`, which `rust_gate.sh` forwards
to clippy and `check_no_allow.py` — but `tools/vendor_lint_ratchet.py` **will**,
because its baseline is pinned to today's counts. Re-run its `--record` and
re-read every `why` after any re-vendor; that re-read is the whole point of the
per-entry reasons.

## [Unreleased]

### Added

- **G1 — `cookies <instance_id>` command**: exports the full in-session cookie jar via the
  Juggler `Browser.getCookies` call. Includes `HttpOnly` cookies (which are inaccessible from
  JavaScript). `--json` returns full cookie objects (name, value, domain, path, httpOnly,
  secure, sameSite, expires, session, size). The exported jar can be passed directly to
  host-side HTTP clients (e.g. `curl --cookie`) to fetch session-gated content without
  requiring in-session XHR round-trips.

- **G3 — `navigate --wait-until load|domcontentloaded`**: the `navigate` command now accepts
  a `--wait-until` flag that blocks until the specified browser lifecycle event fires, bounded
  by `--timeout` seconds. Accepted values: `load` (fires after all resources on the page have
  loaded) and `domcontentloaded` (fires once the HTML is parsed, before sub-resources). Any
  other value is rejected with an error at dispatch time.

- **G4 — `navigate` reports `status_code`**: `--json` output now includes `status_code`, the
  final main-document HTTP status code after following all redirects. `status_code` is `null`
  when the status is uncapturable (e.g. `about:` pages, navigation that errors before a
  network response is received). Navigate never fails on 4xx/5xx responses — the result is
  always `"ok": true` and the caller inspects `status_code` to decide how to proceed.

### Motivation

These three gaps (G1, G3, G4) were identified during the legal-agent eCourts spike:
`docs/superpowers/findings/2026-05-29-ecourts-spike-findings.md` (in the `legal-agent` repo).
The eCourts portal gates case documents behind session cookies set by authenticated page
navigations; extracting those cookies and replaying them host-side (G1) eliminates the need
for in-browser XHR workarounds and unblocks robust, resumable corpus fetching. Deterministic
load-event waiting (G3) and redirect-aware status inspection (G4) provide the scaffolding
needed to drive the multi-step login→search→download flow reliably.
