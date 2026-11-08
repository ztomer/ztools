# Local patches to `vendor/camoufox-rs`

`vendor/camoufox-rs` is upstream's code, re-vendored wholesale when upstream moves.
These are the places this repo's copy deliberately DIFFERS from upstream. They live
here, outside the tree a sync replaces, so a re-vendor cannot drop them silently:
`tools/tests/test_vendor_patches.py` (gate step `python3 -m pytest tools/tests`)
fails while any patch below is not applied.

**Re-vendoring:** replace `vendor/camoufox-rs` with the new upstream tree, then
`git apply vendor/patches/camoufox-rs/*.patch` in the same commit. A patch that no
longer applies means upstream changed the code it touches: check whether upstream
fixed the defect itself (then delete the patch and its entry here) or re-cut it
against the new tree. Then re-record `tools/vendor_lint_baseline.jsonc`.

## `0001-order-404-test-by-channel.patch`

- **Upstream state:** unchanged as of 2026-10-17 (checked). Owner ruling: fix locally.
- **What:** `api::main_frame::tests::navigate_captures_404_status_and_returns_ok`
  sent the navigate ack, slept 10 ms, then sent `Network.responseReceived`, so the 404
  was seen only if it landed inside navigate's 80 ms test-build status window. It
  failed on CI run 38841488233 (`status_code must be Some(404)`, left `None`) and
  passed on 38899800424.
- **Fix:** both Network events are queued before the ack. The connection's single
  reader thread dispatches the mock channel in FIFO order, synchronously, so both
  handlers have run before the ack unblocks navigate. Ordering is by the channel,
  not by a clock.
- **Proof:** the old body with its sleep raised to 100 ms (a busy runner) fails;
  the new body with a 200 ms stall inserted between every send passes.
- **Upstream-worthy:** yes. Report it if upstream accepts issues; delete this patch
  when a re-vendor carries the fix.

## `0002-order-status-tests-by-channel.patch`

- **Upstream state:** unchanged as of 2026-10-19. Same class as 0001, same ruling.
- **What:** `navigate_captures_200_status`, `navigate_follows_redirect_chain_reports_final_status`
  and `navigate_subresource_redirect_does_not_hijack_chain` sent the navigate ack FIRST
  and the Network events after it, so their status counted only if every event landed
  inside navigate's 80 ms test-build status window. `navigate_redirect_does_not_leak_handlers`
  asserts no status, but its redirect chain ran only inside that window too.
- **Fix:** the ack is sent LAST in each responder; FIFO dispatch on the single reader
  thread means every handler has run before navigate is unblocked. The post-ack
  nav-id-match branch was never guaranteed by the old order either (the reader could
  dispatch the events before navigate published the nav id).
- **Proof:** the old bodies with a 100 ms stall after the ack fail (3 of 3); the new
  bodies with a 200 ms stall after every send pass 3/3 runs.
- **Upstream-worthy:** yes, together with 0001.

## `0003-nav-id-match-test.patch`

- **Upstream state:** the arm is upstream's code and this repo's copy of it was read on
  2026-10-19; no upstream case reaches it (this repo carries no upstream URL to check
  against, so the claim is scoped to the copy). Owner ruling: fix locally. The change is
  TEST-ONLY, so it cannot alter the crate's behaviour.
- **What:** `MainFrame::navigate`'s `Network.requestWillBeSent` handler compares the event's
  `navigationId` with the id the navigate ack published, and the arm that does the comparing
  -- both ids present -- is what decides whether a request belongs to THIS navigation. No
  case reached it: every status case sends its `Network.requestWillBeSent` BEFORE the ack, so
  the handler still sees "no id published yet" and takes the speculative arm. The ack-first
  order 0001/0002 removed could not reach it either -- the reader thread may dispatch the
  request before navigate publishes the id -- so it was reached by luck, never by
  construction. That arm is the only thing that stops a request of ANOTHER navigation from
  being counted as this navigation's main document, and the status `navigate` returns is that
  request's status: a change that accepted every request would report the wrong page's
  status, and the four existing status cases would stay green, because all four take the
  speculative arm.
- **Fix:** a case that reaches the arm BY CONSTRUCTION. The mock acks first, then waits
  until navigate has registered its `Page.eventFired` handler -- the lifecycle wait runs
  AFTER the id is published, so seeing that handler registered proves the id is published --
  and only then sends the Network events. The lifecycle event goes last, so the status poll
  starts only after every Network handler has run. A request whose navigation id does NOT
  match is ignored; the one whose id matches is the main document, so its status is the
  status reported.
- **Proof:** the arm was instrumented and counted. 0 hits across the six existing navigate
  status cases (`navigate_captures_404_status_and_returns_ok`,
  `navigate_captures_200_status`, `navigate_follows_redirect_chain_reports_final_status`,
  `navigate_subresource_redirect_does_not_hijack_chain`,
  `navigate_redirect_does_not_leak_handlers`,
  `navigate_status_code_is_none_when_no_response_event`); 6 hits across 20 runs of the old
  ack-first order, i.e. only 3 of 20 runs reached the arm at all; 40 hits across 20 runs of
  the new order (2 per run, 20/20 green). The new case is RED with the arm broken both
  ways -- "accept every request" reports `Some(500)` where `Some(200)` is expected, "reject
  every request" reports `None` -- and GREEN with the shipped logic.
- **Upstream-worthy:** yes, with 0001 and 0002: upstream's suite has no case that reaches
  this arm at all.

## `0004-vendored-tests-wait-on-budget-and-conditions.patch`

- **Upstream state:** the code is this repo's copy, read 2026-11-08; no upstream URL to
  check against. Owner ruling (same as 0001-0003): fix locally. TEST-ONLY, so it cannot
  alter the crate's behaviour.
- **What:** one wall-clock upper bound and a family of sleep-then-check cases.
  `navigate_wait_until_combined_timeout_is_single_bounded` asserted
  `elapsed < 400 ms + 150 ms` and failed on CI run 42453070012 attempt 1 at 552.7 ms
  (a loaded runner; the rerun passed). It was also blind to the very bug it names:
  double-counting costs `100 ms ack + 400 ms lifecycle = 500 ms`, which is UNDER that
  550 ms ceiling, so it was green with the bug present and red without it. In
  `protocol::client` seven cases did `sleep(100 ms)` ("give the reader a moment") before
  asserting what the reader had done (event delivered, stray message ignored, transport
  close seen), and one slept 200 ms before counting two global events.
- **Fix:** the combined-timeout case asserts the BUDGET the lifecycle wait was handed
  (`wait_for_lifecycle` names it in its timeout message): at least the ack delay is
  spent before it starts, so it is `<= timeout - ack delay` on any machine; the bug hands
  it the whole `timeout`. The lower bound stays, because a timeout never fires early.
  The client cases use `fence` (a marker event sent last is dispatched only after every
  earlier message, because the single reader thread is FIFO) and `wait_until` (a polled
  condition, 5 s only to turn a hang into a named failure).
- **Proof:** with a 200 ms late wake-up injected into `wait_for_lifecycle`'s timeout arm
  (what a starved runner does), the OLD case fails (`613.67ms (timeout 400ms)`) and the
  new one passes. With the lifecycle wait handed `timeout` instead of the remaining time
  (the double-counting bug) the new case is RED (`handed 400ms of a 400ms timeout`) and the
  old one is green. Real load did not reproduce the failure here (0 of 30 runs at 16
  spinners, 0 of 30 at 64, on a 16-core Mac), so the stall injection is the proof. The new
  suite passed 6 of 6 full runs under 16 spinners.
- **Left as they are, with reasons:** every remaining `elapsed < N` bound is a HANG
  guard with at least 10x slack over the thing it times (`graceful_shutdown_*` 5 s / 2 s
  against a 200 ms / 0.15 s event and a 5-minute / 10 s alternative; `returned too
  late` 5 s against 150-200 ms timeouts; `integration.rs` 10 s against a 30 s timeout,
  live browser, not in the gate); every `elapsed >=` bound is load-proof. The
  `integration.rs` 500 ms cookie settle needs a live Camoufox and is not run by the gate.
- **Upstream-worthy:** yes, with 0001-0003.
