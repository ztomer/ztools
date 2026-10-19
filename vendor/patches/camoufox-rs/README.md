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
