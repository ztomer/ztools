"""What this machine can actually give a model, and whether it is already thrashing.

THE READING THAT STARTED THIS. A 64GB Mac, fifty minutes idle, was reported as
having "21.9GB available" and a 28.8GB model was said not to fit. Both were wrong.
21.9GB was `Pages free` from vm_stat, which is not available memory on macOS:

    free 26.7 + inactive 16.9 + speculative 2.0 = 45.6 GB   <- the real figure
    Pages free alone                            = 26.7 GB

macOS holds reclaimable memory in `inactive` and `speculative`, and a box with
7.2GB genuinely in use reads as nearly full if you look only at `free`.

THE DEEPER PROBLEM, which is why this module exists rather than a one-line fix.
`eval/samples.py` had already worked this out and written it down:

    After a sweep the page cache legitimately holds tens of GB of model weights
    and "available" drops to ~12GB on a perfectly healthy box, so a headroom
    threshold refuses to record on exactly the machine you most want readings
    from.

which is why sample cleanliness gates on PRESSURE -- swap and compressor -- and
not on headroom at all. The oversize refusal added later gated on headroom, the
quantity that docstring warns against, and so could refuse a model that would run
fine simply because the page cache still held the previous model's weights.

So both gates now ask the same two questions, in the same order, from one place:

    1. Is the machine ALREADY thrashing?  (swap, compressor -- unambiguous)
    2. Does the model fit in what is RECLAIMABLE?  (not merely in what is free)

Question 1 is the load-bearing one. A clean file-backed page holding model weights
evicts instantly and costs nothing; swap traffic and a full compressor are the
states where a timing describes the machine instead of the model.
"""

import sys
from typing import Optional, Tuple


class NotSupportedHere(RuntimeError):
    """Raised when asked to read memory on a platform this repo does not target.

    House rule #3: an unsupported platform is a HARD FAILURE, never a fallback
    that continues anyway. The eval path is macOS-only end to end -- osaurus,
    Metal, the GPU lock -- so a Linux "degrade gracefully" branch here would be
    dead code whose only effect is to turn a missing tool into a plausible
    number. Every quantity in this module comes from vm_stat.
    """


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise NotSupportedHere(
            f"memory readings require macOS vm_stat; this is {sys.platform}. "
            "The eval path (osaurus, Metal, the GPU lock) is macOS-only."
        )

#: Above these the machine is swapping or compressing hard, and any timing taken
#: on it describes the contention rather than the model. Calibrated against the
#: two observed states: during the 31GB leak swap was 12.88GB and the compressor
#: held 29.3GB; healthy after a full sweep they were 1.43GB and 5.1GB.
MAX_CLEAN_SWAP_GB = 8.0
MAX_CLEAN_COMPRESSOR_GB = 15.0

_BYTES_PER_GB = 1024**3
#: vm_stat reports counts of pages; Apple silicon uses 16KB pages. Read from the
#: header rather than assumed, because a wrong page size scales every figure here.
_DEFAULT_PAGE_SIZE = 16384


def _vm_stat() -> dict:
    """vm_stat's counters in GB, keyed by their own labels. {} when unreadable."""
    import re
    import subprocess

    _require_macos()
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
    page = _DEFAULT_PAGE_SIZE
    header = re.search(r"page size of (\d+)", out)
    if header:
        page = int(header.group(1))
    stats = {}
    for line in out.splitlines():
        match = re.match(r'^"?([A-Za-z][^:"]*)"?:\s+(\d+)', line)
        if match:
            stats[match.group(1).strip()] = int(match.group(2)) * page / _BYTES_PER_GB
    return stats


def pressure() -> Optional[Tuple[float, float]]:
    """(swap_gb, compressor_gb), or None when they cannot be read.

    None means "cannot tell" and every caller must treat it as such rather than
    as "fine". Inventing a healthy reading here is how a contended machine's
    numbers got recorded as clean in the first place.
    """
    try:
        import psutil

        swap_gb = psutil.swap_memory().used / _BYTES_PER_GB
        compressor_gb = _vm_stat().get("Pages occupied by compressor", 0.0)
        return swap_gb, compressor_gb
    except Exception:
        return None


def is_thrashing() -> Optional[bool]:
    """Is the machine paging hard enough that timings describe IT, not the model?

    Tri-state on purpose: None is "cannot tell", and is not the same as False.
    """
    reading = pressure()
    if reading is None:
        return None
    swap_gb, compressor_gb = reading
    return swap_gb > MAX_CLEAN_SWAP_GB or compressor_gb > MAX_CLEAN_COMPRESSOR_GB


def reclaimable_available_gb() -> float:
    """Memory a model can have, counting what the kernel would evict to give it.

    psutil's `available` on macOS is free + inactive + speculative, which already
    covers most of the page cache. What it MISSES is file-backed pages that are
    currently `active` -- clean pages holding a previously-loaded model's weights,
    evictable at no cost, and precisely what is resident right after a sweep.

    The active-file-backed estimate subtracts inactive+speculative from the
    file-backed total, which assumes every inactive and speculative page is
    file-backed. That over-subtracts, so the estimate is CONSERVATIVE: it can
    understate what is reclaimable, never overstate it. Understating means the
    gate occasionally refuses something that would have fit, which is the safe
    direction for a gate whose failure mode is producing a wrong number.

    Raises rather than degrading. An earlier version caught everything and
    returned psutil's figure alone, which turned "vm_stat is broken" into a
    number that looks fine and is simply wrong -- the failure this whole module
    exists to stop.
    """
    import psutil

    available = psutil.virtual_memory().available / _BYTES_PER_GB
    stats = _vm_stat()
    file_backed = stats.get("File-backed pages", 0.0)
    inactive = stats.get("Pages inactive", 0.0)
    speculative = stats.get("Pages speculative", 0.0)
    purgeable = stats.get("Pages purgeable", 0.0)
    active_file_backed = max(0.0, file_backed - inactive - speculative)
    return available + active_file_backed + purgeable
