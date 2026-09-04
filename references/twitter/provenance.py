"""Which backend produced a summary, and whether it was degraded.

Class C9 in `docs/REPORT_WEAKNESS_CLASSES.md`. The Osaurus -> restart -> mlx-vlm
-> mlx_lm fallback chain worked, but the artifact recorded nothing about which
tier answered: a summary from the fourth-tier on-device fallback was
byte-for-byte indistinguishable from one produced by the primary 35B model. For
an unattended run that is the worst case -- nobody is watching at 08:00, and a
plausible-but-degraded digest reads exactly like a good one.

House rule (honest placeholders): a degraded state must be visually distinct and
say WHY. So a fallback summary now carries a banner naming the backend and the
reason, and `tw` exits non-zero on request rather than silently shipping it.

`SummaryText` is a `str` subclass so every existing caller, comparison and
`in` test keeps working unchanged; the provenance rides along as an attribute.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Provenance", "SummaryText", "PRIMARY", "FALLBACK", "LAST_RESORT", "FAILED"]

PRIMARY = "primary"
FALLBACK = "fallback"
LAST_RESORT = "last-resort"
FAILED = "failed"
UNKNOWN = "unrecorded"

# Anything that is not a confirmed primary run is degraded. UNKNOWN is included
# deliberately: "nobody recorded it" must not read as "it was fine", or an
# unattended run silently regains the exact ambiguity this class is about.
_DEGRADED_TIERS = frozenset({FALLBACK, LAST_RESORT, FAILED, UNKNOWN})


@dataclass(frozen=True)
class Provenance:
    """How a summary was produced. `backend` is the runtime, `model` the weights."""

    backend: str = "unknown"
    model: str = "unknown"
    tier: str = PRIMARY
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def degraded(self) -> bool:
        return self.tier in _DEGRADED_TIERS

    def describe(self) -> str:
        return f"{self.model} ({self.backend}, {self.tier})"

    def banner(self) -> str:
        """The provenance block for the top of the report.

        A normal run gets one quiet line. A degraded run gets a block that is
        impossible to mistake for a normal one and states the reason.
        """
        if not self.degraded:
            return f"**Model:** {self.describe()}"

        lines = [
            "> ⚠ **DEGRADED OUTPUT** — this summary was NOT produced by the primary model.",
            ">",
            f"> **Backend:** {self.describe()}",
        ]
        for reason in self.reasons:
            lines.append(f"> **Why:** {reason}")
        lines.append(
            "> Treat the content below as lower quality than a normal run: the "
            "fallback models have smaller context windows and weaker summarisation."
        )
        return "\n".join(lines)


class SummaryText(str):
    """A summary string that remembers which backend produced it.

    Subclasses `str` deliberately: `summarize_with_llm` has many callers and
    tests that treat its result as a plain string, and silently changing that
    contract to a tuple would be a bigger change than the bug warrants.
    """

    provenance: Provenance
    processed_count: int | None

    def __new__(
        cls,
        text: str,
        provenance: Provenance | None = None,
        processed_count: int | None = None,
    ) -> "SummaryText":
        obj = super().__new__(cls, text)
        obj.provenance = provenance or Provenance()
        obj.processed_count = processed_count
        return obj


def provenance_of(summary: object) -> Provenance:
    """Provenance for any summary value, including a plain `str`.

    A bare `str` means nobody recorded provenance, which is itself unknown rather
    than primary -- so it is reported as such instead of being assumed good.
    """
    found = getattr(summary, "provenance", None)
    if isinstance(found, Provenance):
        return found
    return Provenance(
        tier=UNKNOWN, reasons=("provenance was not recorded by the summariser",)
    )
