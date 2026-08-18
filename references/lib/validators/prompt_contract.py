"""What the PROMPT asked for, read from the prompt itself.

A validator's contract must equal the prompt's contract. When they diverge, the
validator punishes obedience -- and the resulting score looks like a model failure,
which is how it survives.

The instance that prompted this module: `weekend_fixed_mixed` puts 12 signal venues in
front of the model and asks it to "find 10". `validate_mixed_signal` computed recall
over all 12, so the obedient answer scored

    100 * (0.5 * 10/12 + 0.5 * 1.0) = 91

while ignoring the instruction and returning all 12 scored 100. Every one of eleven
models landed on exactly 91, and it was recorded in docs/BACKLOG.md as "all eleven
missed the identical 2 of 12 signal items -- a fixture or prompt defect rather than
eleven coincidences". They had not missed anything. They had done as they were told,
and the task paid them less for it.

Reading the count OUT OF THE PROMPT rather than configuring it beside the task is
deliberate: a number configured next to a prompt drifts from it the first time
somebody edits the wording, and the drift is silent because both halves still look
reasonable on their own. Parsing the prompt makes the two impossible to separate.
"""

import re
from typing import List, Optional, Tuple

#: "find 10 year-round fixed activities", "list 5-10 events", "return 8 items".
_REQUESTED_COUNT_RE = re.compile(
    r"(?:find|list|return|output|provide)\s+(\d+)(?:\s*(?:-|to)\s*(\d+))?", re.I
)

__all__ = ["parse_signal_noise", "requested_item_count"]


def requested_item_count(source_text: str) -> Optional[int]:
    """How many items the prompt asked for, or None if it did not say.

    A RANGE ("find 5-10 events") returns its LOWER bound. That is the smallest answer
    which still obeys the instruction, and not punishing obedience is the entire
    reason this number is read at all -- scoring a compliant 5 against a denominator
    of 10 would reintroduce the defect this module exists to remove.

    None means the prompt set no count, and the caller should fall back to whatever it
    would have used before. An absent instruction is not an instruction to return
    everything, but it is also not evidence of a limit, so this must not invent one.
    """
    match = _REQUESTED_COUNT_RE.search(source_text or "")
    if not match:
        return None
    return int(match.group(1))


def parse_signal_noise(source_text: str) -> Tuple[List[str], List[str]]:
    """Split a mixed prompt (signal + NOISE section) into (signal_names, noise_names)."""
    if "NOISE" not in source_text:
        return [], []
    signal_part, noise_part = source_text.split("NOISE", 1)
    return _extract_bullets(signal_part), _extract_bullets(noise_part)


def _extract_bullets(text: str) -> List[str]:
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("- "):
            content = line[2:].strip()
            name = content.split(":")[0].strip()
            if name:
                out.append(name)
    return out
