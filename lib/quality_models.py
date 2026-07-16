from dataclasses import dataclass, field
from typing import FrozenSet, List

GENERIC_FILENAMES: FrozenSet[str] = frozenset(
    {
        "filename.txt",
        "file.txt",
        "text.txt",
        "output.txt",
        "document.txt",
        "note.txt",
        "image.png",
        "screenshot.png",
        "unnamed",
        "file",
        "filename",
        "output",
        "document",
        "image",
        "photo",
        "screenshot",
    }
)


@dataclass
class Score:
    name: str
    score: float
    weight: float
    failures: List[str] = field(default_factory=list)

    @property
    def weighted(self) -> float:
        return self.score * self.weight


@dataclass
class ScoreCard:
    model: str
    task: str
    case_id: str
    dimensions: List[Score]
    output: str
    elapsed: float = 0.0

    @property
    def composite(self) -> float:
        if not self.dimensions:
            return 0.0
        return sum(s.weighted for s in self.dimensions)

    @property
    def total_weight(self) -> float:
        return sum(s.weight for s in self.dimensions)

    def report(self) -> str:
        comp = self.composite
        lines = [
            f"  {self.task:12s} {comp:5.1f}%  ({self.elapsed:5.1f}s)  {self.case_id}",
        ]
        for d in self.dimensions:
            if d.failures:
                lines.append(f"    {d.name:18s} {d.score:5.1f}%  FAIL: {'; '.join(d.failures)}")
            else:
                lines.append(f"    {d.name:18s} {d.score:5.1f}%")
        return "\n".join(lines)


@dataclass
class TestCase:
    __test__ = False
    task: str
    input_text: str
    reference: str
    description: str


def _str(x):
    return str(x) if x is not None else ""


def _lower(x):
    return _str(x).lower()
