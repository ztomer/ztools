"""The collect-parity comparator that only bin/ab_test holds, proved from the gate.

WHY THIS FILE EXISTS. `bin/ab_test`'s `compare_collect_records` is the ONE
implementation of the shared-record collect comparator: it decides whether two
collector runs agree on the records they BOTH saw. It is a `python3 -c` program
embedded in the script, so before this file it was exercised by exactly one
thing -- ab_test itself, which no gate step runs, and which release drives with
`cargo test --quiet` (a different command from gate step 4's
`cargo test --manifest-path rust/Cargo.toml --all-features`). A comparator with
one caller, in no gate, is a comparator whose bite nobody is watching.

Set equality was the first criterion and it was the WRONG instrument: the same
Rust collector run twice minutes apart on the same account and 24h window shared
7 of ~50 IDs, because X serves the Following endpoint as a per-load sample and no
two loads see the same tweets. What is deterministic is the record built from a
tweet both legs captured -- author, text and timestamp must agree, and the live
engagement counts must be present on both sides. So a test that only checked the
happy path would assert the wrong property: below, every case in
TestTheComparatorBites asserts a NON-ZERO exit, and the fixture red-proof (the
one ab_test itself carries at its lines 152-160) is one of them.

THE COMPARATOR IS EXTRACTED, NOT COPIED. A second implementation here would be a
third thing to keep in sync, and drift is the failure this file exists to
prevent: a copy would keep passing after ab_test's comparator was broken. So the
program text is lifted out of ab_test's own source (the single-quoted `python3 -c`
argument of `compare_collect_records`) and executed verbatim. An edit to ab_test's
comparator is therefore picked up here immediately, and `bin/ab_test` stays
read-only to this file.

SANDBOX DECISION -- the `sandboxed_server_script` marker IS applied, and not
because this test happens to reach the machine. It does not: the program under
test is stdlib json+datetime over two files. The marker is applied because the
program is lifted out of a script whose full form runs the INSTALLED ztools
binary, six `--help` probes and `cargo test`, so "this copy cannot touch the
machine" is a claim about a peer's script rather than about this file. The marker
makes conftest.py's `no_real_server_restart` guard prove it instead: `osaurus`,
`pgrep`, `lsof` and `curl` are shadowed by tripwires that must go unused, the
machine-wide GPU lock must come out byte-identical, and ZTOOLS_GPU_LOCK_DIR is
forced at tmp_path. Without the marker that fixture yields immediately, so a
comparator that grew a `curl` or a lock acquisition would do it during a gate run
instead of failing here.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent  # tools/tests/ -> repo root
AB_TEST = REPO / "bin" / "ab_test"
FIXTURES = REPO / "tests" / "fixtures" / "collect_parity"

#: This module drives a program lifted out of bin/ab_test -- the script that runs
#: the real binary and the real suite -- so it DECLARES that, and the declaration
#: switches on conftest.py's `no_real_server_restart` proof (see the module
#: docstring). Without it the sandbox below is not checked, only claimed.
pytestmark = pytest.mark.sandboxed_server_script

#: The comparator is stdlib-only over two small files and settles in well under a
#: second. Bounded anyway: an unbounded subprocess.run in a gate step hangs the
#: whole suite at 0% CPU with no indication of where, which is how a hung lock
#: once looked like a merely slow test.
COMPARATOR_TIMEOUT = 30

#: The argv ab_test appends after the embedded program. It is what makes this `-c`
#: program THE comparator rather than some other `-c` in the file, so it is the
#: selector as well as the thing asserted.
TRAILER = '"$rust_file" "$py_file"'


def extract_comparator(source: str) -> str:
    """The program text bash hands to `python3 -c` in compare_collect_records.

    Every `-c '<program>'` in `source` is tried and only the one whose argv
    continues with TRAILER is accepted, so a comment quoting the marker -- or a
    future second `-c` -- cannot become the program under test. Exactly one match
    is required: zero means the shape changed and any extraction would be a guess,
    two means this picked one at random. Both raise naming the count rather than
    returning something plausible.
    """
    marker = "-c '"
    bodies = []
    index = source.find(marker)
    while index >= 0:
        start = index + len(marker)
        cursor = start
        # A bash single-quoted string has no escapes; the only way to embed a
        # quote is the '\'' idiom, so scan for a quote that does not open one.
        while True:
            end = source.find("'", cursor)
            if end < 0:
                raise AssertionError(f"unterminated -c program at offset {start} in ab_test")
            if source[end : end + 3] == "'\\''":
                cursor = end + 3
                continue
            break
        if TRAILER in source[end:].split("\n", 1)[0]:
            bodies.append(source[start:end])
        index = source.find(marker, end)
    if len(bodies) != 1:
        raise AssertionError(
            f"expected exactly 1 embedded comparator in bin/ab_test (the -c program "
            f"invoked with {TRAILER}), found {len(bodies)}. This file drives the "
            f"comparator by lifting it out of ab_test, so if ab_test's shape changed, "
            f"fix the extraction here rather than copying the program in."
        )
    return bodies[0]


def make_compare(body: str):
    """A callable running the extracted program as ab_test runs it:
    `python3 -c <program> <left> <right>`, from the repo root, with the exit
    status as the observation."""

    def compare(left, right):
        # sys.executable, not "python3": the program is stdlib-only, so the
        # interpreter cannot change its verdict, and pinning the one running the
        # suite keeps the gate from measuring a different python3 than the one
        # that reports the result.
        try:
            return subprocess.run(
                [sys.executable, "-c", body, str(left), str(right)],
                capture_output=True,
                text=True,
                cwd=str(REPO),
                timeout=COMPARATOR_TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired as expired:
            raise AssertionError(
                f"comparator did not finish within {COMPARATOR_TIMEOUT}s over "
                f"{left} and {right} — it is looping, not merely slow."
            ) from expired

    return compare


@pytest.fixture(scope="module")
def compare():
    body = extract_comparator(AB_TEST.read_text(encoding="utf-8"))
    # Compiled here rather than at import, so a peer editing the comparator into
    # something that is not Python fails with this message instead of every case
    # in all three classes failing for one unexplained reason.
    compile(body, str(AB_TEST), "exec")
    return make_compare(body)


# ── the three verdict helpers ────────────────────────────────────────────────
# The comparator's contract is three exit codes, and each is a distinct claim, so
# each gets its own assertion rather than a bare `!= 0`: 0 is parity, 1 is a
# rejected or unreadable pair, 2 is "nothing shared", which proves nothing at all.
# The output rides in the message, because a rejection that fires without saying
# which record or field drifted leaves the reader to diff two fixtures by hand.


def parity(result):
    assert result.returncode == 0, (
        f"expected parity (exit 0), got {result.returncode}\n{result.stdout}{result.stderr}"
    )


def rejected(result):
    """A shared record differs, or an input could not be read."""
    assert result.returncode == 1, (
        f"expected rejection (exit 1), got {result.returncode}\n{result.stdout}{result.stderr}"
    )


def inconclusive(result):
    """Nothing shared: the comparator measured no record, so it may not report parity."""
    assert result.returncode == 2, (
        f"expected inconclusive (exit 2), got {result.returncode}\n{result.stdout}"
    )


def mentions(result, needle):
    """The verdict alone is not enough — assert the output names the thing."""
    assert needle in result.stdout, (
        f"comparator output does not mention {needle!r} (exit {result.returncode}):\n"
        f"{result.stdout}"
    )


# ── record shapes ────────────────────────────────────────────────────────────


def write_json(directory: Path, name: str, payload) -> Path:
    path = directory / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def rust_record(**overrides):
    """One tweet as the Rust collector writes it: `id`, and `created_at` in the
    GraphQL form ("Thu Aug 20 12:00:00 +0000 2026")."""
    record = {
        "id": "1827394000000000001",
        "screen_name": "rustlang",
        "text": "Rust 1.80 is released!",
        "created_at": "Thu Aug 20 12:00:00 +0000 2026",
        "favorite_count": 1500,
        "retweet_count": 350,
        "reply_to": None,
    }
    record.update(overrides)
    return record


def py_record(**overrides):
    """The same tweet as the Python collector writes it: `id_str`, and
    `created_at` in ISO-8601. The two name ONE instant in two spellings, which is
    what makes the timestamp comparison an instant comparison rather than a
    string one."""
    record = {
        "id_str": "1827394000000000001",
        "screen_name": "rustlang",
        "text": "Rust 1.80 is released!",
        "created_at": "2026-08-20T12:00:00+00:00",
        "favorite_count": 1500,
        "retweet_count": 350,
    }
    record.update(overrides)
    return record


class TestTheComparatorAgrees:
    """The property being preserved: the records both legs captured must match."""

    def test_identical_shared_records_are_parity(self, compare):
        """tests/fixtures/collect_parity/{rust_a,py_a}.json — four tweets a side,
        three carrying an ID (the fourth carries none on either side), three
        shared. The two files also spell `created_at` differently, so this doubles
        as the cross-format case."""
        result = compare(FIXTURES / "rust_a.json", FIXTURES / "py_a.json")
        parity(result)
        # The shared SET is part of the claim: a comparator that silently stopped
        # intersecting by ID moves this count, and nothing else would notice.
        mentions(result, "3 Rust / 3 Python tweets, 3 shared")
        mentions(result, "all 3 shared records agree")

    def test_partial_overlap_with_agreeing_records_is_parity(self, compare):
        """rust_a vs py_b share 2 records; the rest are tweets the other collector
        never saw. Set inequality is NOT drift — the feed is a per-load sample — so
        this must pass, and a comparator demanding equality would reject every
        real pair."""
        result = compare(FIXTURES / "rust_a.json", FIXTURES / "py_b.json")
        parity(result)
        mentions(result, "3 Rust / 4 Python tweets, 2 shared")
        mentions(result, "all 2 shared records agree")

    def test_the_same_instant_in_two_formats_agrees(self, compare, tmp_path):
        """A timestamp must be compared as an INSTANT. A string comparison would
        report every real record as drift, so this is the case that catches a
        regression to one, and it is asserted on synthetic input because the
        fixture pair also differs in nothing else."""
        left = write_json(tmp_path, "rust.json", [rust_record()])
        right = write_json(tmp_path, "py.json", [py_record()])
        result = compare(left, right)
        parity(result)
        mentions(result, "all 1 shared records agree")


class TestTheComparatorBites:
    """The red-proof this file exists for. A comparator that cannot reject a
    differing record green-lights exactly the parse drift it is here to catch, so
    every case here asserts a NON-ZERO exit and the record it names."""

    def test_a_differing_shared_record_is_rejected(self, compare):
        """The fixture red-proof: rust_a vs py_c share tweet ...0002, whose text
        drifted by " (drifted)". This is the assertion a happy-path-only test
        cannot make."""
        result = compare(FIXTURES / "rust_a.json", FIXTURES / "py_c.json")
        rejected(result)
        mentions(result, "1827394000000000002: text differ")
        mentions(result, "1 of 3 shared records differ")

    def test_a_differing_author_is_rejected(self, compare, tmp_path):
        """Text is not the only field; an author that disagrees means the two legs
        read a different record under one ID."""
        left = write_json(tmp_path, "rust.json", [rust_record()])
        right = write_json(tmp_path, "py.json", [py_record(screen_name="rustlang2")])
        result = compare(left, right)
        rejected(result)
        mentions(result, "1827394000000000001: screen_name differ")

    def test_a_differing_instant_is_rejected(self, compare, tmp_path):
        """Same author, same text, half an hour apart: the record both legs built
        is not the same record, whatever the spelling."""
        left = write_json(tmp_path, "rust.json", [rust_record()])
        right = write_json(tmp_path, "py.json", [py_record(created_at="2026-08-20T12:30:00+00:00")])
        result = compare(left, right)
        rejected(result)
        mentions(result, "1827394000000000001: created_at differ")

    def test_live_counts_are_compared_for_presence(self, compare, tmp_path):
        """Engagement counts drift between loads by nature, so a differing VALUE
        is not drift and must pass; a count PRESENT on one side and ABSENT on the
        other means the two collectors disagree about the shape of the record and
        must not pass. Both directions asserted, because the rule is presence and a
        test holding only the lenient half would accept a comparator that ignored
        the counts entirely."""
        left = write_json(tmp_path, "rust.json", [rust_record()])
        drifted_values = write_json(
            tmp_path, "values.json", [py_record(favorite_count=999999, retweet_count=7)]
        )
        agree = compare(left, drifted_values)
        parity(agree)

        absent = write_json(
            tmp_path,
            "absent.json",
            [{k: v for k, v in py_record().items() if k != "favorite_count"}],
        )
        mismatch = compare(left, absent)
        rejected(mismatch)
        mentions(mismatch, "1827394000000000001: favorite_count differ")

    def test_disjoint_legs_are_inconclusive_not_parity(self, compare):
        """rust_a vs py_d share nothing (py_d's IDs are ...901-903). Exit 2 is the
        honest verdict: no shared record means the comparator measured nothing, and
        under `set -e` a bare non-zero call used to kill ab_test before it could
        report one."""
        result = compare(FIXTURES / "rust_a.json", FIXTURES / "py_d.json")
        inconclusive(result)
        mentions(result, "no shared tweets")
        mentions(result, "inconclusive")

    def test_an_unreadable_input_is_rejected(self, compare, tmp_path):
        """Exit 1, not a crash and not a pass: an input nobody could read has not
        been shown to agree with anything."""
        result = compare(tmp_path / "does-not-exist.json", FIXTURES / "py_a.json")
        rejected(result)
        mentions(result, "cannot read collect outputs")

    def test_a_non_array_input_is_rejected(self, compare, tmp_path):
        """A mapping of tweets is a shape the comparator does not understand. It
        must say so rather than compare zero records and call it parity, which is
        what a permissive loader would do."""
        left = write_json(tmp_path, "rust.json", {"tweets": [rust_record()]})
        right = write_json(tmp_path, "py.json", [py_record()])
        result = compare(left, right)
        rejected(result)


class TestTheExtraction:
    """The seam itself. Were it to pick the wrong text, every verdict above would
    be about a program ab_test does not run."""

    def test_a_mention_of_the_marker_elsewhere_is_not_mistaken_for_the_program(self):
        """A comment quoting the marker, in the style of ab_test's own header
        comments, placed FIRST so a "first occurrence wins" extractor would take
        it and every case above would quietly describe that comment."""
        source = AB_TEST.read_text(encoding="utf-8")
        decoy = "# see `python3 -c 'print(1)'` above for the shape\n"
        assert extract_comparator(decoy + source) == extract_comparator(source)

    def test_an_unrecognised_shape_fails_loudly(self):
        """ab_test rewritten to call a script file instead of embedding one. The
        count in the message is what tells a reader whether to fix the extraction
        here or the script there — a silent fallback would hand this file a
        program nobody runs."""
        source = AB_TEST.read_text(encoding="utf-8").replace('"$PY" -c \'', '"$PY" cmp.py')
        with pytest.raises(AssertionError, match="found 0"):
            extract_comparator(source)


def test_release_runs_the_suite_with_the_gates_feature_set() -> None:
    """Release's test run builds what the gate built.

    `tools/release.sh` verifies an installed binary by running `bin/ab_test`, which
    ran `cargo test --quiet` -- the DEFAULT feature set -- while every gate step that
    runs the suite passes `--all-features`. Code behind a feature was therefore
    certified by the gate and never tested by the release. Every `cargo test` line in
    ab_test must carry the gate's flag.
    """
    lines = [
        line.strip()
        for line in AB_TEST.read_text().splitlines()
        if "cargo test" in line and not line.lstrip().startswith("#")
    ]
    assert lines, "bin/ab_test runs no cargo test at all: this check would be vacuous"
    assert [line for line in lines if "--all-features" not in line] == []
