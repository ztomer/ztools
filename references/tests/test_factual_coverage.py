"""Guards for fact-coverage scoring.

Every model failed `summarize_factual_coverage` (best score 33/100), which read
as a shared model weakness and was not one: the validator matched key facts as
exact substrings while the task prompt orders the model to reword. It was
scoring verbatim copying and calling it coverage.

Two controls proved it, and both are kept here as tests, because either one
would have caught the defect years earlier than five model runs did:

  * the SOURCE TIMELINE is the trivial upper bound -- it contains every key
    fact by construction, so anything under 100 means the fact list is wrong,
    with no model involved. It scored 94: 'Amazon launches drone delivery in
    Toronto' is not a substring of its own source line, which says '...in
    select Toronto neighborhoods'.
  * a summary covering all 18 topics IN ITS OWN WORDS scored 5.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.tasks_prompts import FALSEHOOD_PHRASES, KEY_FACTS, TWITTER_PROMPT
from lib.validators.text_match import identifying_tokens, phrase_overlap
from lib.validators.text_validator_mixed import (
    COVERAGE_TOKEN_RATIO,
    validate_factual_accuracy,
    validate_factual_coverage,
)

SOURCE_TIMELINE = TWITTER_PROMPT.split("<timeline>")[1].split("</timeline>")[0]

# Covers all 18 topics but Tesla, in the narrative voice the prompt demands:
# every fact reworded, several numbers spelled out. A coverage metric that
# cannot credit this is not measuring coverage.
PARAPHRASED_SUMMARY = """## Executive Summary
A heavy day for AI launches, strong tech earnings, and Toronto transit trouble.

## AI and Models
- OpenAI unveiled its next flagship reasoning model, GPT 5, shipping next month (@TechCrunch | 08:00)
- Google followed with Gemini 2.5 Pro and its million-token window (@TechCrunch | 08:30)
- Meta released Llama 4 under an open commercial license (@TechCrunch | 10:30)
- Microsoft bought an AI startup for $2B to strengthen Azure (@TechCrunch | 09:30)

## Hardware
- Apple began mass production of the Vision Pro 2 for a fall release (@TheVerge | 08:15)
- Samsung's Galaxy S25 Ultra debuts a titanium frame and an AI camera (@TheVerge | 09:45)
- Intel launched its Core Ultra chips (@TheVerge | 12:15)
- IBM revealed a quantum computer exceeding 1000+ qubits (@Wired | 12:00)

## Markets
- NVIDIA stock reached an all-time high on data center revenue (@Wired | 08:45)
- Bitcoin climbed past $75K (@Bloomberg | 11:45)
- Shopify posted 40% revenue growth (@CNBC | 12:30)
- Databricks went public in an IPO valuing it at $60B (@Bloomberg | 12:45)
- TD Bank reported strong Q2 earnings (@CNBC | 13:15)
- Canadian GDP expanded 0.5% (@Bloomberg | 13:00)
- Adobe closed its $20B purchase of Figma (@TechCrunch | 13:30)

## Logistics
- Amazon began drone delivery across Toronto (@TorontoStar | 13:45)
- Uber started an autonomous taxi service in Phoenix (@Wired | 14:00)
"""

# The same voice, four topics deep instead of eighteen.
PARTIAL_SUMMARY = "\n".join(PARAPHRASED_SUMMARY.splitlines()[:8])


def test_every_key_fact_is_reachable_from_its_own_source():
    """The structural gate: the timeline must score 100 against its own facts.

    This is the check that fails the moment someone edits KEY_FACTS or a tweet
    and leaves a fact nothing can ever match -- the exact defect that made one
    of the eighteen unattainable and capped the whole task at 94.
    """
    score, reason = validate_factual_coverage(SOURCE_TIMELINE, key_facts=KEY_FACTS)
    unreachable = [
        f for f in KEY_FACTS if phrase_overlap(f, SOURCE_TIMELINE.lower()) < COVERAGE_TOKEN_RATIO
    ]
    assert unreachable == [], f"key facts absent from the source timeline: {unreachable}"
    assert score == 100, reason


def test_paraphrase_is_credited_and_still_separated_from_partial_coverage():
    """A metric that credits paraphrase must not credit everything.

    Under substring matching these scored 5 and 5 -- indistinguishable, and
    both indistinguishable from an empty answer.
    """
    full, _ = validate_factual_coverage(PARAPHRASED_SUMMARY, key_facts=KEY_FACTS)
    partial, _ = validate_factual_coverage(PARTIAL_SUMMARY, key_facts=KEY_FACTS)
    none, _ = validate_factual_coverage("A quiet day; nothing was reported.", key_facts=KEY_FACTS)

    assert full >= 80, f"reworded full coverage scored {full}"
    assert none <= 10, f"a summary with no facts scored {none}"
    assert full - partial >= 50, f"full {full} vs partial {partial} is not a separation"


def test_identifying_tokens_keeps_acronyms_and_digits():
    """Short tokens are dropped as noise unless they carry the identity.

    A plain length rule discarded 'gdp', leaving 'Canadian GDP grows 0.5%' with
    only 'canadian' and 'grows' -- so a summary about Canadian weather matched
    it. Digits matter for the same reason: 'Llama 4' is not 'Llama 3'.
    """
    assert "gdp" in identifying_tokens("Canadian GDP grows 0.5%")
    assert "ipo" in identifying_tokens("Databricks IPO values company at $60B")
    assert "4" in identifying_tokens("Meta announces Llama 4")
    # 'for' and 'at' identify nothing and would match any English text.
    assert "for" not in identifying_tokens("Adobe acquires Figma for $20B")
    assert "at" not in identifying_tokens("Databricks IPO values company at $60B")


def test_a_phrase_with_no_identifying_tokens_falls_back_to_substring():
    """Nothing survives the token filter, so there is nothing to match on.

    'a of the' is all short common words: no 4+ token, no digit, no acronym.
    Scoring it 1.0 would credit every output for covering it, and 0.0 would
    make it permanently unreachable, so it falls back to asking literally.
    """
    assert phrase_overlap("a of the", "well, a of the sort") == 1.0
    assert phrase_overlap("a of the", "nothing similar here") == 0.0
    assert identifying_tokens("a of the") == []


def test_wrong_version_number_is_not_counted_as_coverage():
    """Token matching must not become a name-drop detector."""
    out = "meta announces llama 3 today".lower()
    assert phrase_overlap("Meta announces Llama 4", out) < 1.0


@pytest.mark.parametrize(
    "text,expect_clean",
    [
        (PARAPHRASED_SUMMARY, True),
        ("Jensen Huang was arrested for insider trading, per @BSNews.", False),
        ("Google confirmed layoffs of 100,000 employees across search.", False),
    ],
)
def test_falsehood_detection_keeps_its_sensitivity(text, expect_clean):
    """The shared token extractor must not have loosened the hoax check.

    Coverage and falsehood detection share how tokens are found but not how
    they decide, precisely because the safe direction is opposite: a missed
    fact under-credits a summary, a missed falsehood passes a hoax as clean.
    """
    score, reason = validate_factual_accuracy(text, falsehood_phrases=FALSEHOOD_PHRASES)
    if expect_clean:
        assert score == 100, reason
    else:
        assert score < 100, "planted falsehood was repeated and not flagged"


def test_key_facts_are_distinguishable_from_each_other():
    """No two facts may share an identifying-token set.

    If they did, one summary sentence would score as covering both, and the
    metric would over-credit forever without anything looking wrong.
    """
    seen = {}
    for fact in KEY_FACTS:
        key = frozenset(identifying_tokens(fact))
        assert key, f"{fact!r} has no identifying tokens"
        assert key not in seen, f"{fact!r} is indistinguishable from {seen[key]!r}"
        seen[key] = fact


def test_source_lines_are_not_matched_by_unrelated_facts():
    """Each key fact matches its OWN source line and not the others.

    Guards the loosening direction: a threshold low enough to credit paraphrase
    is also low enough, if set carelessly, to credit the wrong tweet.
    """
    lines = [ln for ln in SOURCE_TIMELINE.splitlines() if re.match(r"\[@", ln)]
    for fact in KEY_FACTS:
        matches = [ln for ln in lines if phrase_overlap(fact, ln.lower()) >= COVERAGE_TOKEN_RATIO]
        assert len(matches) == 1, f"{fact!r} matched {len(matches)} source lines: {matches}"
