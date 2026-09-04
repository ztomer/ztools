"""Matching a known statement inside text that has been reworded.

Extracted from text_validator_mixed.py, which reached 497 of the repo's 500-line
limit. These three functions are one idea and have more than one caller, so they
are a module rather than private helpers.

The problem they exist for: a validator that asks "did the summary cover this
fact?" cannot answer it with `fact in output`. The task prompt orders the model
to reword ("use narrative verbs and connecting phrases"), so a substring test
measures verbatim COPYING and reports it as coverage. It rated a summary
covering all 18 key facts in its own words at 5/100, and rated the source
timeline -- which contains every fact by construction -- at 94, because one key
fact was not a substring of its own source line.
"""

import re


def tokenize(text: str) -> list:
    """Split text into comparable tokens, keeping the marks that carry meaning.

    '$', '%', '+' and an INTERIOR '.' stay attached, because '$75K', '40%',
    '1000+' and '2.5' are each one thing; a trailing '.' is sentence
    punctuation and is dropped, so '$75K.' still equals '$75K'.

    Both sides of a comparison go through this, and that symmetry is the point.
    Matching tokens against raw text instead was wrong twice, both times too
    generous: 'GPT-5' reduces to ('gpt', '5'), and a naked substring test found
    that '5' in all 27 tweets -- every timestamp contains one -- so 'GPT-5'
    scored as covered by the Toronto transit bulletin. Tightening it to a regex
    word boundary still matched the '.5' inside 'Gemini 2.5' and '0.5%'.
    Comparing token to token cannot make that mistake at all.
    """
    raw = re.sub(r"[^a-z0-9$%.+]+", " ", text.lower()).split()
    return [t for t in (tok.strip(".") for tok in raw) if t]


def identifying_tokens(phrase: str) -> list:
    """The tokens that identify `phrase`, for matching it inside a paraphrase.

    One implementation, because two callers previously each grew their own and
    one of them was wrong. What counts as identifying:

      - any token of 4+ characters      (entities, nouns, verbs)
      - any token containing a digit    ('Llama 4' and 'Gemini 2.5 Pro' are
                                        distinguished by exactly those digits)
      - any acronym, recognised from the phrase's OWN capitalisation rather
        than a hand-written list. A length rule alone discards 'GDP', 'IPO'
        and 'TD', which left 'Canadian GDP grows 0.5%' with nothing
        identifying in it at all.
    """
    acronyms = {w.lower() for w in re.findall(r"\b[A-Z]{2,}\b", phrase)}
    return [
        t
        for t in tokenize(phrase)
        if len(t) >= 4 or any(c.isdigit() for c in t) or t in acronyms
    ]


def phrase_overlap(phrase: str, out_lower: str) -> float:
    """Fraction of `phrase`'s identifying tokens present in `out_lower`.

    Returns 1.0 for a verbatim quote, and degrades smoothly as the text is
    reworded -- which is the whole point: an exact-substring test answers
    "was this copied", not "was this covered".

    Callers pick their own threshold, deliberately: the safe direction flips
    with the use. Missing a fact merely under-credits a summary, while missing
    a planted falsehood lets a hoax through.
    """
    tokens = identifying_tokens(phrase)
    if not tokens:
        stripped = phrase.lower().strip()
        return 1.0 if stripped and stripped in out_lower else 0.0
    present = set(tokenize(out_lower))
    return sum(1 for t in tokens if t in present) / len(tokens)
