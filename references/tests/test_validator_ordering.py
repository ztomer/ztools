"""No validator may rank bad output above good output.

Every scoring bug this repo has hit was an ordering failure wearing a plausible
number: `validate_summary` scored `@user 1 @user 2` padding above a
prompt-perfect summary, `validate_file_summary` scored "a python script" as a
detailed description, and `fix_json_years` corrupted the values the grounding
check then compared. A per-dimension unit test does not catch that class — only
comparing a known-good output against a known-bad one does.

These are deliberately coarse: they assert the ORDER, not the number, so they
survive rescoring but fail the moment a validator inverts.
"""

import pytest
from eval.validate import validate_file_summary
from lib.validators.json_validator import validate_detailed_json
from lib.validators_lib import validate_filename, validate_summary

GOOD_SUMMARY = (
    "## Executive Summary\n"
    "Funding and model releases dominated the week, with inference cost recurring\n"
    "across threads. Participants reported benchmarks and confirmed pricing.\n\n"
    "## Funding\n"
    "- Series B closed at $40M (@TechCrunch | Mar 15 08:00)\n"
    "- Follow-on announced for infrastructure (@benedictevans | Mar 15 09:30)\n\n"
    "## Models\n"
    "- Lower latency confirmed (@simonw | Mar 16 11:05)\n"
    "- Early evaluation numbers shared (@karpathy | Mar 16 14:20)\n"
)
BAD_SUMMARY = "stuff happened. things were said. no idea who or when."

GOOD_FILENAME = "quarterly_revenue_report"
BAD_FILENAME = "Here is the filename: IMG 1234.PNG"

GOOD_FILE_SUMMARY = [
    {"path": "lib/config_loader.py", "desc": "Parses TOML config and validates required keys"},
    {"path": "lib/api_client.py", "desc": "Sends chat requests and handles retries"},
    {"path": "lib/report.py", "desc": "Renders scorecards into markdown and writes them"},
    {"path": "lib/extract.py", "desc": "Extracts JSON from noisy model output"},
]
BAD_FILE_SUMMARY = [
    {"path": "lib/config_loader.py", "desc": "a python script"},
    {"path": "lib/api_client.py", "desc": "a config file"},
    {"path": "lib/report.py", "desc": "report"},
    {"path": "lib/extract.py", "desc": "extract"},
]

# The invented item must share NO field values with the source, or the
# grounding ratio counts its price/ages/weather as matches and the cap never
# applies — the first draft of this test asserted against exactly that mistake.
SOURCE = (
    "Spring Festival at Vaughan Mills on Saturday, free admission, all ages, outdoor. "
    "Winter Market at Dufferin Grove on Sunday, $10 entry, ages 6-12, indoor."
)
GROUNDED_JSON = [
    {
        "name": "Spring Festival",
        "location": "Vaughan Mills",
        "day": "Saturday",
        "price": "Free",
        "target_ages": "All",
        "weather": "outdoor",
    }
]
INVENTED_JSON = [
    {
        "name": "Moon Rave",
        "location": "Atlantis",
        "day": "Tuesday",
        "price": "$999",
        "target_ages": "40-50",
        "weather": "underwater",
    }
]


@pytest.mark.parametrize(
    "name, score_good, score_bad",
    [
        ("summary", lambda: validate_summary(GOOD_SUMMARY), lambda: validate_summary(BAD_SUMMARY)),
        (
            "filename",
            lambda: validate_filename(GOOD_FILENAME),
            lambda: validate_filename(BAD_FILENAME),
        ),
        (
            "file_summary",
            lambda: validate_file_summary(GOOD_FILE_SUMMARY),
            lambda: validate_file_summary(BAD_FILE_SUMMARY),
        ),
        (
            "detailed_json",
            lambda: validate_detailed_json(GROUNDED_JSON, source_text=SOURCE),
            lambda: validate_detailed_json(INVENTED_JSON, source_text=SOURCE),
        ),
    ],
)
def test_good_output_outscores_bad(name, score_good, score_bad):
    good = score_good()[0]
    bad = score_bad()[0]
    assert good > bad, f"{name} ranks bad output ({bad}) at or above good ({good})"


def test_prompt_conformant_summary_is_not_merely_adequate():
    """The summarize gate is 90; conformant output must clear it, not scrape by.

    This is the regression that started it: a summary following the prompt
    exactly capped at 85 while `@user N` padding scored 100.
    """
    assert validate_summary(GOOD_SUMMARY)[0] >= 90


def test_padding_the_summary_with_user_tokens_does_not_help():
    padded = GOOD_SUMMARY + "\n- @user 1 responded, @user 2 asked, @user 3 confirmed\n"
    assert validate_summary(padded)[0] <= validate_summary(GOOD_SUMMARY)[0]


def test_hallucinated_json_is_capped_well_below_grounded():
    """Schema-shaped invention must not approach a grounded answer."""
    grounded = validate_detailed_json(GROUNDED_JSON, source_text=SOURCE)[0]
    invented = validate_detailed_json(INVENTED_JSON, source_text=SOURCE)[0]
    assert invented < grounded / 2, (invented, grounded)
