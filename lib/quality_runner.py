import json
import time
from typing import List, Optional

from lib import osaurus_lib
from lib.config import Task, _safe_format_prompt, get_model_prompt
from lib.quality_models import ScoreCard, TestCase, _str
from lib.quality_scorers import score_output
from lib.tui import FAIL, STEP, WARN

LLM_TIMEOUT = 600

FILENAME_CASES = [
    TestCase(
        task="filename",
        input_text="Screenshot showing login error: Invalid credentials. Please try again.",
        reference="login_error_invalid_credentials",
        description="Login error screenshot",
    ),
    TestCase(
        task="filename",
        input_text="Summer Festival 2024 - Family Fun Day at Central Park",  # check-ok: year
        reference="summer_festival_2024_family_fun_day",
        description="Event with clear subject",
    ),
    TestCase(
        task="filename",
        input_text="10 powerful sentences by Scott Adams navigating failure and ambition",
        reference="scott_adams_sentences_failure_ambition",
        description="Quote with multiple keywords",
    ),
    TestCase(
        task="filename",
        input_text="Meeting Notes - Project Alpha - Q1 Review",
        reference="meeting_notes_project_alpha_q1_review",
        description="Work meeting notes",
    ),
    TestCase(
        task="filename",
        input_text="Screen Shot 2024-03-15 at 14.30.22.png",  # check-ok: year
        reference="screen_shot_20240315_143022",
        description="Generic screenshot (less info)",
    ),
]

SUMMARIZE_CASES = [
    TestCase(
        task="summarize",
        input_text=(
            "[@user1 | 10:00] Just launched our new product! Excited to share.\n"
            "[@user2 | 10:15] Looks great! How do I get it?\n"
            "[@user1 | 10:30] Check the website for early access.\n"
            "[@user3 | 10:45] Got it, thanks!\n"
            "[@user2 | 11:00] Anyone tried the beta yet?\n"
            "[@user4 | 11:15] Been using it all morning, very smooth.\n"
            "[@user1 | 11:30] Great feedback!"
        ),
        reference=(
            "This conversation follows a product launch through four stages: "
            "announcement, access setup, beta testing, and community feedback. "
            "@user1 drives the narrative — announcing the product, directing "
            "users to early access, and acknowledging feedback. "
            "@user2 acts as the engaged user asking questions, @user3 confirms "
            "receipt, and @user4 provides the positive beta review.\n\n"
            "## Launch & Access\n"
            "- @user1 announced a new product at 10:00\n"
            "- @user2 asked how to get it at 10:15\n"
            "- @user1 directed users to the website for early access at 10:30\n"
            "- @user3 confirmed receiving the info at 10:45\n\n"
            "## Beta & Feedback\n"
            "- @user2 asked about beta testing at 11:00\n"
            "- @user4 reported smooth beta experience at 11:15\n"
            "- @user1 thanked everyone for the feedback at 11:30"
        ),
        description="Product launch with user interaction",
    ),
    TestCase(
        task="summarize",
        input_text=(
            "[@user5 | 09:00] Server migration starting. ETA 2 hours.\n"
            "[@user6 | 09:15] Database backup complete.\n"
            "[@user7 | 09:30] DNS propagation initiated.\n"
            "[@user5 | 10:00] Config sync in progress.\n"
            "[@user6 | 10:30] All services restored. Monitoring active.\n"
            "[@user7 | 11:00] No issues detected. Migration complete.\n"
            "[@user5 | 11:30] Post-mortem scheduled for tomorrow."
        ),
        reference=(
            "A server migration completed successfully over 2.5 hours. "
            "@user5 led the process with @user6 and @user7 handling backup, "
            "DNS, and monitoring phases. "
            "All services were restored by 10:30 with no issues detected.\n\n"
            "## Migration Steps\n"
            "- @user5 started server migration at 09:00\n"
            "- @user6 completed database backup at 09:15\n"
            "- @user7 initiated DNS propagation at 09:30\n"
            "- @user5 ran config sync at 10:00\n\n"
            "## Completion & Monitoring\n"
            "- @user6 restored all services at 10:30\n"
            "- @user7 confirmed no issues at 11:00\n"
            "- @user5 scheduled post-mortem at 11:30"
        ),
        description="Server migration timeline",
    ),
]

FILE_SUMMARY_CASES = [
    TestCase(
        task="file_summary",
        input_text=json.dumps(
            [
                {"path": "eval_lib.py", "desc": "model evaluation functions"},
                {"path": "validators.py", "desc": "validation logic for JSON and text output"},
                {"path": "config.py", "desc": "configuration management and model prompts"},
                {"path": "osaurus_lib.py", "desc": "LLM API client library"},
            ]
        ),
        reference=json.dumps(
            [
                {"path": "eval_lib.py", "desc": "model evaluation functions for quality scoring"},
                {"path": "validators.py", "desc": "validation logic for JSON and text output"},
                {"path": "config.py", "desc": "configuration management and model prompts"},
                {
                    "path": "osaurus_lib.py",
                    "desc": "LLM API client for Ollama/OAI-compatible servers",
                },
            ]
        ),
        description="4 files with known descriptions",
    ),
]

_QR_YEAR = "2026"  # check-ok: year

WEEKEND_TRANSIENT_PROMPT = f"""
Current Context for the upcoming weekend:
Dates: April 20 to April 22, {_QR_YEAR}
Friday: 15.0°C, Clear (0mm)
Saturday: 12.0°C, Precipitation (5mm)
Sunday: 14.0°C, Clear (0mm)

High-Signal Transient Events (Filter these strictly! Ensure they match the Dates provided!):
- Spring Festival at Downsview Park: Outdoor rides and games. April 20-22. All ages.
- Indoor Coding Workshop for Kids: Learn Python. April 21. Ages 8-14.
- Outdoor Movie Night: Watch a movie under the stars. April 21. All ages.
- Farmers Market at Maple Village: Fresh produce and local crafts. April 20. All ages.
- Pottery Wheel Workshop: Create clay art. April 22. Ages 12+.
- Puppet Show at Vaughan Library: "The Magical Forest". April 20. Ages 4-10.
- Kids Yoga in the Park: Morning yoga for families. April 20. Ages 5-12.
- Magic Show at Markham Theatre: Illusionist show. April 21. All ages.
- Nature Walk at Boyd Conservation: guided family hike. April 22. All ages.
- Board Game Marathon at Community Centre: Family games. April 21. All ages.
- Pizza Making Class: Learn to make pizza. April 22. Ages 8-16.
- Easter Egg Hunt at Raccoon Creek: Egg hunt and crafts. April 20. Ages 3-10.
"""

WEEKEND_FIXED_PROMPT = f"""
Current Context for the upcoming weekend:
Dates: April 20 to April 22, {_QR_YEAR}
Friday: 15.0°C, Clear (0mm)
Saturday: 12.0°C, Precipitation (5mm)
Sunday: 14.0°C, Clear (0mm)

Potential Venues and Current Exhibits:
- Vaughan Sports Arena: Indoor trampoline and dodgeball. All ages.
- High Park: Large outdoor playground and zoo. All ages.
- Aga Khan Museum: Islamic art and culture. Indoor. All ages.
- McMichael Canadian Art Collection: Canadian art exhibits. Indoor. All ages.
- Gibson Park: Playground and splash pad. Outdoor. Ages 3-12.
- Richmond Hill Centre for the Performing Arts: Live theater. Indoor. All ages.
- Maplewood Park Conservation: Hiking trails and picnic area. Outdoor. All ages.
- Ezra Avenue Skatepark: Skateboarding and BMX. Outdoor. Ages 10+.
- Oakridge Arts Festival: Art installations and workshops. April 20-22. Indoor/outdoor. All ages.
- Toronto Fun Zone: Indoor play centre with wall climb. Indoor. Ages 4-14.
- Lake Simcoe Sugar Bush: Maple syrup tours. Outdoor. All ages.
- Markham Museum: Heritage buildings and events. Indoor/outdoor. All ages.

Output JSON with schema: [{{"name": "str", "location": "str",
"target_ages": "str", "price": "str", "weather": "str"}}]
Include 8-10 items. Use "indoor" for rainy days, "outdoor" for clear days.
Focus on venues appropriate for kids ages 6-13.
Output ONLY JSON. No explanations.
"""

WEEKEND_TRANSIENT_REF = json.dumps(
    {
        "weather": {"friday": "clear", "saturday": "rain", "sunday": "clear"},
        "age_range": [6, 13],
        "source_item_names": [
            "spring festival",
            "coding workshop",
            "movie night",
            "farmers market",
            "pottery wheel",
            "puppet show",
            "kids yoga",
            "magic show",
            "nature walk",
            "board game marathon",
            "pizza making",
            "easter egg hunt",
        ],
        "expected_count": [5, 10],
    }
)

WEEKEND_FIXED_REF = json.dumps(
    {
        "weather": {"friday": "clear", "saturday": "rain", "sunday": "clear"},
        "age_range": [6, 13],
        "source_item_names": [
            "vaughan sports arena",
            "high park",
            "aga khan museum",
            "mcmichael",
            "gibson park",
            "richmond hill centre",
            "maplewood park",
            "ezra avenue",
            "oakridge arts",
            "toronto fun zone",
            "lake simcoe sugar bush",
            "markham museum",
        ],
        "exclude": ["canada's wonderland", "ontario science centre", "toronto zoo"],
        "expected_count": [8, 10],
    }
)

WEEKEND_TRANSIENT_CASES = [
    TestCase(
        task="weekend_transient",
        input_text=WEEKEND_TRANSIENT_PROMPT,
        reference=WEEKEND_TRANSIENT_REF,
        description="Transient events with mixed weather",
    ),
]

WEEKEND_FIXED_CASES = [
    TestCase(
        task="weekend_fixed",
        input_text=WEEKEND_FIXED_PROMPT,
        reference=WEEKEND_FIXED_REF,
        description="Fixed venues with mixed weather",
    ),
]

ALL_TEST_CASES = (
    FILENAME_CASES
    + SUMMARIZE_CASES
    + FILE_SUMMARY_CASES
    + WEEKEND_TRANSIENT_CASES
    + WEEKEND_FIXED_CASES
)


def query_model(model: str, prompt: str, input_text: str, task: str) -> Optional[str]:
    try:
        filled = _safe_format_prompt(prompt, input_text)
        result = osaurus_lib.call(
            model=model,
            messages=[{"role": "user", "content": filled}],
            timeout=LLM_TIMEOUT,
            task=task,
        )
        return _str(result.get("content"))
    except Exception:
        return None


def query_model_direct(model: str, full_prompt: str) -> Optional[str]:
    try:
        result = osaurus_lib.call(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            timeout=LLM_TIMEOUT,
        )
        return _str(result.get("content"))
    except Exception:
        return None


def run_suite(
    models: List[str], cases: List[TestCase] = None, verbose: bool = True
) -> List[ScoreCard]:
    if cases is None:
        cases = ALL_TEST_CASES

    results = []

    for i, model in enumerate(models):
        for j, case in enumerate(cases):
            if verbose:
                print(
                    f"  {STEP} {model[:30]:30s} {case.task:12s} {case.description}",
                    end=" ",
                    flush=True,
                )

            t0 = time.time()

            if case.task in ("weekend_transient", "weekend_fixed"):
                output = query_model_direct(model, case.input_text)
            else:
                prompt = get_model_prompt(model, Task(case.task))
                if not prompt:
                    if verbose:
                        print(f"{STEP} skip")
                    continue
                output = query_model(model, prompt, case.input_text, case.task)

            elapsed = time.time() - t0

            if output is None:
                if verbose:
                    print(FAIL)
                results.append(
                    ScoreCard(
                        model=model,
                        task=case.task,
                        case_id=case.description,
                        dimensions=[],
                        output="",
                        elapsed=elapsed,
                    )
                )
                continue

            sc = score_output(output, case.task, case)
            sc.model = model
            sc.elapsed = elapsed
            results.append(sc)

            if verbose:
                comp = sc.composite
                if sc.dimensions:
                    worst = min(d.score for d in sc.dimensions)
                    prefix = STEP if worst >= 60 else (WARN if comp >= 40 else FAIL)
                    failures = [d.failures for d in sc.dimensions if d.failures]
                    all_fails = [f for dim_fails in failures for f in dim_fails]
                    fail_str = f" [{'; '.join(all_fails)}]" if all_fails else ""
                    print(f"{prefix}  {comp:5.1f}%  ({elapsed:.1f}s){fail_str}")
                else:
                    print(f"{FAIL}  0.0%")

    return results
