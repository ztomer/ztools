import json
import time
from typing import List, Optional

from lib.config import get_model_prompt, Task, _safe_format_prompt
from lib.osaurus_lib import call as llm_call
from lib.quality_models import TestCase, ScoreCard, _str
from lib.quality_scorers import score_output
from lib.tui import STEP, WARN, FAIL


FILENAME_CASES = [
    TestCase(
        task="filename",
        input_text="Screenshot showing login error: Invalid credentials. Please try again.",
        reference="login_error_invalid_credentials",
        description="Login error screenshot",
    ),
    TestCase(
        task="filename",
        input_text="Summer Festival 2024 - Family Fun Day at Central Park",
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
        input_text="Screen Shot 2024-03-15 at 14.30.22.png",
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
            "This conversation follows a product launch through four stages: announcement, access setup, beta testing, and community feedback. "
            "@user1 drives the narrative — announcing the product, directing users to early access, and acknowledging feedback. "
            "@user2 acts as the engaged user asking questions, @user3 confirms receipt, and @user4 provides the positive beta review.\n\n"
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
            "@user5 led the process with @user6 and @user7 handling backup, DNS, and monitoring phases. "
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
        input_text=json.dumps([
            {"path": "eval_lib.py", "desc": "model evaluation functions"},
            {"path": "validators.py", "desc": "validation logic for JSON and text output"},
            {"path": "config.py", "desc": "configuration management and model prompts"},
            {"path": "osaurus_lib.py", "desc": "LLM API client library"},
        ]),
        reference=json.dumps([
            {"path": "eval_lib.py", "desc": "model evaluation functions for quality scoring"},
            {"path": "validators.py", "desc": "validation logic for JSON and text output"},
            {"path": "config.py", "desc": "configuration management and model prompts"},
            {"path": "osaurus_lib.py", "desc": "LLM API client for Ollama/OAI-compatible servers"},
        ]),
        description="4 files with known descriptions",
    ),
]

ALL_TEST_CASES = FILENAME_CASES + SUMMARIZE_CASES + FILE_SUMMARY_CASES


def query_model(model: str, prompt: str, input_text: str, task: str) -> Optional[str]:
    try:
        filled = _safe_format_prompt(prompt, input_text)
        result = llm_call(
            model=model,
            messages=[{"role": "user", "content": filled}],
            timeout=600,
            task=task,
        )
        return _str(result.get("content"))
    except Exception:
        return None


def run_suite(models: List[str], cases: List[TestCase] = None,
              verbose: bool = True) -> List[ScoreCard]:
    if cases is None:
        cases = ALL_TEST_CASES

    results = []
    total = len(models) * len(cases)

    for i, model in enumerate(models):
        for j, case in enumerate(cases):
            if verbose:
                print(f"  {STEP} {model[:30]:30s} {case.task:12s} {case.description}",
                      end=" ", flush=True)

            prompt = get_model_prompt(model, Task(case.task))
            if not prompt:
                if verbose:
                    print("- skip")
                continue

            t0 = time.time()
            output = query_model(model, prompt, case.input_text, case.task)
            elapsed = time.time() - t0

            if output is None:
                if verbose:
                    print(FAIL)
                results.append(ScoreCard(
                    model=model, task=case.task, case_id=case.description,
                    dimensions=[], output="", elapsed=elapsed,
                ))
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
