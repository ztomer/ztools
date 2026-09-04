from lib.tui import FAIL, STEP, WARN


def print_header(models, all_cases):
    total_cases = sum(len(c[2]) for c in all_cases)
    print(f"  QUALITY BENCHMARK - {len(models)} models x {total_cases} cases")


def print_model_header(model):
    print(f"\n  Model: {model}")


def print_case_result(human_score, auto_score, elapsed, case_desc, output, failures):
    prefix = STEP if human_score >= 70 else (WARN if human_score >= 30 else FAIL)
    print(f"      {prefix} H:{human_score:3} A:{auto_score:3}  {elapsed:.1f}s  {case_desc}")
    if human_score < 70:
        print(f"         output: {repr(output[:80])}")
        if failures:
            print(f"         issues:  {'; '.join(failures)}")
    else:
        print(f"         output: {repr(output[:80])}")


def print_model_summary(model, avg_human, avg_auto, model_count):
    print(f"\n    Summary for {model}:")
    print(f"      Avg Human: {avg_human:.0f}/100")
    print(f"      Avg Auto:  {avg_auto:.0f}/100")
    print(f"      Gap:       {avg_auto - avg_human:+2.0f} pts")


def print_cross_model_comparison(results):
    if len(results) > 1:
        print("\n  CROSS-MODEL COMPARISON")
        print(f"  {'Model':35s} {'Human':>7} {'Auto':>7} {'Gap':>5}")
        print(f"  {'-' * 35} {'-' * 7} {'-' * 7} {'-' * 5}")
        for model, res in sorted(results.items(), key=lambda x: -x[1]["avg_human"]):
            print(
                f"  {model:35s} {res['avg_human']:6.0f}  {res['avg_auto']:6.0f}  {res['gap']:+3.0f}"
            )
