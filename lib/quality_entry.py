from lib.quality_models import Score, ScoreCard
from lib.quality_report import (
    BASELINE_PATH,
    generate_report,
    save_baseline,
    load_baseline,
    compare_to_baseline,
)
from lib.quality_runner import ALL_TEST_CASES, run_suite


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quality evaluation suite")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Models to test (default: all with prompts)")
    parser.add_argument("--tasks", nargs="*", default=None,
                        choices=["filename", "summarize", "file_summary"],
                        help="Tasks to test (default: all)")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save results as baseline for regression detection")
    parser.add_argument("--regression-only", action="store_true",
                        help="Only compare to baseline, don't re-run models")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    args = parser.parse_args()

    cases = ALL_TEST_CASES
    if args.tasks:
        cases = [c for c in cases if c.task in args.tasks]

    models = args.models or ["foundation", "qwopus3.6-27b-v2-mlx-4bit",
                             "nemotron-3-nano-omni-30b-a3b-mxfp4"]

    if args.regression_only:
        baseline = load_baseline()
        if not baseline:
            print("No baseline found at", BASELINE_PATH)
            return
        print(f"Loaded baseline with {len(baseline)} entries from {BASELINE_PATH}")

        scorecards = []
        for key, prev in baseline.items():
            parts = key.split("::", 2)
            if len(parts) == 3:
                model, task, case_id = parts
                dims = []
                for dname, dscore in prev.get("dimensions", {}).items():
                    dims.append(Score(dname, dscore, 0.0))
                sc = ScoreCard(model, task, case_id, dims, "", prev.get("elapsed", 0.0))
                scorecards.append(sc)

        warnings = compare_to_baseline(scorecards)
        if warnings:
            for w in warnings:
                print(w)
        else:
            print("  No regressions detected against baseline.")
        return

    print(f"Quality Suite: {len(models)} models × {len(cases)} cases")
    results = run_suite(models, cases, verbose=not args.quiet)

    print(generate_report(results))

    print("\n  Regression Check:")
    warnings = compare_to_baseline(results)
    if warnings:
        for w in warnings:
            print(w)
    else:
        print("  No regressions detected.")

    if args.save_baseline:
        save_baseline(results)
        print(f"\n  Baseline saved to {BASELINE_PATH}")


if __name__ == "__main__":
    main()
