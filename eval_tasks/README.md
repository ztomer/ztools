# eval_tasks — task rubrics & task data

The old `eval_tasks` package was split out of `eval/` years ago; its run/analyze/
validator shims (`run.py`, `analyze.py`, `validators.py`, `__main__.py`) were
retired 2026-09-13 once nothing imported them (the Rust `eval/` runner and the
Python `eval/` package own those behaviours).

What survives is exactly what the scoring layer needs as static input, not code:

```
eval_tasks/
├── __init__.py     # backward-compat shim: TASKS + load_tasks_from_config → eval package
├── data/
│   └── taxes/      # sanitized TaxJSON rubrics — scoring input, not code
└── README.md       # this file
```

- `eval_tasks_path("data", "taxes", ...)` (from `lib/paths.py`) resolves the real
  rubric files for `eval.tasks_core`, `lib.validators.taxes_validator` and
  `lib.validators.taxes_grounded`. Rust `task_loader_tests.rs` reads the same tree.
- `__init__.py` remains only as a read-only compat surface (`TASKS`,
  `load_tasks_from_config`, `get_tasks`); it copies, never aliases, so its callers
  cannot mutate the shared task table through it.

Task *definitions* (prompts, validators, parse flags) live in `eval/tasks_core.py`;
adding a task means editing there, not here.