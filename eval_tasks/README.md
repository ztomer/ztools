# eval_tasks — task data

This directory holds static scoring input, not code. The eval itself is Rust:

- **The task table** is `rust/src/ztools/eval/tasks.rs` (`roster()`). Its prompts are
  `eval/prompts/`, its validators `eval/validators/`, and the row-for-row pin of names,
  message roles, `parse_json` flags and validators is `eval/tasks_tests.rs`. Adding a task
  means editing there, not here.
- **The probe tasks** `refusal_*` (`eval/probe_refusal.rs`) and `hebrew_*`
  (`eval/probe_hebrew.rs`) join the roster after `json` (LM.PROBES). Their fixtures are
  synthetic and live in those files. Each `refusal_*` prompt batches five requests and the
  refusal scorer (`eval/validators/refusal.rs`) reads the answer item by item. The controls'
  own real answers, each labelled by hand, are pinned in
  `tests/fixtures/refusal/real_answers.json`, and the scorer must read every one as labelled.
- **The taxes snapshots** are the only files here:

```
eval_tasks/
├── data/
│   └── taxes/      # sanitized TaxJSON rubrics: taxes_*.sanitized.json
└── README.md       # this file
```

`load_all_eval_tasks` (`eval/task_loader.rs`) appends one `taxes_<task>` task per
`taxes_*.json` file under `data/taxes/` to the roster, graded by the taxes validators
(`validators/taxes_grounded/`, `validators/taxes_rubric.rs`). The directory is found through
the `eval_tasks_dirs` config key (default: `eval_tasks/data` beside the binary or the
checkout), or `--tasks-dir` on `ztools model-eval --suite full`. `rust/tests/eval_signals_prune.rs`
and `task_loader_tests.rs` read the same tree.

The Python `eval_tasks` package (`__init__.py`, `run.py`, `analyze.py`, `validators.py`) is
gone: nothing imports it, and no file here is code.
