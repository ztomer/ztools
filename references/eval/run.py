#!/usr/bin/env python3
"""
Evaluation runner module.
Contains the main eval loop, model calling, and validation orchestration.

Shim: split across run_transport (the model call and the temperature pin),
run_validate (scoring one result), run_attempt (one task's retry policy and
reasoning-overrun escalation), run_loop (the across-tasks loop: prefill probe,
heartbeat, stall check, infrastructure abandonment) and run_summary (end-of-run
reporting), to stay under the 500-line limit with room to explain itself. Every
name they define is re-exported here, so existing imports are unaffected.

Patching, though, is NOT re-exported -- rebinding a name on this shim rebinds a
copy nobody reads. Patch the module that OWNS the function: `eval.run_transport`
for `call` / `mlx_call`, `eval.run_attempt` for `MAX_RETRIES` /
`_validate_result` / `save_output`, `eval.run_validate` for its debug `console`,
`eval.run_loop` for `measure_prefill_rate` / `contended_server_warning` /
`gpu_lock`. Each module's docstring says what it owns. The conftest socket gate
fails a test that patches the wrong one rather than letting it reach the live
server, which is how that rule stays enforced instead of remembered.
"""

from eval.result_format import _quality_results_to_eval_format  # noqa: F401
from eval.run_attempt import (  # noqa: F401
    MAX_RETRIES,
    TaskAttempts,
    run_task_attempts,
)
from eval.run_loop import (  # noqa: F401
    MAX_CONSECUTIVE_INFRA_FAILURES,
    MEMORY_WARNING_THRESHOLD,
    run_eval,
    run_eval_quick,
)
from eval.run_summary import (  # noqa: F401
    print_signal_noise_summary,
    print_source_matching_summary,
)
from eval.run_transport import (  # noqa: F401
    EVAL_TEMPERATURE,
    _call_model,
    call,
    mlx_call,
)
from eval.run_validate import _validate_result  # noqa: F401
from eval.signals import DEFAULT_EVAL_TIMEOUT  # noqa: F401

__all__ = [
    "DEFAULT_EVAL_TIMEOUT",
    "EVAL_TEMPERATURE",
    "MAX_CONSECUTIVE_INFRA_FAILURES",
    "MAX_RETRIES",
    "MEMORY_WARNING_THRESHOLD",
    "TaskAttempts",
    "_call_model",
    "_quality_results_to_eval_format",
    "_validate_result",
    "print_signal_noise_summary",
    "print_source_matching_summary",
    "run_eval",
    "run_eval_quick",
    "run_task_attempts",
]
