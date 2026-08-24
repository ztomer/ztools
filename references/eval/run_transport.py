#!/usr/bin/env python3
"""Transport for the eval loop: one model call, no validation and no policy.

Split out of eval/run.py for the 500-line limit; `eval.run` re-exports every
name here, so `from eval.run import _call_model` keeps working -- the same shim
pattern the other splits in this repo use.

This module OWNS the transport names `call` and `mlx_call`. The prefill probe
and the per-task calls both resolve them HERE, at call time, so one
`patch.object(eval.run_transport, "call", ...)` covers a whole run. That is the
point of the module, not an accident of where the lines landed: `from
lib.osaurus_lib import call` binds a COPY, and while the probe and the task loop
each held their own alias, eighteen "mocked" tests went to the live server
through whichever alias nobody had thought to patch.
"""

import os

from lib.mlx_lib import call as mlx_call
from lib.osaurus_lib import call

from eval.signals import DEFAULT_EVAL_TIMEOUT

# Greedy decoding, because a leaderboard has to be reproducible.
#
# The eval inherited DEFAULT_TEMPERATURE (0.1) and never pinned it, so every run
# sampled. Running ornith twice on an unchanged image_rename scored 100% (190s)
# and then 0% (523s) -- a 100-point swing on identical input. Ranking models on
# single sampled runs measures the sampler, and any best_models derived that way
# would be noise wearing a number.
#
# Production still runs at 0.1: this measures the model's best behaviour rather
# than its average, which is the right basis for comparing models and for
# telling whether a prompt change helped. It does not eliminate GPU batching
# non-determinism, only the sampling that dominated it.
EVAL_TEMPERATURE = float(os.environ.get("EVAL_TEMPERATURE", "0"))


def _call_model(
    model: str,
    task_cfg: dict,
    task_name: str,
    host: str,
    port: int,
    backend: str,
    timeout: int = None,
    max_tokens: int = None,
) -> dict:
    """Call model via the appropriate backend (pure transport, no validation)."""
    effective_timeout = timeout or DEFAULT_EVAL_TIMEOUT
    if backend == "mlx":
        return mlx_call(
            model,
            messages=task_cfg["messages"],
            host=host,
            port=port,
            temperature=EVAL_TEMPERATURE,
            timeout=effective_timeout,
        )
    else:
        return call(
            model=model,
            messages=task_cfg["messages"],
            host=host,
            port=port,
            task=task_name,
            parse_json=task_cfg["parse_json"],
            temperature=EVAL_TEMPERATURE,
            timeout=effective_timeout,
            max_tokens=max_tokens,
            # Watch the stream and cut a run that has reasoned past the point where
            # the remaining budget could hold an answer. Without it, a reasoning
            # overrun is only detectable AFTER paying for the whole budget, and a
            # sweep pays that once per affected task per model.
            stream_guard=True,
        )


__all__ = ["EVAL_TEMPERATURE", "_call_model", "call", "mlx_call"]
