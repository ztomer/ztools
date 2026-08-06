"""
Shared LLM fallback orchestration — try server models, restart on failure, fall back to MLX.
"""

import time
from typing import Any, Callable, Optional

from lib.tui import WARN

# Sleep between restart and retry (seconds)
RETRY_SLEEP = 2
DEFAULT_LOCAL_SERVER_URL = "http://localhost:1337"


def call_with_fallback(
    model_list: list[str],
    call_fn: Callable[[str], Optional[Any]],
    *,
    restart_fn: Optional[Callable[[], bool]] = None,
    mlx_fn: Optional[Callable[[], Optional[Any]]] = None,
    foundation_fn: Optional[Callable[[], Optional[Any]]] = None,
    max_server_retries: int = 1,
    label: str = "model",
) -> Optional[Any]:
    """Try server models, restart on failure, fall back to MLX then Foundation.

    Args:
        model_list: Ordered list of model names to try.
        call_fn: Called with each model name. Return truthy value on success.
        restart_fn: Called before retries to restart the server. Optional.
        mlx_fn: Called when all server models fail. Optional.
        foundation_fn: Called after mlx_fn (on-device Apple Foundation Models).
            Optional.
        max_server_retries: Number of restart+retry cycles per model.
        label: Human-readable label for status messages.

    Returns:
        Result from call_fn, mlx_fn, foundation_fn, or None if all fail.
    """
    if restart_fn is None:
        from lib.osaurus_lib import check_llm_availability as _check_llm_availability
        from lib.osaurus_lib import ensure_server as _ensure_server

        def restart_fn() -> bool:
            return bool(_ensure_server() or _check_llm_availability(DEFAULT_LOCAL_SERVER_URL))

    for model in model_list:
        for attempt in range(max_server_retries + 1):
            result = call_fn(model)
            if result:
                return result
            if attempt < max_server_retries:
                print(f"{WARN} {model} failed, restarting server...")
                restart_fn()
                time.sleep(RETRY_SLEEP)

    if mlx_fn:
        mlx_result = mlx_fn()
        if mlx_result:
            return mlx_result

    if foundation_fn:
        foundation_result = foundation_fn()
        if foundation_result:
            return foundation_result

    return None
