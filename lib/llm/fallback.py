"""
Shared LLM fallback orchestration — try server models, restart on failure, fall back to MLX.
"""

import time
from typing import Optional, Callable, Any

from lib.osaurus_lib import ensure_server, check_llm_availability
from lib.tui import WARN


def call_with_fallback(
    model_list: list[str],
    call_fn: Callable[[str], Optional[Any]],
    *,
    restart_fn: Callable[[], bool] = lambda: ensure_server() or check_llm_availability("http://localhost:1337"),
    mlx_fn: Optional[Callable[[], Optional[Any]]] = None,
    max_server_retries: int = 1,
    label: str = "model",
) -> Optional[Any]:
    """Try server models, restart on failure, fall back to MLX.

    Args:
        model_list: Ordered list of model names to try.
        call_fn: Called with each model name. Return truthy value on success.
        restart_fn: Called before retries to restart the server.
        mlx_fn: Called when all server models fail. Optional.
        max_server_retries: Number of restart+retry cycles per model.
        label: Human-readable label for status messages.

    Returns:
        Result from call_fn or mlx_fn, or None if all fail.
    """
    for model in model_list:
        for attempt in range(max_server_retries + 1):
            result = call_fn(model)
            if result:
                return result
            if attempt < max_server_retries:
                print(f"{WARN} {model} failed, restarting server...")
                restart_fn()
                time.sleep(2)

    if mlx_fn:
        mlx_result = mlx_fn()
        if mlx_result:
            return mlx_result

    return None
