# LLM library
# Re-exports from new module structure

from lib.llm.client import (
    call,
    get_api_url,
    get_models,
    is_server_running,
)

from lib.llm.quirks import (
    apply_model_quirks,
)

from lib.llm.parsing import (
    extract_json,
    safe_content,
    clean_output,
)

from lib.llm.constants import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
)

__all__ = [
    # Core functions
    "call",
    "get_api_url",
    "get_models",
    "is_server_running",
    # Quirks
    "apply_model_quirks",
    # Parsing
    "extract_json",
    "safe_content",
    "clean_output",
    # Constants
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT",
]