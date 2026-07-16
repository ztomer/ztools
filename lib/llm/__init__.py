# LLM library
# Re-exports from new module structure

from lib.llm.client import (
    call,
    get_api_url,
    get_models,
    is_server_running,
)
from lib.llm.constants import (
    DEFAULT_HOST,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PORT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
)
from lib.llm.parsing import (
    clean_output,
    extract_json,
    safe_content,
)
from lib.llm.quirks import (
    apply_model_quirks,
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
