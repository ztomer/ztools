"""
ZTools Library - Common utilities for LLM access and validation.

Modules:
  - validators_lib: Quality validators for model outputs
  - osaurus_lib: Generic LLM server utilities
  - mlx_lib: MLX model utilities
  - content_processing: Shared output cleaning utilities
  - logging_config: Logging configuration and loggers
"""

from .config import init_config
from .content_processing import (
    clean_model_output,
    extract_content_from_code_blocks,
    remove_inline_thinking,
    remove_markdown_blocks,
    remove_stats_tokens,
    remove_thinking_blocks,
    strip_backtick_value,
)
from .logging_config import (
    content_logger,
    get_logger,
    lib_logger,
    mlx_logger,
    osaurus_logger,
    validators_logger,
)
from .mlx_lib import (
    call_mlx,
    find_best_mlx_model,
    find_mlx_model,
    find_text_mlx_model,
    list_mlx_models,
    process_mlx_content,
)
from .osaurus_lib import (
    apply_model_quirks,
    call,
    clean_output,
    ensure_server,
    extract_json,
    get_api_url,
    get_base_url,
    get_best_model,
    get_models,
    is_server_running,
)
from .validators_lib import (
    validate_detailed_json,
    validate_filename,
    validate_json,
    validate_summary,
)

__all__ = [
    "init_config",
    # validators
    "validate_json",
    "validate_detailed_json",
    "validate_summary",
    "validate_filename",
    # osaurus
    "call",
    "get_api_url",
    "get_base_url",
    "get_models",
    "is_server_running",
    "extract_json",
    "clean_output",
    "get_best_model",
    "ensure_server",
    "apply_model_quirks",
    # mlx
    "find_mlx_model",
    "find_best_mlx_model",
    "find_text_mlx_model",
    "call_mlx",
    "process_mlx_content",
    "list_mlx_models",
    # content processing
    "remove_thinking_blocks",
    "remove_inline_thinking",
    "remove_stats_tokens",
    "remove_markdown_blocks",
    "extract_content_from_code_blocks",
    "clean_model_output",
    "strip_backtick_value",
    # logging
    "get_logger",
    "lib_logger",
    "osaurus_logger",
    "mlx_logger",
    "validators_logger",
    "content_logger",
]
