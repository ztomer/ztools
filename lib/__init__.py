"""
ZTools Library - Common utilities for LLM access and validation.

Modules:
  - validators_lib: Quality validators for model outputs
  - osaurus_lib: Generic LLM server utilities
  - mlx_lib: MLX model utilities
  - content_processing: Shared output cleaning utilities
  - logging_config: Logging configuration and loggers
"""

from .validators_lib import (
    validate_json,
    validate_detailed_json,
    validate_summary,
    validate_filename,
    VALIDATORS,
)
from .osaurus_lib import (
    call,
    get_api_url,
    get_base_url,
    get_models,
    is_server_running,
    extract_json,
    clean_output,
    get_best_model,
    ensure_server,
)
from .mlx_lib import (
    find_mlx_model,
    find_best_mlx_model,
    find_text_mlx_model,
    call_mlx,
    call_mlx_text,
    process_mlx_content,
    list_mlx_models,
)
from .content_processing import (
    remove_thinking_blocks,
    remove_stats_tokens,
    remove_markdown_blocks,
    extract_content_from_code_blocks,
    clean_model_output,
)
from .config import init_config
from .logging_config import (
    get_logger,
    lib_logger,
    osaurus_logger,
    mlx_logger,
    validators_logger,
    content_logger,
)

__all__ = [
    "init_config",
    # validators
    "validate_json",
    "validate_detailed_json",
    "validate_summary",
    "validate_filename",
    "VALIDATORS",
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
    # mlx
    "find_mlx_model",
    "find_best_mlx_model",
    "find_text_mlx_model",
    "call_mlx",
    "call_mlx_text",
    "process_mlx_content",
    "list_mlx_models",
    # content processing
    "remove_thinking_blocks",
    "remove_stats_tokens",
    "remove_markdown_blocks",
    "extract_content_from_code_blocks",
    "clean_model_output",
    # logging
    "get_logger",
    "lib_logger",
    "osaurus_logger",
    "mlx_logger",
    "validators_logger",
    "content_logger",
]

