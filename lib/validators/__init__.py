# Validators library
# Re-exports from new module structure

from lib.validators.json_validator import (
    validate_json,
    validate_detailed_json,
    extract_list_from_dict,
    check_source_extraction,
    get_source_matching_details,
    has_item_details,
)

from lib.validators.text_validator import (
    validate_summary,
    validate_filename,
    validate_file_summary,
)

from lib.validators.helpers import (
    has_text_headers,
    count_content_lines,
    strip_backtick_value,
    normalize_whitespace,
    extract_json_list,
    is_valid_filename_char,
    has_filename_format,
)

# Also export constants
from lib.validators.constants import (
    MAX_SCORE,
    JSON_ITEMS_WEIGHT,
    JSON_SCHEMA_WEIGHT,
    JSON_COMPLETENESS_WEIGHT,
    JSON_QUALITY_WEIGHT,
    FILENAME_LENGTH_MIN,
    FILENAME_LENGTH_MAX,
    FILENAME_LENGTH_WEIGHT,
    FILENAME_CHARS_WEIGHT,
    FILENAME_FORMAT_WEIGHT,
    SUMMARY_HEADERS_WEIGHT,
    SUMMARY_LENGTH_GOOD,
    SUMMARY_LENGTH_OK,
    SUMMARY_LENGTH_THRESHOLD,
    SUMMARY_LENGTH_THRESHOLD_OK,
    SUMMARY_CONTENT_WEIGHT,
    SUMMARY_LINES_GOOD,
    SUMMARY_LINES_OK,
    FILE_SUMMARY_ITEMS_WEIGHT,
    FILE_SUMMARY_QUALITY_WEIGHT,
    FILE_SUMMARY_MIN_ITEMS,
)

__all__ = [
    # Validators
    "validate_json",
    "validate_detailed_json",
    "validate_summary",
    "validate_filename",
    "validate_file_summary",
    # Helpers
    "extract_list_from_dict",
    "check_source_extraction",
    "get_source_matching_details",
    "has_item_details",
    "has_text_headers",
    "count_content_lines",
    "strip_backtick_value",
    "normalize_whitespace",
    "extract_json_list",
    "is_valid_filename_char",
    "has_filename_format",
]