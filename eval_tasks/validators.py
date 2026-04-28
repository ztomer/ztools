# Validators for eval tasks
# Copy from model_eval.py for standalone use

from typing import Tuple, Any

def validate_detailed_json(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Score objects with details based on quality, not count."""
    # Placeholder - actual implementation in lib/validators_lib.py
    from lib.validators_lib import validate_detailed_json as _vd
    return _vd(data, source_text)


def validate_summary(data: Any) -> Tuple[int, str]:
    """Score summaries based on structure and content quality."""
    from lib.validators_lib import validate_summary as _vs
    return _vs(data)


def validate_filename(data: Any) -> Tuple[int, str]:
    """Score filenames based on quality."""
    from lib.validators_lib import validate_filename as _vf
    return _vf(data)


def validate_file_summary(data: Any, source_text: str = "") -> Tuple[int, str]:
    """Validate file summary quality."""
    from lib.validators_lib import validate_file_summary as _vfs
    return _vfs(data, source_text)