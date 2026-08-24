"""Fixture and structural-gate modules imported into conftest.py's namespace.

Split out of conftest.py for the 500-line cap (no test exemption; see CLAUDE.md).
Pytest discovers autouse fixtures by scanning conftest.py's own module
namespace, so an import here does nothing on its own -- every name has to be
imported BY NAME into conftest.py itself for pytest to see and apply it. See
conftest.py's own comment for the current list.
"""
