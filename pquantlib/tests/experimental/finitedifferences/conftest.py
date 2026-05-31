"""Shared fixtures for Phase 11 W5-B finite-difference tests.

Loads the C++ probe reference JSON ``cluster/w5b`` once per module.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.testing import reference_reader


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    """Phase 11 W5-B C++ probe reference values."""
    return reference_reader.load("cluster/w5b")
