"""Shared C++ v1.43 reference fixture for the standalone ql/math classes.

Reference: ``migration-harness/references/v143/math/tail.json``, produced by
``migration-harness/cpp/probes/v143_math_tail/probe.cpp``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.testing import reference_reader


@pytest.fixture(scope="session")
def v143_tail() -> dict[str, Any]:
    return reference_reader.load("v143/math/tail")
