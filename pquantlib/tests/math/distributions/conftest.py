"""Shared C++ v1.43 reference fixture for the distribution tests.

Reference: ``migration-harness/references/v143/math/distributions.json``,
produced by ``migration-harness/cpp/probes/v143_math_distributions/probe.cpp``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.testing import reference_reader


@pytest.fixture(scope="session")
def v143() -> dict[str, Any]:
    return reference_reader.load("v143/math/distributions")
