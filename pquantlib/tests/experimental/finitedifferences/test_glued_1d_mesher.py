"""Tests for Glued1dMesher.

# C++ parity: ql/experimental/finitedifferences/glued1dmesher.hpp.

Reference values: migration-harness/references/cluster/w5a.json.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.finitedifferences.glued_1d_mesher import Glued1dMesher
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import (
    Uniform1dMesher,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def refs() -> dict[str, Any]:
    return reference_reader.load("cluster/w5a")


def test_glued_with_common_point(refs: dict[str, Any]) -> None:
    """Left[0,1]/4 + Right[1,3]/3 glued: 6 nodes (one shared).

    TIGHT against the C++ probe — locations and step sizes are bit-stable
    floats from the underlying ``Uniform1dMesher`` spacings.
    """
    left = Uniform1dMesher(0.0, 1.0, 4)
    right = Uniform1dMesher(1.0, 3.0, 3)
    glued = Glued1dMesher(left, right)

    tolerance.exact(float(glued.size()), refs["glued_common_size"])
    tolerance.tight(float(glued.locations()[0]), refs["glued_common_loc0"])
    tolerance.tight(float(glued.locations()[1]), refs["glued_common_loc1"])
    tolerance.tight(float(glued.locations()[2]), refs["glued_common_loc2"])
    tolerance.tight(float(glued.locations()[3]), refs["glued_common_loc3"])
    tolerance.tight(float(glued.locations()[4]), refs["glued_common_loc4"])
    tolerance.tight(float(glued.locations()[5]), refs["glued_common_loc5"])
    tolerance.tight(glued.dplus(2), refs["glued_common_dplus2"])
    tolerance.tight(glued.dminus(3), refs["glued_common_dminus3"])


def test_glued_no_overlap(refs: dict[str, Any]) -> None:
    """Left[0,1]/3 + Right[2,4]/3 glued: 6 nodes (no shared point).

    The midpoint dplus[2] should equal Right[0] - Left[end] = 1.0.

    TIGHT against C++ probe.
    """
    left = Uniform1dMesher(0.0, 1.0, 3)
    right = Uniform1dMesher(2.0, 4.0, 3)
    glued = Glued1dMesher(left, right)

    tolerance.exact(float(glued.size()), refs["glued_no_overlap_size"])
    tolerance.tight(float(glued.locations()[2]), refs["glued_no_overlap_loc2"])
    tolerance.tight(float(glued.locations()[3]), refs["glued_no_overlap_loc3"])
    tolerance.tight(glued.dplus(2), refs["glued_no_overlap_dplus2"])


def test_glued_boundary_sentinels() -> None:
    """dplus at last node + dminus at first node are NaN sentinels.

    # C++ parity: ``Null<Real>`` -> ``math.nan`` in Python.
    """
    left = Uniform1dMesher(0.0, 1.0, 4)
    right = Uniform1dMesher(1.0, 3.0, 3)
    glued = Glued1dMesher(left, right)
    assert math.isnan(glued.dplus(glued.size() - 1))
    assert math.isnan(glued.dminus(0))


def test_glued_left_rightmost_exceeds_right_leftmost_raises() -> None:
    """``left.locations()[-1] > right.locations()[0]`` must raise.

    # C++ parity: QL_REQUIRE in constructor.
    """
    left = Uniform1dMesher(0.0, 3.0, 4)  # ends at 3
    right = Uniform1dMesher(1.0, 2.0, 3)  # starts at 1
    with pytest.raises(LibraryException):
        Glued1dMesher(left, right)
