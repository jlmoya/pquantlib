"""Concentrating1dMesher cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/zabr/mesher.json
Probe:     migration-harness/cpp/probes/v143_zabr_mesher/probe.cpp

Ported as a dependency of ``ZabrModel.fd_price`` / ``full_fd_price``,
which build every grid from it. Cross-validated in its own right so a
ZABR price discrepancy localises to the PDE rather than to the grid.

Both C++ constructors are covered: the single-critical-point sinh map
(with and without the required-point reparameterisation, and with the
critical point sitting on either endpoint) and the multi-critical-point
ODE construction.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.finitedifferences.glued_1d_mesher import Glued1dMesher
from pquantlib.methods.finitedifferences.meshers.concentrating_1d_mesher import (
    Concentrating1dMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/zabr/mesher")


def _mesher(key: str) -> Fdm1dMesher:
    if key == "cpoint_not_required":
        return Concentrating1dMesher(0.0, 1.0, 21, (0.3, 0.1), False)
    if key == "cpoint_required":
        return Concentrating1dMesher(0.0, 1.0, 21, (0.3, 0.1), True)
    if key == "c_at_start":
        return Concentrating1dMesher(0.0, 1.0, 15, (0.0, 0.05), True)
    if key == "c_at_end":
        return Concentrating1dMesher(0.0, 1.0, 15, (1.0, 0.05), True)
    if key == "no_cpoint":
        return Concentrating1dMesher(0.0, 1.0, 11, (None, None), False)
    if key == "even_size":
        return Concentrating1dMesher(0.0, 1.0, 20, (0.5, 0.25), True)
    if key == "zabr_fd_grid":
        return Concentrating1dMesher(0.00001, 0.18, 500, (0.03, 0.1), True)
    if key == "zabr_fullfd_left":
        return Concentrating1dMesher(0.00502, 0.04, 55, (0.03, 0.1), True)
    if key == "zabr_fullfd_right":
        return Concentrating1dMesher(0.04, 0.1793, 46, (0.05, 0.1), True)
    if key == "zabr_fullfd_glued":
        return Glued1dMesher(_mesher("zabr_fullfd_left"), _mesher("zabr_fullfd_right"))
    if key == "multi_two_points":
        return Concentrating1dMesher(
            0.0, 1.0, 21, critical_points=[(0.25, 0.05, False), (0.75, 0.10, False)]
        )
    if key == "multi_required":
        return Concentrating1dMesher(
            0.0, 1.0, 21, critical_points=[(0.25, 0.05, True), (0.75, 0.10, True)]
        )
    raise AssertionError(f"unknown case {key}")


_CASES = [
    "cpoint_not_required",
    "cpoint_required",
    "c_at_start",
    "c_at_end",
    "no_cpoint",
    "even_size",
    "zabr_fd_grid",
    "zabr_fullfd_left",
    "zabr_fullfd_right",
    "zabr_fullfd_glued",
    "multi_two_points",
    "multi_required",
]


@pytest.mark.parametrize("key", _CASES)
def test_locations_match_cpp(cpp: dict[str, Any], key: str) -> None:
    block = cpp[key]
    m = _mesher(key)
    assert m.size() == int(block["size"])
    for i, expected in enumerate(block["locations"]):
        tolerance.tight(m.location(i), float(expected))


@pytest.mark.parametrize("key", _CASES)
def test_spacings_match_cpp(cpp: dict[str, Any], key: str) -> None:
    """dplus/dminus are what the FD operators actually read."""
    block = cpp[key]
    m = _mesher(key)
    for i, expected in enumerate(block["dplus"]):
        tolerance.tight(m.dplus(i), float(expected))
    for i, expected in enumerate(block["dminus"]):
        tolerance.tight(m.dminus(i + 1), float(expected))


def test_required_point_is_exactly_on_the_grid() -> None:
    """That is the whole purpose of ``require_c_point``."""
    m = _mesher("cpoint_required")
    locations = [m.location(i) for i in range(m.size())]
    assert any(abs(loc - 0.3) < 1e-14 for loc in locations)
    # Without the flag it is generally not a node.
    plain = _mesher("cpoint_not_required")
    plain_locations = [plain.location(i) for i in range(plain.size())]
    assert all(abs(loc - 0.3) > 1e-6 for loc in plain_locations)


def test_null_critical_point_gives_a_uniform_grid() -> None:
    m = _mesher("no_cpoint")
    for i in range(m.size()):
        tolerance.tight(m.location(i), i / 10.0)


def test_density_is_scaled_by_the_span() -> None:
    """C++ multiplies the density by ``end - start`` inside the constructor.

    Two meshers whose spans differ by a factor of 10 but whose densities
    are equal must therefore be exact rescalings of one another.
    """
    narrow = Concentrating1dMesher(0.0, 1.0, 21, (0.3, 0.1), False)
    wide = Concentrating1dMesher(0.0, 10.0, 21, (3.0, 0.1), False)
    for i in range(narrow.size()):
        tolerance.tight(wide.location(i), 10.0 * narrow.location(i))


def test_endpoints_are_exact() -> None:
    m = _mesher("zabr_fd_grid")
    tolerance.exact(m.location(0), 0.00001)
    tolerance.exact(m.location(m.size() - 1), 0.18)


def test_end_must_exceed_start() -> None:
    with pytest.raises(LibraryException, match="end must be larger than start"):
        Concentrating1dMesher(1.0, 1.0, 10, (0.5, 0.1), False)


def test_critical_point_outside_range_raises() -> None:
    with pytest.raises(LibraryException, match="cPoint must be between"):
        Concentrating1dMesher(0.0, 1.0, 10, (1.5, 0.1), False)


def test_required_without_critical_point_raises() -> None:
    with pytest.raises(LibraryException, match="cPoint is required"):
        Concentrating1dMesher(0.0, 1.0, 10, (None, None), True)


def test_critical_point_without_density_raises() -> None:
    with pytest.raises(LibraryException, match="density must be given"):
        Concentrating1dMesher(0.0, 1.0, 10, (0.5, None), False)


def test_both_constructor_forms_are_mutually_exclusive() -> None:
    with pytest.raises(LibraryException, match="not both"):
        Concentrating1dMesher(
            0.0, 1.0, 10, (0.5, 0.1), False, critical_points=[(0.5, 0.1, True)]
        )
