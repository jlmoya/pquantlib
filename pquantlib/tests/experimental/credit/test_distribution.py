"""Cross-validate Distribution against C++.

Probe source: migration-harness/cpp/probes/cluster_w3a/probe.cpp
Reference:    migration-harness/references/cluster/w3a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.experimental.credit.distribution import (
    Distribution,
    convolve_distributions,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3a")


def _build_probe_distribution() -> Distribution:
    """Mirror the C++ probe: 5 buckets in [0, 5], samples added at 0.25/0.5/1.5/2.5/2.7/3.9."""
    d = Distribution(5, 0.0, 5.0)
    for v in (0.25, 0.5, 1.5, 2.5, 2.7, 3.9):
        d.add(v)
    d.normalize()
    return d


def test_distribution_grid_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    d = _build_probe_distribution()
    ref = cpp_ref["distribution_basics"]
    assert d.size() == ref["size"]
    tolerance.tight(d.x(0), ref["x_0"])
    tolerance.tight(d.x(3), ref["x_3"])
    tolerance.tight(d.dx(0), ref["dx_0"])
    tolerance.tight(d.dx(4), ref["dx_4"])


def test_distribution_density_cumulative_match_cpp(cpp_ref: dict[str, Any]) -> None:
    d = _build_probe_distribution()
    ref = cpp_ref["distribution_basics"]
    tolerance.tight(d.density(0), ref["density_0"])
    tolerance.tight(d.density(2), ref["density_2"])
    tolerance.tight(d.cumulative(0), ref["cumulative_0"])
    tolerance.tight(d.cumulative(4), ref["cumulative_4"])


def test_distribution_expected_value_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    d = _build_probe_distribution()
    ref = cpp_ref["distribution_basics"]
    tolerance.tight(d.expected_value(), ref["expected_value"])


def test_distribution_locate_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    d = _build_probe_distribution()
    ref = cpp_ref["distribution_basics"]
    assert d.locate(2.3) == ref["locate_2_3"]


def test_distribution_average_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    d = _build_probe_distribution()
    ref = cpp_ref["distribution_basics"]
    tolerance.tight(d.average(0), ref["average_0"])
    tolerance.tight(d.average(2), ref["average_2"])


def test_distribution_add_density_and_add_average_round_trip() -> None:
    d = Distribution(5, 0.0, 5.0)
    d.add_density(0, 0.5)
    d.add_average(0, 0.25)
    # Before normalize, density is just the accumulator value.
    # After normalize, with no count_ events the density stays at the raw value.
    tolerance.tight(d.density(0), 0.5)


def test_distribution_underflow_overflow() -> None:
    d = Distribution(5, 0.0, 5.0)
    d.add(-1.0)
    d.add(6.0)
    d.add(2.5)
    assert d.underflow() == 1
    assert d.overflow() == 1


def test_distribution_normalize_idempotent() -> None:
    d = _build_probe_distribution()
    snap1 = [d.density(i) for i in range(d.size())]
    d.normalize()  # second call should be a no-op
    snap2 = [d.density(i) for i in range(d.size())]
    assert snap1 == snap2


def test_distribution_convolve_basic_shape() -> None:
    """Convolution of two distributions sized (4, 0, 4) → result of size 7."""
    d1 = Distribution(4, 0.0, 4.0)
    d1.add_density(0, 0.25)
    d1.add_density(1, 0.25)
    d1.add_density(2, 0.25)
    d1.add_density(3, 0.25)
    d1.normalize()
    d2 = Distribution(4, 0.0, 4.0)
    d2.add_density(0, 0.5)
    d2.add_density(1, 0.5)
    d2.normalize()
    out = convolve_distributions(d1, d2)
    # Convolved size = d1.size + d2.size - 1 = 7.
    assert out.size() == 7
