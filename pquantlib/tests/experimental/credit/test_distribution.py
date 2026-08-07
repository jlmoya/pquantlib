"""Cross-validate Distribution against C++.

Probe source: migration-harness/cpp/probes/cluster_w3a/probe.cpp
Reference:    migration-harness/references/cluster/w3a.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
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


# =============================================================================
# v1.43 cross-validation — block A of the creditloss probe.
#
# Probe source: migration-harness/cpp/probes/v143_experimental_creditloss/probe.cpp
# Reference:    migration-harness/references/v143/experimental/creditloss.json
#
# The probe pins every observable array rather than a summary statistic, so a
# mismatch localises to a bucket. All TIGHT: Distribution does no root finding
# and no quadrature.
# =============================================================================

from pquantlib.experimental.credit.distribution import (  # noqa: E402
    ManipulateDistribution,
)

#: probe.cpp:313-314.
_A_SAMPLES = [
    -0.5, 0.05, 0.3, 0.7, 1.2, 1.9, 2.1, 2.4, 3.3, 3.7, 3.9, 4.5, 0.6, 2.2, 2.25,
]


@pytest.fixture(scope="module")
def v143_ref() -> dict[str, Any]:
    return reference_reader.load("v143/experimental/creditloss")


def _assert_arrays(dist: Distribution, prefix: str, ref: dict[str, Any]) -> None:
    n = dist.size()
    assert n == ref[f"{prefix}_size"]
    for suffix, getter in (
        ("_x", dist.x),
        ("_dx", dist.dx),
        ("_density", dist.density),
        ("_cumulative", dist.cumulative),
        ("_excess", dist.excess),
        ("_cumulative_excess", dist.cumulative_excess),
        ("_average", dist.average),
    ):
        expected = ref[prefix + suffix]
        for i in range(n):
            tolerance.tight(getter(i), expected[i])


def _sampled() -> Distribution:
    d = Distribution(8, 0.0, 4.0)
    for v in _A_SAMPLES:
        d.add(v)
    return d


def test_v143_fresh_grid_matches_cpp(v143_ref: dict[str, Any]) -> None:
    """The last bucket width is snapped to xmax; that is pinned too."""
    _assert_arrays(Distribution(8, 0.0, 4.0), "distA_fresh", v143_ref)


def test_v143_locate_matches_cpp(v143_ref: dict[str, Any]) -> None:
    d = Distribution(8, 0.0, 4.0)
    xs = v143_ref["distA_locate_x"]
    assert [d.locate(x) for x in xs] == v143_ref["distA_locate"]
    for x, expected in zip(xs, v143_ref["distA_dx_at"], strict=True):
        tolerance.tight(d.dx_at(x), expected)


def test_v143_sampled_distribution_matches_cpp(v143_ref: dict[str, Any]) -> None:
    """add() sampling, including the underflow (-0.5) and overflow (4.5) paths."""
    _assert_arrays(_sampled(), "distA_sampled", v143_ref)


def test_v143_sampled_statistics_match_cpp(v143_ref: dict[str, Any]) -> None:
    d = _sampled()
    tolerance.tight(d.expected_value(), v143_ref["distA_sampled_expected_value"])
    tolerance.tight(
        d.tranche_expected_value(1.0, 3.0), v143_ref["distA_sampled_tranche_ev_1_3"]
    )
    tolerance.tight(d.confidence_level(0.5), v143_ref["distA_sampled_conf_50"])
    tolerance.tight(d.confidence_level(0.9), v143_ref["distA_sampled_conf_90"])
    tolerance.tight(
        d.cumulative_density(1.5), v143_ref["distA_sampled_cum_density_1_5"]
    )
    tolerance.tight(
        d.cumulative_density(3.1), v143_ref["distA_sampled_cum_density_3_1"]
    )
    tolerance.tight(
        d.cumulative_excess_probability(0.5, 3.0),
        v143_ref["distA_sampled_cum_excess_0_5_3_0"],
    )
    tolerance.tight(d.expected_shortfall(0.5), v143_ref["distA_sampled_esf_50"])
    tolerance.tight(d.expected_shortfall(0.9), v143_ref["distA_sampled_esf_90"])


def test_v143_density_built_matches_cpp(v143_ref: dict[str, Any]) -> None:
    """add_density / add_average path, bypassing add()."""
    d = Distribution(5, 0.0, 5.0)
    for i in range(5):
        d.add_density(i, 0.05 * (i + 1))
        d.add_average(i, 0.5 + i)
    _assert_arrays(d, "distA_density_built", v143_ref)
    tolerance.tight(
        d.expected_value(), v143_ref["distA_density_built_expected_value"]
    )


def test_v143_expected_value_of_functional() -> None:
    """``expectedValue(F&)`` at f = identity must equal ``expectedValue()``.

    # C++ parity: the member template at distribution.hpp:80-89 versus the
    # plain overload at distribution.cpp:161-171 — the same sum with f(x) = x.
    """
    d = _sampled()
    tolerance.exact(d.expected_value_of(lambda x: x), d.expected_value())


def test_v143_manipulate_distribution_convolve_matches_cpp(
    v143_ref: dict[str, Any],
) -> None:
    c1 = Distribution(4, 0.0, 4.0)
    c2 = Distribution(3, 0.0, 3.0)
    for i in range(4):
        c1.add_density(i, 0.1 * (4 - i))
    for i in range(3):
        c2.add_density(i, 0.2 * (i + 1))
    c1.normalize()
    c2.normalize()
    _assert_arrays(ManipulateDistribution.convolve(c1, c2), "distA_convolve", v143_ref)


def test_v143_manipulate_distribution_matches_free_function() -> None:
    """The static and the module-level entry point are the same computation."""
    c1 = Distribution(4, 0.0, 4.0)
    c2 = Distribution(3, 0.0, 3.0)
    for i in range(4):
        c1.add_density(i, 0.1 * (4 - i))
    for i in range(3):
        c2.add_density(i, 0.2 * (i + 1))
    c1.normalize()
    c2.normalize()
    a = ManipulateDistribution.convolve(c1, c2)
    b = convolve_distributions(c1, c2)
    for i in range(a.size()):
        tolerance.exact(a.density(i), b.density(i))


def test_v143_tranche_matches_cpp(v143_ref: dict[str, Any]) -> None:
    """``tranche`` is destructive; the C++ arrays it leaves behind are pinned."""
    d = _sampled()
    d.normalize()
    d.tranche(1.0, 3.0)
    n = d.size()
    assert n == v143_ref["distA_tranche_size"]
    for suffix, getter in (
        ("_x", d.x),
        ("_dx", d.dx),
        ("_density", d.density),
        ("_cumulative", d.cumulative),
        ("_excess", d.excess),
    ):
        expected = v143_ref["distA_tranche" + suffix]
        for i in range(n):
            tolerance.tight(getter(i), expected[i])
    tolerance.tight(d.expected_shortfall(0.5), v143_ref["distA_tranche_esf_50"])


def test_v143_tranche_rejects_inverted_limits() -> None:
    """# C++ parity: distribution.cpp:238-239."""
    d = _sampled()
    d.normalize()
    with pytest.raises(LibraryException, match="attachment >= detachment"):
        d.tranche(3.0, 1.0)


def test_v143_tranche_rejects_limits_beyond_the_grid() -> None:
    """# C++ parity: distribution.cpp:240-242."""
    d = _sampled()
    d.normalize()
    with pytest.raises(LibraryException, match="attachment or detachment too large"):
        d.tranche(1.0, 9.0)
