"""Cross-validate the C++ ``<random>`` primitives the swarm optimizers use.

Probe source: migration-harness/cpp/probes/v143_experimental_pso/probe.cpp
Reference:    migration-harness/references/v143/experimental/pso.json (blocks A-C)

``ql/experimental/math`` reaches past QuantLib's own RNGs into ``<random>``:
``ClubsTopology`` uses ``std::uniform_int_distribution``, ``LevyFlightDistribution``
uses ``std::uniform_real_distribution``, and ``GaussianWalk`` uses
``std::normal_distribution``. Those distributions are specified by their
statistics, not their algorithms, so reproducing QuantLib means reproducing the
particular standard library the reference binary was built against.

Everything here is asserted **exactly**. These are bit streams; a rounding
tolerance would defeat the purpose. If the reference binary is ever rebuilt
against a different standard library, these tests are what fail first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.math.isotropic_random_walk import IsotropicRandomWalk
from pquantlib.experimental.math.levy_flight_distribution import LevyFlightDistribution
from pquantlib.experimental.math.std_random import (
    StdMt19937,
    StdNormalDistribution,
    StdUniformIntDistribution,
    StdUniformRealDistribution,
    generate_canonical,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def ref() -> dict[str, Any]:
    return reference_reader.load("v143/experimental/pso")


# --------------------------------------------------------------------------
# Block A — the <random> primitives. All exact.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 42, 12345])
def test_std_mt19937_is_the_quantlib_mt(ref: dict[str, Any], seed: int) -> None:
    """std::mt19937 and QuantLib's MT are the same engine, bit for bit.

    Everything else in ``std_random`` rests on this, so the probe emits both
    C++ sequences and asserts their equality there too.
    """
    assert ref[f"A1_same_engine_seed{seed}"] is True
    g = StdMt19937(seed)
    assert [g() for _ in range(8)] == ref[f"A1_std_mt19937_seed{seed}"]


def test_generate_canonical(ref: dict[str, Any]) -> None:
    """generate_canonical<double,53> consumes two words and combines them."""
    g = StdMt19937(42)
    assert [generate_canonical(g) for _ in range(8)] == ref[
        "A2_generate_canonical_53_seed42"
    ]


def test_uniform_real_distribution(ref: dict[str, Any]) -> None:
    """uniform_real_distribution over (0,1) and (-1,1)."""
    g = StdMt19937(42)
    u01 = StdUniformRealDistribution(0.0, 1.0)
    assert [u01(g) for _ in range(8)] == ref["A2_uniform_real_0_1_seed42"]
    g = StdMt19937(7)
    um11 = StdUniformRealDistribution(-1.0, 1.0)
    assert [um11(g) for _ in range(8)] == ref["A2_uniform_real_m1_1_seed7"]


@pytest.mark.parametrize(
    ("lo", "hi"),
    [(1, 8), (1, 5), (1, 4), (1, 3), (1, 2), (0, 9), (1, 1), (2, 17), (0, 1000000)],
)
def test_uniform_int_distribution(ref: dict[str, Any], lo: int, hi: int) -> None:
    """uniform_int_distribution<Size>, including the degenerate 1-value range.

    ``(1, 1)`` is the case that consumes no engine words at all; ``(0, 1000000)``
    exercises the ``__independent_bits_engine`` rejection loop.
    """
    g = StdMt19937(1234)
    dist = StdUniformIntDistribution(lo, hi)
    assert [dist(g) for _ in range(12)] == ref[f"A3_uniform_int_{lo}_{hi}_seed1234"]


def test_uniform_int_distribution_param_override(ref: dict[str, Any]) -> None:
    """Per-call reparametrisation on a shared engine, as ClubsTopology does it."""
    g = StdMt19937(99)
    dist = StdUniformIntDistribution(1, 8)
    got = [dist(g, 1, 1 + (i % 5)) for i in range(12)]
    assert got == ref["A3_uniform_int_paramtype_seed99"]


@pytest.mark.parametrize(("sigma", "tag"), [(1.0, "1"), (0.25, "0p25")])
def test_normal_distribution(ref: dict[str, Any], sigma: float, tag: str) -> None:
    """normal_distribution — Marsaglia polar, generated in pairs.

    Ten draws is deliberately odd-numbered on the pair boundary so the cached
    second variate is exercised in both states.
    """
    g = StdMt19937(5)
    nd = StdNormalDistribution(0.0, sigma)
    assert [nd(g) for _ in range(10)] == ref[f"A4_normal_sigma{tag}_seed5"]


def test_normal_distribution_reset_drops_the_cache() -> None:
    """reset() discards the cached variate, so the next call draws afresh."""
    g1 = StdMt19937(5)
    nd1 = StdNormalDistribution()
    first = nd1(g1)
    second = nd1(g1)  # served from the cache, consumes nothing

    g2 = StdMt19937(5)
    nd2 = StdNormalDistribution()
    assert nd2(g2) == first
    nd2.reset()
    # After reset the cache is gone, so this draws a fresh pair.
    assert nd2(g2) != second


def test_std_mt19937_rejects_seed_zero() -> None:
    """Seed 0 would silently fall through to the clock-seeded generator."""
    with pytest.raises(LibraryException, match="nonzero explicit seed"):
        StdMt19937(0)


# --------------------------------------------------------------------------
# Block B/C — LevyFlightDistribution and IsotropicRandomWalk over std::mt19937.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("xm", "alpha", "seed", "tag"),
    [(1.0, 1.0, 42, "xm1_a1"), (1.0, 1.5, 42, "xm1_a1p5"), (0.5, 0.7, 11, "xm0p5_a0p7")],
)
def test_levy_variates_over_std_engine(
    ref: dict[str, Any], xm: float, alpha: float, seed: int, tag: str
) -> None:
    """xm * u^(-1/alpha) with u from std::uniform_real_distribution.

    Exact: the transform is two multiplications over a bit-identical uniform.
    """
    g = StdMt19937(seed)
    dist = LevyFlightDistribution(xm, alpha)
    assert [dist(g) for _ in range(8)] == ref[f"B_levy_variates_{tag}"]


@pytest.mark.parametrize(
    ("xm", "alpha", "tag"),
    [(1.0, 1.0, "xm1_a1"), (1.0, 1.5, "xm1_a1p5"), (0.5, 0.7, "xm0p5_a0p7")],
)
def test_levy_pdf(ref: dict[str, Any], xm: float, alpha: float, tag: str) -> None:
    """Closed-form pdf, including the zero below support."""
    d = LevyFlightDistribution(xm, alpha)
    for i, x in enumerate([0.1, 0.5, 1.0, 2.0, 10.0]):
        tolerance.tight(d.pdf(x), float(ref[f"B_levy_pdf_{tag}"][i]), reason=f"pdf[{i}]")


@pytest.mark.parametrize("dim", [1, 2, 3, 4])
def test_isotropic_walk_over_std_engine(ref: dict[str, Any], dim: int) -> None:
    """IsotropicRandomWalk with a std::mt19937 radius and a QuantLib-MT angle.

    The two streams are independent and seeded differently; consuming either
    one out of step shows up immediately.
    """
    walk = IsotropicRandomWalk(
        engine=StdMt19937(2024),
        distribution=LevyFlightDistribution(1.0, 1.3),
        dim=dim,
        weights=np.ones(dim, dtype=np.float64),
        seed=7777,
    )
    out: list[float] = []
    step = np.zeros(dim, dtype=np.float64)
    for _ in range(5):
        walk.next_real(step)
        out.extend(float(v) for v in step)
    expected = ref[f"C_isowalk_dim{dim}_seed2024_ang7777"]
    for i, (got, want) in enumerate(zip(out, expected, strict=True)):
        tolerance.tight(got, float(want), reason=f"isowalk dim{dim}[{i}]")


def test_isotropic_walk_box_weighted(ref: dict[str, Any]) -> None:
    """setDimension(dim, lower, upper) rescales the sphere into a box ellipsoid."""
    walk = IsotropicRandomWalk(
        engine=StdMt19937(2024),
        distribution=LevyFlightDistribution(1.0, 1.3),
        dim=1,
        weights=np.ones(1, dtype=np.float64),
        seed=7777,
    )
    walk.set_dimension_bounded(
        3,
        np.array([-1.0, 0.0, -5.0]),
        np.array([3.0, 1.0, 5.0]),
    )
    out: list[float] = []
    step = np.zeros(3, dtype=np.float64)
    for _ in range(5):
        walk.next_real(step)
        out.extend(float(v) for v in step)
    expected = ref["C_isowalk_boxweighted_dim3"]
    for i, (got, want) in enumerate(zip(out, expected, strict=True)):
        tolerance.tight(got, float(want), reason=f"isowalk boxweighted[{i}]")
