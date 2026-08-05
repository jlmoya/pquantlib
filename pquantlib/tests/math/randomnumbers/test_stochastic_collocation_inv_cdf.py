"""Cross-validate StochasticCollocationInvCDF against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/randomnumbers.json``,
section ``stochastic_collocation_inv_cdf``.

Four configurations are pinned: an order-8 fit of the normal inverse itself
(where the answer should be near the identity, so the collocation error is
directly readable), an order-10 lognormal, the same with ``p_max`` set (which
changes ``sigma`` and hence the whole mapping), and an order-6 shifted normal
with ``p_min`` set.

.. rubric:: Tolerance derivation

Inside the collocation range the barycentric interpolant is well conditioned
and the only discrepancy is FMA contraction in the C++ Release build (~1e-15
relative), so TIGHT applies.

Outside it the interpolant is *extrapolating*, and the Lebesgue function
grows fast: at ``u = 1e-6`` in the ``p_max`` case the evaluation point is
6.211 while the outermost node is 4.859, and the Lebesgue function there is
3.5e3. Multiplying that by the largest collocation ordinate and by the ~1e-16
relative agreement of the ordinates bounds the difference at a few times
1e-12 absolute — which is what is observed (4.0e-12), and which is 1.7e-11
relative at a value of 0.23. The test therefore computes the Lebesgue
function at each evaluation point and derives the tolerance from it, rather
than hard-coding a number that happens to pass.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

import pytest

from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.integrals.gaussian_quadrature import GaussHermiteIntegration
from pquantlib.math.randomnumbers.stochastic_collocation_inv_cdf import (
    StochasticCollocationInvCDF,
)
from pquantlib.testing import reference_reader, tolerance

_ICN = InverseCumulativeNormal()

#: The four ``invCDF`` callables the probe was built with.
_INV_CDF: dict[str, Callable[[float], float]] = {
    "normal_order8": _ICN,
    "lognormal_order10": lambda p: math.exp(0.3 * _ICN(p) - 0.045),
    "lognormal_order10_pmax": lambda p: math.exp(0.3 * _ICN(p) - 0.045),
    "shifted_order6": lambda p: 2.0 * _ICN(p) + 1.5,
}

#: Double-precision relative agreement of the collocation ordinates between
#: the two builds (FMA contraction only).
_ORDINATE_EPS = 2.3e-16


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/randomnumbers")


def _make(case: dict[str, Any]) -> StochasticCollocationInvCDF:
    return StochasticCollocationInvCDF(
        _INV_CDF[case["label"]], case["order"], case["p_max"], case["p_min"]
    )


def _lebesgue(nodes: list[float], z: float) -> float:
    """Sum of |Lagrange basis| at ``z`` — the amplification of ordinate error."""
    total = 0.0
    for i, xi in enumerate(nodes):
        li = 1.0
        for j, xj in enumerate(nodes):
            if i != j:
                li *= (z - xj) / (xi - xj)
        total += abs(li)
    return total


def _assert_at(f: StochasticCollocationInvCDF, x: float, actual: float, expected: float) -> None:
    nodes = [float(v) for v in f.x()]
    scale = max(abs(float(v)) for v in f.y())
    bound = _lebesgue(nodes, x * f.sigma()) * scale * _ORDINATE_EPS
    if bound <= 1e-14:
        tolerance.tight(actual, expected)
        return
    tolerance.custom(
        actual,
        expected,
        abs_tol=10.0 * bound,
        rel_tol=1e-12,
        reason=(
            f"Lagrange extrapolation: Lebesgue function {bound / (scale * _ORDINATE_EPS):.3g} "
            f"at the evaluation point times ordinate scale {scale:.3g} times the "
            f"{_ORDINATE_EPS:.1e} FMA-level agreement of the ordinates bounds the "
            f"difference at {bound:.3g}"
        ),
    )


def test_all_four_configurations_are_pinned(cpp: dict[str, Any]) -> None:
    labels = {c["label"] for c in cpp["stochastic_collocation_inv_cdf"]}
    assert labels == set(_INV_CDF)


def test_value_matches_cpp(cpp: dict[str, Any]) -> None:
    """``value(x)`` — the normal-argument entry point."""
    for case in cpp["stochastic_collocation_inv_cdf"]:
        f = _make(case)
        for x, expected in zip(case["xs"], case["values"], strict=True):
            _assert_at(f, x, f.value(x), expected)


def test_call_matches_cpp(cpp: dict[str, Any]) -> None:
    """``__call__(u)`` — the probability entry point, including both tails."""
    for case in cpp["stochastic_collocation_inv_cdf"]:
        f = _make(case)
        for u, expected in zip(case["us"], case["calls"], strict=True):
            _assert_at(f, _ICN(u), f(u), expected)


def test_normal_fit_is_close_to_the_identity(cpp: dict[str, Any]) -> None:
    """Collocating the normal inverse against itself must reproduce ``x``.

    The residual is the collocation error, not a bug: an order-8 fit of an
    exactly-polynomial-representable function is exact up to conditioning, so
    the departures from ``x`` in the C++ reference are 1e-8-ish at |x| = 3 and
    are themselves worth pinning as the method's accuracy.
    """
    case = next(c for c in cpp["stochastic_collocation_inv_cdf"] if c["label"] == "normal_order8")
    for x, value in zip(case["xs"], case["values"], strict=True):
        assert abs(value - x) < 1e-7


def test_pmax_changes_sigma(cpp: dict[str, Any]) -> None:
    """``p_max`` rescales the normal argument; without it ``sigma`` is 1."""
    plain = _make(
        next(c for c in cpp["stochastic_collocation_inv_cdf"] if c["label"] == "lognormal_order10")
    )
    pinned = _make(
        next(
            c
            for c in cpp["stochastic_collocation_inv_cdf"]
            if c["label"] == "lognormal_order10_pmax"
        )
    )
    tolerance.exact(plain.sigma(), 1.0)
    assert pinned.sigma() != 1.0


def test_collocation_nodes_are_scaled_gauss_hermite(cpp: dict[str, Any]) -> None:
    """The abscissae are ``sqrt(2)`` times the Gauss-Hermite nodes.

    The scaling is what makes them the quadrature points of a *standard
    normal* rather than of ``exp(-x^2)``; dropping it would shrink the
    collocation range by 41% and silently change every extrapolated value.
    """
    for case in cpp["stochastic_collocation_inv_cdf"]:
        f = _make(case)
        gh = GaussHermiteIntegration(case["order"]).x()
        assert len(f.x()) == case["order"]
        for a, e in zip(f.x(), gh, strict=True):
            tolerance.tight(float(a), math.sqrt(2.0) * float(e))


def test_monotone_on_the_lognormal_fit(cpp: dict[str, Any]) -> None:
    """An inverse CDF must be non-decreasing in ``u`` over the fitted range."""
    case = next(
        c for c in cpp["stochastic_collocation_inv_cdf"] if c["label"] == "lognormal_order10"
    )
    values = [_make(case)(u) for u in case["us"]]
    assert all(b >= a for a, b in itertools.pairwise(values))
