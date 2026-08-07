"""The seven fitting methods, cross-validated against C++ v1.43.

Reference: migration-harness/references/v143/ts/fittedbond.json
Probe:     migration-harness/cpp/probes/v143_ts_fittedbond/probe.cpp

Everything here is LAYER 1 of the probe: ``FittingMethod::discount(x, t)``,
``size()`` and the basis functions, evaluated at hand-chosen parameter vectors.
No bonds, no optimizer, so nothing in this module is path-dependent and every
assertion is TIGHT or better.

The sampled times deliberately straddle the cutoffs (probe.cpp:106-108): 0.0
and 1e-6 sit below every ``min_cutoff_time`` used, 20.0 and 30.0 above every
``max_cutoff_time``, so both flat-forward extrapolation branches of
``FittingMethod::discount`` (fittedbonddiscountcurve.hpp:358-371) are live.

No evaluation-date fixture: not one object built here reads
``Settings::evaluationDate``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.bernstein_polynomial import BernsteinPolynomial
from pquantlib.termstructures.yield_.fitted_bond_discount_curve import FittingMethod
from pquantlib.termstructures.yield_.nonlinear_fitting_methods import (
    CubicBSplinesFitting,
    ExponentialSplinesFitting,
    NaturalCubicFitting,
    NelsonSiegelFitting,
    SimplePolynomialFitting,
    SvenssonFitting,
)
from pquantlib.testing import reference_reader, tolerance

CPP: dict[str, Any] = reference_reader.load("v143/ts/fittedbond")
TIMES: list[float] = CPP["meta"]["times"]

# probe.cpp:271-272 and 296-297.
KNOTS12: list[float] = [-30.0, -20.0, -10.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]
KNOTS9: list[float] = [-10.0, -5.0, 0.0, 4.0, 8.0, 12.0, 20.0, 30.0, 40.0]


def _size_cases() -> dict[str, FittingMethod]:
    """One instance per ``layer1_size`` key. probe.cpp:245-290.

    ``spread_over_svensson`` is absent: ``SpreadFittingMethod`` needs a
    discount curve, so it is exercised in test_fitted_bond_discount_curve.py.
    """
    return {
        "exp_c1_kfree_n9": ExponentialSplinesFitting(True),
        "exp_c0_kfree_n9": ExponentialSplinesFitting(False),
        "exp_c1_kfix_n9": ExponentialSplinesFitting(True, num_coeffs=9, fixed_kappa=0.05),
        "exp_c0_kfix_n9": ExponentialSplinesFitting(False, num_coeffs=9, fixed_kappa=0.05),
        "exp_c1_kfree_n5": ExponentialSplinesFitting(True, num_coeffs=5, fixed_kappa=None),
        "exp_c1_kfix_n2": ExponentialSplinesFitting(True, num_coeffs=2, fixed_kappa=0.05),
        "exp_ctor_l2_c1_n7_kfix": ExponentialSplinesFitting(
            True, num_coeffs=7, fixed_kappa=0.05
        ),
        "nelson_siegel": NelsonSiegelFitting(),
        "svensson": SvenssonFitting(),
        "bspline_k12_c1": CubicBSplinesFitting(KNOTS12, True),
        "bspline_k12_c0": CubicBSplinesFitting(KNOTS12, False),
        "bspline_k9_c1": CubicBSplinesFitting(KNOTS9, True),
        "natcubic_with_zero": NaturalCubicFitting([0.0, 1.0, 3.0, 7.0, 15.0]),
        "natcubic_without_zero": NaturalCubicFitting([1.0, 3.0, 7.0, 15.0]),
        "natcubic_unsorted_dup": NaturalCubicFitting([7.0, 1.0, 3.0, 1.0, 15.0, 0.0]),
        "natcubic_near_dup_1e15": NaturalCubicFitting([1.0, 1.0 + 1e-15, 3.0]),
        "poly_d3_c1": SimplePolynomialFitting(3, True),
        "poly_d3_c0": SimplePolynomialFitting(3, False),
        "poly_d1_c1": SimplePolynomialFitting(1, True),
    }


def _discount_cases() -> dict[str, FittingMethod]:
    """One instance per ``layer1_discount`` key. probe.cpp:294-378."""
    return {
        "exp_c1_kfree": ExponentialSplinesFitting(True),
        "exp_c0_kfree": ExponentialSplinesFitting(False),
        "exp_c1_kfix": ExponentialSplinesFitting(True, num_coeffs=9, fixed_kappa=0.05),
        "exp_c0_kfix": ExponentialSplinesFitting(False, num_coeffs=9, fixed_kappa=0.05),
        "nelson_siegel": NelsonSiegelFitting(),
        "nelson_siegel_cutoff": NelsonSiegelFitting(min_cutoff_time=1.0, max_cutoff_time=5.0),
        "svensson": SvenssonFitting(),
        "svensson_cutoff": SvenssonFitting(min_cutoff_time=0.5, max_cutoff_time=8.0),
        "bspline_c1": CubicBSplinesFitting(KNOTS12, True),
        "bspline_c0": CubicBSplinesFitting(KNOTS12, False),
        "natural_cubic": NaturalCubicFitting([1.0, 3.0, 7.0, 15.0, 30.0]),
        "natural_cubic_dedup": NaturalCubicFitting([7.0, 1.0, 3.0, 1.0, 15.0, 0.0]),
        "poly_c1": SimplePolynomialFitting(3, True),
        "poly_c0": SimplePolynomialFitting(3, False),
    }


# ---------------------------------------------------------------------------
# size()
# ---------------------------------------------------------------------------


def test_every_probed_size_case_is_covered() -> None:
    covered = set(_size_cases()) | {"spread_over_svensson"}
    assert covered == set(CPP["layer1_size"])


@pytest.mark.parametrize("key", sorted(_size_cases()))
def test_size_matches_cpp(key: str) -> None:
    assert _size_cases()[key].size() == CPP["layer1_size"][key], key


def test_exponential_splines_size_depends_on_both_switches() -> None:
    """The discriminating shape of ``ExponentialSplinesFitting::size``.

    cpp:69-73 makes ``size()`` a function of BOTH ``constrainAtZero`` and
    whether ``fixedKappa`` is null; a port that dropped either dependency
    would still agree on some of the four cells, so all four are pinned.
    """
    sizes = CPP["layer1_size"]
    assert sizes["exp_c0_kfree_n9"] == sizes["exp_c1_kfree_n9"] + 1
    assert sizes["exp_c1_kfix_n9"] == sizes["exp_c1_kfree_n9"] - 1
    assert sizes["exp_c0_kfix_n9"] == sizes["exp_c0_kfree_n9"] - 1


def test_natural_cubic_normalises_the_knot_vector() -> None:
    """0.0 is appended, the result sorted, then de-duplicated at 1e-14.

    cpp:279-284. Four knot lists that describe the same knot set once
    normalised must give the same ``size()``, and a pair 1e-15 apart must
    collapse to one knot.
    """
    sizes = CPP["layer1_size"]
    assert sizes["natcubic_with_zero"] == sizes["natcubic_without_zero"]
    assert sizes["natcubic_unsorted_dup"] == sizes["natcubic_with_zero"]
    # {1, 1+1e-15, 3} + {0} de-dups to {0, 1, 3} -> size 2.
    assert sizes["natcubic_near_dup_1e15"] == 2


# ---------------------------------------------------------------------------
# discount(x, t)
# ---------------------------------------------------------------------------


def test_every_probed_discount_case_is_covered() -> None:
    assert set(_discount_cases()) == set(CPP["layer1_discount"])


@pytest.mark.parametrize("key", sorted(_discount_cases()))
def test_discount_matches_cpp(key: str) -> None:
    method = _discount_cases()[key]
    block = CPP["layer1_discount"][key]
    x: npt.NDArray[np.float64] = np.array(block["x"], dtype=np.float64)
    assert method.size() == block["size"], key
    for t, expected in zip(TIMES, block["d"], strict=True):
        tolerance.tight(method.discount(x, t), expected, reason=f"{key} t={t}")


# The subset whose discount function accumulates with ``d += x[i] * ...`` and
# is therefore FMA-contracted by Clang. These must stay BIT-exact: the whole
# reason nonlinear_fitting_methods.py calls math.fma is that a half-ulp here
# flips a simplex comparison later. See that module's docstring.
_FMA_KEYS = ("exp_c1_kfree", "exp_c0_kfree", "exp_c1_kfix", "exp_c0_kfix", "poly_c1", "poly_c0")


@pytest.mark.parametrize("key", _FMA_KEYS)
def test_fma_contracted_discount_functions_are_bit_exact(key: str) -> None:
    method = _discount_cases()[key]
    block = CPP["layer1_discount"][key]
    x: npt.NDArray[np.float64] = np.array(block["x"], dtype=np.float64)
    for t, expected in zip(TIMES, block["d"], strict=True):
        tolerance.exact(method.discount(x, t), expected, reason=f"{key} t={t}")


def test_cutoffs_bend_the_curve_only_outside_the_band() -> None:
    """The cutoff variants must agree with the plain ones INSIDE the band.

    Between ``min_cutoff_time`` and ``max_cutoff_time`` the discount is the
    bare ``discountFunction``; the two branches only fire outside. Sampled
    times 1.0/3.0/5.0 lie inside Nelson-Siegel's [1, 5] band and 0.25/7.5/20/30
    outside it, so the same reference file proves both halves.

    ``t = 0`` is the one outside point where the two agree, and necessarily so:
    the min-cutoff branch returns ``exp(log(d(min))/min * 0) == 1`` and the
    plain function returns ``exp(-r*0) == 1``.
    """
    plain = CPP["layer1_discount"]["nelson_siegel"]["d"]
    cutoff = CPP["layer1_discount"]["nelson_siegel_cutoff"]["d"]
    inside = [i for i, t in enumerate(TIMES) if 1.0 <= t <= 5.0]
    outside = [i for i, t in enumerate(TIMES) if not 1.0 <= t <= 5.0 and t != 0.0]
    assert inside
    assert outside
    for i in inside:
        tolerance.exact(cutoff[i], plain[i], reason=f"t={TIMES[i]} inside the band")
    zero_index = TIMES.index(0.0)
    assert cutoff[zero_index] == 1.0
    assert plain[zero_index] == 1.0
    assert all(cutoff[i] != plain[i] for i in outside), "cutoff never bent the curve"


# ---------------------------------------------------------------------------
# basis functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("i", range(8))
def test_cubic_bspline_basis_function_matches_cpp(i: int) -> None:
    method = CubicBSplinesFitting(KNOTS12, True)
    for t, expected in zip(TIMES, CPP["layer1_bspline_basis"][f"N{i}"], strict=True):
        tolerance.tight(method.basis_function(i, t), expected, reason=f"N{i} t={t}")


@pytest.mark.parametrize("i", range(5))
def test_bernstein_basis_matches_cpp(i: int) -> None:
    """``B_i^i(t)``, the only Bernstein call SimplePolynomialFitting makes.

    cpp:378-385. ``B_n^n(t) = C(n,n) t^n (1-t)^0``, so these are the monomials
    — including for ``t > 1``, where the ``(1-t)^0`` factor is exactly 1 and
    the two spellings coincide.
    """
    for t, expected in zip(TIMES, CPP["layer1_bernstein"][f"B{i}_{i}"], strict=True):
        tolerance.exact(BernsteinPolynomial.get(i, i, t), expected, reason=f"B{i}_{i} t={t}")


# ---------------------------------------------------------------------------
# constructor guards
# ---------------------------------------------------------------------------


def test_exponential_splines_rejects_zero_free_coefficients() -> None:
    """cpp:41 — ``size() > 0``; with kappa fixed, numCoeffs 1 leaves none."""
    with pytest.raises(LibraryException, match="At least 1 unconstrained coefficient required"):
        ExponentialSplinesFitting(True, num_coeffs=1, fixed_kappa=0.05)


def test_cubic_bsplines_rejects_a_short_knot_vector() -> None:
    """cpp:197-198 — at least 8 knots.

    With 8 knots exactly it constructs; with 7 the BSpline built in the
    member-initialiser list (cpp:195) rejects it first, for its own reason
    (``p <= n`` fails at p = 3, n = 2), which is why the two raise different
    messages.
    """
    CubicBSplinesFitting([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], False)
    with pytest.raises(LibraryException, match="must have p <= n"):
        CubicBSplinesFitting([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], False)


def test_natural_cubic_rejects_degenerate_knot_sets() -> None:
    """cpp:286-298 — at least two knots, strictly increasing after de-dup."""
    with pytest.raises(LibraryException, match="at least two knot times required"):
        NaturalCubicFitting([0.0])
    with pytest.raises(LibraryException, match="at least two knot times required"):
        NaturalCubicFitting([1e-15])


def test_cubic_bsplines_rejects_knots_with_a_null_nth_basis_at_zero() -> None:
    """cpp:208-209 — constrainAtZero needs ``|N_1(0)| > QL_EPSILON``.

    N_1 is supported on ``[knots[1], knots[5]]``; pushing that window entirely
    to the right of 0 makes the coefficient solved for at t=0 blow up, and C++
    refuses rather than producing an ill-conditioned problem.
    """
    with pytest.raises(LibraryException, match="must be nonzero at t=0"):
        CubicBSplinesFitting([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], True)


# ---------------------------------------------------------------------------
# clone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(_discount_cases()))
def test_clone_is_an_independent_object_of_the_same_class(key: str) -> None:
    """``Clone<FittingMethod>`` copy semantics (fittedbonddiscountcurve.hpp:164).

    The clone must be a different object of the SAME class, must price
    identically, and must not share the parameter arrays — C++ gets the deep
    array copy from ``Array``'s copy constructor, NumPy needs it asked for.
    """
    original = _discount_cases()[key]
    clone = original.clone()
    assert clone is not original
    assert type(clone) is type(original)
    assert clone.size() == original.size()

    block = CPP["layer1_discount"][key]
    x: npt.NDArray[np.float64] = np.array(block["x"], dtype=np.float64)
    for t in TIMES:
        tolerance.exact(clone.discount(x, t), original.discount(x, t), reason=f"{key} t={t}")

    # Writing through the clone's weights must not reach the original.
    clone._weights = np.array([1.0, 2.0])  # pyright: ignore[reportPrivateUsage]
    assert original.weights().size == 0
