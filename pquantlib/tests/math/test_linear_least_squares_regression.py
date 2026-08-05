"""Cross-validate the least-squares regression family against the v1.43 probe.

Reference: ``migration-harness/references/v143/math/tail.json`` —
``general_linear_least_squares``, ``linear_regression_1d`` and
``linear_regression_nd``.

Two designs are fitted with the same data: a well-conditioned quadratic basis,
and a deliberately rank-deficient one whose third basis function is twice the
second. The second is the discriminating case — C++ drops every direction
whose singular value fails ``w[i] > n * eps * w[0]``, so it returns the
minimum-norm coefficients of the surviving subspace rather than diverging.
A normal-equation solve would blow up there and ``numpy.linalg.lstsq`` would
cut off somewhere else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pquantlib.math.general_linear_least_squares import GeneralLinearLeastSquares
from pquantlib.math.linear_least_squares_regression import (
    LinearFct,
    LinearFcts,
    LinearLeastSquaresRegression,
    LinearRegression,
)
from pquantlib.testing import tolerance

_BASES: dict[str, list[Callable[[float], float]]] = {
    "quadratic": [lambda _v: 1.0, lambda v: v, lambda v: v * v],
    "rank_deficient": [lambda _v: 1.0, lambda v: v, lambda v: 2.0 * v],
}


def test_general_fit_matches_cpp_tight(v143_tail: dict[str, Any]) -> None:
    block = v143_tail["general_linear_least_squares"]
    x, y = block["x"], block["y"]
    for case in block["cases"]:
        fit = GeneralLinearLeastSquares(x, y, _BASES[case["basis"]])
        assert fit.size() == case["size"]
        assert fit.dim() == case["dim"]
        for name, actual in (
            ("coefficients", fit.coefficients()),
            ("error", fit.error()),
            ("standard_errors", fit.standard_errors()),
            ("residuals", fit.residuals()),
        ):
            for i, (a, e) in enumerate(zip(actual, case[name], strict=True)):
                tolerance.tight(float(a), e, reason=f"{case['basis']}.{name}[{i}]")


def test_rank_deficient_basis_drops_the_degenerate_direction(v143_tail: dict[str, Any]) -> None:
    """With ``v_2 = 2 v_1`` the fit must stay finite and split the slope 1:2.

    Pinned as a structural property on top of the numeric comparison, because
    "returns finite garbage" and "returns the minimum-norm solution" both look
    like passing if only the coefficients are compared to a reference that was
    itself generated wrong.
    """
    block = v143_tail["general_linear_least_squares"]
    assert any(c["basis"] == "rank_deficient" for c in block["cases"])
    fit = GeneralLinearLeastSquares(block["x"], block["y"], _BASES["rank_deficient"])
    a = fit.coefficients()
    assert all(abs(float(v)) < 1e3 for v in a)
    # minimum-norm split of the effective slope a1 + 2 a2 across the two
    # collinear directions is a2 = 2 a1
    tolerance.tight(float(a[2]), 2.0 * float(a[1]))


def test_linear_regression_1d_matches_cpp_tight(v143_tail: dict[str, Any]) -> None:
    block = v143_tail["general_linear_least_squares"]
    for case in v143_tail["linear_regression_1d"]:
        fit = LinearRegression(block["x"], block["y"], case["intercept"])
        assert fit.dim() == case["dim"]
        for i, (a, e) in enumerate(zip(fit.coefficients(), case["coefficients"], strict=True)):
            tolerance.tight(float(a), e, reason=f"intercept={case['intercept']}, coef[{i}]")
        for i, (a, e) in enumerate(zip(fit.standard_errors(), case["standard_errors"], strict=True)):
            tolerance.tight(float(a), e, reason=f"intercept={case['intercept']}, se[{i}]")


def test_zero_intercept_drops_the_constant_term(v143_tail: dict[str, Any]) -> None:
    """``intercept=0.0`` removes the basis function; it does not fit it to zero."""
    cases = {c["intercept"]: c for c in v143_tail["linear_regression_1d"]}
    assert cases[0.0]["dim"] == cases[1.0]["dim"] - 1


def test_nonunit_intercept_rescales_the_constant(v143_tail: dict[str, Any]) -> None:
    """``intercept=2.5`` makes the basis function the constant 2.5, so the
    fitted coefficient is the unit-intercept one divided by 2.5.
    """
    cases = {c["intercept"]: c for c in v143_tail["linear_regression_1d"]}
    tolerance.tight(cases[2.5]["coefficients"][0] * 2.5, cases[1.0]["coefficients"][0])


def test_linear_regression_nd_matches_cpp_tight(v143_tail: dict[str, Any]) -> None:
    block = v143_tail["linear_regression_nd"]
    for case in block["cases"]:
        fit = LinearRegression(block["x"], block["y"], case["intercept"])
        assert fit.dim() == case["dim"]
        for i, (a, e) in enumerate(zip(fit.coefficients(), case["coefficients"], strict=True)):
            tolerance.tight(float(a), e, reason=f"intercept={case['intercept']}, coef[{i}]")
        for i, (a, e) in enumerate(zip(fit.residuals(), case["residuals"], strict=True)):
            tolerance.tight(float(a), e, reason=f"intercept={case['intercept']}, resid[{i}]")


def test_linear_fct_projects_a_component() -> None:
    assert LinearFct(0)([3.0, 5.0]) == 3.0
    assert LinearFct(1)([3.0, 5.0]) == 5.0


def test_linear_fcts_basis_shape() -> None:
    """The scalar and vector branches of the C++ ``if constexpr`` dispatch."""
    assert len(LinearFcts([1.0, 2.0, 3.0], 1.0).fcts()) == 2
    assert len(LinearFcts([1.0, 2.0, 3.0], 0.0).fcts()) == 1
    assert len(LinearFcts([[1.0, 2.0], [3.0, 4.0]], 1.0).fcts()) == 3
    assert len(LinearFcts([[1.0, 2.0], [3.0, 4.0]], 0.0).fcts()) == 2


def test_backward_compatible_alias_fits_identically(v143_tail: dict[str, Any]) -> None:
    """``LinearLeastSquaresRegression`` is the same fit under the retired name."""
    block = v143_tail["general_linear_least_squares"]
    a = GeneralLinearLeastSquares(block["x"], block["y"], _BASES["quadratic"]).coefficients()
    b = LinearLeastSquaresRegression(block["x"], block["y"], _BASES["quadratic"]).coefficients()
    for x, y in zip(a, b, strict=True):
        tolerance.exact(float(x), float(y))
