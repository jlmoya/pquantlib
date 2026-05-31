"""Tests for the W9-A discounters + swap/forward mappings + correlation base.

Cross-validates against ``migration-harness/references/cluster/w9a.json``.

C++ parity:
  ql/models/marketmodels/discounter.{hpp,cpp}
  ql/models/marketmodels/pathwisediscounter.{hpp,cpp}
  ql/models/marketmodels/swapforwardmappings.{hpp,cpp}
  ql/models/marketmodels/forwardforwardmappings.{hpp,cpp}
  ql/models/marketmodels/piecewiseconstantcorrelation.hpp
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.discounter import MarketModelDiscounter
from pquantlib.models.marketmodels.forward_forward_mappings import ForwardForwardMappings
from pquantlib.models.marketmodels.pathwise_discounter import (
    MarketModelPathwiseDiscounter,
)
from pquantlib.models.marketmodels.piecewise_constant_correlation import (
    PiecewiseConstantCorrelation,
)
from pquantlib.models.marketmodels.swap_forward_mappings import SwapForwardMappings
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w9a")


def _rate_times6() -> list[float]:
    return [0.5 * (i + 1) for i in range(6)]


def _fwds5() -> list[float]:
    return [0.04, 0.045, 0.05, 0.055, 0.06]


def _lmm() -> LMMCurveState:
    cs = LMMCurveState(_rate_times6())
    cs.set_on_forward_rates(_fwds5())
    return cs


# --- SwapForwardMappings -----------------------------------------------------


def test_coterminal_swap_forward_jacobian(ref: dict[str, Any]) -> None:
    cs = _lmm()
    jac = SwapForwardMappings.coterminal_swap_forward_jacobian(cs)
    # TIGHT: analytic jacobian — pure algebra.
    tight(float(jac[0, 0]), ref["jac_0_0"])
    tight(float(jac[0, 4]), ref["jac_0_4"])
    tight(float(jac[2, 2]), ref["jac_2_2"])
    tight(float(jac[2, 4]), ref["jac_2_4"])
    tight(float(jac[4, 4]), ref["jac_4_4"])
    # below-diagonal entry must be zero
    tight(float(jac[4, 0]), ref["jac_4_0"])


def test_annuity_and_swap_derivative(ref: dict[str, Any]) -> None:
    cs = _lmm()
    tight(SwapForwardMappings.annuity(cs, 0, 5, 5), ref["jac_annuity_0_5_5"])
    tight(
        SwapForwardMappings.swap_derivative(cs, 0, 5, 2),
        ref["jac_swapderiv_0_5_2"],
    )


def test_swap_derivative_out_of_range_is_zero() -> None:
    cs = _lmm()
    # forward index before start or at/after end -> 0
    assert SwapForwardMappings.swap_derivative(cs, 1, 4, 0) == 0.0
    assert SwapForwardMappings.swap_derivative(cs, 1, 4, 4) == 0.0


def test_cm_swap_zed_matrix_shape() -> None:
    cs = _lmm()
    z = SwapForwardMappings.cm_swap_zed_matrix(cs, 2, 0.0)
    assert z.shape == (5, 5)
    # lower-triangular strictly-below-diagonal entries untouched (zero jacobian)
    assert float(z[4, 0]) == 0.0


# --- MarketModelDiscounter ---------------------------------------------------


def test_market_model_discounter(ref: dict[str, Any]) -> None:
    cs = _lmm()
    rt = _rate_times6()
    # payment at a rate time
    tight(MarketModelDiscounter(2.0, rt).numeraire_bonds(cs, 5), ref["disc_pay2_num5"])
    # payment between rate times (log-linear)
    tight(
        MarketModelDiscounter(1.75, rt).numeraire_bonds(cs, 5),
        ref["disc_pay175_num5"],
    )
    # payment after the last rate time (clamped)
    tight(MarketModelDiscounter(3.5, rt).numeraire_bonds(cs, 5), ref["disc_pay35_num5"])
    # different numeraire
    tight(
        MarketModelDiscounter(1.75, rt).numeraire_bonds(cs, 0),
        ref["disc_pay175_num0"],
    )


# --- MarketModelPathwiseDiscounter -------------------------------------------


def test_pathwise_discounter_value_matches_curve_state() -> None:
    # The pathwise discounter's factors[0] (the discount) should match the
    # money-market discount of the curve-state discount ratios at step 0.
    cs = _lmm()
    rt = _rate_times6()
    n_rates = len(rt) - 1
    # Discounts row: P(t_0, t_j) = discount_ratio(j, 0) for the single step.
    discounts = np.zeros((1, n_rates + 1), dtype=np.float64)
    for j in range(n_rates + 1):
        discounts[0, j] = cs.discount_ratio(j, 0)
    libor = np.zeros((1, n_rates), dtype=np.float64)
    pd = MarketModelPathwiseDiscounter(1.75, rt)
    factors = [0.0] * (n_rates + 1)
    pd.get_factors(libor, discounts, 0, factors)
    # factors[0] is the discount; compare to the non-pathwise discounter
    # rebased to numeraire 0 (== P(payment, t_0) discount).
    expected = MarketModelDiscounter(1.75, rt).numeraire_bonds(cs, 0)
    tight(factors[0], expected)


# --- PiecewiseConstantCorrelation --------------------------------------------


def test_piecewise_constant_correlation_concrete_accessor() -> None:
    # Cannot instantiate the ABC directly.
    with pytest.raises(TypeError):
        PiecewiseConstantCorrelation()  # type: ignore[abstract]

    class _Const(PiecewiseConstantCorrelation):
        def __init__(self) -> None:
            self._c = [np.eye(3, dtype=np.float64), 0.5 * np.eye(3, dtype=np.float64)]

        def times(self) -> list[float]:
            return [1.0, 2.0]

        def rate_times(self) -> list[float]:
            return [1.0, 2.0, 3.0]

        def correlations(self) -> list[Any]:
            return self._c

        def number_of_rates(self) -> int:
            return 3

    c = _Const()
    assert float(c.correlation(0)[0, 0]) == 1.0
    assert float(c.correlation(1)[0, 0]) == 0.5
    with pytest.raises(LibraryException):
        c.correlation(2)


# --- ForwardForwardMappings --------------------------------------------------


def test_restrict_curve_state_roundtrip() -> None:
    # Restricting with multiplier=1, offset=0 should reproduce the same
    # discount ratios (identity restriction).
    cs = _lmm()
    restricted = ForwardForwardMappings.restrict_curve_state(cs, 1, 0)
    for i in range(6):
        tight(restricted.discount_ratio(i, 0), cs.discount_ratio(i, 0))
