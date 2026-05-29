"""PiecewiseTimeDependentHestonModel behavioral + cross-validation tests.

Cross-validates against ``migration-harness/references/cluster/w1d.json``.

C++ parity:
ql/models/equity/piecewisetimedependenthestonmodel.{hpp,cpp} @ v1.42.1.

Tolerance choice:
* Parameter accessors at given t: EXACT (piecewise lookup is exact).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.math.optimization.constraint import (
    BoundaryConstraint,
    PositiveConstraint,
)
from pquantlib.models.equity.piecewise_time_dependent_heston_model import (
    PiecewiseTimeDependentHestonModel,
)
from pquantlib.models.parameter import PiecewiseConstantParameter
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib.time.time_grid import TimeGrid


def _build_degenerate_model() -> PiecewiseTimeDependentHestonModel:
    """All segments share the same params (collapses to plain Heston)."""
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)

    breaks = [0.5]
    theta = PiecewiseConstantParameter(breaks, PositiveConstraint())
    theta.set_param(0, 0.04)
    theta.set_param(1, 0.04)
    kappa = PiecewiseConstantParameter(breaks, PositiveConstraint())
    kappa.set_param(0, 2.0)
    kappa.set_param(1, 2.0)
    sigma = PiecewiseConstantParameter(breaks, PositiveConstraint())
    sigma.set_param(0, 0.3)
    sigma.set_param(1, 0.3)
    rho = PiecewiseConstantParameter(breaks, BoundaryConstraint(-1.0, 1.0))
    rho.set_param(0, -0.7)
    rho.set_param(1, -0.7)

    return PiecewiseTimeDependentHestonModel(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(100.0),
        v0=0.04,
        theta=theta,
        kappa=kappa,
        sigma=sigma,
        rho=rho,
        time_grid=TimeGrid.with_mandatory([0.5, 1.0]),
    )


def _build_two_segment_model() -> PiecewiseTimeDependentHestonModel:
    """Different params per segment."""
    dc = Actual365Fixed()
    ref = Date.from_ymd(15, Month.June, 2026)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)

    breaks = [0.5]
    theta = PiecewiseConstantParameter(breaks, PositiveConstraint())
    theta.set_param(0, 0.06)
    theta.set_param(1, 0.04)
    kappa = PiecewiseConstantParameter(breaks, PositiveConstraint())
    kappa.set_param(0, 2.5)
    kappa.set_param(1, 1.5)
    sigma = PiecewiseConstantParameter(breaks, PositiveConstraint())
    sigma.set_param(0, 0.4)
    sigma.set_param(1, 0.2)
    rho = PiecewiseConstantParameter(breaks, BoundaryConstraint(-1.0, 1.0))
    rho.set_param(0, -0.8)
    rho.set_param(1, -0.5)

    return PiecewiseTimeDependentHestonModel(
        risk_free_rate=rf,
        dividend_yield=div,
        s0=SimpleQuote(100.0),
        v0=0.04,
        theta=theta,
        kappa=kappa,
        sigma=sigma,
        rho=rho,
        time_grid=TimeGrid.with_mandatory([0.5, 1.0]),
    )


@pytest.fixture
def cpp_refs() -> dict[str, Any]:
    return load_reference("cluster/w1d")


def test_degenerate_params_at_segment_midpoints(cpp_refs: dict[str, Any]) -> None:
    """Constant params recover the same value across segments.

    # C++ parity: piecewisetimedependenthestonmodel.hpp:57-65.
    """
    model = _build_degenerate_model()
    r = cpp_refs["ptd_heston_degenerate"]
    exact(model.theta(0.25), r["theta_seg0"])
    exact(model.kappa(0.25), r["kappa_seg0"])
    exact(model.sigma(0.25), r["sigma_seg0"])
    exact(model.rho(0.25), r["rho_seg0"])
    exact(model.theta(0.75), r["theta_seg1"])
    exact(model.v0(), r["v0"])
    exact(model.s0(), r["s0"])


def test_two_segment_distinct_params_per_segment(
    cpp_refs: dict[str, Any],
) -> None:
    """2-segment params evaluate distinctly per midpoint.

    # C++ parity: piecewisetimedependenthestonmodel.hpp:57-65.
    """
    model = _build_two_segment_model()
    r = cpp_refs["ptd_heston_2segment"]
    exact(model.theta(0.25), r["theta_seg0"])
    exact(model.kappa(0.25), r["kappa_seg0"])
    exact(model.sigma(0.25), r["sigma_seg0"])
    exact(model.rho(0.25), r["rho_seg0"])
    exact(model.theta(0.75), r["theta_seg1"])
    exact(model.kappa(0.75), r["kappa_seg1"])
    exact(model.sigma(0.75), r["sigma_seg1"])
    exact(model.rho(0.75), r["rho_seg1"])


def test_arguments_layout_has_five_slots() -> None:
    """params() returns 5 logical slots (theta, kappa, sigma, rho, v0).

    # C++ parity: piecewisetimedependenthestonmodel.cpp:36 — CalibratedModel(5).
    """
    model = _build_two_segment_model()
    # 4 piecewise (size 2 each) + 1 constant (size 1) = 9 free params.
    assert len(model.arguments) == 5
    assert model.params().shape == (4 * 2 + 1,)


def test_time_grid_back_is_one_year() -> None:
    """time_grid.back() == 1.0."""
    model = _build_two_segment_model()
    exact(model.time_grid().back(), 1.0)
