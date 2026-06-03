"""Tests for the legacy FD PDE / grid-operator construction layer (WS3-FD2).

# Retired-API compat layer — see package docstring.

The log-grid ``BSMOperator.from_grid`` path is cross-validated against the
uniform-grid ``BSMOperator`` coefficient formula: on a *log-uniform* grid the
non-uniform stencil (``dxm == dxp == h``, ``dx == 2h``) reduces algebraically to
the uniform stencil, so the interior coefficient bands must match
``BSMOperator(size, h, r, q, sigma)`` exactly (the uniform op is itself TIGHT-
validated against C++ in WS3-FD1). The ``TransformedGrid`` spacings and the
``PdeBSM`` coefficient delegation are checked against hand-derived values.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import tolerance
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month
from pquantlib_helpers.methods.finitedifferences.bsm_operator import BSMOperator
from pquantlib_helpers.methods.finitedifferences.pde import (
    PdeBSM,
    PdeConstantCoeff,
)
from pquantlib_helpers.methods.finitedifferences.pde_operator import (
    BSMTermOperator,
    OperatorFactory,
)
from pquantlib_helpers.methods.finitedifferences.transformed_grid import (
    LogGrid,
    TransformedGrid,
)


def _dc() -> Actual365Fixed:
    return Actual365Fixed()


def _flat_process(
    *, s0: float, r: float, q: float, sigma: float
) -> GeneralizedBlackScholesProcess:
    today = Date.from_ymd(15, Month.January, 2024)
    dc = _dc()
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(s0),
        dividend_ts=FlatForward.from_rate(today, q, dc),
        risk_free_ts=FlatForward.from_rate(today, r, dc),
        black_vol_ts=BlackConstantVol(
            reference_date=today,
            calendar=NullCalendar(),
            volatility=sigma,
            day_counter=dc,
        ),
    )


# --- TransformedGrid / LogGrid ----------------------------------------------


def test_transformed_grid_spacings_uniform() -> None:
    grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    tg = TransformedGrid(grid)
    # interior spacings all == 1 (backward/forward), total == 2
    for i in (1, 2, 3):
        tolerance.exact(tg.dxm(i), 1.0)
        tolerance.exact(tg.dxp(i), 1.0)
        tolerance.exact(tg.dx(i), 2.0)
    # boundary spacings left at zero
    tolerance.exact(tg.dxm(0), 0.0)
    tolerance.exact(tg.dxp(4), 0.0)


def test_log_grid_transforms_with_log() -> None:
    grid = np.array([1.0, math.e, math.e**2, math.e**3])
    lg = LogGrid(grid)
    for i in range(4):
        tolerance.tight(lg.log_grid(i), float(i))
    # spacings of log(grid) are uniform == 1
    tolerance.tight(lg.dxm(1), 1.0)
    tolerance.tight(lg.dxp(1), 1.0)


# --- PdeBSM coefficients ----------------------------------------------------


def test_pde_bsm_diffusion_drift_delegate_to_process() -> None:
    proc = _flat_process(s0=100.0, r=0.05, q=0.01, sigma=0.20)
    pde = PdeBSM(proc)
    t, x = 0.5, math.log(100.0)
    tolerance.tight(pde.diffusion(t, x), proc.diffusion_1d(t, x))
    tolerance.tight(pde.drift(t, x), proc.drift_1d(t, x))


def test_pde_constant_coeff_freezes_values() -> None:
    proc = _flat_process(s0=100.0, r=0.05, q=0.01, sigma=0.20)
    pde = PdeBSM(proc)
    t0, x0 = 0.5, math.log(100.0)
    cc = PdeConstantCoeff(pde, t0, x0)
    # frozen: same value regardless of query point
    tolerance.tight(cc.diffusion(1.0, 7.0), pde.diffusion(t0, x0))
    tolerance.tight(cc.drift(1.0, 7.0), pde.drift(t0, x0))
    tolerance.tight(cc.discount(1.0, 7.0), pde.discount(t0, x0))


# --- BSMOperator.from_grid (log-grid) vs uniform formula --------------------


def test_bsm_from_loguniform_grid_matches_uniform_operator() -> None:
    r, q, sigma = 0.05, 0.01, 0.20
    proc = _flat_process(s0=100.0, r=r, q=q, sigma=sigma)
    # build a log-uniform grid: S = exp(log(S0) + j*h)
    h = 0.1
    n = 9
    centre = math.log(100.0)
    log_nodes = centre + (np.arange(n) - n // 2) * h
    grid = np.exp(log_nodes)

    residual_time = 0.0  # discount(t=0) == r for a flat curve
    op = BSMOperator.from_grid(grid, proc, residual_time)
    # the constant-coeff log-uniform stencil must equal the uniform formula
    ref = BSMOperator(n, h, r, q, sigma)
    # interior rows only (boundary rows differ: from_grid leaves them zero too).
    # LOOSE tier (not TIGHT): the grid is built via exp(log_nodes) and the
    # operator internally re-applies log(), so the recovered log-spacings carry
    # ~1e-12 relative round-trip error vs the exact uniform step `h`. The stencil
    # *logic* is identical; only the fp round-trip differs.
    for i in range(1, n - 1):
        tolerance.loose(float(op.diagonal()[i]), float(ref.diagonal()[i]))
        tolerance.loose(float(op.lower_diagonal()[i - 1]), float(ref.lower_diagonal()[i - 1]))
        tolerance.loose(float(op.upper_diagonal()[i]), float(ref.upper_diagonal()[i]))


def test_operator_factory_constant_path_returns_bsm_operator() -> None:
    proc = _flat_process(s0=100.0, r=0.05, q=0.01, sigma=0.20)
    h = 0.1
    n = 7
    grid = np.exp(math.log(100.0) + (np.arange(n) - n // 2) * h)
    op = OperatorFactory.get_operator(proc, grid, 0.0, time_dependent=False)
    assert isinstance(op, BSMOperator)
    assert op.size() == n


def test_operator_factory_time_dependent_path_returns_term_operator() -> None:
    proc = _flat_process(s0=100.0, r=0.05, q=0.01, sigma=0.20)
    h = 0.1
    n = 7
    grid = np.exp(math.log(100.0) + (np.arange(n) - n // 2) * h)
    op = OperatorFactory.get_operator(proc, grid, 0.5, time_dependent=True)
    assert isinstance(op, BSMTermOperator)
    assert op.is_time_dependent()
    # interior coefficients are populated (non-zero diagonal)
    assert abs(float(op.diagonal()[n // 2])) > 0.0


def test_bsm_term_operator_resets_on_set_time() -> None:
    proc = _flat_process(s0=100.0, r=0.05, q=0.01, sigma=0.20)
    h = 0.1
    n = 7
    grid = np.exp(math.log(100.0) + (np.arange(n) - n // 2) * h)
    op = BSMTermOperator(grid, proc, 0.5)
    before = float(op.diagonal()[n // 2])
    op.set_time(1.0)  # flat curve -> same discount, coefficients unchanged
    after = float(op.diagonal()[n // 2])
    tolerance.tight(after, before)
