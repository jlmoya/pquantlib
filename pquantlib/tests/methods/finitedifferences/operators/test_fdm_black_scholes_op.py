"""Tests for FdmBlackScholesOp (1-D BSM operator in log-spot).

# C++ parity: ql/methods/finitedifferences/operators/fdmblackscholesop.{hpp,cpp}
# @ v1.42.1.

The L5-D BSM operator implements

    L = (r - q - 0.5 sigma^2) D_x + 0.5 sigma^2 D_{xx} - r I

in log-spot coordinates. Tests check the operator is invariant
under the no-arbitrage condition (constant function maps to -r * const)
and that ``solve_splitting`` is the inverse of ``(I + dt * L)``.
"""

from __future__ import annotations

import numpy as np

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import (
    FdmBlackScholesOp,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing.tolerance import loose, tight
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month


def _build_op_and_mesher() -> tuple[FdmBlackScholesOp, FdmMesherComposite, float]:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    expiry = ref + 365
    spot_q = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    process = GeneralizedBlackScholesProcess(x0=spot_q, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol)
    maturity = process.time(expiry)
    bs_mesher = FdmBlackScholesMesher(21, process, maturity, 100.0)
    mesher = FdmMesherComposite(bs_mesher)
    op = FdmBlackScholesOp(mesher, process, 100.0)
    return op, mesher, maturity


def test_apply_to_constant_is_minus_r_times_constant() -> None:
    """At interior nodes, L applied to f(x)=1 gives -r * 1 (since
    D_x[1] = 0 and D_xx[1] = 0).

    TIGHT-tier: pure interior — no boundary upwinding interferes.
    """
    op, mesher, _ = _build_op_and_mesher()
    op.set_time(0.0, 1.0)
    n = mesher.layout().size()
    f = np.ones(n, dtype=np.float64)
    out = op.apply(f)
    # Interior: -r * 1 = -0.05.
    for i in range(2, n - 2):  # skip near-boundary because first-derivative
        # upwinding leaks at indices 0 and 1
        tight(float(out[i]), -0.05)


def test_solve_splitting_is_inverse_of_implicit_step() -> None:
    """For any r, ``solve_splitting(0, r, -dt)`` returns x such that
    ``(I - dt*L) x = r``. Round-trip: y = (I - dt L) x must equal r
    (within LOOSE — depends on the operator condition number).
    """
    op, mesher, _ = _build_op_and_mesher()
    op.set_time(0.0, 1.0)
    # Use a smooth-ish rhs (avoid sharp boundaries).
    spots = np.exp(mesher.locations(0))
    rhs = np.array([max(s - 100.0, 0.0) for s in spots], dtype=np.float64)

    dt = 0.01
    x = op.solve_splitting(0, rhs, -dt)  # solve (I - dt*L) x = rhs
    # Apply (I - dt*L) to x and verify ≈ rhs.
    y = x - dt * op.apply(x)
    for a, b in zip(y, rhs, strict=True):
        loose(float(a), float(b))


def test_size_is_one() -> None:
    op, _, _ = _build_op_and_mesher()
    assert op.size() == 1


def test_apply_direction_nonzero_returns_unchanged() -> None:
    """The 1-D op only handles direction == 0. ``solve_splitting`` on
    direction != 0 returns rhs unchanged (multi-D fallback).
    """
    op, mesher, _ = _build_op_and_mesher()
    op.set_time(0.0, 1.0)
    rhs = np.linspace(1.0, 10.0, mesher.layout().size(), dtype=np.float64)
    result = op.solve_splitting(1, rhs, -0.01)
    # No-op pass-through.
    for a, b in zip(result, rhs, strict=True):
        tight(float(a), float(b))


def test_preconditioner_matches_solve_splitting() -> None:
    op, mesher, _ = _build_op_and_mesher()
    op.set_time(0.0, 1.0)
    rhs = np.ones(mesher.layout().size(), dtype=np.float64)
    pre = op.preconditioner(rhs, -0.01)
    ss = op.solve_splitting(0, rhs, -0.01)
    for a, b in zip(pre, ss, strict=True):
        tight(float(a), float(b))
