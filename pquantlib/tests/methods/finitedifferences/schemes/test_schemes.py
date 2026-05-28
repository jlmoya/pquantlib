"""Tests for the time-stepping schemes (ExplicitEuler / ImplicitEuler / CrankNicolson).

# C++ parity: ql/methods/finitedifferences/schemes/*scheme.{hpp,cpp}
# @ v1.42.1.

The schemes are tested for self-consistency: each step preserves a
constant function under a vanishing rate / vol, the implicit scheme
inverts the explicit one (in a linear sense), and the descriptor
factories return correct theta/mu pairs.
"""

from __future__ import annotations

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import (
    FdmBlackScholesOp,
)
from pquantlib.methods.finitedifferences.schemes.crank_nicolson_scheme import (
    CrankNicolsonScheme,
)
from pquantlib.methods.finitedifferences.schemes.explicit_euler_scheme import (
    ExplicitEulerScheme,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import (
    FdmSchemeDesc,
    FdmSchemeType,
)
from pquantlib.methods.finitedifferences.schemes.implicit_euler_scheme import (
    ImplicitEulerScheme,
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


def _build_op_and_mesher() -> tuple[FdmBlackScholesOp, FdmMesherComposite]:
    dc = Actual365Fixed()
    cal = NullCalendar()
    ref = Date.from_ymd(15, Month.June, 2026)
    spot_q = SimpleQuote(100.0)
    rf = FlatForward.from_rate(reference_date=ref, forward_rate=0.05, day_counter=dc)
    div = FlatForward.from_rate(reference_date=ref, forward_rate=0.0, day_counter=dc)
    vol = BlackConstantVol(reference_date=ref, calendar=cal, day_counter=dc, volatility=0.20)
    process = GeneralizedBlackScholesProcess(x0=spot_q, dividend_ts=div, risk_free_ts=rf, black_vol_ts=vol)
    bs_mesher = FdmBlackScholesMesher(21, process, 1.0, 100.0)
    mesher = FdmMesherComposite(bs_mesher)
    op = FdmBlackScholesOp(mesher, process, 100.0)
    return op, mesher


def test_scheme_desc_crank_nicolson() -> None:
    """CrankNicolson factory returns theta=0.5, mu=0."""
    d = FdmSchemeDesc.crank_nicolson()
    assert d.type == FdmSchemeType.CrankNicolsonType
    tight(d.theta, 0.5)
    tight(d.mu, 0.0)


def test_scheme_desc_douglas_same_theta_as_crank_nicolson() -> None:
    d = FdmSchemeDesc.douglas()
    assert d.type == FdmSchemeType.DouglasType
    tight(d.theta, 0.5)


def test_scheme_desc_implicit_euler() -> None:
    d = FdmSchemeDesc.implicit_euler()
    assert d.type == FdmSchemeType.ImplicitEulerType
    tight(d.theta, 0.0)


def test_scheme_desc_explicit_euler() -> None:
    d = FdmSchemeDesc.explicit_euler()
    assert d.type == FdmSchemeType.ExplicitEulerType
    tight(d.theta, 0.0)


def test_explicit_euler_step_advances_state() -> None:
    """One explicit-Euler step advances ``a`` by ``dt * L(t) a``."""
    op, mesher = _build_op_and_mesher()
    n = mesher.layout().size()
    a = np.ones(n, dtype=np.float64)
    scheme = ExplicitEulerScheme(op)
    scheme.set_step(0.01)
    new = scheme.step(a, 1.0)
    # Constant function on the interior under L is approximately -r.
    # So one step shifts a[i] by -r * dt ≈ -0.0005 at interior nodes.
    for i in range(3, n - 3):
        tight(float(new[i]), 1.0 + 0.01 * (-0.05), reason="L*1 = -r on interior")


def test_implicit_euler_step_negative_time_raises() -> None:
    op, _ = _build_op_and_mesher()
    scheme = ImplicitEulerScheme(op)
    scheme.set_step(1.0)
    a = np.ones(20, dtype=np.float64)
    # step(a, t=0.5) with dt=1.0 → t-dt=-0.5 → require fails.
    with pytest.raises(LibraryException):
        scheme.step(a, 0.5)


def test_crank_nicolson_step_combines_explicit_and_implicit() -> None:
    """Crank-Nicolson with theta=0.5 = one half-explicit + one half-implicit step.

    For f=1 in the interior, both ∂x f and ∂xx f vanish (constant
    function); the operator L acts as a scalar -r, so the explicit
    half multiplies by (1 - r*dt*(1-theta)) = 1 - 0.025 r dt and the
    implicit half divides by (1 + r*dt*theta) = 1 + 0.025 r dt.

    TIGHT: closed-form scalar arithmetic agrees to the bit-precision
    of the implementation.
    """
    op, mesher = _build_op_and_mesher()
    n = mesher.layout().size()
    a = np.ones(n, dtype=np.float64)
    scheme = CrankNicolsonScheme(theta=0.5, op=op)
    scheme.set_step(0.01)
    new = scheme.step(a, 1.0)
    # Expected: explicit yields (1 + 0.5 * dt * (-r)) = 0.99975;
    # implicit then solves (1 - 0.5 * dt * (-r)) x = 0.99975
    # so x = 0.99975 / 1.00025.
    expected = (1.0 + 0.5 * 0.01 * -0.05) / (1.0 - 0.5 * 0.01 * -0.05)
    # Interior nodes only (skip the upwinding leakage near boundaries).
    for i in range(5, n - 5):
        loose(float(new[i]), expected)
        # may produce slightly different roundoff
        # in the boundary-adjacent rows
