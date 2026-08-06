"""``FdmBlackScholesOp``'s local-vol and quanto branches, against C++ v1.43.

# C++ parity: migration-harness/cpp/probes/v143_methods_bsop/probe.cpp
# @ v1.43 (submodule 6b57206e0).

``FdmBlackScholesOp::setTime`` has four arms — {constant vol, local vol} x
{no quanto, quanto} — plus the ``illegalLocalVolOverwrite`` sub-branch of
the local-vol loop. The Python port had only the first arm; the other two
constructor arguments were accepted and dropped, and
``FdmBlackScholesSolver`` refused to run with them at all.

Each arm is pinned twice:

* on the operator, through ``apply`` on three interleaved indicator
  vectors — for a tridiagonal operator those three products determine
  every band, so this compares the coefficients themselves rather than
  something downstream of them;
* end to end, through a 100-step ``Fdm1DimSolver`` rollback.

The strike-dependent cases use an **external** local-vol term structure
(:class:`RampLocalVol`, a closed form both sides evaluate identically)
rather than a Dupire surface. That is deliberate: what is under test here
is ``FdmBlackScholesOp``'s own arithmetic — the ``exp(locations)`` spot
grid, the per-node variance loop, the per-node ``axpyb`` and the vector
overload of ``quantoAdjustment`` — and a Dupire surface would put the
quality of ``LocalVolSurface`` in the way of that. (``LocalVolSurface``
has its own defect, recorded in the branch report: it assumes a zero
risk-free rate and zero dividend yield when forming the forward.)

``flat_lv`` does exercise the derived path: with a ``BlackConstantVol``,
``GeneralizedBlackScholesProcess.local_volatility()`` returns a
``LocalConstantVol``, and the local-vol arm must then reproduce the
constant-vol arm.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import FdmBlackScholesOp
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver import Fdm1DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmLogInnerValue,
)
from pquantlib.methods.finitedifferences.utilities.fdm_quanto_helper import FdmQuantoHelper
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import BlackConstantVol
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.null_calendar import NullCalendar
from pquantlib.time.date import Date
from pquantlib.time.month import Month

_TODAY = Date.from_ymd(15, Month.January, 2024)
_DC = Actual365Fixed()
_N = 25
_STRIKE = 100.0


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/methods/bsop")


class RampLocalVol(LocalVolTermStructure):
    """``sigma(t, S) = 0.15 + 0.05 log(S/100) + 0.02 t``.

    # C++ parity: ``RampLocalVol`` in probe.cpp — the same closed form, so
    # the comparison isolates FdmBlackScholesOp rather than a surface.
    """

    def __init__(self) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=_TODAY,
            calendar=NullCalendar(),
            day_counter=_DC,
        )

    def max_date(self) -> Date:
        return Date.max_date()

    def min_strike(self) -> float:
        return 0.0

    def max_strike(self) -> float:
        return math.inf

    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        return 0.15 + 0.05 * math.log(underlying_level / 100.0) + 0.02 * t


class ThrowingLocalVol(RampLocalVol):
    """Same ramp, but refuses to quote below S = 80.

    # C++ parity: ``ThrowingLocalVol`` in probe.cpp.
    """

    def _local_vol_impl(self, t: float, underlying_level: float) -> float:
        if underlying_level <= 80.0:
            raise LibraryException("no local vol below 80")
        return super()._local_vol_impl(t, underlying_level)


def _flat(rate: float) -> FlatForward:
    return FlatForward.from_rate(_TODAY, rate, _DC)


def _black_vol() -> BlackConstantVol:
    return BlackConstantVol(
        reference_date=_TODAY, calendar=NullCalendar(), volatility=0.20, day_counter=_DC
    )


def _mesher(x_min: float = math.log(50.0), x_max: float = math.log(150.0)) -> FdmMesherComposite:
    return FdmMesherComposite(Uniform1dMesher(x_min, x_max, _N))


def _process(local_vol_ts: LocalVolTermStructure | None) -> GeneralizedBlackScholesProcess:
    if local_vol_ts is None:
        return BlackScholesMertonProcess(
            x0=SimpleQuote(100.0),
            dividend_ts=_flat(0.02),
            risk_free_ts=_flat(0.05),
            black_vol_ts=_black_vol(),
        )
    return GeneralizedBlackScholesProcess(
        x0=SimpleQuote(100.0),
        dividend_ts=_flat(0.02),
        risk_free_ts=_flat(0.05),
        black_vol_ts=_black_vol(),
        local_vol_ts=local_vol_ts,
    )


def _quanto_helper() -> FdmQuantoHelper:
    """domestic 5%, foreign 3%, fx vol 12%, correlation 0.4, ATM fx 1.25."""
    return FdmQuantoHelper(
        _flat(0.05),
        _flat(0.03),
        BlackConstantVol(
            reference_date=_TODAY, calendar=NullCalendar(), volatility=0.12, day_counter=_DC
        ),
        0.4,
        1.25,
    )


def _check_apply(cpp: dict[str, Any], prefix: str, op: FdmBlackScholesOp) -> None:
    """Three interleaved indicator vectors pin every band of the operator."""
    for k in range(3):
        e: Array = np.zeros(_N, dtype=np.float64)
        e[k::3] = 1.0
        y = op.apply(e)
        tight(float(y[0]), cpp[f"{prefix}apply{k}_0"])
        tight(float(y[1]), cpp[f"{prefix}apply{k}_1"])
        tight(float(y[_N // 2]), cpp[f"{prefix}apply{k}_mid"])
        tight(float(y[_N - 1]), cpp[f"{prefix}apply{k}_last"])


#: name -> (uses the process's local_volatility(), external ramp, quanto)
_CASES: dict[str, tuple[bool, bool, bool]] = {
    "flat_plain": (False, False, False),
    "flat_lv": (True, False, False),
    "ramp_plain": (False, True, False),
    "ramp_lv": (True, True, False),
    "flat_plain_q": (False, False, True),
    "ramp_lv_q": (True, True, True),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_v143_fdm_black_scholes_op_branches(cpp: dict[str, Any], name: str) -> None:
    """All four ``setTime`` arms, at the operator and after a full rollback.

    # C++ parity: probe.cpp ``block``.
    """
    local_vol, external, quanto = _CASES[name]
    mesher = _mesher()
    process = _process(RampLocalVol() if external else None)
    helper = _quanto_helper() if quanto else None
    prefix = f"{name}_"

    op = FdmBlackScholesOp(mesher, process, _STRIKE, local_vol, -1.0, 0, helper)
    op.set_time(0.0, 0.25)
    _check_apply(cpp, prefix, op)

    # A second set_time must recompute the coefficients — for the ramp they
    # also move with the mid-point time.
    op.set_time(0.5, 0.75)
    y = op.apply(np.ones(_N, dtype=np.float64))
    tight(float(y[0]), cpp[prefix + "t2_apply_0"])
    tight(float(y[_N // 2]), cpp[prefix + "t2_apply_mid"])
    tight(float(y[_N - 1]), cpp[prefix + "t2_apply_last"])

    calc = FdmLogInnerValue(PlainVanillaPayoff(OptionType.Call, _STRIKE), mesher, 0)
    desc = FdmSolverDesc(
        mesher, FdmStepConditionComposite([], []), calc.avg_inner_value, 1.0, 100, 0
    )
    solver = Fdm1DimSolver(
        desc,
        FdmSchemeDesc.douglas(),
        FdmBlackScholesOp(mesher, process, _STRIKE, local_vol, -1.0, 0, helper),
    )
    tight(solver.interpolate_at(math.log(90.0)), cpp[prefix + "value_90"])
    tight(solver.interpolate_at(math.log(100.0)), cpp[prefix + "value_100"])
    tight(solver.interpolate_at(math.log(110.0)), cpp[prefix + "value_110"])
    tight(solver.derivative_x(math.log(100.0)), cpp[prefix + "dx_100"])


def test_v143_local_vol_arm_is_not_the_constant_vol_arm(cpp: dict[str, Any]) -> None:
    """The reference itself separates the arms — this test has teeth.

    ``ramp_plain`` and ``ramp_lv`` differ only in the ``local_vol`` flag and
    C++ prices them 9.1948 against 7.7129, so an operator that ignored the
    flag (as the Python port did) fails ``ramp_lv``. Likewise the quanto
    flag moves ``flat_plain`` 9.1948 -> 7.5753.
    """
    assert abs(cpp["ramp_plain_value_100"] - cpp["ramp_lv_value_100"]) > 1.0
    assert abs(cpp["flat_plain_value_100"] - cpp["flat_plain_q_value_100"]) > 1.0
    assert abs(cpp["ramp_lv_value_100"] - cpp["ramp_lv_q_value_100"]) > 1.0
    # ... and with a *constant* Black vol the derived local vol is that same
    # constant, so those two arms must agree instead. C++ agrees to 2.4e-15
    # relative; the two arms take different code paths to the same number.
    assert abs(cpp["flat_plain_value_100"] - cpp["flat_lv_value_100"]) < 1e-12


def test_v143_illegal_local_vol_overwrite(cpp: dict[str, Any]) -> None:
    """A non-negative override substitutes for a local vol that raises.

    # C++ parity: probe.cpp ``blockOverwrite`` — the ``try/catch`` in
    # ``setTime``'s local-vol loop, with ``illegalLocalVolOverwrite_ = 0.35``
    # standing in for every node at or below S = 80.
    """
    mesher = _mesher()
    process = _process(ThrowingLocalVol())
    op = FdmBlackScholesOp(mesher, process, _STRIKE, True, 0.35, 0, None)
    op.set_time(0.0, 0.25)
    _check_apply(cpp, "ovr_", op)


def test_v143_negative_overwrite_lets_the_local_vol_error_propagate() -> None:
    """A negative override is C++'s "do not catch" sentinel.

    # C++ parity: ``if (illegalLocalVolOverwrite_ < 0.0) { ... }`` — the
    # branch without the ``try``. C++'s callers pass ``-Null<Real>()``.
    """
    op = FdmBlackScholesOp(_mesher(), _process(ThrowingLocalVol()), _STRIKE, True, -1.0, 0, None)
    with pytest.raises(LibraryException, match="no local vol below 80"):
        op.set_time(0.0, 0.25)
