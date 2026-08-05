"""Cross-validate ``CashFlows.IrrFinder`` and the ``CashFlows.irr`` solve.

Probe: ``v143/cf/irr``.

``IrrFinder`` is private in C++, so it is pinned through the public
``CashFlows::yield``: the root, the objective and derivative evaluated at that
root, and the constructor-time ``checkSign`` guard that rejects a
(leg, market price) pair for which an IRR is meaningless.

The legs are rebuilt here exactly as the probe builds them — coupon payment
dates equal to accrual end dates (C++ ``withPaymentAdjustment(Unadjusted)``),
plus a redemption on the last schedule date — and the pinned per-flow listing
is asserted first, so a matching yield cannot hide a differently-shaped leg.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.cashflows.cash_flow import CashFlow
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.cashflows.duration import Duration
from pquantlib.cashflows.fixed_rate_coupon import FixedRateCoupon
from pquantlib.cashflows.simple_cash_flow import SimpleCashFlow
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.daycounters.thirty_360 import Convention as Thirty360Convention
from pquantlib.daycounters.thirty_360 import Thirty360
from pquantlib.exceptions import LibraryException
from pquantlib.interest_rate import InterestRate
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import custom, loose, tight
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.month import Month
from pquantlib.time.period import Period
from pquantlib.time.schedule import MakeSchedule
from pquantlib.time.time_unit import TimeUnit

_SETTLEMENT = Date.from_ymd(15, Month.June, 2026)
_T360 = Thirty360(Thirty360Convention.BondBasis)
_A365 = Actual365Fixed()


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/cf/irr")


def _build_leg(coupon_rate: float, notional: float, years: int, freq: Frequency) -> list[CashFlow]:
    """Mirror of the probe's ``buildLeg``."""
    schedule = (
        MakeSchedule()
        .from_date(_SETTLEMENT)
        .to(_SETTLEMENT + Period(years, TimeUnit.Years))
        .with_frequency(freq)
        .with_calendar(TARGET())
        .with_convention(BusinessDayConvention.Unadjusted)
        .backwards()
        .build()
    )
    leg: list[CashFlow] = [
        FixedRateCoupon.from_rate(
            schedule.date(i + 1),  # payment == accrual end (Unadjusted)
            notional,
            coupon_rate,
            _T360,
            schedule.date(i),
            schedule.date(i + 1),
        )
        for i in range(len(schedule) - 1)
    ]
    leg.append(SimpleCashFlow(notional, schedule.back()))
    return leg


_FIVE_YEAR_KEYS = (
    "par_5y_semi_guess_005",
    "par_5y_semi_guess_001",
    "par_5y_semi_guess_020",
    "premium_5y_semi",
    "discount_5y_semi",
    "check_sign_negative_target",
)
_TEN_YEAR_KEYS = (
    "par_10y_annual_compounded",
    "par_10y_annual_continuous",
    "par_10y_annual_simple",
    "par_10y_simple_then_comp",
    "par_10y_comp_then_simple",
)


def _leg_for(key: str) -> list[CashFlow]:
    if key in _FIVE_YEAR_KEYS:
        return _build_leg(0.04, 100.0, 5, Frequency.Semiannual)
    if key in _TEN_YEAR_KEYS:
        return _build_leg(0.055, 100.0, 10, Frequency.Annual)
    if key == "check_sign_single_flow_zero_target":
        return [SimpleCashFlow(100.0, _SETTLEMENT + Period(1, TimeUnit.Years))]
    msg = f"unknown probe case {key}"
    raise AssertionError(msg)


def _day_counter_for(key: str) -> DayCounter:
    # The probe uses Thirty360 for the two Compounded 30/360 bond cases and
    # Actual365Fixed for the rest; mirror that mapping exactly.
    return _T360 if key in {*_FIVE_YEAR_KEYS, "par_10y_annual_compounded"} else _A365


@pytest.mark.parametrize("key", [*_FIVE_YEAR_KEYS, *_TEN_YEAR_KEYS, "check_sign_single_flow_zero_target"])
def test_leg_structure_matches_cpp(cpp: dict[str, Any], key: str) -> None:
    """The leg that produced the reference yield, flow by flow."""
    ref = cpp[key]
    leg = _leg_for(key)
    assert len(leg) == len(ref["cashflows"])
    for cf, flow in zip(leg, ref["cashflows"], strict=True):
        assert cf.date().serial_number() == flow["date_serial"]
        tight(cf.amount(), flow["amount"])


@pytest.mark.parametrize("key", [*_FIVE_YEAR_KEYS, *_TEN_YEAR_KEYS, "check_sign_single_flow_zero_target"])
def test_irr_matches_cpp(cpp: dict[str, Any], key: str) -> None:
    ref = cpp[key]
    leg = _leg_for(key)
    dc = _day_counter_for(key)
    comp = Compounding(ref["compounding"])
    freq = Frequency(ref["frequency"])

    if ref["raises"]:
        # C++ IrrFinder::checkSign rejects the pair in its constructor.
        with pytest.raises(LibraryException) as excinfo:
            CashFlows.irr(
                leg,
                ref["target_npv"],
                dc,
                comp,
                freq,
                False,
                _SETTLEMENT,
                _SETTLEMENT,
                guess=ref["guess"],
            )
        assert ref["error"] in str(excinfo.value)
        return

    y = CashFlows.irr(
        leg,
        ref["target_npv"],
        dc,
        comp,
        freq,
        False,
        _SETTLEMENT,
        _SETTLEMENT,
        guess=ref["guess"],
    )
    # LOOSE: the root is the output of an iterative solve run to accuracy
    # 1e-10 in y; NewtonSafe stops on |dx| < accuracy, so agreement is bounded
    # by that accuracy and not by machine epsilon.
    loose(y, ref["yield"])


@pytest.mark.parametrize("key", [*_FIVE_YEAR_KEYS[:5], *_TEN_YEAR_KEYS])
def test_irr_finder_objective_and_derivative(cpp: dict[str, Any], key: str) -> None:
    """``IrrFinder.__call__`` and ``.derivative`` at the C++ root.

    Evaluated at the *pinned C++ root*, not at ours, so this isolates the
    objective from the solver: a sign-flipped objective or a derivative built
    from the wrong duration flavour fails here even when the root agrees.
    """
    ref = cpp[key]
    leg = _leg_for(key)
    dc = _day_counter_for(key)
    comp = Compounding(ref["compounding"])
    freq = Frequency(ref["frequency"])

    finder = CashFlows.IrrFinder(leg, ref["target_npv"], dc, comp, freq, False, _SETTLEMENT, _SETTLEMENT)
    y = ref["yield"]
    # The objective is ``npv(≈100) - target(100)``: a difference of two nearly
    # equal quantities, so its own relative accuracy is meaningless — the
    # residual is pure cancellation. The bound follows from the arithmetic:
    # the residual's error IS the NPV's error, and the NPV is asserted at
    # TIGHT (rel 1e-12) below, so abs(d_residual) <= 1e-12 * abs(npv) ~= 1e-10 for the
    # 100-notional legs used here. rel_tol is left at the TIGHT value so that
    # a residual which is NOT near zero is still held to the tight tier.
    custom(
        finder(y),
        ref["objective_at_root"],
        abs_tol=1e-10,
        rel_tol=1e-12,
        reason="objective at the root is a cancelling difference; bound is "
        "the TIGHT relative bound on the NPV (1e-12) times |npv| ≈ 100",
    )
    # Structural arithmetic over identical inputs → TIGHT.
    tight(finder.derivative(y), ref["derivative_at_root"])

    rate = InterestRate(y, dc, comp, freq)
    tight(
        CashFlows.npv_yield(leg, rate, False, _SETTLEMENT, _SETTLEMENT),
        ref["npv_at_root"],
    )
    tight(
        CashFlows.duration(leg, rate, Duration.Modified, False, _SETTLEMENT, _SETTLEMENT),
        ref["modified_duration_at_root"],
    )


def test_guess_reaches_the_solver(cpp: dict[str, Any]) -> None:
    """``guess`` is not decorative: it seeds the bracket and the step (guess/10).

    All three probe cases solve the same leg for the same target from three
    different guesses and reach the same root; a port that dropped ``guess``
    would still pass that. So this additionally asserts that a guess the
    bracket search cannot rescue fails — i.e. the value genuinely reaches
    ``Solver1D.solve``.
    """
    leg = _leg_for("par_5y_semi_guess_005")
    for key in ("par_5y_semi_guess_005", "par_5y_semi_guess_001", "par_5y_semi_guess_020"):
        loose(
            CashFlows.irr(
                leg,
                100.0,
                _T360,
                Compounding.Compounded,
                Frequency.Semiannual,
                False,
                _SETTLEMENT,
                _SETTLEMENT,
                guess=cpp[key]["guess"],
            ),
            cpp[key]["yield"],
        )

    # max_evaluations must reach the solver too: one evaluation cannot
    # converge, and Solver1D reports that rather than returning the guess.
    with pytest.raises(LibraryException):
        CashFlows.irr(
            leg,
            100.0,
            _T360,
            Compounding.Compounded,
            Frequency.Semiannual,
            False,
            _SETTLEMENT,
            _SETTLEMENT,
            max_iterations=1,
        )


def test_custom_solver_is_used() -> None:
    """The ``solver`` argument is C++'s ``template <typename Solver> yield``."""
    leg = _leg_for("par_5y_semi_guess_005")
    args = (
        leg,
        100.0,
        _T360,
        Compounding.Compounded,
        Frequency.Semiannual,
        False,
        _SETTLEMENT,
        _SETTLEMENT,
    )
    with_default = CashFlows.irr(*args)
    with_brent = CashFlows.irr(*args, solver=Brent())
    # Same root from a derivative-free solver — LOOSE for the same reason as
    # above (both stop at accuracy 1e-10 in y, from different iterate paths).
    loose(with_brent, with_default)
