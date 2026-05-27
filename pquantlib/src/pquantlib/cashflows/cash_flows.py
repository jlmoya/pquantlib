"""CashFlows — namespace-style aggregator of cashflow-analysis functions.

# C++ parity: ql/cashflows/cashflows.hpp + .cpp (v1.42.1).

The C++ class is delete-constructor + all-static-methods — a namespace
pattern. The Python port matches by exposing a class with classmethods
only and a private ``__init__`` raising ``TypeError``.

L2-D coverage (port these):
- ``npv(leg, discount_curve, ...)`` — discount via a YieldTermStructureProtocol.
- ``npv(leg, yield, day_counter, comp, freq, ...)`` — discount via an
  InterestRate (flat-curve flavour).
- ``bps(leg, discount_curve, ...)``
- ``yield_`` (named ``irr`` in the L2-D spec) — solve for the yield
  that reproduces a target NPV; iterative, Newton-style.
- ``duration(leg, rate, Duration.Type, ...)`` — Simple/Macaulay/Modified.
- ``convexity(leg, rate, ...)``

Deferred carve-outs:
- ``atmRate`` (requires Visitor dispatch for BPSCalculator).
- ``zSpread`` (requires ZeroSpreadedTermStructure — L2-B).
- ``basisPointValue`` / ``yieldValueBasisPoint``.
- ``startDate`` / ``maturityDate`` / ``previousCashFlow`` / ``nextCashFlow``
  helpers (trivial — port on demand).
- Settings.evaluationDate fallback for ``settlement_date=None`` (callers
  must supply explicitly).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

from pquantlib import qassert
from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.duration import Duration
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.interest_rate import InterestRate
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


_BASIS_POINT: float = 1.0e-4
# Default IRR solver settings (match C++ ql/cashflows/cashflows.cpp:911).
_DEFAULT_IRR_GUESS: float = 0.05
_DEFAULT_IRR_ACCURACY: float = 1.0e-10
_DEFAULT_IRR_MAX_ITER: int = 100
_DEFAULT_IRR_STEP: float = 1.0e-4


def _stepwise_discount_time(
    cf: CashFlow,
    dc: DayCounter,
    npv_date: Date,
    last_date: Date,
) -> float:
    """Mirror of C++ ``getStepwiseDiscountTime`` (cashflows.cpp:568-600).

    Handles Coupon vs non-Coupon split: for a coupon, use its
    reference-period dates as the ref args; for a plain cashflow,
    fake a 1y window when we don't have a previous coupon date.
    """
    cf_date = cf.date()
    if isinstance(cf, Coupon):
        ref_start = cf.reference_period_start()
        ref_end = cf.reference_period_end()
        if last_date != cf.accrual_start_date():
            coupon_period = dc.year_fraction(
                cf.accrual_start_date(), cf_date, ref_start, ref_end
            )
            accrued_period = dc.year_fraction(
                cf.accrual_start_date(), last_date, ref_start, ref_end
            )
            return coupon_period - accrued_period
        return dc.year_fraction(last_date, cf_date, ref_start, ref_end)
    # Non-Coupon CashFlow.
    ref_start = (
        cf_date - Period(1, TimeUnit.Years) if last_date == npv_date else last_date
    )
    ref_end = cf_date
    return dc.year_fraction(last_date, cf_date, ref_start, ref_end)


def _simple_duration_impl(
    leg: Sequence[CashFlow],
    y: InterestRate,
    include_settlement_date_flows: bool,
    settlement_date: Date,
    npv_date: Date,
) -> float:
    """C++ parity: cashflows.cpp:602-640."""
    p_val = 0.0
    dpdy = 0.0
    t = 0.0
    last_date = npv_date
    dc = y.day_counter()
    for cf in leg:
        if cf.has_occurred(settlement_date, include_settlement_date_flows):
            continue
        c = cf.amount()
        if cf.trading_ex_coupon(settlement_date):
            c = 0.0
        t += _stepwise_discount_time(cf, dc, npv_date, last_date)
        b = y.discount_factor(t)
        p_val += c * b
        dpdy += t * c * b
        last_date = cf.date()
    if p_val == 0.0:
        return 0.0
    return dpdy / p_val


def _modified_duration_impl(
    leg: Sequence[CashFlow],
    y: InterestRate,
    include_settlement_date_flows: bool,
    settlement_date: Date,
    npv_date: Date,
) -> float:
    """C++ parity: cashflows.cpp:642-707."""
    p_val = 0.0
    t = 0.0
    dpdy = 0.0
    r = y.rate()
    n = float(y.frequency())
    last_date = npv_date
    dc = y.day_counter()
    for cf in leg:
        if cf.has_occurred(settlement_date, include_settlement_date_flows):
            continue
        c = cf.amount()
        if cf.trading_ex_coupon(settlement_date):
            c = 0.0
        t += _stepwise_discount_time(cf, dc, npv_date, last_date)
        b = y.discount_factor(t)
        p_val += c * b
        if y.compounding() == Compounding.Simple:
            dpdy -= c * b * b * t
        elif y.compounding() == Compounding.Compounded:
            dpdy -= c * t * b / (1.0 + r / n)
        elif y.compounding() == Compounding.Continuous:
            dpdy -= c * b * t
        elif y.compounding() == Compounding.SimpleThenCompounded:
            if t <= 1.0 / n:
                dpdy -= c * b * b * t
            else:
                dpdy -= c * t * b / (1.0 + r / n)
        elif y.compounding() == Compounding.CompoundedThenSimple:
            if t > 1.0 / n:
                dpdy -= c * b * b * t
            else:
                dpdy -= c * t * b / (1.0 + r / n)
        else:
            qassert.fail(f"unknown compounding convention ({int(y.compounding())})")
        last_date = cf.date()
    if p_val == 0.0:
        return 0.0
    return -dpdy / p_val


def _macaulay_duration_impl(
    leg: Sequence[CashFlow],
    y: InterestRate,
    include_settlement_date_flows: bool,
    settlement_date: Date,
    npv_date: Date,
) -> float:
    """C++ parity: cashflows.cpp:709-722."""
    qassert.require(
        y.compounding() == Compounding.Compounded, "compounded rate required"
    )
    return (1.0 + y.rate() / float(y.frequency())) * _modified_duration_impl(
        leg, y, include_settlement_date_flows, settlement_date, npv_date
    )


class CashFlows:
    """Namespace-only class — direct construction is disabled.

    C++ parity: ql/cashflows/cashflows.hpp:41 deleted ctor.
    """

    def __init__(self) -> NoReturn:
        msg = "CashFlows is a namespace; use classmethods only"
        raise TypeError(msg)

    # ===================================================================
    # NPV
    # ===================================================================

    @classmethod
    def npv_curve(
        cls,
        leg: Sequence[CashFlow],
        discount_curve: YieldTermStructureProtocol,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
    ) -> float:
        """NPV of a leg against a discount curve.

        C++ parity: ql/cashflows/cashflows.cpp:424-447 ``npv(leg, YieldTermStructure&, ...)``.
        """
        if not leg:
            return 0.0
        settle = settlement_date if settlement_date is not None else discount_curve.reference_date()
        npv_d = npv_date if npv_date is not None else settle
        total = 0.0
        for cf in leg:
            if not cf.has_occurred(settle, include_settlement_date_flows) and not cf.trading_ex_coupon(
                settle
            ):
                total += cf.amount() * discount_curve.discount(cf.date())
        return total / discount_curve.discount(npv_d)

    @classmethod
    def npv_yield(
        cls,
        leg: Sequence[CashFlow],
        yield_rate: InterestRate,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
    ) -> float:
        """NPV of a leg against a flat yield curve (InterestRate-based).

        C++ parity: ql/cashflows/cashflows.cpp:811-853 ``npv(leg, InterestRate&, ...)``.
        """
        if not leg:
            return 0.0
        qassert.require(
            settlement_date is not None, "settlement_date is required (no Settings.evaluationDate)"
        )
        assert settlement_date is not None
        npv_d = npv_date if npv_date is not None else settlement_date
        npv = 0.0
        discount = 1.0
        last_date = npv_d
        dc = yield_rate.day_counter()
        for cf in leg:
            if cf.has_occurred(settlement_date, include_settlement_date_flows):
                continue
            amount = cf.amount()
            if cf.trading_ex_coupon(settlement_date):
                amount = 0.0
            b = yield_rate.discount_factor(
                _stepwise_discount_time(cf, dc, npv_d, last_date)
            )
            discount *= b
            last_date = cf.date()
            npv += amount * discount
        return npv

    # ===================================================================
    # BPS
    # ===================================================================

    @classmethod
    def bps(
        cls,
        leg: Sequence[CashFlow],
        discount_curve: YieldTermStructureProtocol,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
    ) -> float:
        """Basis-Point Sensitivity — derivative of NPV w.r.t. a parallel rate shift.

        C++ parity: ql/cashflows/cashflows.cpp:449-470 ``bps(leg, YieldTermStructure&, ...)``.

        This implementation foregoes the C++ Visitor-based BPSCalculator
        (visitor dispatch is deferred carve-out in this port) and walks
        the leg directly, summing ``nominal * accrual_period * df`` for
        each Coupon. Non-Coupon cashflows contribute zero (matching the
        C++ visitor's ``visit(Coupon&)`` body — see cashflows.cpp:400-419).
        """
        if not leg:
            return 0.0
        settle = settlement_date if settlement_date is not None else discount_curve.reference_date()
        npv_d = npv_date if npv_date is not None else settle
        bps_sum = 0.0
        for cf in leg:
            if cf.has_occurred(settle, include_settlement_date_flows) or cf.trading_ex_coupon(settle):
                continue
            if isinstance(cf, Coupon):
                bps_sum += cf.nominal() * cf.accrual_period() * discount_curve.discount(cf.date())
        return _BASIS_POINT * bps_sum / discount_curve.discount(npv_d)

    # ===================================================================
    # Duration
    # ===================================================================

    @classmethod
    def duration(
        cls,
        leg: Sequence[CashFlow],
        rate: InterestRate,
        duration_type: Duration = Duration.Modified,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
    ) -> float:
        """Duration of a leg.

        C++ parity: ql/cashflows/cashflows.cpp:924-956.
        """
        if not leg:
            return 0.0
        qassert.require(
            settlement_date is not None,
            "settlement_date is required (no Settings.evaluationDate)",
        )
        assert settlement_date is not None
        npv_d = npv_date if npv_date is not None else settlement_date
        if duration_type == Duration.Simple:
            return _simple_duration_impl(
                leg, rate, include_settlement_date_flows, settlement_date, npv_d
            )
        if duration_type == Duration.Modified:
            return _modified_duration_impl(
                leg, rate, include_settlement_date_flows, settlement_date, npv_d
            )
        if duration_type == Duration.Macaulay:
            return _macaulay_duration_impl(
                leg, rate, include_settlement_date_flows, settlement_date, npv_d
            )
        qassert.fail(f"unknown duration type ({int(duration_type)})")

    # ===================================================================
    # Convexity
    # ===================================================================

    @classmethod
    def convexity(
        cls,
        leg: Sequence[CashFlow],
        rate: InterestRate,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
    ) -> float:
        """Convexity — second derivative of NPV w.r.t. yield.

        C++ parity: ql/cashflows/cashflows.cpp:973-1041.
        """
        if not leg:
            return 0.0
        qassert.require(
            settlement_date is not None,
            "settlement_date is required (no Settings.evaluationDate)",
        )
        assert settlement_date is not None
        npv_d = npv_date if npv_date is not None else settlement_date
        dc = rate.day_counter()
        p_val = 0.0
        t = 0.0
        d2pdy2 = 0.0
        r = rate.rate()
        n = float(rate.frequency())
        last_date = npv_d
        for cf in leg:
            if cf.has_occurred(settlement_date, include_settlement_date_flows):
                continue
            c = cf.amount()
            if cf.trading_ex_coupon(settlement_date):
                c = 0.0
            t += _stepwise_discount_time(cf, dc, npv_d, last_date)
            b = rate.discount_factor(t)
            p_val += c * b
            if rate.compounding() == Compounding.Simple:
                d2pdy2 += c * 2.0 * b * b * b * t * t
            elif rate.compounding() == Compounding.Compounded:
                d2pdy2 += c * b * t * (n * t + 1.0) / (n * (1.0 + r / n) * (1.0 + r / n))
            elif rate.compounding() == Compounding.Continuous:
                d2pdy2 += c * b * t * t
            elif rate.compounding() == Compounding.SimpleThenCompounded:
                if t <= 1.0 / n:
                    d2pdy2 += c * 2.0 * b * b * b * t * t
                else:
                    d2pdy2 += c * b * t * (n * t + 1.0) / (n * (1.0 + r / n) * (1.0 + r / n))
            elif rate.compounding() == Compounding.CompoundedThenSimple:
                if t > 1.0 / n:
                    d2pdy2 += c * 2.0 * b * b * b * t * t
                else:
                    d2pdy2 += c * b * t * (n * t + 1.0) / (n * (1.0 + r / n) * (1.0 + r / n))
            else:
                qassert.fail(
                    f"unknown compounding convention ({int(rate.compounding())})"
                )
            last_date = cf.date()
        if p_val == 0.0:
            return 0.0
        return d2pdy2 / p_val

    # ===================================================================
    # IRR (yield that reproduces a target NPV)
    # ===================================================================

    @classmethod
    def irr(
        cls,
        leg: Sequence[CashFlow],
        target_npv: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
        accuracy: float = _DEFAULT_IRR_ACCURACY,
        max_iterations: int = _DEFAULT_IRR_MAX_ITER,
        guess: float = _DEFAULT_IRR_GUESS,
    ) -> float:
        """Solve for the yield that reproduces ``target_npv``.

        C++ parity: ql/cashflows/cashflows.cpp:903-921 (``yield`` function).
        The C++ default uses NewtonSafe; we use Brent (already ported)
        to avoid requiring derivative-aware setup.
        """
        qassert.require(
            settlement_date is not None,
            "settlement_date is required (no Settings.evaluationDate)",
        )
        assert settlement_date is not None

        def objective(y: float) -> float:
            ir = InterestRate(y, day_counter, compounding, frequency)
            npv = cls.npv_yield(
                leg, ir, include_settlement_date_flows, settlement_date, npv_date
            )
            return target_npv - npv

        solver = Brent()
        solver.set_max_evaluations(max_iterations)
        return solver.solve(objective, accuracy, guess, _DEFAULT_IRR_STEP)
