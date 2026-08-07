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

The remaining surface — the one ``BondFunctions`` delegates to — was added
in the v1.43 ``bondswap`` wave and is cross-validated against C++ v1.43 in
``pquantlib/tests/pricingengines/bond/test_bond_functions.py``:
``previous_cash_flow`` / ``next_cash_flow`` (+ ``*_amount``), ``nominal``,
``accrual_start_date`` / ``accrual_end_date`` / ``reference_period_start`` /
``reference_period_end`` / ``accrual_period`` / ``accrual_days`` /
``accrued_period`` / ``accrued_days``, ``atm_rate``, ``bps_yield``,
``basis_point_value``, ``yield_value_basis_point``, ``npv_z_spread`` and
``z_spread``.

Deferred carve-outs:
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
from pquantlib.math.solvers1d.newton_safe import NewtonSafe
from pquantlib.math.solvers1d.solver_1d import Solver1D
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.termstructures.yield_.zero_spreaded_term_structure import (
    ZeroSpreadedTermStructure,
)
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure


_BASIS_POINT: float = 1.0e-4


def _sign(x: float) -> int:
    """C++ parity: ``ql/math/comparison.hpp`` ``sign`` used by ``IrrFinder::checkSign``."""
    if x == 0.0:
        return 0
    return 1 if x > 0.0 else -1


# Default IRR solver settings (match C++ ql/cashflows/cashflows.cpp:911).
_DEFAULT_IRR_GUESS: float = 0.05
_DEFAULT_IRR_ACCURACY: float = 1.0e-10
_DEFAULT_IRR_MAX_ITER: int = 100


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
            coupon_period = dc.year_fraction(cf.accrual_start_date(), cf_date, ref_start, ref_end)
            accrued_period = dc.year_fraction(cf.accrual_start_date(), last_date, ref_start, ref_end)
            return coupon_period - accrued_period
        return dc.year_fraction(last_date, cf_date, ref_start, ref_end)
    # Non-Coupon CashFlow.
    ref_start = cf_date - Period(1, TimeUnit.Years) if last_date == npv_date else last_date
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
    qassert.require(y.compounding() == Compounding.Compounded, "compounded rate required")
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
    # IrrFinder — the solver objective behind ``irr``
    # ===================================================================

    class IrrFinder:
        """Objective for the yield solve: ``f(y) = NPV(leg, y) - target``.

        C++ parity: ``CashFlows::IrrFinder``
        (ql/cashflows/cashflows.hpp:43-66, .cpp:733-800).

        Nested exactly as in C++. It carries the derivative that
        ``NewtonSafe`` needs (``-modifiedDuration * P``) and runs the
        constructor-time ``checkSign`` guard, which rejects a
        (leg, target NPV) pair whose signs make an IRR meaningless.
        """

        def __init__(
            self,
            leg: Sequence[CashFlow],
            npv: float,
            day_counter: DayCounter,
            compounding: Compounding,
            frequency: Frequency,
            include_settlement_date_flows: bool,
            settlement_date: Date,
            npv_date: Date | None = None,
        ) -> None:
            self._leg = leg
            self._npv = npv
            self._day_counter = day_counter
            self._compounding = compounding
            self._frequency = frequency
            self._include_settlement_date_flows = include_settlement_date_flows
            self._settlement_date = settlement_date
            self._npv_date = npv_date if npv_date is not None else settlement_date
            self._check_sign()

        def _rate(self, y: float) -> InterestRate:
            return InterestRate(y, self._day_counter, self._compounding, self._frequency)

        def __call__(self, y: float) -> float:
            # C++ parity: cashflows.cpp:754-760 — ``NPV - npv_``, in that order.
            npv = CashFlows.npv_yield(
                self._leg,
                self._rate(y),
                self._include_settlement_date_flows,
                self._settlement_date,
                self._npv_date,
            )
            return npv - self._npv

        def derivative(self, y: float) -> float:
            # C++ parity: cashflows.cpp:762-769.
            rate = self._rate(y)
            p = CashFlows.npv_yield(
                self._leg,
                rate,
                self._include_settlement_date_flows,
                self._settlement_date,
                self._npv_date,
            )
            return (
                -CashFlows.duration(
                    self._leg,
                    rate,
                    Duration.Modified,
                    self._include_settlement_date_flows,
                    self._settlement_date,
                    self._npv_date,
                )
                * p
            )

        def _check_sign(self) -> None:
            """Reject a (leg, price) pair for which an IRR is nonsensical.

            C++ parity: cashflows.cpp:771-790 ``IrrFinder::checkSign``.
            Cash flows of the sign opposite to the market price must be
            present, otherwise no yield reproduces it.
            """
            last_sign = _sign(-self._npv)
            sign_changes = 0
            for cf in self._leg:
                if cf.has_occurred(
                    self._settlement_date, self._include_settlement_date_flows
                ) or cf.trading_ex_coupon(self._settlement_date):
                    continue
                this_sign = _sign(cf.amount())
                if last_sign * this_sign < 0:
                    sign_changes += 1
                if this_sign != 0:
                    last_sign = this_sign
            qassert.require(
                sign_changes > 0,
                "the given cash flows cannot result in the given market price due to their sign",
            )

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
    def npvbps(
        cls,
        leg: Sequence[CashFlow],
        discount_curve: YieldTermStructureProtocol,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
    ) -> tuple[float, float]:
        """NPV and BPS of a leg, computed together.

        C++ parity: ``CashFlows::npvbps`` (cashflows.cpp). Upstream computes
        both in a single pass "for performance reason" and returns
        ``std::pair<Real, Real>``; here a ``(npv, bps)`` tuple.

        Required by ``DiscountingConstNotionalCrossCurrencySwapEngine``, new in
        v1.43.
        """
        npv = 0.0
        bps = 0.0
        if not leg:
            return (npv, bps)
        settle = settlement_date if settlement_date is not None else discount_curve.reference_date()
        npv_d = npv_date if npv_date is not None else settle
        for cf in leg:
            if not cf.has_occurred(settle, include_settlement_date_flows) and not cf.trading_ex_coupon(
                settle
            ):
                df = discount_curve.discount(cf.date())
                npv += cf.amount() * df
                if isinstance(cf, Coupon):
                    bps += cf.nominal() * cf.accrual_period() * df
        d = discount_curve.discount(npv_d)
        return (npv / d, _BASIS_POINT * bps / d)

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
            b = yield_rate.discount_factor(_stepwise_discount_time(cf, dc, npv_d, last_date))
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
            return _simple_duration_impl(leg, rate, include_settlement_date_flows, settlement_date, npv_d)
        if duration_type == Duration.Modified:
            return _modified_duration_impl(leg, rate, include_settlement_date_flows, settlement_date, npv_d)
        if duration_type == Duration.Macaulay:
            return _macaulay_duration_impl(leg, rate, include_settlement_date_flows, settlement_date, npv_d)
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
                qassert.fail(f"unknown compounding convention ({int(rate.compounding())})")
            last_date = cf.date()
        if p_val == 0.0:
            return 0.0
        return d2pdy2 / p_val

    # ===================================================================
    # Leg-walking helpers used by the Bond base class
    # ===================================================================

    @classmethod
    def start_date(cls, leg: Sequence[CashFlow]) -> Date:
        """Earliest accrual_start_date among coupons (else earliest cf date).

        C++ parity: ql/cashflows/cashflows.cpp:38-50.
        """
        qassert.require(len(leg) > 0, "empty leg")
        # Use max Date as initial sentinel, mirror C++.
        d = Date.max_date()
        for cf in leg:
            cf_d = cf.accrual_start_date() if isinstance(cf, Coupon) else cf.date()
            d = min(d, cf_d)
        return d

    @classmethod
    def maturity_date(cls, leg: Sequence[CashFlow]) -> Date:
        """Latest accrual_end_date among coupons (else latest cf date).

        C++ parity: ql/cashflows/cashflows.cpp:52-64.
        """
        qassert.require(len(leg) > 0, "empty leg")
        d = Date.min_date()
        for cf in leg:
            cf_d = cf.accrual_end_date() if isinstance(cf, Coupon) else cf.date()
            d = max(d, cf_d)
        return d

    @classmethod
    def is_expired(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> bool:
        """All cash flows have occurred at ``settlement_date``.

        C++ parity: ql/cashflows/cashflows.cpp:66-81.
        """
        if not leg:
            return True
        # Walk from the end since the latest cashflow is most likely
        # still pending — mirrors C++ reverse-iteration optimisation.
        return all(cf.has_occurred(settlement_date, include_settlement_date_flows) for cf in reversed(leg))

    @classmethod
    def previous_cash_flow_date(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> Date:
        """Date of the most recent already-occurred cash flow (or null Date).

        C++ parity: ql/cashflows/cashflows.cpp:119-129.
        """
        for cf in reversed(leg):
            if cf.has_occurred(settlement_date, include_settlement_date_flows):
                return cf.date()
        return Date()

    @classmethod
    def next_cash_flow_date(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> Date:
        """Date of the next-to-occur cash flow (or null Date).

        C++ parity: ql/cashflows/cashflows.cpp:131-141.
        """
        for cf in leg:
            if not cf.has_occurred(settlement_date, include_settlement_date_flows):
                return cf.date()
        return Date()

    @classmethod
    def _next_cash_flow_index(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> int:
        """Index of the next-to-occur cash flow, or ``len(leg)`` if none.

        Internal helper used by accrued_amount / nominal / accrual_*_date
        and the next_coupon_rate aggregator.
        """
        for i, cf in enumerate(leg):
            if not cf.has_occurred(settlement_date, include_settlement_date_flows):
                return i
        return len(leg)

    @classmethod
    def _previous_cash_flow_index(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> int:
        """Index of the most-recent already-occurred cf, or ``-1`` if none.

        Internal helper for previous_coupon_rate aggregation.
        """
        for i in range(len(leg) - 1, -1, -1):
            if leg[i].has_occurred(settlement_date, include_settlement_date_flows):
                return i
        return -1

    @classmethod
    def accrued_amount(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> float:
        """Accrued amount across coupons sharing the next-payment date.

        C++ parity: ql/cashflows/cashflows.cpp:376-393.
        """
        idx = cls._next_cash_flow_index(leg, include_settlement_date_flows, settlement_date)
        if idx == len(leg):
            return 0.0
        payment_date = leg[idx].date()
        result = 0.0
        for cf in leg[idx:]:
            if cf.date() != payment_date:
                break
            if isinstance(cf, Coupon):
                result += cf.accrued_amount(settlement_date)
        return result

    @classmethod
    def _aggregate_rate(cls, leg: Sequence[CashFlow], walk: Sequence[int]) -> float:
        """C++ parity: cashflows.cpp:178-210 (anonymous-namespace ``aggregateRate``).

        ``walk`` is the C++ iterator range expressed as the indices to visit
        **in iteration order** — forward (``nextCouponRate``) or backward
        (``previousCouponRate``, which is handed a ``const_reverse_iterator``).
        The direction matters: on the final payment date the leg holds
        ``[.., lastCoupon, redemption]``, so walking backward from the
        redemption still reaches the coupon while walking forward does not.

        The C++ body SUMS ``cp->rate()`` over the same-date coupons (it does
        not average them) and requires that they agree on nominal, accrual
        period and day counter; it then ``QL_ENSURE``s that at least one
        Coupon was seen, so a payment date carrying only a Redemption raises.
        """
        if not walk:
            return 0.0
        payment_date = leg[walk[0]].date()
        first_coupon_found = False
        nominal = 0.0
        accrual_period = 0.0
        day_counter: DayCounter | None = None
        result = 0.0
        for i in walk:
            cf = leg[i]
            if cf.date() != payment_date:
                break
            if not isinstance(cf, Coupon):
                continue
            if first_coupon_found:
                qassert.require(
                    nominal == cf.nominal()
                    and accrual_period == cf.accrual_period()
                    and day_counter == cf.day_counter(),
                    f"cannot aggregate two different coupons on {payment_date}",
                )
            else:
                first_coupon_found = True
                nominal = cf.nominal()
                accrual_period = cf.accrual_period()
                day_counter = cf.day_counter()
            result += cf.rate()
        qassert.require(first_coupon_found, f"no coupon paid at cashflow date {payment_date}")
        return result

    @classmethod
    def next_coupon_rate(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> float:
        """Aggregate next-coupon rate (across same-date coupons).

        C++ parity: ql/cashflows/cashflows.cpp:223-229.
        """
        idx = cls._next_cash_flow_index(leg, include_settlement_date_flows, settlement_date)
        return cls._aggregate_rate(leg, range(idx, len(leg)))

    @classmethod
    def previous_coupon_rate(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> float:
        """Aggregate previous-coupon rate (across same-date coupons).

        C++ parity: ql/cashflows/cashflows.cpp:214-221. Note the C++ walks
        *backwards* from the previous cashflow (``leg.rbegin()``-based
        iterator through to ``leg.rend()``).
        """
        idx = cls._previous_cash_flow_index(leg, include_settlement_date_flows, settlement_date)
        if idx < 0:
            return 0.0
        return cls._aggregate_rate(leg, range(idx, -1, -1))

    # ===================================================================
    # CashFlow / Coupon inspectors used by BondFunctions
    # ===================================================================

    @classmethod
    def previous_cash_flow(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> CashFlow | None:
        """Most recent already-occurred cashflow, or ``None``.

        C++ parity: cashflows.cpp:83-99. The C++ returns a
        ``Leg::const_reverse_iterator`` and signals "none" with ``leg.rend()``;
        Python returns ``None``.
        """
        idx = cls._previous_cash_flow_index(leg, include_settlement_date_flows, settlement_date)
        return None if idx < 0 else leg[idx]

    @classmethod
    def next_cash_flow(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> CashFlow | None:
        """Next-to-occur cashflow, or ``None``.

        C++ parity: cashflows.cpp:101-117 (``leg.end()`` means "none").
        """
        idx = cls._next_cash_flow_index(leg, include_settlement_date_flows, settlement_date)
        return None if idx == len(leg) else leg[idx]

    @classmethod
    def previous_cash_flow_amount(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> float:
        """Total amount paid on the previous payment date.

        C++ parity: cashflows.cpp:143-157 — walks BACKWARD from the previous
        cashflow, summing every amount sharing its date; ``Real()`` (0.0) if
        there is no previous cashflow.
        """
        idx = cls._previous_cash_flow_index(leg, include_settlement_date_flows, settlement_date)
        if idx < 0:
            return 0.0
        payment_date = leg[idx].date()
        result = 0.0
        for i in range(idx, -1, -1):
            if leg[i].date() != payment_date:
                break
            result += leg[i].amount()
        return result

    @classmethod
    def next_cash_flow_amount(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> float:
        """Total amount paid on the next payment date.

        C++ parity: cashflows.cpp:159-173.
        """
        idx = cls._next_cash_flow_index(leg, include_settlement_date_flows, settlement_date)
        if idx == len(leg):
            return 0.0
        payment_date = leg[idx].date()
        result = 0.0
        for cf in leg[idx:]:
            if cf.date() != payment_date:
                break
            result += cf.amount()
        return result

    @classmethod
    def _first_coupon_at_next_payment_date(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> Coupon | None:
        """First ``Coupon`` sharing the next cashflow's payment date, else ``None``.

        C++ parity: the loop body repeated verbatim in cashflows.cpp:246-374
        by accrualStartDate / accrualEndDate / referencePeriodStart /
        referencePeriodEnd / accrualPeriod / accrualDays / accruedPeriod /
        accruedDays / nominal.
        """
        idx = cls._next_cash_flow_index(leg, include_settlement_date_flows, settlement_date)
        if idx == len(leg):
            return None
        payment_date = leg[idx].date()
        for cf in leg[idx:]:
            if cf.date() != payment_date:
                break
            if isinstance(cf, Coupon):
                return cf
        return None

    @classmethod
    def nominal(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> float:
        """C++ parity: cashflows.cpp:231-244."""
        cp = cls._first_coupon_at_next_payment_date(
            leg, include_settlement_date_flows, settlement_date
        )
        return 0.0 if cp is None else cp.nominal()

    @classmethod
    def accrual_start_date(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> Date:
        """C++ parity: cashflows.cpp:246-260 — null ``Date`` when no Coupon."""
        cp = cls._first_coupon_at_next_payment_date(
            leg, include_settlement_date_flows, settlement_date
        )
        return Date() if cp is None else cp.accrual_start_date()

    @classmethod
    def accrual_end_date(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> Date:
        """C++ parity: cashflows.cpp:262-276."""
        cp = cls._first_coupon_at_next_payment_date(
            leg, include_settlement_date_flows, settlement_date
        )
        return Date() if cp is None else cp.accrual_end_date()

    @classmethod
    def reference_period_start(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> Date:
        """C++ parity: cashflows.cpp:278-292."""
        cp = cls._first_coupon_at_next_payment_date(
            leg, include_settlement_date_flows, settlement_date
        )
        return Date() if cp is None else cp.reference_period_start()

    @classmethod
    def reference_period_end(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> Date:
        """C++ parity: cashflows.cpp:294-308."""
        cp = cls._first_coupon_at_next_payment_date(
            leg, include_settlement_date_flows, settlement_date
        )
        return Date() if cp is None else cp.reference_period_end()

    @classmethod
    def accrual_period(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> float:
        """C++ parity: cashflows.cpp:310-323."""
        cp = cls._first_coupon_at_next_payment_date(
            leg, include_settlement_date_flows, settlement_date
        )
        return 0.0 if cp is None else cp.accrual_period()

    @classmethod
    def accrual_days(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> int:
        """C++ parity: cashflows.cpp:325-338."""
        cp = cls._first_coupon_at_next_payment_date(
            leg, include_settlement_date_flows, settlement_date
        )
        return 0 if cp is None else cp.accrual_days()

    @classmethod
    def accrued_period(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> float:
        """C++ parity: cashflows.cpp:340-356."""
        cp = cls._first_coupon_at_next_payment_date(
            leg, include_settlement_date_flows, settlement_date
        )
        return 0.0 if cp is None else cp.accrued_period(settlement_date)

    @classmethod
    def accrued_days(
        cls,
        leg: Sequence[CashFlow],
        include_settlement_date_flows: bool | None,
        settlement_date: Date,
    ) -> int:
        """C++ parity: cashflows.cpp:358-374."""
        cp = cls._first_coupon_at_next_payment_date(
            leg, include_settlement_date_flows, settlement_date
        )
        return 0 if cp is None else cp.accrued_days(settlement_date)

    # ===================================================================
    # atmRate / basisPointValue / yieldValueBasisPoint / z-spread
    # ===================================================================

    @classmethod
    def atm_rate(
        cls,
        leg: Sequence[CashFlow],
        discount_curve: YieldTermStructureProtocol,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
        target_npv: float | None = None,
    ) -> float:
        """At-the-money rate reproducing ``target_npv``.

        C++ parity: cashflows.cpp:509-551. ``target_npv=None`` is C++'s
        ``Null<Real>()``: the target becomes the leg's own NPV minus the
        non-rate-sensitive NPV (i.e. the par coupon). With a target the
        C++ multiplies it by ``discount(npvDate)`` FIRST and only then
        subtracts the non-sensitive NPV.

        The C++ walks the leg with a ``BPSCalculator`` visitor; acyclic
        Visitor dispatch sends a ``Coupon`` to ``visit(Coupon&)`` (which
        feeds ``bps_`` only) and every other cashflow to ``visit(CashFlow&)``
        (which feeds ``nonSensNPV_`` only), so the split below is exact.
        """
        if not leg:
            return 0.0
        settle = settlement_date if settlement_date is not None else discount_curve.reference_date()
        npv_d = npv_date if npv_date is not None else settle
        npv = 0.0
        bps_sum = 0.0
        non_sens_npv = 0.0
        for cf in leg:
            if cf.has_occurred(settle, include_settlement_date_flows) or cf.trading_ex_coupon(settle):
                continue
            df = discount_curve.discount(cf.date())
            npv += cf.amount() * df
            if isinstance(cf, Coupon):
                bps_sum += cf.nominal() * cf.accrual_period() * df
            else:
                non_sens_npv += cf.amount() * df
        if target_npv is None:
            target = npv - non_sens_npv
        else:
            target = target_npv * discount_curve.discount(npv_d) - non_sens_npv
        if target == 0.0:
            return 0.0
        qassert.require(bps_sum != 0.0, "null bps: impossible atm rate")
        return target / bps_sum

    @classmethod
    def bps_yield(
        cls,
        leg: Sequence[CashFlow],
        yield_rate: InterestRate,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
    ) -> float:
        """BPS against a flat yield.

        C++ parity: cashflows.cpp:870-890 — builds a ``FlatForward`` anchored
        at the settlement date from the ``InterestRate`` and delegates to the
        discount-curve ``bps``.
        """
        if not leg:
            return 0.0
        qassert.require(
            settlement_date is not None,
            "settlement_date is required (no Settings.evaluationDate)",
        )
        assert settlement_date is not None
        flat_rate = FlatForward.from_rate(
            settlement_date,
            yield_rate.rate(),
            yield_rate.day_counter(),
            yield_rate.compounding(),
            yield_rate.frequency(),
        )
        return cls.bps(leg, flat_rate, include_settlement_date_flows, settlement_date, npv_date)

    @classmethod
    def basis_point_value(
        cls,
        leg: Sequence[CashFlow],
        yield_rate: InterestRate,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
    ) -> float:
        """Second-order price change for a one-basis-point yield shift.

        C++ parity: cashflows.cpp:1059-1091.
        """
        if not leg:
            return 0.0
        qassert.require(
            settlement_date is not None,
            "settlement_date is required (no Settings.evaluationDate)",
        )
        assert settlement_date is not None
        npv_d = npv_date if npv_date is not None else settlement_date
        npv = cls.npv_yield(leg, yield_rate, include_settlement_date_flows, settlement_date, npv_d)
        modified_duration = cls.duration(
            leg, yield_rate, Duration.Modified, include_settlement_date_flows, settlement_date, npv_d
        )
        convexity = cls.convexity(
            leg, yield_rate, include_settlement_date_flows, settlement_date, npv_d
        )
        delta = -modified_duration * npv
        gamma = (convexity / 100.0) * npv
        shift = 0.0001
        delta *= shift
        gamma *= shift * shift
        return delta + 0.5 * gamma

    @classmethod
    def yield_value_basis_point(
        cls,
        leg: Sequence[CashFlow],
        yield_rate: InterestRate,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
    ) -> float:
        """Yield change for a one-price-point move.

        C++ parity: cashflows.cpp:1106-1130 — note the ``shift`` is 0.01,
        not 0.0001, and the result is ``shift / (-npv * modifiedDuration)``.
        """
        if not leg:
            return 0.0
        qassert.require(
            settlement_date is not None,
            "settlement_date is required (no Settings.evaluationDate)",
        )
        assert settlement_date is not None
        npv_d = npv_date if npv_date is not None else settlement_date
        npv = cls.npv_yield(leg, yield_rate, include_settlement_date_flows, settlement_date, npv_d)
        modified_duration = cls.duration(
            leg, yield_rate, Duration.Modified, include_settlement_date_flows, settlement_date, npv_d
        )
        shift = 0.01
        return (1.0 / (-npv * modified_duration)) * shift

    @classmethod
    def npv_z_spread(
        cls,
        leg: Sequence[CashFlow],
        discount_curve: YieldTermStructure,
        z_spread: float,
        compounding: Compounding,
        frequency: Frequency,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
    ) -> float:
        """NPV against ``discount_curve`` shifted by a zero-rate spread.

        C++ parity: cashflows.cpp:1146-1175.
        """
        if not leg:
            return 0.0
        qassert.require(
            settlement_date is not None,
            "settlement_date is required (no Settings.evaluationDate)",
        )
        assert settlement_date is not None
        spreaded = ZeroSpreadedTermStructure(
            discount_curve, SimpleQuote(z_spread), compounding, frequency
        )
        return cls.npv_curve(
            leg, spreaded, include_settlement_date_flows, settlement_date, npv_date
        )

    @classmethod
    def z_spread(
        cls,
        leg: Sequence[CashFlow],
        npv: float,
        discount_curve: YieldTermStructure,
        compounding: Compounding,
        frequency: Frequency,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
        accuracy: float = _DEFAULT_IRR_ACCURACY,
        max_iterations: int = _DEFAULT_IRR_MAX_ITER,
        guess: float = 0.0,
    ) -> float:
        """Solve for the zero-rate spread reproducing ``npv``.

        C++ parity: cashflows.cpp:1190-1225. The solver is ``Brent`` with a
        step of 0.01 and the objective is ``npv - NPV(spread)`` — that sign
        order matters for a bracketing solver's first step.
        """
        qassert.require(
            settlement_date is not None,
            "settlement_date is required (no Settings.evaluationDate)",
        )
        assert settlement_date is not None
        quote = SimpleQuote(0.0)
        spreaded = ZeroSpreadedTermStructure(discount_curve, quote, compounding, frequency)

        def objective(spread: float) -> float:
            quote.set_value(spread)
            return npv - cls.npv_curve(
                leg, spreaded, include_settlement_date_flows, settlement_date, npv_date
            )

        solver = Brent()
        solver.set_max_evaluations(max_iterations)
        return solver.solve(objective, accuracy, guess, 0.01)

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
        solver: Solver1D | None = None,
    ) -> float:
        """Solve for the yield that reproduces ``target_npv``.

        C++ parity: ql/cashflows/cashflows.cpp:905-923 (``CashFlows::yield``)
        plus the ``template <typename Solver> yield(solver, ...)`` overload
        at cashflows.hpp:276-292. Defaults to ``NewtonSafe`` exactly as C++
        does; ``solver`` is the Python spelling of the template parameter.

        The objective is :class:`CashFlows.IrrFinder`, whose constructor
        runs C++'s ``checkSign`` guard — a (leg, target NPV) pair with no
        sign change raises ``LibraryException`` rather than silently
        returning a meaningless root.
        """
        if solver is None:
            solver = NewtonSafe()
        solver.set_max_evaluations(max_iterations)
        return cls.irr_with_solver(
            solver,
            leg,
            target_npv,
            day_counter,
            compounding,
            frequency,
            include_settlement_date_flows,
            settlement_date,
            npv_date,
            accuracy,
            guess,
        )

    @classmethod
    def irr_with_solver(
        cls,
        solver: Solver1D,
        leg: Sequence[CashFlow],
        target_npv: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        include_settlement_date_flows: bool = False,
        settlement_date: Date | None = None,
        npv_date: Date | None = None,
        accuracy: float = _DEFAULT_IRR_ACCURACY,
        guess: float = _DEFAULT_IRR_GUESS,
    ) -> float:
        """Solver-parameterised IRR.

        C++ parity: ``template <typename Solver> CashFlows::yield(solver, ...)``
        (cashflows.hpp:276-292). Note it does NOT touch the solver's max
        evaluations — the caller configures the solver — and there is
        deliberately no ``maxIterations`` parameter.
        """
        qassert.require(
            settlement_date is not None,
            "settlement_date is required (no Settings.evaluationDate)",
        )
        assert settlement_date is not None

        obj_function = cls.IrrFinder(
            leg,
            target_npv,
            day_counter,
            compounding,
            frequency,
            include_settlement_date_flows,
            settlement_date,
            npv_date,
        )
        # C++ parity: cashflows.hpp:291 — the step is guess/10, not a constant.
        return solver.solve(obj_function, accuracy, guess, guess / 10.0)
