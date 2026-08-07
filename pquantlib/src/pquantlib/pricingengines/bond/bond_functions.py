"""BondFunctions — Bond adapters of the CashFlows functions.

# C++ parity: ql/pricingengines/bond/bondfunctions.{hpp,cpp} (v1.43),
#             ``struct BondFunctions`` — an all-static free-function surface.

Every method here calls into :class:`~pquantlib.cashflows.cash_flows.CashFlows`
passing the bond's cashflows, the bond settlement date (unless another date is
given), zero ex-dividend days, and ``include_settlement_date_flows=False``.

Prices are always **clean** and always expressed **per 100 of the current
(amortised) notional**: the C++ rescales by ``100.0 / bond.notional(settlement)``,
so for a bond with a face value other than 100 any price input must be given per
100 (``quote = cash price * 100 / face``) and returned prices are per 100 too.

Almost every method is guarded by
``QL_REQUIRE(isTradable(bond, settlement), "non tradable at ...")`` where
``isTradable`` is exactly ``bond.notional(settlement) != 0.0``. Because
``Bond.notional`` returns 0 **on** the redemption date, a bond is not tradable on
its own maturity date and these methods raise there. ``accrued_amount`` is the
single exception — it returns ``0.0`` instead of raising (bondfunctions.cpp:229).

Overload mapping (C++ has overloads, Python does not)
-----------------------------------------------------
Date inspectors::

    Date startDate(const Bond&)                                  -> start_date
    Date maturityDate(const Bond&)                               -> maturity_date
    bool isTradable(const Bond&, Date)                            -> is_tradable

CashFlow inspectors (the two iterator-returning ones return ``CashFlow | None``
in Python; C++ signals "none" with ``leg.rend()`` / ``leg.end()``)::

    Leg::const_reverse_iterator previousCashFlow(const Bond&, Date)
                                                                 -> previous_cash_flow
    Leg::const_iterator nextCashFlow(const Bond&, Date)           -> next_cash_flow
    Date previousCashFlowDate(const Bond&, Date)                  -> previous_cash_flow_date
    Date nextCashFlowDate(const Bond&, Date)                      -> next_cash_flow_date
    Real previousCashFlowAmount(const Bond&, Date)                -> previous_cash_flow_amount
    Real nextCashFlowAmount(const Bond&, Date)                    -> next_cash_flow_amount

Coupon inspectors::

    Rate previousCouponRate(const Bond&, Date)                    -> previous_coupon_rate
    Rate nextCouponRate(const Bond&, Date)                        -> next_coupon_rate
    Date accrualStartDate(const Bond&, Date)                      -> accrual_start_date
    Date accrualEndDate(const Bond&, Date)                        -> accrual_end_date
    Date referencePeriodStart(const Bond&, Date)                  -> reference_period_start
    Date referencePeriodEnd(const Bond&, Date)                    -> reference_period_end
    Time accrualPeriod(const Bond&, Date)                         -> accrual_period
    Date::serial_type accrualDays(const Bond&, Date)              -> accrual_days
    Time accruedPeriod(const Bond&, Date)                         -> accrued_period
    Date::serial_type accruedDays(const Bond&, Date)              -> accrued_days
    Real accruedAmount(const Bond&, Date)                         -> accrued_amount

YieldTermStructure group — the plain names::

    Real cleanPrice(const Bond&, const YieldTermStructure&, Date) -> clean_price
    Real dirtyPrice(const Bond&, const YieldTermStructure&, Date) -> dirty_price
    Real bps(const Bond&, const YieldTermStructure&, Date)        -> bps
    Rate atmRate(const Bond&, const YieldTermStructure&, Date, Bond::Price)
                                                                 -> atm_rate

InterestRate group — ``*_from_yield`` (the argument is an ``InterestRate``)::

    Real cleanPrice(const Bond&, const InterestRate&, Date)       -> clean_price_from_yield
    Real dirtyPrice(const Bond&, const InterestRate&, Date)       -> dirty_price_from_yield
    Real bps(const Bond&, const InterestRate&, Date)              -> bps_from_yield
    Time duration(const Bond&, const InterestRate&, Duration::Type, Date)
                                                                 -> duration_from_yield
    Real convexity(const Bond&, const InterestRate&, Date)        -> convexity_from_yield
    Real basisPointValue(const Bond&, const InterestRate&, Date)  -> basis_point_value_from_yield
    Real yieldValueBasisPoint(const Bond&, const InterestRate&, Date)
                                                          -> yield_value_basis_point_from_yield

Rate + DayCounter + Compounding + Frequency group — ``*_from_rate``. Each of
these C++ overloads does nothing but build ``InterestRate(yield, dc, comp, freq)``
and forward, so the Python versions are one-liners with the same contract::

    Real cleanPrice(const Bond&, Rate, const DayCounter&, Compounding, Frequency, Date)
                                                                 -> clean_price_from_rate
    Real dirtyPrice(...)                                          -> dirty_price_from_rate
    Real bps(...)                                                 -> bps_from_rate
    Time duration(..., Duration::Type, Date)                      -> duration_from_rate
    Real convexity(...)                                           -> convexity_from_rate
    Real basisPointValue(...)                                     -> basis_point_value_from_rate
    Real yieldValueBasisPoint(...)                        -> yield_value_basis_point_from_rate

Yield (IRR). ``yield`` is a Python keyword, so the two C++ overloads become::

    Rate yield(const Bond&, Bond::Price, const DayCounter&, Compounding, Frequency,
               Date, Real accuracy, Size maxIterations, Rate guess)
                                                                 -> bond_yield
    template <typename Solver>
    Rate yield(const Solver&, const Bond&, Bond::Price, const DayCounter&,
               Compounding, Frequency, Date, Real accuracy, Rate guess)
                                                                 -> bond_yield_with_solver

Note the template overload has **no** ``maxIterations`` parameter — the caller
pre-configures the solver via ``set_max_evaluations``. ``bond_yield`` is exactly
``bond_yield_with_solver`` with a ``NewtonSafe`` whose max evaluations is set.

Z-spread group::

    Real cleanPrice(const Bond&, const shared_ptr<YieldTermStructure>&, Spread,
                    Compounding, Frequency, Date)                -> clean_price_from_z_spread
    Real dirtyPrice(...)                                          -> dirty_price_from_z_spread
    Spread zSpread(const Bond&, Bond::Price, const shared_ptr<YieldTermStructure>&,
                   Compounding, Frequency, Date, Real, Size, Rate)
                                                                 -> z_spread

The three ``[[deprecated]]`` v1.42 overloads of those same three functions take
an extra ``const DayCounter&`` between the spread and the compounding and
**discard it** — in bondfunctions.cpp:500-508 / 530-538 / 570-582 the parameter
is unnamed and the body forwards to the 5-argument overload. They are reproduced
here as the optional ``day_counter`` keyword argument on the same three methods,
which is likewise accepted and ignored. Passing it changes nothing; that is C++
behaviour, cross-validated in
``fixed_zspread_deprecated_daycounter_ignored``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.cash_flows import CashFlows
from pquantlib.cashflows.duration import Duration
from pquantlib.instruments.bond import BondPrice, BondPriceType
from pquantlib.interest_rate import InterestRate
from pquantlib.math.solvers1d.newton_safe import NewtonSafe
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.instruments.bond import Bond
    from pquantlib.math.solvers1d.solver_1d import Solver1D
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure
    from pquantlib.time.compounding import Compounding
    from pquantlib.time.frequency import Frequency

_NULL_DATE: Date = Date()

# C++ parity: bondfunctions.hpp:164-166 / 278-280 default arguments.
_DEFAULT_YIELD_ACCURACY: float = 1.0e-10
_DEFAULT_YIELD_MAX_ITERATIONS: int = 100
_DEFAULT_YIELD_GUESS: float = 0.05
_DEFAULT_ZSPREAD_GUESS: float = 0.0


def _settle(bond: Bond, settlement_date: Date | None) -> Date:
    """C++ ``if (settlement == Date()) settlement = bond.settlementDate();``."""
    if settlement_date is None or settlement_date == _NULL_DATE:
        return bond.settlement_date()
    return settlement_date


class BondFunctions:
    """Namespace-only class — direct construction is disabled.

    # C++ parity: ``struct BondFunctions`` (bondfunctions.hpp:59) has no
    # members and only static methods, i.e. a namespace. The Python port
    # matches by exposing static methods only and refusing construction,
    # exactly as :class:`~pquantlib.cashflows.cash_flows.CashFlows` does.
    """

    def __init__(self) -> None:
        msg = "BondFunctions is a namespace; use static methods only"
        raise TypeError(msg)

    # ------------------------------------------------------------------
    # Date inspectors
    # ------------------------------------------------------------------

    @staticmethod
    def start_date(bond: Bond) -> Date:
        """C++ parity: bondfunctions.cpp:31-33."""
        return CashFlows.start_date(bond.cashflows())

    @staticmethod
    def maturity_date(bond: Bond) -> Date:
        """C++ parity: bondfunctions.cpp:35-37."""
        return CashFlows.maturity_date(bond.cashflows())

    @staticmethod
    def is_tradable(bond: Bond, settlement_date: Date | None = None) -> bool:
        """C++ parity: bondfunctions.cpp:39-45 — ``notional(settlement) != 0``."""
        return bond.notional(_settle(bond, settlement_date)) != 0.0

    @staticmethod
    def _require_tradable(bond: Bond, settlement: Date) -> None:
        qassert.require(
            BondFunctions.is_tradable(bond, settlement),
            f"non tradable at {settlement} (maturity being {bond.maturity_date()})",
        )

    # ------------------------------------------------------------------
    # CashFlow inspectors
    # ------------------------------------------------------------------

    @staticmethod
    def previous_cash_flow(bond: Bond, ref_date: Date | None = None) -> CashFlow | None:
        """C++ parity: bondfunctions.cpp:47-55."""
        settlement = _settle(bond, ref_date)
        return CashFlows.previous_cash_flow(bond.cashflows(), False, settlement)

    @staticmethod
    def next_cash_flow(bond: Bond, ref_date: Date | None = None) -> CashFlow | None:
        """C++ parity: bondfunctions.cpp:57-64."""
        settlement = _settle(bond, ref_date)
        return CashFlows.next_cash_flow(bond.cashflows(), False, settlement)

    @staticmethod
    def previous_cash_flow_date(bond: Bond, ref_date: Date | None = None) -> Date:
        """C++ parity: bondfunctions.cpp:66-73."""
        settlement = _settle(bond, ref_date)
        return CashFlows.previous_cash_flow_date(bond.cashflows(), False, settlement)

    @staticmethod
    def next_cash_flow_date(bond: Bond, ref_date: Date | None = None) -> Date:
        """C++ parity: bondfunctions.cpp:75-82."""
        settlement = _settle(bond, ref_date)
        return CashFlows.next_cash_flow_date(bond.cashflows(), False, settlement)

    @staticmethod
    def previous_cash_flow_amount(bond: Bond, ref_date: Date | None = None) -> float:
        """C++ parity: bondfunctions.cpp:84-91."""
        settlement = _settle(bond, ref_date)
        return CashFlows.previous_cash_flow_amount(bond.cashflows(), False, settlement)

    @staticmethod
    def next_cash_flow_amount(bond: Bond, ref_date: Date | None = None) -> float:
        """C++ parity: bondfunctions.cpp:93-100."""
        settlement = _settle(bond, ref_date)
        return CashFlows.next_cash_flow_amount(bond.cashflows(), False, settlement)

    # ------------------------------------------------------------------
    # Coupon inspectors
    # ------------------------------------------------------------------

    @staticmethod
    def previous_coupon_rate(bond: Bond, settlement_date: Date | None = None) -> float:
        """C++ parity: bondfunctions.cpp:102-109. Note this one is NOT guarded."""
        settlement = _settle(bond, settlement_date)
        return CashFlows.previous_coupon_rate(bond.cashflows(), False, settlement)

    @staticmethod
    def next_coupon_rate(bond: Bond, settlement_date: Date | None = None) -> float:
        """C++ parity: bondfunctions.cpp:111-118. Note this one is NOT guarded."""
        settlement = _settle(bond, settlement_date)
        return CashFlows.next_coupon_rate(bond.cashflows(), False, settlement)

    @staticmethod
    def accrual_start_date(bond: Bond, settlement_date: Date | None = None) -> Date:
        """C++ parity: bondfunctions.cpp:120-131."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return CashFlows.accrual_start_date(bond.cashflows(), False, settlement)

    @staticmethod
    def accrual_end_date(bond: Bond, settlement_date: Date | None = None) -> Date:
        """C++ parity: bondfunctions.cpp:133-144."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return CashFlows.accrual_end_date(bond.cashflows(), False, settlement)

    @staticmethod
    def reference_period_start(bond: Bond, settlement_date: Date | None = None) -> Date:
        """C++ parity: bondfunctions.cpp:146-157."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return CashFlows.reference_period_start(bond.cashflows(), False, settlement)

    @staticmethod
    def reference_period_end(bond: Bond, settlement_date: Date | None = None) -> Date:
        """C++ parity: bondfunctions.cpp:159-170."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return CashFlows.reference_period_end(bond.cashflows(), False, settlement)

    @staticmethod
    def accrual_period(bond: Bond, settlement_date: Date | None = None) -> float:
        """C++ parity: bondfunctions.cpp:172-183."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return CashFlows.accrual_period(bond.cashflows(), False, settlement)

    @staticmethod
    def accrual_days(bond: Bond, settlement_date: Date | None = None) -> int:
        """C++ parity: bondfunctions.cpp:185-196."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return CashFlows.accrual_days(bond.cashflows(), False, settlement)

    @staticmethod
    def accrued_period(bond: Bond, settlement_date: Date | None = None) -> float:
        """C++ parity: bondfunctions.cpp:198-209."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return CashFlows.accrued_period(bond.cashflows(), False, settlement)

    @staticmethod
    def accrued_days(bond: Bond, settlement_date: Date | None = None) -> int:
        """C++ parity: bondfunctions.cpp:211-222."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return CashFlows.accrued_days(bond.cashflows(), False, settlement)

    @staticmethod
    def accrued_amount(bond: Bond, settlement_date: Date | None = None) -> float:
        """Accrued amount per 100 of current notional.

        C++ parity: bondfunctions.cpp:224-235. The only inspector that
        RETURNS 0.0 when the bond is not tradable instead of raising.
        """
        settlement = _settle(bond, settlement_date)
        if not BondFunctions.is_tradable(bond, settlement):
            return 0.0
        return (
            CashFlows.accrued_amount(bond.cashflows(), False, settlement)
            * 100.0
            / bond.notional(settlement)
        )

    # ------------------------------------------------------------------
    # YieldTermStructure group
    # ------------------------------------------------------------------

    @staticmethod
    def clean_price(
        bond: Bond,
        discount_curve: YieldTermStructureProtocol,
        settlement_date: Date | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:239-246."""
        settlement = _settle(bond, settlement_date)
        return BondFunctions.dirty_price(bond, discount_curve, settlement) - bond.accrued_amount(
            settlement
        )

    @staticmethod
    def dirty_price(
        bond: Bond,
        discount_curve: YieldTermStructureProtocol,
        settlement_date: Date | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:248-263."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return (
            CashFlows.npv_curve(bond.cashflows(), discount_curve, False, settlement)
            * 100.0
            / bond.notional(settlement)
        )

    @staticmethod
    def bps(
        bond: Bond,
        discount_curve: YieldTermStructureProtocol,
        settlement_date: Date | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:265-278."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return (
            CashFlows.bps(bond.cashflows(), discount_curve, False, settlement)
            * 100.0
            / bond.notional(settlement)
        )

    @staticmethod
    def atm_rate(
        bond: Bond,
        discount_curve: YieldTermStructureProtocol,
        settlement_date: Date | None = None,
        price: BondPrice | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:280-304.

        ``price=None`` is the C++ default ``Bond::Price{}`` (invalid), which
        makes ``atmRate`` pass ``Null<Real>()`` as the target NPV.
        """
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        target_npv: float | None = None
        if price is not None and price.is_valid():
            dirty = price.amount() + (
                bond.accrued_amount(settlement) if price.type() == BondPriceType.Clean else 0.0
            )
            target_npv = dirty / 100.0 * bond.notional(settlement)
        return CashFlows.atm_rate(
            bond.cashflows(), discount_curve, False, settlement, settlement, target_npv
        )

    # ------------------------------------------------------------------
    # InterestRate group
    # ------------------------------------------------------------------

    @staticmethod
    def clean_price_from_yield(
        bond: Bond, yield_rate: InterestRate, settlement_date: Date | None = None
    ) -> float:
        """C++ parity: bondfunctions.cpp:306-310.

        Note the C++ does NOT default the settlement date before calling
        ``bond.accruedAmount(settlement)`` — it relies on ``accruedAmount``
        defaulting a null date itself. Same net effect.
        """
        settlement = _settle(bond, settlement_date)
        return BondFunctions.dirty_price_from_yield(
            bond, yield_rate, settlement
        ) - bond.accrued_amount(settlement)

    @staticmethod
    def dirty_price_from_yield(
        bond: Bond, yield_rate: InterestRate, settlement_date: Date | None = None
    ) -> float:
        """C++ parity: bondfunctions.cpp:322-336."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return (
            CashFlows.npv_yield(bond.cashflows(), yield_rate, False, settlement)
            * 100.0
            / bond.notional(settlement)
        )

    @staticmethod
    def bps_from_yield(
        bond: Bond, yield_rate: InterestRate, settlement_date: Date | None = None
    ) -> float:
        """C++ parity: bondfunctions.cpp:348-361."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return (
            CashFlows.bps_yield(bond.cashflows(), yield_rate, False, settlement)
            * 100.0
            / bond.notional(settlement)
        )

    @staticmethod
    def duration_from_yield(
        bond: Bond,
        yield_rate: InterestRate,
        duration_type: Duration = Duration.Modified,
        settlement_date: Date | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:389-403."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return CashFlows.duration(bond.cashflows(), yield_rate, duration_type, False, settlement)

    @staticmethod
    def convexity_from_yield(
        bond: Bond, yield_rate: InterestRate, settlement_date: Date | None = None
    ) -> float:
        """C++ parity: bondfunctions.cpp:416-428."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return CashFlows.convexity(bond.cashflows(), yield_rate, False, settlement)

    @staticmethod
    def basis_point_value_from_yield(
        bond: Bond, yield_rate: InterestRate, settlement_date: Date | None = None
    ) -> float:
        """C++ parity: bondfunctions.cpp:440-452."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return CashFlows.basis_point_value(bond.cashflows(), yield_rate, False, settlement)

    @staticmethod
    def yield_value_basis_point_from_yield(
        bond: Bond, yield_rate: InterestRate, settlement_date: Date | None = None
    ) -> float:
        """C++ parity: bondfunctions.cpp:464-476."""
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return CashFlows.yield_value_basis_point(bond.cashflows(), yield_rate, False, settlement)

    # ------------------------------------------------------------------
    # Rate + DayCounter + Compounding + Frequency group
    # ------------------------------------------------------------------

    @staticmethod
    def clean_price_from_rate(
        bond: Bond,
        yield_rate: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:312-320."""
        y = InterestRate(yield_rate, day_counter, compounding, frequency)
        return BondFunctions.clean_price_from_yield(bond, y, settlement_date)

    @staticmethod
    def dirty_price_from_rate(
        bond: Bond,
        yield_rate: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:338-346."""
        y = InterestRate(yield_rate, day_counter, compounding, frequency)
        return BondFunctions.dirty_price_from_yield(bond, y, settlement_date)

    @staticmethod
    def bps_from_rate(
        bond: Bond,
        yield_rate: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:363-371."""
        y = InterestRate(yield_rate, day_counter, compounding, frequency)
        return BondFunctions.bps_from_yield(bond, y, settlement_date)

    @staticmethod
    def duration_from_rate(
        bond: Bond,
        yield_rate: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        duration_type: Duration = Duration.Modified,
        settlement_date: Date | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:405-414."""
        y = InterestRate(yield_rate, day_counter, compounding, frequency)
        return BondFunctions.duration_from_yield(bond, y, duration_type, settlement_date)

    @staticmethod
    def convexity_from_rate(
        bond: Bond,
        yield_rate: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:430-438."""
        y = InterestRate(yield_rate, day_counter, compounding, frequency)
        return BondFunctions.convexity_from_yield(bond, y, settlement_date)

    @staticmethod
    def basis_point_value_from_rate(
        bond: Bond,
        yield_rate: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:454-462."""
        y = InterestRate(yield_rate, day_counter, compounding, frequency)
        return BondFunctions.basis_point_value_from_yield(bond, y, settlement_date)

    @staticmethod
    def yield_value_basis_point_from_rate(
        bond: Bond,
        yield_rate: float,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:478-486."""
        y = InterestRate(yield_rate, day_counter, compounding, frequency)
        return BondFunctions.yield_value_basis_point_from_yield(bond, y, settlement_date)

    # ------------------------------------------------------------------
    # Yield (IRR)
    # ------------------------------------------------------------------

    @staticmethod
    def bond_yield_with_solver(
        solver: Solver1D,
        bond: Bond,
        price: BondPrice,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
        accuracy: float = _DEFAULT_YIELD_ACCURACY,
        guess: float = _DEFAULT_YIELD_GUESS,
    ) -> float:
        """Solver-parameterised yield.

        C++ parity: bondfunctions.hpp:167-195, ``template <typename Solver>
        static Rate yield(const Solver& solver, ...)``. There is deliberately
        no ``maxIterations`` argument: the caller configures the solver.
        """
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        amount = price.amount()
        if price.type() == BondPriceType.Clean:
            amount += bond.accrued_amount(settlement)
        amount /= 100.0 / bond.notional(settlement)
        return CashFlows.irr_with_solver(
            solver,
            bond.cashflows(),
            amount,
            day_counter,
            compounding,
            frequency,
            False,
            settlement,
            settlement,
            accuracy,
            guess,
        )

    @staticmethod
    def bond_yield(
        bond: Bond,
        price: BondPrice,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
        accuracy: float = _DEFAULT_YIELD_ACCURACY,
        max_iterations: int = _DEFAULT_YIELD_MAX_ITERATIONS,
        guess: float = _DEFAULT_YIELD_GUESS,
    ) -> float:
        """Yield from a price, using ``NewtonSafe``.

        C++ parity: bondfunctions.cpp:373-387 — a ``NewtonSafe`` with
        ``setMaxEvaluations(maxIterations)`` handed to the template overload.
        """
        solver = NewtonSafe()
        solver.set_max_evaluations(max_iterations)
        return BondFunctions.bond_yield_with_solver(
            solver, bond, price, day_counter, compounding, frequency, settlement_date, accuracy,
            guess,
        )

    # ------------------------------------------------------------------
    # Z-spread group
    # ------------------------------------------------------------------

    @staticmethod
    def clean_price_from_z_spread(
        bond: Bond,
        discount_curve: YieldTermStructure,
        z_spread: float,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
        day_counter: DayCounter | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:488-498.

        ``day_counter`` is the deprecated-v1.42 overload's extra argument
        (bondfunctions.cpp:500-508); C++ ignores it and so does this.
        """
        del day_counter  # C++ parity: the deprecated overload discards it.
        settlement = _settle(bond, settlement_date)
        return BondFunctions.dirty_price_from_z_spread(
            bond, discount_curve, z_spread, compounding, frequency, settlement
        ) - bond.accrued_amount(settlement)

    @staticmethod
    def dirty_price_from_z_spread(
        bond: Bond,
        discount_curve: YieldTermStructure,
        z_spread: float,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
        day_counter: DayCounter | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:510-528 (+ 530-538 deprecated)."""
        del day_counter  # C++ parity: the deprecated overload discards it.
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        return (
            CashFlows.npv_z_spread(
                bond.cashflows(),
                discount_curve,
                z_spread,
                compounding,
                frequency,
                False,
                settlement,
            )
            * 100.0
            / bond.notional(settlement)
        )

    @staticmethod
    def z_spread(
        bond: Bond,
        price: BondPrice,
        discount_curve: YieldTermStructure,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
        accuracy: float = _DEFAULT_YIELD_ACCURACY,
        max_iterations: int = _DEFAULT_YIELD_MAX_ITERATIONS,
        guess: float = _DEFAULT_ZSPREAD_GUESS,
        day_counter: DayCounter | None = None,
    ) -> float:
        """C++ parity: bondfunctions.cpp:540-568 (+ 570-582 deprecated)."""
        del day_counter  # C++ parity: the deprecated overload discards it.
        settlement = _settle(bond, settlement_date)
        BondFunctions._require_tradable(bond, settlement)
        dirty = price.amount() + (
            bond.accrued_amount(settlement) if price.type() == BondPriceType.Clean else 0.0
        )
        dirty /= 100.0 / bond.notional(settlement)
        return CashFlows.z_spread(
            bond.cashflows(),
            dirty,
            discount_curve,
            compounding,
            frequency,
            False,
            settlement,
            settlement,
            accuracy,
            max_iterations,
            guess,
        )


__all__ = ["BondFunctions", "BondPrice", "BondPriceType"]
