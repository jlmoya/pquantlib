"""CallableBond — bond with an embedded call/put schedule.

# C++ parity: ql/experimental/callablebonds/callablebond.{hpp,cpp} (v1.42.1).

Base callable-bond class for fixed-rate and zero-coupon bonds.  Only
European/Bermudan put/call schedules are supported (no American
optionality), as in the C++ source.

Beyond the instrument plumbing (``setup_arguments`` filling the
callable-bond engine arguments), this ports the analytics:

- ``implied_volatility`` — Black implied forward-yield vol (Brent solve
  over a ``BlackCallableFixedRateBondEngine``).
- ``oas`` / ``clean_price_oas`` — option-adjusted spread round-trip
  (Brent solve over the bond's *current* engine; the tree engine applies
  the continuous spread to the lattice).
- ``effective_duration`` / ``effective_convexity`` — bump-and-revalue of
  the OAS-implied clean price.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.coupon import Coupon
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.instruments.bond import (
    Bond,
    BondArguments,
    BondPrice,
    BondPriceType,
    BondResults,
)
from pquantlib.interest_rate import InterestRate
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.instruments.callability import Callability
    from pquantlib.pricingengines.pricing_engine import PricingEngineArguments
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure
    from pquantlib.time.calendar import Calendar
    from pquantlib.time.schedule import Schedule

_NULL_DATE: Date = Date()


class CallableBondArguments(BondArguments):
    """Engine argument carrier for callable bonds.

    # C++ parity: ``CallableBond::arguments`` (callablebond.hpp:151-171).
    """

    def __init__(self) -> None:
        super().__init__()
        self.coupon_dates: list[Date] = []
        self.coupon_amounts: list[float] = []
        self.face_amount: float = 0.0
        self.redemption: float = 0.0
        self.redemption_date: Date = _NULL_DATE
        self.payment_day_counter: DayCounter | None = None
        self.frequency: Frequency = Frequency.NoFrequency
        self.put_call_schedule: list[Callability] = []
        self.callability_prices: list[float] = []
        self.callability_dates: list[Date] = []
        # Continuous spread added to the model (only applied by the tree engine).
        self.spread: float = 0.0

    def validate(self) -> None:
        # C++ parity: callablebond.cpp:56-70.
        qassert.require(self.settlement_date != _NULL_DATE, "null settlement date")
        qassert.require(self.redemption >= 0.0, "positive redemption required")
        qassert.require(
            len(self.callability_dates) == len(self.callability_prices),
            "different number of callability dates and prices",
        )
        qassert.require(
            len(self.coupon_dates) == len(self.coupon_amounts),
            "different number of coupon dates and amounts",
        )


class CallableBondResults(BondResults):
    """Results carrier for callable bonds (no extra fields).

    # C++ parity: ``CallableBond::results`` (callablebond.hpp:174-177).
    """


class CallableBond(Bond):
    """Abstract base for callable/puttable bonds.

    # C++ parity: ``class CallableBond : public Bond`` (callablebond.hpp:53).

    Derived classes populate ``self._cashflows`` + ``self._frequency``
    before delegating to the protected helpers.
    """

    def __init__(
        self,
        settlement_days: int,
        maturity_date: Date,
        calendar: Calendar,
        payment_day_counter: DayCounter,
        face_amount: float,
        issue_date: Date | None = None,
        put_call_schedule: Sequence[Callability] | None = None,
    ) -> None:
        Bond.__init__(self, settlement_days, calendar, issue_date)
        self._payment_day_counter: DayCounter = payment_day_counter
        self._put_call_schedule: list[Callability] = (
            list(put_call_schedule) if put_call_schedule else []
        )
        self._face_amount: float = face_amount
        self._frequency: Frequency = Frequency.NoFrequency
        self._maturity_date = maturity_date

        if self._put_call_schedule:
            final_option_date = Date.min_date()
            for c in self._put_call_schedule:
                final_option_date = max(final_option_date, c.date())
            qassert.require(
                final_option_date <= self._maturity_date,
                "Bond cannot mature before last call/put date",
            )

    # ------------------------------------------------------------------
    # Inspectors
    # ------------------------------------------------------------------

    def callability(self) -> list[Callability]:
        # C++ parity: callablebond.hpp:62-64.
        return list(self._put_call_schedule)

    def frequency(self) -> Frequency:
        return self._frequency

    # ------------------------------------------------------------------
    # setup_arguments
    # ------------------------------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        # C++ parity: callablebond.cpp:404-476.
        Bond.setup_arguments(self, args)
        qassert.require(isinstance(args, CallableBondArguments), "no arguments given")
        assert isinstance(args, CallableBondArguments)

        settlement = args.settlement_date
        args.face_amount = self._face_amount
        args.redemption = self.redemption().amount()
        args.redemption_date = self.redemption().date()

        cfs = self.cashflows()
        args.coupon_dates = []
        args.coupon_amounts = []
        # C++ iterates cfs[0 .. size-2] (the last cashflow is the redemption).
        for cf in cfs[:-1]:
            if not cf.has_occurred(settlement, False) and not cf.trading_ex_coupon(settlement):
                args.coupon_dates.append(cf.date())
                args.coupon_amounts.append(cf.amount())

        args.callability_prices = []
        args.callability_dates = []
        args.payment_day_counter = self._payment_day_counter
        args.frequency = self._frequency
        args.put_call_schedule = list(self._put_call_schedule)

        for c in self._put_call_schedule:
            # C++ ``Event::hasOccurred(settlement, false)`` — a call/put right
            # is "still live" iff its date is strictly after the settlement.
            if not _occurred(c.date(), settlement):
                args.callability_dates.append(c.date())
                price = c.price().amount()
                if c.price().type() == BondPriceType.Clean:
                    # Convert clean call price to dirty using accrued at the
                    # call date (issuer action — ignore ex-coupon conventions).
                    # C++ parity: callablebond.cpp:447-471.
                    call_date = c.date()
                    call_accrued = 0.0
                    for cf in self._cashflows:
                        if not cf.has_occurred(call_date, False):
                            if isinstance(cf, Coupon):
                                acc = cf.accrued_amount(call_date)
                                if cf.trading_ex_coupon(call_date):
                                    acc = cf.amount() + acc
                                call_accrued = acc / self.notional(call_date) * 100.0
                            break
                    price = price + call_accrued
                args.callability_prices.append(price)

        args.spread = 0.0

    # ------------------------------------------------------------------
    # Implied volatility (Black engine)
    # ------------------------------------------------------------------

    def implied_volatility(
        self,
        target_price: BondPrice,
        discount_curve: YieldTermStructureProtocol,
        accuracy: float,
        max_evaluations: int,
        min_vol: float,
        max_vol: float,
    ) -> float:
        """Black implied forward-yield volatility.

        # C++ parity: callablebond.cpp:112-139.
        """
        qassert.require(not self.is_expired(), "instrument expired")
        if target_price.type() == BondPriceType.Dirty:
            dirty_target = target_price.amount()
        elif target_price.type() == BondPriceType.Clean:
            dirty_target = target_price.amount() + self.accrued_amount()
        else:  # pragma: no cover - exhaustive
            qassert.fail("unknown price type")
        target_value = dirty_target * self._face_amount / 100.0
        guess = 0.5 * (min_vol + max_vol)

        # Lazy imports break the callable_bond <-> black engine cycle.
        from pquantlib.pricingengines.bond.black_callable_bond_engine import (  # noqa: PLC0415
            BlackCallableFixedRateBondEngine,
        )
        from pquantlib.quotes.simple_quote import SimpleQuote  # noqa: PLC0415

        vol = SimpleQuote(0.0)
        engine = BlackCallableFixedRateBondEngine(vol, discount_curve)
        self.setup_arguments(engine.get_arguments())
        results = engine.get_results()

        def objective(x: float) -> float:
            vol.set_value(x)
            engine.calculate()
            # matchNPV=False -> use settlement_value (C++ ImpliedVolHelper).
            assert isinstance(results, CallableBondResults)
            sv = results.settlement_value
            assert sv is not None
            return sv - target_value

        solver = Brent()
        solver.set_max_evaluations(max_evaluations)
        return solver.solve(objective, accuracy, guess, min_vol, max_vol)

    # ------------------------------------------------------------------
    # OAS / clean price / duration / convexity
    # ------------------------------------------------------------------

    def oas(
        self,
        clean_price: float,
        engine_ts: YieldTermStructure,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
        accuracy: float = 1.0e-10,
        max_iterations: int = 100,
        guess: float = 0.0,
    ) -> float:
        """Option-adjusted spread reproducing ``clean_price``.

        # C++ parity: callablebond.cpp:279-310.
        """
        settle = settlement_date if settlement_date not in (None, _NULL_DATE) else self.settlement_date()
        assert settle is not None
        dirty_price = clean_price + self.accrued_amount(settle)
        dirty_price /= 100.0 / self.notional(settle)

        npv_helper = self._npv_spread_helper()

        def objective(x: float) -> float:
            return dirty_price - npv_helper(x)

        solver = Brent()
        solver.set_max_evaluations(max_iterations)
        step = 0.001
        oas_cont = solver.solve(objective, accuracy, guess, step)
        return _continuous_to_conv(
            oas_cont, self, engine_ts, day_counter, compounding, frequency
        )

    def clean_price_oas(
        self,
        oas: float,
        engine_ts: YieldTermStructure,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        settlement_date: Date | None = None,
    ) -> float:
        """Clean price implied by a (conventional) OAS over ``engine_ts``.

        # C++ parity: callablebond.cpp:314-336.
        """
        settle = settlement_date if settlement_date not in (None, _NULL_DATE) else self.settlement_date()
        assert settle is not None
        oas_cont = _conv_to_continuous(
            oas, self, engine_ts, day_counter, compounding, frequency
        )
        npv_helper = self._npv_spread_helper()
        p = npv_helper(oas_cont) * 100.0 / self.notional(settle) - self.accrued_amount(settle)
        return p

    def effective_duration(
        self,
        oas: float,
        engine_ts: YieldTermStructure,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        bump: float = 2e-4,
    ) -> float:
        """Effective duration — first differential of dirty price vs a shift.

        # C++ parity: callablebond.cpp:338-368.
        """
        p = self.clean_price_oas(oas, engine_ts, day_counter, compounding, frequency)
        ppp = self.clean_price_oas(oas + bump, engine_ts, day_counter, compounding, frequency)
        pmm = self.clean_price_oas(oas - bump, engine_ts, day_counter, compounding, frequency)
        if p == 0.0:
            return 0.0
        return (pmm - ppp) / (2 * p * bump)

    def effective_convexity(
        self,
        oas: float,
        engine_ts: YieldTermStructure,
        day_counter: DayCounter,
        compounding: Compounding,
        frequency: Frequency,
        bump: float = 2e-4,
    ) -> float:
        """Effective convexity — second differential of dirty price vs a shift.

        # C++ parity: callablebond.cpp:370-401.
        """
        p = self.clean_price_oas(oas, engine_ts, day_counter, compounding, frequency)
        ppp = self.clean_price_oas(oas + bump, engine_ts, day_counter, compounding, frequency)
        pmm = self.clean_price_oas(oas - bump, engine_ts, day_counter, compounding, frequency)
        if p == 0.0:
            return 0.0
        return (ppp + pmm - 2 * p) / (bump**2 * p)

    # ------------------------------------------------------------------
    # NPVSpreadHelper — re-runs the bond's current engine with a spread
    # ------------------------------------------------------------------

    def _npv_spread_helper(self) -> Callable[[float], float]:
        # C++ parity: ``CallableBond::NPVSpreadHelper`` (callablebond.cpp:253-277).
        engine = self._engine
        qassert.require(engine is not None, "no pricing engine set")
        assert engine is not None
        self.setup_arguments(engine.get_arguments())
        results = engine.get_results()

        def helper(x: float) -> float:
            args = engine.get_arguments()
            assert isinstance(args, CallableBondArguments)
            orig = args.spread
            try:
                args.spread = x
                engine.calculate()
                assert isinstance(results, BondResults)
                val = results.value
                assert val is not None
                return val
            finally:
                args.spread = orig

        return helper


def _occurred(d: Date, ref: Date) -> bool:
    # C++ Event::hasOccurred(settlement, false): an event occurs if its date
    # is on or before the reference (with includeRefDate=false → strictly <=).
    return d <= ref


# ----------------------------------------------------------------------
# Continuous <-> conventional spread conversions (callablebond.cpp:180-248)
# ----------------------------------------------------------------------


def _continuous_to_conv(
    oas: float,
    bond: Bond,
    yts: YieldTermStructure,
    day_counter: DayCounter,
    compounding: Compounding,
    frequency: Frequency,
) -> float:
    # C++ parity: callablebond.cpp:180-211.
    zz = yts.zero_rate(
        bond.maturity_date(),
        Compounding.Continuous,
        Frequency.NoFrequency,
        result_day_counter=day_counter,
    ).rate()
    base_rate = InterestRate(zz, day_counter, Compounding.Continuous, Frequency.NoFrequency)
    spreaded_rate = InterestRate(oas + zz, day_counter, Compounding.Continuous, Frequency.NoFrequency)
    br = base_rate.equivalent_rate_dates(
        day_counter, compounding, frequency, yts.reference_date(), bond.maturity_date()
    ).rate()
    sr = spreaded_rate.equivalent_rate_dates(
        day_counter, compounding, frequency, yts.reference_date(), bond.maturity_date()
    ).rate()
    return sr - br


def _conv_to_continuous(
    oas: float,
    bond: Bond,
    yts: YieldTermStructure,
    day_counter: DayCounter,
    compounding: Compounding,
    frequency: Frequency,
) -> float:
    # C++ parity: callablebond.cpp:216-248.
    zz = yts.zero_rate(
        bond.maturity_date(),
        compounding,
        frequency,
        result_day_counter=day_counter,
    ).rate()
    base_rate = InterestRate(zz, day_counter, compounding, frequency)
    spreaded_rate = InterestRate(oas + zz, day_counter, compounding, frequency)
    br = base_rate.equivalent_rate_dates(
        day_counter,
        Compounding.Continuous,
        Frequency.NoFrequency,
        yts.reference_date(),
        bond.maturity_date(),
    ).rate()
    sr = spreaded_rate.equivalent_rate_dates(
        day_counter,
        Compounding.Continuous,
        Frequency.NoFrequency,
        yts.reference_date(),
        bond.maturity_date(),
    ).rate()
    return sr - br


# ----------------------------------------------------------------------
# Concrete callable bonds
# ----------------------------------------------------------------------


class CallableFixedRateBond(CallableBond):
    """Callable/puttable fixed-rate bond.

    # C++ parity: ``CallableFixedRateBond`` (callablebond.cpp:479-509).
    """

    def __init__(
        self,
        settlement_days: int,
        face_amount: float,
        schedule: Schedule,
        coupons: Sequence[float],
        accrual_day_counter: DayCounter,
        payment_convention: BusinessDayConvention | None = None,
        redemption: float = 100.0,
        issue_date: Date | None = None,
        put_call_schedule: Sequence[Callability] | None = None,
    ) -> None:
        pay_conv = (
            payment_convention
            if payment_convention is not None
            else BusinessDayConvention.Following
        )
        CallableBond.__init__(
            self,
            settlement_days,
            schedule.dates[-1],
            schedule.calendar,
            accrual_day_counter,
            face_amount,
            issue_date,
            put_call_schedule,
        )
        self._frequency = (
            schedule.tenor.frequency() if schedule.has_tenor() else Frequency.NoFrequency
        )
        self._cashflows = list(
            fixed_rate_leg(
                schedule,
                nominals=[face_amount],
                rates=list(coupons),
                day_counter=accrual_day_counter,
                compounding=Compounding.Simple,
                frequency=Frequency.Annual,
                payment_adjustment=pay_conv,
                payment_calendar=schedule.calendar,
            )
        )
        self._add_redemptions_to_cashflows([redemption])
        for cf in self._cashflows:
            cf.register_with(self)


class CallableZeroCouponBond(CallableBond):
    """Callable/puttable zero-coupon bond.

    # C++ parity: ``CallableZeroCouponBond`` (callablebond.cpp:512-530).
    """

    def __init__(
        self,
        settlement_days: int,
        face_amount: float,
        calendar: Calendar,
        maturity_date: Date,
        day_counter: DayCounter,
        payment_convention: BusinessDayConvention | None = None,
        redemption: float = 100.0,
        issue_date: Date | None = None,
        put_call_schedule: Sequence[Callability] | None = None,
    ) -> None:
        pay_conv = (
            payment_convention
            if payment_convention is not None
            else BusinessDayConvention.Following
        )
        CallableBond.__init__(
            self,
            settlement_days,
            maturity_date,
            calendar,
            day_counter,
            face_amount,
            issue_date,
            put_call_schedule,
        )
        self._frequency = Frequency.Once
        redemption_date = calendar.adjust(self._maturity_date, pay_conv)
        self._set_single_redemption(face_amount, redemption, redemption_date)
        for cf in self._cashflows:
            cf.register_with(self)


__all__ = [
    "CallableBond",
    "CallableBondArguments",
    "CallableBondResults",
    "CallableFixedRateBond",
    "CallableZeroCouponBond",
]
