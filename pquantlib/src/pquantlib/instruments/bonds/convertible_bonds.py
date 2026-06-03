"""Convertible bonds — base + fixed-coupon / zero-coupon variants.

# C++ parity: ql/instruments/bonds/convertiblebonds.{hpp,cpp} (v1.42.1).

A :class:`ConvertibleBond` is a :class:`~pquantlib.instruments.bond.Bond`
carrying an embedded conversion right (``conversion_ratio``) plus an optional
call/put schedule (:class:`CallabilitySchedule`). It is priced on a
Tsiveriotis-Fernandes credit-adjusted binomial lattice by the
:class:`~pquantlib.pricingengines.bond.binomial_convertible_engine.BinomialConvertibleEngine`.

Per the C++ warning, all the yield-based ``Bond`` methods (clean/dirty price,
yield) refer to the *underlying plain-vanilla bond* and ignore convertibility
+ callability — only the engine NPV accounts for the optionality.

Variants:

* :class:`ConvertibleZeroCouponBond` — single redemption, notional forced to
  100.
* :class:`ConvertibleFixedCouponBond` — fixed-rate coupon leg, notional forced
  to 100.

# C++ parity divergence — notional:
# the C++ ctors force the notional to 100 (the conversion ratio + redemption
# are quoted per-100), independent of any face amount. The Python port keeps
# the same forced-100 behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.fixed_rate_leg import fixed_rate_leg
from pquantlib.cashflows.simple_cash_flow import Redemption
from pquantlib.instruments.bond import (
    Bond,
    BondArguments,
    BondPriceType,
    BondResults,
)
from pquantlib.instruments.callability import CallabilityType
from pquantlib.instruments.soft_callability import SoftCallability
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency

if TYPE_CHECKING:
    from pquantlib.cashflows.cash_flow import CashFlow
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.exercise import Exercise
    from pquantlib.instruments.callability import Callability
    from pquantlib.pricingengines.pricing_engine import (
        PricingEngineArguments,
        PricingEngineResults,
    )
    from pquantlib.time.schedule import Schedule

_NULL_DATE: Date = Date()


class ConvertibleBondArguments(BondArguments):
    """Engine-arguments carrier for a convertible bond.

    # C++ parity: ``ConvertibleBond::arguments`` (convertiblebonds.hpp:148-166).
    """

    def __init__(self) -> None:
        super().__init__()
        self.exercise: Exercise | None = None
        self.conversion_ratio: float | None = None
        self.callability_dates: list[Date] = []
        self.callability_types: list[CallabilityType] = []
        self.callability_prices: list[float] = []
        self.callability_triggers: list[float | None] = []
        self.issue_date: Date = _NULL_DATE
        self.settlement_days: int | None = None
        self.redemption: float | None = None

    def validate(self) -> None:
        # C++ parity: convertiblebonds.cpp:197-220.
        qassert.require(self.exercise is not None, "no exercise given")
        qassert.require(self.conversion_ratio is not None, "null conversion ratio")
        assert self.conversion_ratio is not None
        qassert.require(
            self.conversion_ratio > 0.0,
            f"positive conversion ratio required: {self.conversion_ratio} not allowed",
        )
        qassert.require(self.redemption is not None, "null redemption")
        assert self.redemption is not None
        qassert.require(
            self.redemption >= 0.0,
            f"positive redemption required: {self.redemption} not allowed",
        )
        qassert.require(self.settlement_date != _NULL_DATE, "null settlement date")
        qassert.require(self.settlement_days is not None, "null settlement days")
        qassert.require(
            len(self.callability_dates) == len(self.callability_types),
            "different number of callability dates and types",
        )
        qassert.require(
            len(self.callability_dates) == len(self.callability_prices),
            "different number of callability dates and prices",
        )
        qassert.require(
            len(self.callability_dates) == len(self.callability_triggers),
            "different number of callability dates and triggers",
        )
        qassert.require(len(self.cashflows) > 0, "no cashflows given")


class ConvertibleBondResults(BondResults):
    """Engine-results carrier for a convertible bond.

    # C++ parity: ``ConvertibleBond::results`` (== ``Bond::results``).
    """


class ConvertibleBond(Bond):
    """Base class for convertible bonds.

    # C++ parity: ``class ConvertibleBond : public Bond``
    # (convertiblebonds.{hpp,cpp}).
    """

    def __init__(
        self,
        exercise: Exercise,
        conversion_ratio: float,
        callability: Sequence[Callability],
        issue_date: Date | None,
        settlement_days: int,
        schedule: Schedule,
        redemption: float,
    ) -> None:
        # C++ parity: convertiblebonds.cpp:32-50.
        super().__init__(settlement_days, schedule.calendar, issue_date)
        self._exercise: Exercise = exercise
        self._conversion_ratio: float = conversion_ratio
        self._callability: list[Callability] = list(callability)
        self._redemption_amount: float = redemption

        self._maturity_date = schedule.end_date

        if self._callability:
            qassert.require(
                self._callability[-1].date() <= self._maturity_date,
                f"last callability date ({self._callability[-1].date()}) later "
                f"than maturity ({self._maturity_date})",
            )

    # -- inspectors -------------------------------------------------------

    def conversion_ratio(self) -> float:
        """C++ parity: ``ConvertibleBond::conversionRatio`` (hpp:58)."""
        return self._conversion_ratio

    def callability(self) -> list[Callability]:
        """C++ parity: ``ConvertibleBond::callability`` (hpp:59)."""
        return list(self._callability)

    # -- engine plumbing --------------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        # C++ parity: convertiblebonds.cpp:153-194.
        qassert.require(
            isinstance(args, ConvertibleBondArguments), "wrong argument type"
        )
        assert isinstance(args, ConvertibleBondArguments)

        args.exercise = self._exercise
        args.conversion_ratio = self._conversion_ratio

        settlement = self.settlement_date()

        args.callability_dates = []
        args.callability_types = []
        args.callability_prices = []
        args.callability_triggers = []
        for c in self._callability:
            if not c.has_occurred(settlement, False):
                args.callability_types.append(c.type())
                args.callability_dates.append(c.date())
                price = c.price().amount()
                if c.price().type() == BondPriceType.Clean:
                    price += self.accrued_amount(c.date())
                args.callability_prices.append(price)
                if isinstance(c, SoftCallability):
                    args.callability_triggers.append(c.trigger())
                else:
                    args.callability_triggers.append(None)

        args.cashflows = self.cashflows()
        args.issue_date = self._issue_date
        args.settlement_date = settlement
        args.settlement_days = self._settlement_days
        args.redemption = self._redemption_amount

    def fetch_results(self, results: PricingEngineResults) -> None:
        # Convertible results are Bond results (NPV + settlement value).
        super().fetch_results(results)


class ConvertibleZeroCouponBond(ConvertibleBond):
    """Convertible zero-coupon bond.

    # C++ parity: ``ConvertibleZeroCouponBond`` (convertiblebonds.cpp:53-73).
    """

    def __init__(
        self,
        exercise: Exercise,
        conversion_ratio: float,
        callability: Sequence[Callability],
        issue_date: Date | None,
        settlement_days: int,
        day_counter: DayCounter,
        schedule: Schedule,
        redemption: float = 100.0,
    ) -> None:
        super().__init__(
            exercise,
            conversion_ratio,
            callability,
            issue_date,
            settlement_days,
            schedule,
            redemption,
        )
        # !!! notional forcibly set to 100 (C++ convertiblebonds.cpp:69-72).
        self._cashflows = []
        self._set_single_redemption(100.0, redemption, self._maturity_date)
        for cf in self._cashflows:
            cf.register_with(self)


class ConvertibleFixedCouponBond(ConvertibleBond):
    """Convertible fixed-coupon bond.

    # C++ parity: ``ConvertibleFixedCouponBond`` (convertiblebonds.cpp:76-109).
    """

    def __init__(
        self,
        exercise: Exercise,
        conversion_ratio: float,
        callability: Sequence[Callability],
        issue_date: Date | None,
        settlement_days: int,
        coupons: Sequence[float],
        day_counter: DayCounter,
        schedule: Schedule,
        redemption: float = 100.0,
    ) -> None:
        super().__init__(
            exercise,
            conversion_ratio,
            callability,
            issue_date,
            settlement_days,
            schedule,
            redemption,
        )

        # !!! notional forcibly set to 100. C++ uses
        # FixedRateLeg(schedule).withNotionals(100).withCouponRates(coupons,
        # dayCounter).withPaymentAdjustment(schedule.businessDayConvention()).
        # The fixed_rate_leg InterestRate convention defaults to Simple /
        # Annual, matching FixedRateLeg::withCouponRates(coupons, dc).
        self._cashflows = list(
            fixed_rate_leg(
                schedule,
                nominals=[100.0],
                rates=list(coupons),
                day_counter=day_counter,
                compounding=Compounding.Simple,
                frequency=Frequency.Annual,
                payment_adjustment=schedule.business_day_convention,
                payment_calendar=schedule.calendar,
            )
        )

        self._add_convertible_redemption(redemption)

    def _add_convertible_redemption(self, redemption: float) -> None:
        """Append a single redemption flow on the (forced-100) notional.

        # C++ parity: ``addRedemptionsToCashflows({redemption})`` on a leg
        # built with notional 100. With a single notional the redemption
        # amount is ``redemption/100 * 100 = redemption`` paid at maturity.
        """
        red_cf: CashFlow = Redemption(redemption, self._maturity_date)
        self._notionals = [100.0, 0.0]
        self._notional_schedule = [_NULL_DATE, self._maturity_date]
        self._redemptions = [red_cf]
        self._cashflows.append(red_cf)
        self._cashflows.sort(key=lambda cf: cf.date())
        qassert.require(len(self._redemptions) == 1, "multiple redemptions created")
        for cf in self._cashflows:
            cf.register_with(self)


__all__ = [
    "ConvertibleBond",
    "ConvertibleBondArguments",
    "ConvertibleBondResults",
    "ConvertibleFixedCouponBond",
    "ConvertibleZeroCouponBond",
]
