"""Catastrophe bond instrument.

# C++ parity: ql/experimental/catbonds/catbond.{hpp,cpp} (v1.42.1).

A ``CatBond`` is a ``Bond`` whose notional is eroded by catastrophe losses
through an attached ``NotionalRisk``.  The MonteCarloCatBondEngine fills
the loss / exhaustion probabilities and the expected loss in addition to
the NPV.

``FloatingCatBond`` builds an IBOR-indexed floating leg (the only concrete
cat bond in the C++ source).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.cashflows.ibor_leg import ibor_leg
from pquantlib.instruments.bond import Bond, BondArguments, BondResults
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule

if TYPE_CHECKING:
    from pquantlib.daycounters.day_counter import DayCounter
    from pquantlib.instruments.risky_notional import NotionalRisk
    from pquantlib.pricingengines.pricing_engine import (
        PricingEngineArguments,
        PricingEngineResults,
    )
    from pquantlib.termstructures.protocols import IborIndexProtocol
    from pquantlib.time.calendar import Calendar

_NULL_DATE: Date = Date()


class CatBondArguments(BondArguments):
    """Engine argument carrier for cat bonds.

    # C++ parity: ``CatBond::arguments`` (catbond.hpp:66-71).
    """

    def __init__(self) -> None:
        super().__init__()
        self.start_date: Date = _NULL_DATE
        self.notional_risk: NotionalRisk | None = None

    def validate(self) -> None:
        # C++ parity: catbond.cpp:33-36.
        super().validate()
        qassert.require(self.notional_risk is not None, "null notionalRisk")


class CatBondResults(BondResults):
    """Results carrier for cat bonds.

    # C++ parity: ``CatBond::results`` (catbond.hpp:74-79).
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss_probability: float = 0.0
        self.exhaustion_probability: float = 0.0
        self.expected_loss: float = 0.0

    def reset(self) -> None:
        super().reset()
        self.loss_probability = 0.0
        self.exhaustion_probability = 0.0
        self.expected_loss = 0.0


class CatBond(Bond):
    """Catastrophe bond — a Bond with notional erosion under cat events.

    # C++ parity: ``class CatBond : public Bond`` (catbond.hpp:37-64).
    """

    def __init__(
        self,
        settlement_days: int,
        calendar: Calendar,
        issue_date: Date | None,
        notional_risk: NotionalRisk,
    ) -> None:
        Bond.__init__(self, settlement_days, calendar, issue_date)
        self._notional_risk: NotionalRisk = notional_risk
        self._loss_probability: float = 0.0
        self._exhaustion_probability: float = 0.0
        self._expected_loss: float = 0.0

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        # C++ parity: catbond.cpp:38-47.
        qassert.require(isinstance(args, CatBondArguments), "wrong arguments type")
        assert isinstance(args, CatBondArguments)
        Bond.setup_arguments(self, args)
        args.notional_risk = self._notional_risk
        args.start_date = self.issue_date()

    def fetch_results(self, results: PricingEngineResults) -> None:
        # C++ parity: catbond.cpp:49-58.
        Bond.fetch_results(self, results)
        qassert.require(isinstance(results, CatBondResults), "wrong result type")
        assert isinstance(results, CatBondResults)
        self._loss_probability = results.loss_probability
        self._expected_loss = results.expected_loss
        self._exhaustion_probability = results.exhaustion_probability

    def loss_probability(self) -> float:
        return self._loss_probability

    def expected_loss(self) -> float:
        return self._expected_loss

    def exhaustion_probability(self) -> float:
        return self._exhaustion_probability


class FloatingCatBond(CatBond):
    """Floating-rate (IBOR-indexed) cat bond.

    # C++ parity: ``class FloatingCatBond`` (catbond.{hpp,cpp}).

    The primary constructor takes a pre-built ``Schedule``; use
    :meth:`from_dates` for the C++ second-constructor convenience form
    (startDate / maturityDate / couponFrequency).
    """

    def __init__(
        self,
        settlement_days: int,
        face_amount: float,
        schedule: Schedule,
        ibor_index: IborIndexProtocol,
        accrual_day_counter: DayCounter,
        notional_risk: NotionalRisk,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        fixing_days: int | None = None,
        gearings: float | Sequence[float] = 1.0,
        spreads: float | Sequence[float] = 0.0,
        caps: Sequence[float] | None = None,
        floors: Sequence[float] | None = None,
        in_arrears: bool = False,
        redemption: float = 100.0,
        issue_date: Date | None = None,
    ) -> None:
        del caps, floors  # accepted for parity; ignored by the ibor_leg builder.
        CatBond.__init__(self, settlement_days, schedule.calendar, issue_date, notional_risk)
        self._maturity_date = schedule.end_date

        self._cashflows = list(
            ibor_leg(
                schedule,
                ibor_index,
                nominals=[face_amount],
                payment_day_counter=accrual_day_counter,
                payment_adjustment=payment_convention,
                fixing_days=fixing_days,
                gearings=gearings,
                spreads=spreads,
                in_arrears=in_arrears,
            )
        )
        # The C++ ``IborLeg`` attaches a default ``BlackIborCouponPricer``
        # internally; PQuantLib's ``ibor_leg`` builder leaves the pricer to
        # the caller, so we attach the standard IBOR pricer here to mirror
        # the C++ FloatingCatBond's ready-to-price coupons.
        from pquantlib.cashflows.coupon_pricer import (  # noqa: PLC0415
            IborCouponPricer,
            set_coupon_pricer,
        )

        set_coupon_pricer(self._cashflows, IborCouponPricer())
        self._add_redemptions_to_cashflows([redemption])

        qassert.require(len(self._cashflows) > 0, "bond with no cashflows!")
        qassert.require(len(self._redemptions) == 1, "multiple redemptions created")

        reg = getattr(ibor_index, "register_with", None)
        if callable(reg):
            reg(self)
        for cf in self._cashflows:
            cf.register_with(self)

    @classmethod
    def from_dates(
        cls,
        settlement_days: int,
        face_amount: float,
        start_date: Date,
        maturity_date: Date,
        coupon_frequency: Frequency,
        calendar: Calendar,
        ibor_index: IborIndexProtocol,
        accrual_day_counter: DayCounter,
        notional_risk: NotionalRisk,
        accrual_convention: BusinessDayConvention = BusinessDayConvention.Following,
        payment_convention: BusinessDayConvention = BusinessDayConvention.Following,
        fixing_days: int | None = None,
        gearings: float | Sequence[float] = 1.0,
        spreads: float | Sequence[float] = 0.0,
        in_arrears: bool = False,
        redemption: float = 100.0,
        issue_date: Date | None = None,
        stub_date: Date | None = None,
        rule: DateGeneration = DateGeneration.Backward,
        end_of_month: bool = False,
    ) -> FloatingCatBond:
        """C++ second-constructor form — builds the schedule internally.

        # C++ parity: catbond.cpp:98-166.
        """
        stub = stub_date if stub_date is not None else _NULL_DATE
        if rule == DateGeneration.Backward:
            first_date, next_to_last = _NULL_DATE, stub
        elif rule == DateGeneration.Forward:
            first_date, next_to_last = stub, _NULL_DATE
        else:
            qassert.fail(f"stub date not allowed with {rule} DateGeneration.Rule")

        schedule = Schedule.from_rule(
            start_date,
            maturity_date,
            Period.from_frequency(coupon_frequency),
            calendar,
            accrual_convention,
            accrual_convention,
            rule,
            end_of_month,
            first_date,
            next_to_last,
        )
        return cls(
            settlement_days,
            face_amount,
            schedule,
            ibor_index,
            accrual_day_counter,
            notional_risk,
            payment_convention,
            fixing_days,
            gearings,
            spreads,
            None,
            None,
            in_arrears,
            redemption,
            issue_date,
        )


__all__ = [
    "CatBond",
    "CatBondArguments",
    "CatBondResults",
    "FloatingCatBond",
]
