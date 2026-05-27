"""Forward — abstract base for forward contracts + ForwardTypePayoff.

# C++ parity: ql/instruments/forward.{hpp,cpp} (v1.42.1).

The C++ design:

* ``Forward : public Instrument`` — abstract base that ties together a
  ``Payoff`` + a discount curve + a value/maturity date pair. Concrete
  subclasses must:
  - implement ``spot_value()`` and ``spot_income(income_discount_curve)``,
  - set the protected ``underlying_income`` and ``underlying_spot_value``
    members in ``perform_calculations`` before calling the base.
* ``Forward.perform_calculations`` then computes:
    NPV = payoff(forward_value()) * discount_curve.discount(maturityDate)
  where ``forward_value() = (underlying_spot_value - underlying_income)
                         / discount_curve.discount(maturityDate)``.

PQuantLib divergences:

* ``Settings.evaluationDate()`` is not available as a global; we plumb
  an ``evaluation_date`` through ``settlement_date()`` instead.
* The C++ ``Forward::isExpired()`` checks ``settlement_date()`` against
  the maturity date; the Python port does the same once an evaluation
  date is supplied.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.instrument import Instrument
from pquantlib.interest_rate import InterestRate
from pquantlib.payoffs import Payoff
from pquantlib.position import PositionType
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.time_unit import TimeUnit


class ForwardTypePayoff(Payoff):
    """Linear forward payoff: ``price - strike`` (Long) / ``strike - price`` (Short).

    # C++ parity: ql/instruments/forward.hpp ``class ForwardTypePayoff``.
    """

    def __init__(self, position_type: PositionType, strike: float) -> None:
        qassert.require(strike >= 0.0, "negative strike given")
        self._position_type: PositionType = position_type
        self._strike: float = strike

    def forward_type(self) -> PositionType:
        return self._position_type

    def strike(self) -> float:
        return self._strike

    def name(self) -> str:
        return "Forward"

    def description(self) -> str:
        return f"{self.name()}, {self._strike} strike"

    def __call__(self, price: float) -> float:
        if self._position_type == PositionType.Long:
            return price - self._strike
        return self._strike - price


class Forward(Instrument):
    """Abstract base forward class.

    # C++ parity: ``class Forward : public Instrument`` (forward.hpp).

    Concrete subclasses must:
    - override :meth:`spot_value` and :meth:`spot_income`,
    - set :attr:`_underlying_income` and :attr:`_underlying_spot_value`
      (typically inside :meth:`_perform_calculations` before chaining to
      the base implementation).
    """

    def __init__(
        self,
        day_counter: DayCounter,
        calendar: Calendar,
        business_day_convention: BusinessDayConvention,
        settlement_days: int,
        payoff: Payoff,
        value_date: Date,
        maturity_date: Date,
        discount_curve: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__()
        self._day_counter: DayCounter = day_counter
        self._calendar: Calendar = calendar
        self._business_day_convention: BusinessDayConvention = business_day_convention
        self._settlement_days: int = settlement_days
        self._payoff: Payoff = payoff
        self._value_date: Date = value_date
        # C++ adjusts maturity by the business-day convention in the ctor.
        self._maturity_date: Date = calendar.adjust(
            maturity_date, business_day_convention
        )
        self._discount_curve: YieldTermStructureProtocol | None = discount_curve
        # Filled by subclass perform_calculations before base call.
        self._underlying_income: float = 0.0
        self._underlying_spot_value: float = 0.0

    # --- inspectors --------------------------------------------------------

    def calendar(self) -> Calendar:
        return self._calendar

    def business_day_convention(self) -> BusinessDayConvention:
        return self._business_day_convention

    def day_counter(self) -> DayCounter:
        return self._day_counter

    def discount_curve(self) -> YieldTermStructureProtocol | None:
        return self._discount_curve

    def value_date(self) -> Date:
        return self._value_date

    def maturity_date(self) -> Date:
        return self._maturity_date

    def settlement_date(self, evaluation_date: Date | None = None) -> Date:
        """Settlement date.

        # C++ parity: ``Forward::settlementDate`` advances evaluation
        # date by ``settlementDays`` and takes the max with ``valueDate``.

        Python divergence: C++ pulls evaluation date from
        ``Settings::instance()``; we accept it as an argument because
        there is no Settings singleton yet.  When omitted, defaults to
        the discount-curve reference date (mirrors how callers used
        Settings::evaluationDate before the registration step).
        """
        if evaluation_date is None:
            qassert.require(
                self._discount_curve is not None,
                "evaluation_date must be supplied if no discount curve set",
            )
            assert self._discount_curve is not None
            evaluation_date = self._discount_curve.reference_date()
        d = self._calendar.advance(
            evaluation_date, self._settlement_days, TimeUnit.Days
        )
        return max(d, self._value_date)

    def is_expired(self) -> bool:
        """C++ parity: ``Forward::isExpired``.

        Uses :meth:`settlement_date` (no Settings global) — defaults to
        the discount curve reference date when present.
        """
        if self._discount_curve is None:
            return False
        return self._maturity_date < self.settlement_date()

    # --- subclass hooks ----------------------------------------------------

    @abstractmethod
    def spot_value(self) -> float:
        """Spot value/price of the underlying.

        # C++ parity: ``Forward::spotValue() = 0``.
        """

    @abstractmethod
    def spot_income(
        self, income_discount_curve: YieldTermStructureProtocol | None
    ) -> float:
        """PV of income/dividends/storage of the underlying.

        # C++ parity: ``Forward::spotIncome(...) = 0``.
        """

    # --- calculations ------------------------------------------------------

    def forward_value(self) -> float:
        """Forward price/value of the underlying.

        # C++ parity: ``Forward::forwardValue``.
        """
        self.calculate()
        qassert.require(
            self._discount_curve is not None, "null term structure set to Forward"
        )
        assert self._discount_curve is not None
        return (
            self._underlying_spot_value - self._underlying_income
        ) / self._discount_curve.discount(self._maturity_date)

    def implied_yield(
        self,
        underlying_spot_value: float,
        forward_value: float,
        settlement_date: Date,
        compounding: Compounding,
        day_counter: DayCounter,
    ) -> InterestRate:
        """Simple-yield calculation from spot + forward values.

        # C++ parity: ``Forward::impliedYield``.
        """
        t = day_counter.year_fraction(settlement_date, self._maturity_date)
        compound_factor = forward_value / (
            underlying_spot_value - self.spot_income(self._discount_curve)
        )
        return InterestRate.implied_rate(
            compound_factor, day_counter, compounding, Frequency.Annual, t
        )

    def _perform_calculations(self) -> None:
        """C++ parity: ``Forward::performCalculations``.

        Concrete subclasses MUST set ``_underlying_spot_value`` and
        ``_underlying_income`` BEFORE chaining to this implementation
        via ``super()._perform_calculations()``.
        """
        qassert.require(
            self._discount_curve is not None, "null term structure set to Forward"
        )
        assert self._discount_curve is not None
        # C++ dynamic_pointer_cast<ForwardTypePayoff>(payoff_)
        qassert.require(
            isinstance(self._payoff, ForwardTypePayoff),
            "Forward expects a ForwardTypePayoff",
        )
        assert isinstance(self._payoff, ForwardTypePayoff)
        fwd_value = self.forward_value()
        self._npv = self._payoff(fwd_value) * self._discount_curve.discount(
            self._maturity_date
        )


__all__ = ["Forward", "ForwardTypePayoff"]
