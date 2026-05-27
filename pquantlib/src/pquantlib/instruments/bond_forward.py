"""BondForward — forward contract on a Bond.

# C++ parity: ql/instruments/bondforward.{hpp,cpp} (v1.42.1).

The C++ inheritance ``BondForward : Forward : Instrument`` is collapsed
to a single ``BondForward : Instrument`` in this port because:

- the L3-B scope explicitly defers the ``Forward`` abstract base
  (and its associated ``Position`` enum / ``ForwardTypePayoff`` /
  engine plumbing) to L3-E or a follow-up cluster;
- BondForward only ever uses Forward as a data-carrier — the
  ``performCalculations`` body is local to BondForward + Forward's
  ``forwardValue`` formula, both of which we can inline here without
  losing parity.

A minimal ``BondForwardPosition`` enum (Long / Short) is provided
inline rather than introducing a separate ``position`` module. When
``Forward`` lands later, this enum can be replaced by the shared
``Position.Type`` enum (Python-level structural-rename, no semantic
change).

C++ formulas (bondforward.cpp + forward.cpp):

- ``spotValue = bond.dirtyPrice()``.
- ``spotIncome = sum(coupon.amount * incomeDiscount(coupon.date))``
  for each coupon between settlement and maturity (exclusive).
- ``forwardValue = (spotValue - spotIncome) / discountCurve.discount(maturityDate)``.
- ``forwardPrice == forwardValue``.
- ``cleanForwardPrice = forwardValue - bond.accruedAmount(maturityDate)``.
- ``NPV = (forwardValue - strike) [Long] or (strike - forwardValue)
  [Short]) * discountCurve.discount(maturityDate)``.
"""

from __future__ import annotations

from enum import IntEnum

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.instruments.bond import Bond
from pquantlib.instruments.instrument import Instrument
from pquantlib.patterns.observable_settings import ObservableSettings
from pquantlib.pricingengines.pricing_engine import (
    PricingEngineArguments,
    PricingEngineResults,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.time_unit import TimeUnit

_NULL_DATE: Date = Date()


class BondForwardPosition(IntEnum):
    """C++ parity: ``Position::Type`` enum.

    Forward / Position will be ported as a sibling cluster; for L3-B
    we inline the minimum-needed enum here. See module docstring.
    """

    Long = 0
    Short = 1


class BondForward(Instrument):
    """Forward contract on a bond.

    # C++ parity: ``class BondForward : public Forward``. The port
    # collapses Forward into BondForward (see module docstring).
    """

    def __init__(
        self,
        value_date: Date,
        maturity_date: Date,
        position_type: BondForwardPosition,
        strike: float,
        settlement_days: int,
        day_counter: DayCounter,
        calendar: Calendar,
        business_day_convention: BusinessDayConvention,
        bond: Bond,
        discount_curve: YieldTermStructure,
        income_discount_curve: YieldTermStructure | None = None,
    ) -> None:
        super().__init__()
        qassert.require(strike >= 0.0, "negative strike given")
        self._value_date: Date = value_date
        self._maturity_date: Date = calendar.adjust(maturity_date, business_day_convention)
        self._position_type: BondForwardPosition = position_type
        self._strike: float = strike
        self._settlement_days: int = settlement_days
        self._day_counter: DayCounter = day_counter
        self._calendar: Calendar = calendar
        self._business_day_convention: BusinessDayConvention = business_day_convention
        self._bond: Bond = bond
        self._discount_curve: YieldTermStructure = discount_curve
        self._income_discount_curve: YieldTermStructure = (
            income_discount_curve if income_discount_curve is not None else discount_curve
        )

        ObservableSettings().register_with(self)
        # Both curves and the bond notify us when they update.
        discount_curve.register_with(self)
        self._income_discount_curve.register_with(self)
        bond.register_with(self)

    # ----- inspectors ---------------------------------------------------

    def value_date(self) -> Date:
        return self._value_date

    def maturity_date(self) -> Date:
        return self._maturity_date

    def position_type(self) -> BondForwardPosition:
        return self._position_type

    def strike(self) -> float:
        return self._strike

    def calendar(self) -> Calendar:
        return self._calendar

    def business_day_convention(self) -> BusinessDayConvention:
        return self._business_day_convention

    def day_counter(self) -> DayCounter:
        return self._day_counter

    def discount_curve(self) -> YieldTermStructure:
        return self._discount_curve

    def income_discount_curve(self) -> YieldTermStructure:
        return self._income_discount_curve

    def bond(self) -> Bond:
        return self._bond

    def settlement_date(self, d: Date | None = None) -> Date:
        """Forward-contract settlement date.

        # C++ parity: forward.cpp:48-52.
        """
        d_eff = (
            d if d is not None and d != _NULL_DATE
            else ObservableSettings().evaluation_date_or_today()
        )
        settle = self._calendar.advance(d_eff, self._settlement_days, TimeUnit.Days)
        return max(settle, self._value_date)

    def is_expired(self) -> bool:
        """Maturity date has occurred at the forward's settlement date.

        # C++ parity: forward.cpp:55-58.
        """
        settle = self.settlement_date()
        return self._maturity_date < settle

    # ----- core calculations ---------------------------------------------

    def spot_value(self) -> float:
        """Spot value of the underlying bond (dirty price).

        # C++ parity: bondforward.cpp:86-88.
        """
        return self._bond.dirty_price()

    def spot_income(
        self, income_discount_curve: YieldTermStructure | None = None,
    ) -> float:
        """Income (discounted coupon flows) of the bond between settlement
        and contract maturity.

        # C++ parity: bondforward.cpp:59-83.
        """
        curve = (
            income_discount_curve
            if income_discount_curve is not None
            else self._income_discount_curve
        )
        income = 0.0
        settle = self.settlement_date()
        for cf in self._bond.cashflows():
            if not cf.has_occurred(settle, False):
                if cf.has_occurred(self._maturity_date, False):
                    income += cf.amount() * curve.discount(cf.date())
                else:
                    # cashflow is beyond contract maturity — stop early
                    # (mirrors C++ break, relying on date-sorted cashflows).
                    break
        return income

    def forward_value(self) -> float:
        """Forward value of the underlying.

        # C++ parity: forward.cpp:61-65.
        """
        spot = self.spot_value()
        income = self.spot_income(self._income_discount_curve)
        df = self._discount_curve.discount(self._maturity_date)
        return (spot - income) / df

    def forward_price(self) -> float:
        """Dirty forward price (alias of ``forward_value``).

        # C++ parity: bondforward.cpp:54-56.
        """
        return self.forward_value()

    def clean_forward_price(self) -> float:
        """Clean forward price = forward value - accrued at delivery.

        # C++ parity: bondforward.cpp:49-51.
        """
        return self.forward_value() - self._bond.accrued_amount(self._maturity_date)

    # ----- Instrument plumbing ------------------------------------------

    def setup_arguments(self, args: PricingEngineArguments) -> None:
        """No-op — BondForward has no pricing engine yet.

        # C++ parity divergence: the C++ ``Forward`` populates a generic
        # arguments record. We defer the engine + the arguments carrier
        # to a later cluster; for now, ``NPV()`` is computed directly via
        # :meth:`_perform_calculations` overriding the default lazy
        # lifecycle.
        """

    def fetch_results(self, results: PricingEngineResults) -> None:
        """No-op — no engine returns results yet."""

    def _perform_calculations(self) -> None:
        """Compute NPV inline from the discount curve.

        # C++ parity: forward.cpp:83-92 specialised for the Bond payoff
        # (price - strike for Long, strike - price for Short).
        """
        fwd = self.forward_value()
        payoff = (
            fwd - self._strike
            if self._position_type == BondForwardPosition.Long
            else self._strike - fwd
        )
        df = self._discount_curve.discount(self._maturity_date)
        self._npv = payoff * df
        self._error_estimate = 0.0
        self._valuation_date = self._discount_curve.reference_date()


__all__ = ["BondForward", "BondForwardPosition"]
