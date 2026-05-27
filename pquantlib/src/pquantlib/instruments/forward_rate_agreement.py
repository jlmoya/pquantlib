"""ForwardRateAgreement (FRA) — index-driven forward rate contract.

# C++ parity: ql/instruments/forwardrateagreement.{hpp,cpp} (v1.42.1).

A FRA settles on the value date (the underlying loan/deposit start
date), with payoff:

    amount = sign * notional * (F - K) * tau / (1 + F * tau)

where:
- ``F`` is the index-implied forward rate over [valueDate, maturityDate]
- ``K`` is the contract (strike) rate
- ``tau`` is the year-fraction across [valueDate, maturityDate] on the
  index day counter
- ``sign`` is +1 for Long (FRA buyer / future borrower), -1 for Short.

NPV = ``amount * discount(valueDate)`` on the discount curve (or the
index's forwarding curve if no separate discount curve was provided).

Two constructors:
- (index, valueDate, type, strike, notional [, discountCurve])
  → useIndexedCoupon=TRUE; the maturity is index.maturityDate(valueDate)
    and ``forwardRate`` is computed via ``index.fixing(fixingDate)``.
- (index, valueDate, maturityDate, type, strike, notional
    [, discountCurve])
  → useIndexedCoupon=FALSE; the par-coupon approximation forward rate
    is ``(df(valueDate) / df(maturityDate) - 1) / yearFraction``.

PQuantLib follows the C++ closely but does NOT register the FRA as a
Settings observer (no Settings singleton).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.instruments.instrument import Instrument
from pquantlib.interest_rate import InterestRate
from pquantlib.position import PositionType
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency


class ForwardRateAgreement(Instrument):
    """Forward Rate Agreement on an Ibor index.

    # C++ parity: ``class ForwardRateAgreement : public Instrument``
    # (ql/instruments/forwardrateagreement.hpp).
    """

    def __init__(
        self,
        index: IborIndex,
        value_date: Date,
        position_type: PositionType,
        strike_forward_rate: float,
        notional_amount: float,
        maturity_date: Date | None = None,
        discount_curve: YieldTermStructureProtocol | None = None,
    ) -> None:
        """Construct a FRA.

        # C++ parity: both FRA constructors collapsed into one Python
        # signature.  ``maturity_date=None`` triggers
        # ``useIndexedCoupon=True`` (maturity = index.maturityDate(value)).
        # An explicit ``maturity_date`` triggers the par-coupon branch.
        """
        super().__init__()
        qassert.require(notional_amount > 0.0, "notionalAmount must be positive")

        use_indexed_coupon = maturity_date is None
        if maturity_date is None:
            maturity_date = index.maturity_date(value_date)

        self._fra_type: PositionType = position_type
        self._notional_amount: float = notional_amount
        self._index: IborIndex = index
        self._use_indexed_coupon: bool = use_indexed_coupon
        self._day_counter: DayCounter = index.day_counter()
        self._calendar: Calendar = index.fixing_calendar()
        self._business_day_convention: BusinessDayConvention = (
            index.business_day_convention()
        )
        self._value_date: Date = value_date
        # C++ adjusts maturity by index BDC.
        self._maturity_date: Date = self._calendar.adjust(
            maturity_date, self._business_day_convention
        )
        self._discount_curve: YieldTermStructureProtocol | None = discount_curve

        qassert.require(
            self._value_date < self._maturity_date,
            "valueDate must be earlier than maturityDate",
        )

        # Strike as InterestRate (Simple/Once compounding mirroring C++).
        self._strike_forward_rate: InterestRate = InterestRate(
            strike_forward_rate, index.day_counter(), Compounding.Simple, Frequency.Once
        )

        # Internal lazy state.
        self._forward_rate: InterestRate | None = None
        self._amount: float | None = None

    # --- inspectors --------------------------------------------------------

    def calendar(self) -> Calendar:
        return self._calendar

    def business_day_convention(self) -> BusinessDayConvention:
        return self._business_day_convention

    def day_counter(self) -> DayCounter:
        return self._day_counter

    def discount_curve(self) -> YieldTermStructureProtocol | None:
        return self._discount_curve

    def fixing_date(self) -> Date:
        return self._index.fixing_date(self._value_date)

    def value_date(self) -> Date:
        return self._value_date

    def maturity_date(self) -> Date:
        return self._maturity_date

    # --- Instrument interface ---------------------------------------------

    def is_expired(self) -> bool:
        """A FRA settles on the value date.

        # C++ parity: ``ForwardRateAgreement::isExpired`` uses
        # ``detail::simple_event(valueDate_).hasOccurred()`` which is
        # in turn driven by Settings::evaluationDate.  PQuantLib has no
        # such global, so we always return False (mirrors the
        # "before-settlement" branch).  Callers that need expiry
        # checking can compare value_date() to their own eval date.
        """
        return False

    def amount(self) -> float:
        """The payoff on the value date.

        # C++ parity: ``ForwardRateAgreement::amount``.
        """
        self.calculate()
        qassert.require(self._amount is not None, "amount not computed")
        assert self._amount is not None
        return self._amount

    def forward_rate(self) -> InterestRate:
        """Relevant forward rate associated with the FRA term.

        # C++ parity: ``ForwardRateAgreement::forwardRate``.
        """
        self.calculate()
        qassert.require(self._forward_rate is not None, "forward rate not computed")
        assert self._forward_rate is not None
        return self._forward_rate

    # --- calculations -----------------------------------------------------

    def setup_expired(self) -> None:
        super().setup_expired()
        self._calculate_forward_rate()

    def _perform_calculations(self) -> None:
        """C++ parity: ``ForwardRateAgreement::performCalculations``."""
        self._calculate_amount()
        # Discount curve: explicit one wins, else the index's forwarding
        # curve.
        if self._discount_curve is not None:
            discount = self._discount_curve
        else:
            discount = self._index.forecast_term_structure()
            qassert.require(
                discount is not None,
                "no discount curve set on FRA and index has no forwarding curve",
            )
            assert discount is not None
        assert self._amount is not None
        self._npv = self._amount * discount.discount(self._value_date)

    def _calculate_forward_rate(self) -> None:
        """C++ parity: ``ForwardRateAgreement::calculateForwardRate``."""
        if self._use_indexed_coupon:
            self._forward_rate = InterestRate(
                self._index.fixing(self.fixing_date(), forecast_todays_fixing=True),
                self._index.day_counter(),
                Compounding.Simple,
                Frequency.Once,
            )
        else:
            # Par-coupon approximation
            forecast_curve = self._index.forecast_term_structure()
            qassert.require(
                forecast_curve is not None,
                "no forwarding term structure set on FRA index",
            )
            assert forecast_curve is not None
            d_value = forecast_curve.discount(self._value_date)
            d_maturity = forecast_curve.discount(self._maturity_date)
            year_fraction = self._index.day_counter().year_fraction(
                self._value_date, self._maturity_date
            )
            par_rate = (d_value / d_maturity - 1.0) / year_fraction
            self._forward_rate = InterestRate(
                par_rate,
                self._index.day_counter(),
                Compounding.Simple,
                Frequency.Once,
            )

    def _calculate_amount(self) -> None:
        """C++ parity: ``ForwardRateAgreement::calculateAmount``."""
        self._calculate_forward_rate()
        sign = 1 if self._fra_type == PositionType.Long else -1
        assert self._forward_rate is not None
        f = self._forward_rate.rate()
        k = self._strike_forward_rate.rate()
        t = self._forward_rate.day_counter().year_fraction(
            self._value_date, self._maturity_date
        )
        self._amount = self._notional_amount * sign * (f - k) * t / (1.0 + f * t)


__all__ = ["ForwardRateAgreement"]
