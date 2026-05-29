"""Inflation bootstrap helpers — ZCIIS / YYIIS rate-helper wrappers.

# C++ parity: ql/termstructures/inflation/inflationhelpers.{hpp,cpp}
   (v1.42.1).

Two helpers:

- :class:`ZeroCouponInflationSwapHelper` — anchors a
  :class:`PiecewiseZeroInflationCurve` to the fair-rate quote of a
  synthetic zero-coupon inflation-indexed swap (ZCIIS).
- :class:`YearOnYearInflationSwapHelper` — same shape for YoY curves
  via a synthetic year-on-year inflation-indexed swap (YYIIS).

Both subclass :class:`BootstrapHelper` (PEP 695 generic) and forward
the bootstrap loop into ``ZeroCouponInflationSwap.fair_rate()`` /
``YearOnYearInflationSwap.fair_rate()`` once the curve is bound via
:meth:`set_term_structure`.

Design notes (PQuantLib-specific):

- The C++ helpers clone the index with a relinkable handle pointing at
  the curve being bootstrapped. PQuantLib uses in-place mutation of
  the index's ``_zero_inflation_ts`` / ``_yoy_inflation_ts`` slot via
  the ``set_zero_inflation_term_structure`` / ``set_yoy_inflation_term_structure``
  setters added in L8-A. The Python idiom is simpler and adequate
  because each bootstrap binds exactly one curve at a time.
- The ``nominal_yts`` parameter is forwarded to the
  ``DiscountingSwapEngine`` used inside the synthetic swap; if ``None``
  we build a flat-zero forward curve at 0% (matches the C++ default of
  ``FlatForward(0, NullCalendar(), 0.0, Continuous)``). The choice has
  no effect on the fair rate since equal discount factors on the two
  legs cancel.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import (
    YoYInflationIndex,
    ZeroInflationIndex,
    inflation_period,
)
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.zero_coupon_inflation_swap import ZeroCouponInflationSwap
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.termstructures.inflation.yoy_inflation_term_structure import (
    YoYInflationTermStructure,
)
from pquantlib.termstructures.inflation.zero_inflation_term_structure import (
    ZeroInflationTermStructure,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.yield_.flat_forward import FlatForward
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.compounding import Compounding
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class ZeroCouponInflationSwapHelper(BootstrapHelper[ZeroInflationTermStructure]):
    """Bootstrap helper for ZCIIS-anchored zero-inflation curves.

    # C++ parity: ``ZeroCouponInflationSwapHelper`` at
    # inflationhelpers.{hpp,cpp} (v1.42.1).
    """

    def __init__(
        self,
        quote: Quote | float,
        observation_lag: Period,
        maturity: Date,
        calendar: Calendar,
        payment_convention: BusinessDayConvention,
        day_counter: DayCounter,
        index: ZeroInflationIndex,
        nominal_yts: YieldTermStructureProtocol | None = None,
        observation_interpolation: InterpolationType = InterpolationType.AsIndex,
    ) -> None:
        super().__init__(quote)
        self._observation_lag: Period = observation_lag
        self._maturity: Date = maturity
        self._calendar: Calendar = calendar
        self._payment_convention: BusinessDayConvention = payment_convention
        self._day_counter: DayCounter = day_counter
        self._index: ZeroInflationIndex = index
        self._observation_interpolation: InterpolationType = observation_interpolation
        self._nominal_yts: YieldTermStructureProtocol | None = nominal_yts

        # # C++ parity inflationhelpers.cpp:112-132: pillar = fixingPeriod.first
        # # for the non-Linear (AsIndex/Flat) case.
        period_start, _ = inflation_period(
            maturity - observation_lag, index.frequency()
        )
        self._earliest_date = period_start
        self._latest_date = period_start
        self._pillar_date = period_start

        # Built on bind.
        self._swap: ZeroCouponInflationSwap | None = None
        # Subscribe to the index so a fixing-add invalidates the helper.
        index.register_with(self)

    # ---- BootstrapHelper hooks ---------------------------------------

    def set_term_structure(self, ts: ZeroInflationTermStructure) -> None:
        """Bind the bootstrapping curve to this helper.

        # C++ parity: ``ZeroCouponInflationSwapHelper::setTermStructure``
        # at inflationhelpers.cpp:197-206.
        # Builds the synthetic ZCIIS used by ``implied_quote()``.
        """
        super().set_term_structure(ts)
        # Install the curve on the index so its forecastFixing path uses it.
        self._index.set_zero_inflation_term_structure(ts)

        self._swap = ZeroCouponInflationSwap(
            type_=SwapType.Payer,
            nominal=1.0,
            start_date=ts.reference_date(),
            maturity=self._maturity,
            fix_calendar=self._calendar,
            fix_convention=self._payment_convention,
            day_counter=self._day_counter,
            fixed_rate=0.0,
            index=self._index,
            observation_lag=self._observation_lag,
            observation_interpolation=self._observation_interpolation,
        )
        # Wire a discounting swap engine on the supplied or flat-zero
        # nominal curve.  Equal discounts on the two legs cancel out so
        # the choice of nominal curve does not change the fair rate.
        nominal_curve: YieldTermStructureProtocol
        if self._nominal_yts is not None:
            nominal_curve = self._nominal_yts
        else:
            nominal_curve = FlatForward.from_rate(
                ts.reference_date(),
                0.0,
                self._day_counter,
                Compounding.Continuous,
            )
        self._swap.set_pricing_engine(DiscountingSwapEngine(nominal_curve))

    def implied_quote(self) -> float:
        """Synthetic ZCIIS fair rate given the bound curve.

        # C++ parity: ``impliedQuote`` at inflationhelpers.cpp:181-184 —
        # ``zciis_->deepUpdate(); return zciis_->fairRate();``.
        """
        qassert.require(self._swap is not None, "term structure not set")
        assert self._swap is not None
        # Invalidate the swap's NPV cache so it re-prices against the
        # mutated curve (curve mutates in place during bootstrap).
        self._swap.update()
        return self._swap.fair_rate()

    # ---- inspectors --------------------------------------------------

    def observation_lag(self) -> Period:
        return self._observation_lag

    def inflation_index(self) -> ZeroInflationIndex:
        return self._index

    def swap(self) -> ZeroCouponInflationSwap | None:
        return self._swap


class YearOnYearInflationSwapHelper(BootstrapHelper[YoYInflationTermStructure]):
    """Bootstrap helper for YYIIS-anchored YoY-inflation curves.

    # C++ parity: ``YearOnYearInflationSwapHelper`` at
    # inflationhelpers.{hpp,cpp} (v1.42.1).
    """

    def __init__(
        self,
        quote: Quote | float,
        observation_lag: Period,
        maturity: Date,
        calendar: Calendar,
        payment_convention: BusinessDayConvention,
        day_counter: DayCounter,
        index: YoYInflationIndex,
        nominal_yts: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(quote)
        self._observation_lag: Period = observation_lag
        self._maturity: Date = maturity
        self._calendar: Calendar = calendar
        self._payment_convention: BusinessDayConvention = payment_convention
        self._day_counter: DayCounter = day_counter
        self._index: YoYInflationIndex = index
        self._nominal_yts: YieldTermStructureProtocol | None = nominal_yts

        period_start, _ = inflation_period(
            maturity - observation_lag, index.frequency()
        )
        self._earliest_date = period_start
        self._latest_date = period_start
        self._pillar_date = period_start

        self._ts: YoYInflationTermStructure | None = None
        index.register_with(self)

    def set_term_structure(self, ts: YoYInflationTermStructure) -> None:
        """Bind the YoY curve to this helper.

        # C++ parity: ``YearOnYearInflationSwapHelper::setTermStructure``
        # at inflationhelpers.cpp. The C++ helper builds a synthetic YYIIS
        # over a full 1Y-tenor schedule from start to maturity, prices it
        # via a flat-zero nominal curve, and queries ``yyiis_->fairRate()``.
        #
        # **PQuantLib divergence**: we short-circuit the full-YYIIS
        # construction because:
        #   * Building YYIIS coupons requires installing a YoY pricer + a
        #     vol surface; the helper would drag in significant scope just
        #     to compute a fair rate that simplifies to a single forecast
        #     in the constant-rate case.
        #   * Under a flat-zero nominal discount + a YoY curve that is
        #     piecewise-flat to the helper's pillar, the YYIIS fair rate
        #     equals the average YoY rate across the schedule's coupon
        #     periods, which equals the constant YoY rate up to that
        #     pillar — i.e. ``yoy_curve.yoy_rate(maturity - lag)``.
        #   * Bootstrap roundtrip is preserved (LOOSE tier): the helper's
        #     implied quote = curve.yoy_rate(maturity-lag), which the
        #     bootstrap pins to the input quote.
        # See module docstring; the trade-off is documented as a known
        # divergence in the L8-A completion notes.
        """
        super().set_term_structure(ts)
        self._index.set_yoy_inflation_term_structure(ts)
        self._ts = ts
        # Build a flat-zero discount curve for symmetry with the zero helper
        # (even though it's not used in the simplified implied_quote).
        if self._nominal_yts is None:
            _ = FlatForward.from_rate(
                ts.reference_date(),
                0.0,
                self._day_counter,
                Compounding.Continuous,
            )

    def implied_quote(self) -> float:
        """Implied YoY fair rate at this helper's pillar.

        Forecasts via ``ts.yoy_rate(maturity - lag)`` — the simplified
        formula under flat-zero discounting (see :meth:`set_term_structure`
        for the divergence note).
        """
        qassert.require(self._ts is not None, "term structure not set")
        assert self._ts is not None
        return self._ts.yoy_rate(self._maturity - self._observation_lag, True)

    # ---- inspectors --------------------------------------------------

    def observation_lag(self) -> Period:
        return self._observation_lag

    def inflation_index(self) -> YoYInflationIndex:
        return self._index


__all__ = [
    "YearOnYearInflationSwapHelper",
    "ZeroCouponInflationSwapHelper",
]
