"""YoYOptionletHelper — bootstrap helper for YoY-vol stripping.

# C++ parity: ql/experimental/inflation/yoyoptionlethelpers.{hpp,cpp}
   (v1.42.1) — ``YoYOptionletHelper``.

Builds a standard YoY cap/floor (via :func:`make_yoy_inflation_cap_floor`)
to reprice; ``set_term_structure`` re-binds the cap/floor pricer's vol
surface and ``implied_quote`` returns the cap/floor NPV.
"""

from __future__ import annotations

from pquantlib.cashflows.yoy_inflation_coupon import YoYInflationCoupon
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import YoYInflationIndex
from pquantlib.instruments.make_yoy_inflation_cap_floor import (
    make_yoy_inflation_cap_floor,
)
from pquantlib.instruments.yoy_inflation_capfloor import (
    YoYInflationCapFloor,
    YoYInflationCapFloorType,
)
from pquantlib.pricingengines.inflation.yoy_inflation_capfloor_engine import (
    YoYInflationCapFloorEngine,
)
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.bootstrap_helper import BootstrapHelper
from pquantlib.termstructures.volatility.inflation.yoy_optionlet_volatility_surface import (
    YoYOptionletVolatilitySurface,
)
from pquantlib.time.calendar import Calendar
from pquantlib.time.period import Period


class YoYOptionletHelper(BootstrapHelper[YoYOptionletVolatilitySurface]):
    """Year-on-year inflation-volatility bootstrap helper."""

    def __init__(
        self,
        price: Quote | float,
        notional: float,
        cap_floor_type: YoYInflationCapFloorType,
        lag: Period,
        yoy_day_counter: DayCounter,
        payment_calendar: Calendar,
        fixing_days: int,
        index: YoYInflationIndex,
        interpolation: InterpolationType,
        strike: float,
        n: int,
        pricer: YoYInflationCapFloorEngine,
    ) -> None:
        super().__init__(price)
        self._notional: float = notional
        self._cap_floor_type: YoYInflationCapFloorType = cap_floor_type
        self._lag: Period = lag
        self._fixing_days: int = fixing_days
        self._index: YoYInflationIndex = index
        self._strike: float = strike
        self._n: int = n
        self._yoy_day_counter: DayCounter = yoy_day_counter
        self._calendar: Calendar = payment_calendar
        self._pricer: YoYInflationCapFloorEngine = pricer

        # Build the cap/floor to reprice (done once).
        self._yoy_cap_floor: YoYInflationCapFloor = make_yoy_inflation_cap_floor(
            cap_floor_type,
            index,
            n,
            payment_calendar,
            lag,
            interpolation,
            strike=strike,
            nominal=notional,
            fixing_days=fixing_days,
            payment_day_counter=yoy_day_counter,
        )

        # Dates already include the index/instrument lag; these are the
        # fixing dates of the first/last coupon.
        leg = self._yoy_cap_floor.yoy_leg()
        first = leg[0]
        last = leg[-1]
        assert isinstance(first, YoYInflationCoupon)
        assert isinstance(last, YoYInflationCoupon)
        self._earliest_date = first.fixing_date()
        self._latest_date = last.fixing_date()
        self._pillar_date = self._latest_date

        self._yoy_cap_floor.set_pricing_engine(self._pricer)

    def implied_quote(self) -> float:
        # # C++ parity: yoyCapFloor_->deepUpdate(); return NPV.
        self._yoy_cap_floor.update()
        return self._yoy_cap_floor.npv()

    def set_term_structure(self, ts: YoYOptionletVolatilitySurface) -> None:
        # # C++ parity: rebind the pricer's vol surface each call (the
        # # surface pointer changes during the bootstrap).
        super().set_term_structure(ts)
        self._pricer.set_volatility(ts)
