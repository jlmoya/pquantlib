"""InterpolatingCPICapFloorEngine — CPI cap/floor from a price surface.

# C++ parity: ql/experimental/inflation/cpicapfloorengines.{hpp,cpp}
   (v1.42.1) — ``InterpolatingCPICapFloorEngine``.

This engine prices a :class:`CPICapFloor` purely by reading an
already-built :class:`CPICapFloorTermPriceSurface`; it only adds *timing*
functionality (handling a possible difference between the cap/floor's
observation lag and the surface's).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.experimental.inflation.cpi_cap_floor_term_price_surface import (
    CPICapFloorTermPriceSurface,
)
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import inflation_period
from pquantlib.instruments.cpi_cap_floor import (
    CPICapFloorArguments,
    CPICapFloorResults,
)
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

_ONE_DAY = Period(1, TimeUnit.Days)
_ZERO_MONTHS = Period(0, TimeUnit.Months)


class InterpolatingCPICapFloorEngine(
    GenericEngine[CPICapFloorArguments, CPICapFloorResults]
):
    """Prices a CPI cap/floor by interpolating a market price surface."""

    def __init__(self, price_surface: CPICapFloorTermPriceSurface) -> None:
        super().__init__(CPICapFloorArguments(), CPICapFloorResults())
        self._price_surf: CPICapFloorTermPriceSurface = price_surface

    def name(self) -> str:
        return "InterpolatingCPICapFloorEngine"

    def price_surface(self) -> CPICapFloorTermPriceSurface:
        return self._price_surf

    def calculate(self) -> None:
        # # C++ parity: InterpolatingCPICapFloorEngine::calculate
        # # (cpicapfloorengines.cpp:38-97).
        args = self._arguments
        results = self._results
        results.reset()

        # Difference between the cap/floor's obs lag and the surface's.
        lag_diff = args.observation_lag - self._price_surf.observation_lag()
        qassert.require(
            lag_diff >= _ZERO_MONTHS,
            f"InterpolatingCPICapFloorEngine: lag difference must be "
            f"non-negative: {lag_diff}",
        )

        # Effective maturity used on the surface (its time axis is built
        # from the calibration instruments' maturities).
        effective_maturity = args.pay_date - lag_diff
        is_call = args.type == OptionType.Call

        if args.observation_interpolation == InterpolationType.AsIndex:
            # Same as index -> the surface (which uses the index) is direct.
            if is_call:
                npv = self._price_surf.cap_price(effective_maturity, args.strike)
            else:
                npv = self._price_surf.floor_price(effective_maturity, args.strike)
        else:
            assert args.index is not None
            dd_first, dd_second = inflation_period(
                effective_maturity, args.index.frequency()
            )
            if is_call:
                price_start = self._price_surf.cap_price(dd_first, args.strike)
            else:
                price_start = self._price_surf.floor_price(dd_first, args.strike)

            if args.observation_interpolation == InterpolationType.Flat:
                # Value cannot change after the first day of the period.
                npv = price_start
            else:
                # Linear interpolation across the inflation period.
                period_end = dd_second + _ONE_DAY
                if is_call:
                    price_end = self._price_surf.cap_price(period_end, args.strike)
                else:
                    price_end = self._price_surf.floor_price(period_end, args.strike)
                num = float(effective_maturity - dd_first)
                denom = float(period_end - dd_first)
                npv = price_start + (price_end - price_start) * num / denom

        results.value = npv
