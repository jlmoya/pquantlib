"""InterpolatedYoYOptionletStripper — concrete YoY caplet-vol stripper.

# C++ parity: ql/experimental/inflation/interpolatedyoyoptionletstripper.hpp
   (v1.42.1) — ``InterpolatedYoYOptionletStripper<Interpolator1D>``.

For each quoted strike K the stripper:

1. solves for the initial (shortest-maturity) caplet vol that reprices
   the first cap/floor price (a Brent solve on a 2-point flat vol curve),
2. builds :class:`YoYOptionletHelper` instruments for every maturity at
   strike K, and
3. bootstraps a :class:`PiecewiseYoYOptionletVolatilityCurve` per K.

:meth:`slice` then reads each per-K curve at a query date to return the
``(strikes, vols)`` smile slice.

.. note::
   The upstream C++ class carries a ``\\bug Tests currently fail``
   annotation — the per-K bootstrap is genuinely fragile. PQuantLib ports
   it faithfully; cross-validation is LOOSE-tier (bootstrap convergence).
"""

from __future__ import annotations

from collections.abc import Callable
from math import floor

from pquantlib import qassert
from pquantlib.currencies.currency import Currency
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.experimental.inflation.generic_indexes import YYGenericCPI
from pquantlib.experimental.inflation.piecewise_yoy_optionlet_volatility import (
    PiecewiseYoYOptionletVolatilityCurve,
)
from pquantlib.experimental.inflation.yoy_cap_floor_term_price_surface import (
    YoYCapFloorTermPriceSurface,
)
from pquantlib.experimental.inflation.yoy_inflation_optionlet_volatility_structure2 import (
    InterpolatedYoYOptionletVolatilityCurve,
)
from pquantlib.experimental.inflation.yoy_optionlet_helpers import YoYOptionletHelper
from pquantlib.experimental.inflation.yoy_optionlet_stripper import YoYOptionletStripper
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import YoYInflationIndex
from pquantlib.instruments.make_yoy_inflation_cap_floor import (
    make_yoy_inflation_cap_floor,
)
from pquantlib.instruments.yoy_inflation_capfloor import YoYInflationCapFloorType
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.pricingengines.inflation.yoy_inflation_capfloor_engine import (
    YoYInflationCapFloorEngine,
)
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.inflation.constant_yoy_optionlet_volatility import (
    ConstantYoYOptionletVolatility,
)
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendars.target import TARGET
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

Interpolation1DFactory = Callable[[Array, Array], Interpolation]


class _ObjectiveFunction:
    """Find the initial caplet vol that reprices the shortest cap/floor.

    # C++ parity: ``InterpolatedYoYOptionletStripper::ObjectiveFunction``.
    """

    def __init__(
        self,
        cf_type: YoYInflationCapFloorType,
        slope: float,
        strike: float,
        an_index: YoYInflationIndex,
        surf: YoYCapFloorTermPriceSurface,
        pricer: YoYInflationCapFloorEngine,
        price_to_match: float,
    ) -> None:
        self._slope = slope
        self._frequency = an_index.frequency()
        self._index_is_interpolated = an_index.interpolated()
        self._price_to_match = price_to_match
        self._surf = surf
        self._pricer = pricer
        self._lag = surf.observation_lag()

        n = floor(0.5 + surf.time_from_reference(surf.min_maturity()))
        qassert.require(n > 0, f"first maturity in price surface not > 0: {n}")
        self._capfloor = make_yoy_inflation_cap_floor(
            cf_type,
            an_index,
            n,
            surf.calendar(),
            surf.observation_lag(),
            InterpolationType.AsIndex,
            strike=strike,
            nominal=10000.0,
        )

        # 2-point vol-curve dates/times (base + just past first maturity).
        self._dvec = [surf.base_date(), surf.min_maturity() + Period(7, TimeUnit.Days)]
        self._tvec = [
            surf.day_counter().year_fraction(surf.reference_date(), self._dvec[0]),
            surf.day_counter().year_fraction(surf.reference_date(), self._dvec[1]),
        ]
        self._capfloor.set_pricing_engine(self._pricer)

    def __call__(self, guess: float) -> float:
        v1 = guess
        v0 = guess - self._slope * (self._tvec[1] - self._tvec[0]) * guess
        v_curve = InterpolatedYoYOptionletVolatilityCurve(
            0,
            TARGET(),
            BusinessDayConvention.ModifiedFollowing,
            Actual365Fixed(),
            self._lag,
            self._frequency,
            self._index_is_interpolated,
            self._dvec,
            [v0, v1],
            -1.0,
            3.0,
        )
        self._pricer.set_volatility(v_curve)
        return self._price_to_match - self._capfloor.npv()


class InterpolatedYoYOptionletStripper(YoYOptionletStripper):
    """Interpolated (per-K-bootstrapped) YoY optionlet vol stripper."""

    def __init__(
        self, interpolator: Interpolation1DFactory = LinearInterpolation
    ) -> None:
        self._interpolator: Interpolation1DFactory = interpolator
        self._surface: YoYCapFloorTermPriceSurface | None = None
        self._pricer: YoYInflationCapFloorEngine | None = None
        self._lag: Period | None = None
        self._frequency = None
        self._index_is_interpolated: bool = False
        self._vol_curves: list[PiecewiseYoYOptionletVolatilityCurve] = []

    # ---- YoYOptionletStripper interface ----------------------------------

    def min_strike(self) -> float:
        assert self._surface is not None
        return self._surface.strikes()[0]

    def max_strike(self) -> float:
        assert self._surface is not None
        return self._surface.strikes()[-1]

    def strikes(self) -> list[float]:
        assert self._surface is not None
        return self._surface.strikes()

    def initialize(
        self,
        surface: YoYCapFloorTermPriceSurface,
        pricer: YoYInflationCapFloorEngine,
        slope: float,
    ) -> None:
        # # C++ parity: InterpolatedYoYOptionletStripper::initialize.
        self._surface = surface
        self._pricer = pricer
        self._lag = surface.observation_lag()
        self._frequency = surface.frequency()
        self._index_is_interpolated = surface.index_is_interpolated()
        fixing_days = surface.fixing_days()
        settlement_days = 0
        cal = surface.calendar()
        bdc = surface.business_day_convention()
        dc = surface.day_counter()

        max_floor = surface.floor_strikes()[-1]
        use_type = YoYInflationCapFloorType.Floor
        tp_min = surface.maturities()[0]

        # "fake" Generic YoY index carrying the right lag/frequency.
        an_index = YYGenericCPI(
            self._frequency, False, self._lag, Currency(),
            ts=surface.yoy_index().yoy_inflation_term_structure(),
        )

        self._vol_curves = []
        for strike in surface.strikes():
            if strike > max_floor:
                use_type = YoYInflationCapFloorType.Cap

            # 1. solve for the initial point on the vol curve.
            solver = Brent()
            solver_tol = 1e-7
            lo, hi = 0.00001, 0.08
            guess = (hi + lo) / 2.0
            price_to_match = (
                surface.cap_price(tp_min, strike)
                if use_type == YoYInflationCapFloorType.Cap
                else surface.floor_price(tp_min, strike)
            )
            found = solver.solve(
                _ObjectiveFunction(
                    use_type, slope, strike, an_index, surface,
                    pricer, price_to_match,
                ),
                solver_tol, guess, lo, hi,
            )

            # 2. create helpers for every maturity at this strike.
            notional = 10000.0
            helpers: list[YoYOptionletHelper] = []
            for tp in surface.maturities():
                next_price = (
                    surface.cap_price(tp, strike)
                    if use_type == YoYInflationCapFloorType.Cap
                    else surface.floor_price(tp, strike)
                )
                n_t = floor(
                    surface.time_from_reference(surface.yoy_option_date_from_tenor(tp))
                    + 0.5
                )
                helper = YoYOptionletHelper(
                    SimpleQuote(next_price), notional, use_type, self._lag,
                    dc, cal, fixing_days, an_index, InterpolationType.Flat,
                    strike, n_t, pricer,
                )
                yoy_vol_black = ConstantYoYOptionletVolatility(
                    vol=found, settlement_days=settlement_days, calendar=cal,
                    business_day_convention=bdc, day_counter=dc,
                    observation_lag=self._lag, frequency=self._frequency,
                    index_is_interpolated=False, min_strike=-1.0, max_strike=3.0,
                )
                helper.set_term_structure(yoy_vol_black)
                helpers.append(helper)

            # 3. bootstrap a piecewise curve at this strike.
            t_min = surface.time_from_reference(
                surface.yoy_option_date_from_tenor(tp_min)
            )
            base_yoy_vol = found - slope * t_min * found
            eps = max(strike, 0.02) / 1000.0
            test_pw = PiecewiseYoYOptionletVolatilityCurve(
                settlement_days, cal, bdc, dc, self._lag, self._frequency,
                self._index_is_interpolated, strike - eps, strike + eps,
                base_yoy_vol, helpers,
            )
            self._vol_curves.append(test_pw)

    def slice(self, d: Date) -> tuple[list[float], list[float]]:
        ks = self.strikes()
        vols = [self._vol_curves[i].volatility(d, k) for i, k in enumerate(ks)]
        return list(ks), vols

    def vol_curves(self) -> list[PiecewiseYoYOptionletVolatilityCurve]:
        return list(self._vol_curves)
