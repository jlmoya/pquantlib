"""InterpolatedYoYCapFloorTermPriceSurface — concrete YoY price surface.

# C++ parity: ql/experimental/inflation/yoycapfloortermpricesurface.hpp
   (v1.42.1) — ``InterpolatedYoYCapFloorTermPriceSurface<I2D, I1D>``.

Builds two 2-D interpolations (cap + floor prices over maturity-time x
strike), finds the ATM YoY-swap rate at each maturity as the cap/floor
intersection (a Brent root-find with a heuristic interval search), then
bootstraps a :class:`PiecewiseYoYInflationCurve` from those ATM swap
rates so the surface can return a YoY forecasting curve.

The 2-D interpolator (``I2D``) defaults to :class:`BicubicSpline` and the
1-D ATM-swap-rate interpolator (``I1D``) to :class:`LinearInterpolation`,
matching the canonical C++ usage ``<Bicubic, Linear>``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.inflation.interpolated_cpi_cap_floor_term_price_surface import (
    Interpolator2DFactory,
    Surface2D,
)
from pquantlib.experimental.inflation.yoy_cap_floor_term_price_surface import (
    YoYCapFloorTermPriceSurface,
)
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import YoYInflationIndex
from pquantlib.math.array import Array
from pquantlib.math.interpolations.bicubic_spline import BicubicSpline
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.math.matrix import Matrix
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.inflation.inflation_helpers import (
    YearOnYearInflationSwapHelper,
)
from pquantlib.termstructures.inflation.piecewise_yoy_inflation_curve import (
    PiecewiseYoYInflationCurve,
)
from pquantlib.termstructures.inflation.yoy_inflation_term_structure import (
    YoYInflationTermStructure,
)
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

Interpolation1DFactory = Callable[[Array, Array], Interpolation]


class InterpolatedYoYCapFloorTermPriceSurface(YoYCapFloorTermPriceSurface):
    """Bicubic/Linear-by-default YoY cap/floor price surface."""

    def __init__(
        self,
        fixing_days: int,
        yy_lag: Period,
        yii: YoYInflationIndex,
        interpolation: InterpolationType,
        nominal: YieldTermStructureProtocol,
        day_counter: DayCounter,
        calendar: Calendar,
        bdc: BusinessDayConvention,
        c_strikes: Sequence[float],
        f_strikes: Sequence[float],
        cf_maturities: Sequence[Period],
        c_price: Matrix,
        f_price: Matrix,
        interpolator2d: Interpolator2DFactory = BicubicSpline,
        interpolator1d: Interpolation1DFactory = LinearInterpolation,
    ) -> None:
        super().__init__(
            fixing_days, yy_lag, yii, interpolation, nominal, day_counter,
            calendar, bdc, c_strikes, f_strikes, cf_maturities, c_price, f_price,
        )
        self._interpolator2d: Interpolator2DFactory = interpolator2d
        self._interpolator1d: Interpolation1DFactory = interpolator1d
        self._cap_price: Surface2D | None = None
        self._floor_price: Surface2D | None = None
        self._atm_swap_rate_curve: Interpolation | None = None
        self._atm_time_rates: tuple[list[float], list[float]] = ([], [])
        self._atm_date_rates: tuple[list[Date], list[float]] = ([], [])
        self._yoy: YoYInflationTermStructure | None = None
        self._perform_calculations()

    # ----- inflation term structure interface -----------------------------

    def max_date(self) -> Date:
        assert self._yoy is not None
        return self._yoy.max_date()

    def base_date(self) -> Date:
        assert self._yoy is not None
        return self._yoy.base_date()

    def yoy_ts(self) -> object:
        return self._yoy

    def atm_yoy_swap_time_rates(self) -> tuple[list[float], list[float]]:
        return self._atm_time_rates

    def atm_yoy_swap_date_rates(self) -> tuple[list[Date], list[float]]:
        return self._atm_date_rates

    # ----- price interface ------------------------------------------------

    def _price_at_date(self, d: Date, k: float) -> float:
        atm = self._atm_yoy_swap_rate_at_date(d)
        return self._cap_price_at_date(d, k) if k > atm else self._floor_price_at_date(d, k)

    def _cap_price_at_date(self, d: Date, k: float) -> float:
        assert self._cap_price is not None
        t = self.time_from_reference(d)
        return self._cap_price(t, k, allow_extrapolation=True)

    def _floor_price_at_date(self, d: Date, k: float) -> float:
        assert self._floor_price is not None
        t = self.time_from_reference(d)
        return self._floor_price(t, k, allow_extrapolation=True)

    def _atm_yoy_swap_rate_at_date(self, d: Date, extrapolate: bool = True) -> float:
        assert self._atm_swap_rate_curve is not None
        return self._atm_swap_rate_curve(
            self.time_from_reference(d), allow_extrapolation=extrapolate
        )

    def _atm_yoy_rate_at_date(
        self, d: Date, obs_lag: Period | None = None, extrapolate: bool = True
    ) -> float:
        assert self._yoy is not None
        p = self.observation_lag() if obs_lag is None else obs_lag
        return self._yoy.yoy_rate(d - p, extrapolate)

    # ----- lazy-object-style calculation ----------------------------------

    def _perform_calculations(self) -> None:
        self._intersect()
        self._calculate_yoy_term_structure()

    def _intersect(self) -> None:
        # # C++ parity: InterpolatedYoYCapFloorTermPriceSurface::intersect
        # # (yoycapfloortermpricesurface.hpp:344-518).
        max_search_range = 0.0201
        max_extrapolation_maturity = 5.01
        search_step = 0.0050
        intrinsic_value_add_on = 0.001

        n_mat = len(self._cf_maturities)
        valid_maturity = [False] * n_mat

        self._cf_maturity_times = [
            self.time_from_reference(self.yoy_option_date_from_tenor(m))
            for m in self._cf_maturities
        ]
        xs = np.asarray(self._cf_maturity_times, dtype=np.float64)
        c_strikes = np.asarray(self._c_strikes, dtype=np.float64)
        f_strikes = np.asarray(self._f_strikes, dtype=np.float64)

        self._cap_price = self._interpolator2d(xs, c_strikes, self._c_price)
        self._cap_price.enable_extrapolation()
        self._floor_price = self._interpolator2d(xs, f_strikes, self._f_price)
        self._floor_price.enable_extrapolation()
        cap = self._cap_price
        floor = self._floor_price

        solver = Brent()
        solver_tol = 1e-7
        min_swap_intersection = [0.0] * n_mat
        max_swap_intersection = [0.0] * n_mat
        tmp_swap_maturities: list[float] = []
        tmp_swap_rates: list[float] = []

        for i in range(n_mat):
            t = self._cf_maturity_times[i]
            tmp_min, tmp_max = self._swap_rate_bounds(t, cap, floor)
            min_swap_intersection[i] = tmp_min
            max_swap_intersection[i] = tmp_max

            lo, hi, trials_exceeded = self._bracket_intersection(
                t, cap, floor, max_search_range, search_step
            )
            guess = (hi + lo) / 2.0

            if not trials_exceeded:

                def objective(k: float, _t: float = t) -> float:
                    return cap(_t, k, allow_extrapolation=True) - floor(
                        _t, k, allow_extrapolation=True
                    )

                k_i = solver.solve(objective, solver_tol, guess, lo, hi)
                if k_i <= tmp_min:
                    if t > max_extrapolation_maturity:
                        qassert.fail(
                            f"cap/floor intersection finding failed at t = {t}: "
                            f"intersection below arbitrage-free lower bound {tmp_min}"
                        )
                else:
                    tmp_swap_maturities.append(t)
                    tmp_swap_rates.append(k_i)
                    valid_maturity[i] = True
            elif t > max_extrapolation_maturity:
                qassert.fail(
                    f"cap/floor intersection finding failed at t = {t}: "
                    f"no intersection found inside the admissible range"
                )

        # assemble the ATM swap (time,rate) and (date,rate) pairs
        self._assemble_atm_rates(
            valid_maturity,
            min_swap_intersection,
            max_swap_intersection,
            tmp_swap_maturities,
            tmp_swap_rates,
            intrinsic_value_add_on,
        )

        self._atm_swap_rate_curve = self._interpolator1d(
            np.asarray(self._atm_time_rates[0], dtype=np.float64),
            np.asarray(self._atm_time_rates[1], dtype=np.float64),
        )

    def _assemble_atm_rates(
        self,
        valid_maturity: list[bool],
        min_swap: list[float],
        max_swap: list[float],
        tmp_swap_maturities: list[float],
        tmp_swap_rates: list[float],
        intrinsic_value_add_on: float,
    ) -> None:
        """Build the ATM-swap (time,rate)/(date,rate) pairs from the solve.

        # C++ parity: the final assembly loop in intersect()
        # (yoycapfloortermpricesurface.hpp:490-511) — for invalid maturities
        # it overwrites the swap rate with a heuristic that keeps every
        # option's intrinsic value below its price.
        """
        date_first: list[Date] = []
        time_first: list[float] = []
        time_rates: list[float] = []
        date_rates: list[float] = []
        counter = 0
        for i in range(len(self._cf_maturities)):
            if not valid_maturity[i]:
                ref_plus_mat = self.reference_date() + self._cf_maturities[i]
                date_first.append(ref_plus_mat)
                time_first.append(self.time_from_reference(ref_plus_mat))
                new_rate = min_swap[i] + intrinsic_value_add_on
                if new_rate > max_swap[i]:
                    new_rate = 0.5 * (min_swap[i] + max_swap[i])
                time_rates.append(new_rate)
                date_rates.append(new_rate)
            else:
                time_first.append(tmp_swap_maturities[counter])
                time_rates.append(tmp_swap_rates[counter])
                date_first.append(
                    self.yoy_option_date_from_tenor(self._cf_maturities[counter])
                )
                date_rates.append(tmp_swap_rates[counter])
                counter += 1
        self._atm_time_rates = (time_first, time_rates)
        self._atm_date_rates = (date_first, date_rates)

    def _swap_rate_bounds(
        self, t: float, cap: Surface2D, floor: Surface2D
    ) -> tuple[float, float]:
        """Arbitrage-free min/max ATM-swap-rate band at maturity-time ``t``.

        # C++ parity: the floor/cap loops in intersect() that set
        # ``tmpMin/MaxSwapRateIntersection`` (yoycapfloortermpricesurface.hpp:391-409).
        """
        num_years = round(t)
        sum_discount = 0.0
        for jj in range(num_years):
            sum_discount += self._nominal_ts.discount(jj + 1.0)
        tmp_min = -1.0e10
        tmp_max = 1.0e10
        for k in self._f_strikes:
            price = floor(t, k, allow_extrapolation=True)
            tmp_min = max(tmp_min, k - price / (sum_discount * 10000))
        for k in self._c_strikes:
            price = cap(t, k, allow_extrapolation=True)
            tmp_max = min(tmp_max, k + price / (sum_discount * 10000))
        return tmp_min, tmp_max

    def _bracket_intersection(
        self,
        t: float,
        cap: Surface2D,
        floor: Surface2D,
        max_search_range: float,
        search_step: float,
    ) -> tuple[float, float, bool]:
        """Locate the ``[lo, hi]`` interval bracketing the cap/floor crossing.

        # C++ parity: the interval-search loops in intersect()
        # (yoycapfloortermpricesurface.hpp:414-452). Returns
        # ``(lo, hi, trials_exceeded)``.
        """
        num_trials = int(max_search_range / search_step)
        f_back = self._f_strikes[-1]
        decreasing = floor(t, f_back, allow_extrapolation=True) > cap(
            t, f_back, allow_extrapolation=True
        )
        counter = 1
        stop = False
        strike = 0.0
        trials_exceeded = False
        while not stop:
            if decreasing:
                strike = f_back - counter * search_step
                crossed = floor(t, strike, allow_extrapolation=True) < cap(
                    t, strike, allow_extrapolation=True
                )
            else:
                strike = f_back + counter * search_step
                crossed = floor(t, strike, allow_extrapolation=True) > cap(
                    t, strike, allow_extrapolation=True
                )
            if crossed:
                stop = True
            counter += 1
            if counter == num_trials + 1 and not stop:
                stop = True
                trials_exceeded = True
        if decreasing:
            return strike, strike + search_step, trials_exceeded
        return strike - search_step, strike, trials_exceeded

    def _calculate_yoy_term_structure(self) -> None:
        # # C++ parity: calculateYoYTermStructure (yoycapfloortermpricesurface.hpp:521-570).
        n_years = round(
            self.time_from_reference(self.reference_date() + self._cf_maturities[-1])
        )
        ref = self._nominal_ts.reference_date()
        helpers: list[YearOnYearInflationSwapHelper] = []
        for i in range(1, n_years + 1):
            maturity = ref + Period(i, TimeUnit.Years)
            quote = SimpleQuote(self.atm_yoy_swap_rate(maturity))
            helpers.append(
                YearOnYearInflationSwapHelper(
                    quote=quote,
                    observation_lag=self.observation_lag(),
                    maturity=maturity,
                    calendar=self.calendar(),
                    payment_convention=self._bdc,
                    day_counter=self.day_counter(),
                    index=self._yoy_index,
                    nominal_yts=self._nominal_ts,
                )
            )

        base_yoy_rate = self.atm_yoy_swap_rate(self.reference_date())
        curve = PiecewiseYoYInflationCurve(
            reference_date=ref,
            calendar=self.calendar(),
            day_counter=self.day_counter(),
            observation_lag=self.observation_lag(),
            frequency=self._yoy_index.frequency(),
            base_rate=base_yoy_rate,
            instruments=helpers,
            nominal_yts=self._nominal_ts,
        )
        self._yoy = curve
