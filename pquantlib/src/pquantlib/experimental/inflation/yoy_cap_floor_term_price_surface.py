"""YoYCapFloorTermPriceSurface — abstract YoY cap/floor price surface.

# C++ parity: ql/experimental/inflation/yoycapfloortermpricesurface.{hpp,cpp}
   (v1.42.1) — ``YoYCapFloorTermPriceSurface`` abstract base. The concrete
   ``InterpolatedYoYCapFloorTermPriceSurface`` lives in
   :mod:`pquantlib.experimental.inflation.interpolated_yoy_cap_floor_term_price_surface`.

The abstract base holds the market cap/floor price grids (strikes x
maturities), validates monotonicity, builds the combined strike set
``cfStrikes`` (floors then non-overlapping caps), and dispatches the
``Period``-tenor overloads to the ``Date`` ones (which concretes
implement).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import YoYInflationIndex
from pquantlib.math.matrix import Matrix
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.term_structure import TermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period


def _index_is_interpolated(interp: InterpolationType, index: YoYInflationIndex) -> bool:
    """C++ ``detail::CPI::isInterpolated(type, index)``.

    True iff the effective interpolation resolves to ``Linear``: i.e.
    ``type == Linear`` or (``type == AsIndex`` and the index is itself
    interpolated).
    """
    if interp == InterpolationType.Linear:
        return True
    if interp == InterpolationType.AsIndex:
        return index.interpolated()
    return False


class YoYCapFloorTermPriceSurface(TermStructure):
    """Abstract YoY cap/floor term price surface.

    # C++ parity: ``YoYCapFloorTermPriceSurface`` (yoycapfloortermpricesurface.hpp:42).
    """

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
    ) -> None:
        # # C++ parity: TermStructure(0, cal, dc) — moving mode, 0 settlement days.
        super().__init__(settlement_days=0, calendar=calendar, day_counter=day_counter)
        self._fixing_days: int = fixing_days
        self._bdc: BusinessDayConvention = bdc
        self._yoy_index: YoYInflationIndex = yii
        self._observation_lag: Period = yy_lag
        self._nominal_ts: YieldTermStructureProtocol = nominal
        self._c_strikes: list[float] = list(c_strikes)
        self._f_strikes: list[float] = list(f_strikes)
        self._cf_maturities: list[Period] = list(cf_maturities)
        self._c_price: Matrix = np.ascontiguousarray(c_price, dtype=np.float64)
        self._f_price: Matrix = np.ascontiguousarray(f_price, dtype=np.float64)
        self._index_is_interpolated: bool = _index_is_interpolated(interpolation, yii)
        self._cf_maturity_times: list[float] = []

        f_rows, f_cols = self._f_price.shape
        c_rows, c_cols = self._c_price.shape

        # data consistency checking
        qassert.require(len(self._f_strikes) > 1, "not enough floor strikes")
        qassert.require(len(self._c_strikes) > 1, "not enough cap strikes")
        qassert.require(len(self._cf_maturities) > 1, "not enough maturities")
        qassert.require(
            len(self._f_strikes) == f_rows, "floor strikes vs floor price rows not equal"
        )
        qassert.require(
            len(self._c_strikes) == c_rows, "cap strikes vs cap price rows not equal"
        )
        qassert.require(
            len(self._cf_maturities) == f_cols,
            "maturities vs floor price columns not equal",
        )
        qassert.require(
            len(self._cf_maturities) == c_cols,
            "maturities vs cap price columns not equal",
        )

        # data has correct properties (positive, monotonic)?
        zero = Period(0, self._cf_maturities[0].units)
        for j in range(len(self._cf_maturities)):
            qassert.require(self._cf_maturities[j] > zero, "non-positive maturities")
            if j > 0:
                qassert.require(
                    self._cf_maturities[j] > self._cf_maturities[j - 1],
                    "non-increasing maturities",
                )
            for i in range(f_rows):
                qassert.require(
                    self._f_price[i, j] > 0.0,
                    f"non-positive floor price: {self._f_price[i, j]}",
                )
                if i > 0:
                    qassert.require(
                        self._f_price[i, j] >= self._f_price[i - 1, j],
                        "non-increasing floor prices",
                    )
            for i in range(c_rows):
                qassert.require(
                    self._c_price[i, j] > 0.0,
                    f"non-positive cap price: {self._c_price[i, j]}",
                )
                if i > 0:
                    qassert.require(
                        self._c_price[i, j] <= self._c_price[i - 1, j],
                        "non-decreasing cap prices",
                    )

        # Combined strike set: all floor strikes, then non-overlapping caps.
        self._cf_strikes: list[float] = list(self._f_strikes)
        eps = 0.0000001
        max_f_strike = self._f_strikes[-1]
        for k in self._c_strikes:
            if k > max_f_strike + eps:
                self._cf_strikes.append(k)

        qassert.require(len(self._cf_strikes) > 2, "overall not enough strikes")
        for i in range(1, len(self._cf_strikes)):
            qassert.require(
                self._cf_strikes[i] > self._cf_strikes[i - 1],
                "cfStrikes not increasing",
            )

    # ----- inspectors -----------------------------------------------------

    def index_is_interpolated(self) -> bool:
        return self._index_is_interpolated

    def observation_lag(self) -> Period:
        return self._observation_lag

    def frequency(self) -> Frequency:
        return self._yoy_index.frequency()

    def yoy_index(self) -> YoYInflationIndex:
        return self._yoy_index

    def business_day_convention(self) -> BusinessDayConvention:
        return self._bdc

    def fixing_days(self) -> int:
        return self._fixing_days

    def nominal_term_structure(self) -> YieldTermStructureProtocol:
        return self._nominal_ts

    def strikes(self) -> list[float]:
        return list(self._cf_strikes)

    def cap_strikes(self) -> list[float]:
        return list(self._c_strikes)

    def floor_strikes(self) -> list[float]:
        return list(self._f_strikes)

    def maturities(self) -> list[Period]:
        return list(self._cf_maturities)

    def min_strike(self) -> float:
        return self._cf_strikes[0]

    def max_strike(self) -> float:
        return self._cf_strikes[-1]

    def min_maturity(self) -> Date:
        return self.reference_date() + self._cf_maturities[0]

    def max_maturity(self) -> Date:
        return self.reference_date() + self._cf_maturities[-1]

    def yoy_option_date_from_tenor(self, p: Period) -> Date:
        # # C++ parity: yoycapfloortermpricesurface.cpp:103-106.
        return self.reference_date() + p

    # ----- abstract surface interface (Date forms) ------------------------

    @abstractmethod
    def base_date(self) -> Date: ...

    @abstractmethod
    def yoy_ts(self) -> object: ...

    @abstractmethod
    def atm_yoy_swap_time_rates(self) -> tuple[list[float], list[float]]: ...

    @abstractmethod
    def atm_yoy_swap_date_rates(self) -> tuple[list[Date], list[float]]: ...

    @abstractmethod
    def _price_at_date(self, d: Date, k: float) -> float: ...

    @abstractmethod
    def _cap_price_at_date(self, d: Date, k: float) -> float: ...

    @abstractmethod
    def _floor_price_at_date(self, d: Date, k: float) -> float: ...

    @abstractmethod
    def _atm_yoy_swap_rate_at_date(self, d: Date, extrapolate: bool = True) -> float: ...

    @abstractmethod
    def _atm_yoy_rate_at_date(
        self, d: Date, obs_lag: Period | None = None, extrapolate: bool = True
    ) -> float: ...

    # ----- public Date+Period dispatch ------------------------------------

    def price(self, d: Date | Period, k: float) -> float:
        if isinstance(d, Period):
            return self._price_at_date(self.yoy_option_date_from_tenor(d), k)
        return self._price_at_date(d, k)

    def cap_price(self, d: Date | Period, k: float) -> float:
        if isinstance(d, Period):
            return self._cap_price_at_date(self.yoy_option_date_from_tenor(d), k)
        return self._cap_price_at_date(d, k)

    def floor_price(self, d: Date | Period, k: float) -> float:
        if isinstance(d, Period):
            return self._floor_price_at_date(self.yoy_option_date_from_tenor(d), k)
        return self._floor_price_at_date(d, k)

    def atm_yoy_swap_rate(self, d: Date | Period, extrapolate: bool = True) -> float:
        if isinstance(d, Period):
            return self._atm_yoy_swap_rate_at_date(
                self.yoy_option_date_from_tenor(d), extrapolate
            )
        return self._atm_yoy_swap_rate_at_date(d, extrapolate)

    def atm_yoy_rate(
        self,
        d: Date | Period,
        obs_lag: Period | None = None,
        extrapolate: bool = True,
    ) -> float:
        if isinstance(d, Period):
            return self._atm_yoy_rate_at_date(
                self.yoy_option_date_from_tenor(d), obs_lag, extrapolate
            )
        return self._atm_yoy_rate_at_date(d, obs_lag, extrapolate)
