"""CPICapFloorTermPriceSurface — abstract CPI cap/floor price surface.

# C++ parity: ql/experimental/inflation/cpicapfloortermpricesurface.{hpp,cpp}
   (v1.42.1) — ``CPICapFloorTermPriceSurface`` abstract base. Concrete
   ``InterpolatedCPICapFloorTermPriceSurface`` lives in
   :mod:`pquantlib.experimental.inflation.interpolated_cpi_cap_floor_term_price_surface`.

Unlike the YoY surface this assumes an ATM zero-inflation term structure
is available on the index (CPI cap/floors have a single flow, so no
stripping is required — ATM comes from the curve and put/call parity
extends the surface across all strikes).

Strikes use the **quoting convention** (yearly-average inflation): the
actual option strike is ``(1+quote)^T``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.indexes.inflation.cpi import (
    InterpolationType,
    is_interpolated,
    lagged_fixing,
)
from pquantlib.indexes.inflation.inflation_index import (
    ZeroInflationIndex,
    inflation_year_fraction,
)
from pquantlib.math.matrix import Matrix
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.term_structure import TermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.frequency import Frequency
from pquantlib.time.period import Period


class CPICapFloorTermPriceSurface(TermStructure):
    """Abstract CPI cap/floor term price surface.

    # C++ parity: ``CPICapFloorTermPriceSurface`` (cpicapfloortermpricesurface.hpp:54).
    """

    def __init__(
        self,
        nominal: float,
        base_rate: float,
        observation_lag: Period,
        calendar: Calendar,
        bdc: BusinessDayConvention,
        day_counter: DayCounter,
        zii: ZeroInflationIndex,
        interpolation_type: InterpolationType,
        yts: YieldTermStructureProtocol,
        c_strikes: Sequence[float],
        f_strikes: Sequence[float],
        cf_maturities: Sequence[Period],
        c_price: Matrix,
        f_price: Matrix,
    ) -> None:
        # # C++ parity: TermStructure(0, cal, dc) — moving mode, 0 settlement days.
        super().__init__(settlement_days=0, calendar=calendar, day_counter=day_counter)
        self._zii: ZeroInflationIndex = zii
        self._interpolation_type: InterpolationType = interpolation_type
        self._nominal_ts: YieldTermStructureProtocol = yts
        self._c_strikes: list[float] = list(c_strikes)
        self._f_strikes: list[float] = list(f_strikes)
        self._cf_maturities: list[Period] = list(cf_maturities)
        self._c_price: Matrix = np.ascontiguousarray(c_price, dtype=np.float64)
        self._f_price: Matrix = np.ascontiguousarray(f_price, dtype=np.float64)
        self._nominal: float = nominal
        self._bdc: BusinessDayConvention = bdc
        self._observation_lag: Period = observation_lag
        self._base_rate: float = base_rate
        self._cf_maturity_times: list[float] = []

        # C++ requires a ZITS on the index + a nominal TS (we take concrete
        # types so the null-handle checks are statically guaranteed).
        qassert.require(
            zii.zero_inflation_term_structure() is not None, "ZITS missing from index"
        )

        f_rows, f_cols = self._f_price.shape
        c_rows, c_cols = self._c_price.shape

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

    # ----- InflationTermStructure-ish interface ---------------------------

    def observation_lag(self) -> Period:
        return self._observation_lag

    def frequency(self) -> Frequency:
        return self._zii.frequency()

    def base_rate(self) -> float:
        return self._base_rate

    def nominal(self) -> float:
        return self._nominal

    def business_day_convention(self) -> BusinessDayConvention:
        return self._bdc

    def zero_inflation_index(self) -> ZeroInflationIndex:
        return self._zii

    def nominal_term_structure(self) -> YieldTermStructureProtocol:
        return self._nominal_ts

    # ----- inspectors -----------------------------------------------------

    def strikes(self) -> list[float]:
        return list(self._cf_strikes)

    def cap_strikes(self) -> list[float]:
        return list(self._c_strikes)

    def floor_strikes(self) -> list[float]:
        return list(self._f_strikes)

    def maturities(self) -> list[Period]:
        return list(self._cf_maturities)

    def cap_prices(self) -> Matrix:
        return self._c_price

    def floor_prices(self) -> Matrix:
        return self._f_price

    def min_strike(self) -> float:
        return self._cf_strikes[0]

    def max_strike(self) -> float:
        return self._cf_strikes[-1]

    def min_date(self) -> Date:
        return self.reference_date() + self._cf_maturities[0]

    def max_date(self) -> Date:
        return self.reference_date() + self._cf_maturities[-1]

    def atm_rate(self, maturity: Date) -> float:
        """ATM yearly-average inflation rate from the index's zero curve.

        # C++ parity: ``CPICapFloorTermPriceSurface::atmRate``
        # (cpicapfloortermpricesurface.cpp:112-122).
        """
        f0 = lagged_fixing(
            self._zii, self.reference_date(), self._observation_lag,
            self._interpolation_type,
        )
        f1 = lagged_fixing(
            self._zii, maturity, self._observation_lag, self._interpolation_type
        )
        t = inflation_year_fraction(
            self._zii.frequency(),
            is_interpolated(self._interpolation_type),
            self.day_counter(),
            self.reference_date() - self._observation_lag,
            maturity - self._observation_lag,
        )
        return (f1 / f0) ** (1.0 / t) - 1.0 if t > 0.0 else self._base_rate

    def cpi_option_date_from_tenor(self, p: Period) -> Date:
        # # C++ parity: cpicapfloortermpricesurface.cpp:124-127.
        return self.calendar().adjust(self.reference_date() + p, self._bdc)

    # ----- abstract price interface (Date forms) --------------------------

    @abstractmethod
    def _price_at_date(self, d: Date, k: float) -> float: ...

    @abstractmethod
    def _cap_price_at_date(self, d: Date, k: float) -> float: ...

    @abstractmethod
    def _floor_price_at_date(self, d: Date, k: float) -> float: ...

    # ----- public Date+Period dispatch ------------------------------------

    def price(self, d: Date | Period, k: float) -> float:
        if isinstance(d, Period):
            return self._price_at_date(self.cpi_option_date_from_tenor(d), k)
        return self._price_at_date(d, k)

    def cap_price(self, d: Date | Period, k: float) -> float:
        if isinstance(d, Period):
            return self._cap_price_at_date(self.cpi_option_date_from_tenor(d), k)
        return self._cap_price_at_date(d, k)

    def floor_price(self, d: Date | Period, k: float) -> float:
        if isinstance(d, Period):
            return self._floor_price_at_date(self.cpi_option_date_from_tenor(d), k)
        return self._floor_price_at_date(d, k)
