"""InterpolatedCPICapFloorTermPriceSurface — concrete CPI price surface.

# C++ parity: ql/experimental/inflation/cpicapfloortermpricesurface.hpp
   (v1.42.1) — ``InterpolatedCPICapFloorTermPriceSurface<Interpolator2D>``.

Completes the cap/floor price matrices across **all** combined strikes by
put/call parity (CPI options have a single flow, so parity holds exactly
given the discounted forward), then fits a 2-D interpolation (default
:class:`BilinearInterpolation`) over ``(maturity_time, strike)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.inflation.cpi_cap_floor_term_price_surface import (
    CPICapFloorTermPriceSurface,
)
from pquantlib.indexes.inflation.cpi import InterpolationType
from pquantlib.indexes.inflation.inflation_index import ZeroInflationIndex
from pquantlib.math.closeness import close
from pquantlib.math.interpolations.bilinear import BilinearInterpolation
from pquantlib.math.matrix import Matrix
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class Surface2D(Protocol):
    """Structural type for a 2-D interpolation usable as a price surface.

    Both :class:`BilinearInterpolation` (concrete-only) and the
    :class:`Interpolation2D` hierarchy (e.g. :class:`BicubicSpline`,
    :class:`Polynomial2DSpline`) satisfy this — they share the same
    ``__call__`` / ``enable_extrapolation`` surface but do not share a base.
    """

    def __call__(
        self, x: float, y: float, *, allow_extrapolation: bool = ...
    ) -> float: ...

    def enable_extrapolation(self, b: bool = ...) -> None: ...


# C++ ``interpolator2d_.interpolate(xs, ys, z)`` -> factory taking the x
# grid (maturity times), y grid (strikes) and the z matrix (z[y, x]).
Interpolator2DFactory = Callable[[Matrix, Matrix, Matrix], Surface2D]


class InterpolatedCPICapFloorTermPriceSurface(CPICapFloorTermPriceSurface):
    """Bilinear-by-default CPI cap/floor price surface."""

    def __init__(
        self,
        nominal: float,
        start_rate: float,
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
        interpolator2d: Interpolator2DFactory = BilinearInterpolation,
    ) -> None:
        super().__init__(
            nominal, start_rate, observation_lag, calendar, bdc, day_counter,
            zii, interpolation_type, yts, c_strikes, f_strikes, cf_maturities,
            c_price, f_price,
        )
        self._interpolator2d: Interpolator2DFactory = interpolator2d
        self._cap_price: Surface2D | None = None
        self._floor_price: Surface2D | None = None
        self._perform_calculations()

    def _perform_calculations(self) -> None:
        # # C++ parity: InterpolatedCPICapFloorTermPriceSurface::performCalculations
        # # (cpicapfloortermpricesurface.hpp:233-303).
        n_k = len(self._cf_strikes)
        n_m = len(self._cf_maturities)
        # B matrices are [strike, maturity] (rows=strike, cols=maturity).
        c_b = np.full((n_k, n_m), np.nan, dtype=np.float64)
        f_b = np.full((n_k, n_m), np.nan, dtype=np.float64)

        for j in range(n_m):
            mat = self._cf_maturities[j]
            opt_date = self.cpi_option_date_from_tenor(mat)
            df = self._nominal_ts.discount(opt_date)
            atm_quote = self.atm_rate(opt_date)
            atm = (1.0 + atm_quote) ** mat.length
            s = atm * df
            for i in range(n_k):
                k_quote = self._cf_strikes[i]
                k = (1.0 + k_quote) ** mat.length
                # Locate this combined strike in the floor/cap input strikes.
                ind_f = _find_close(self._f_strikes, k_quote)
                ind_c = _find_close(self._c_strikes, k_quote)
                is_floor = ind_f is not None
                is_cap = ind_c is not None
                if is_floor:
                    assert ind_f is not None
                    f_b[i, j] = self._f_price[ind_f, j]
                    if not is_cap:
                        c_b[i, j] = self._f_price[ind_f, j] + s - k * df
                if is_cap:
                    assert ind_c is not None
                    c_b[i, j] = self._c_price[ind_c, j]
                    if not is_floor:
                        f_b[i, j] = self._c_price[ind_c, j] + k * df - s

        qassert.require(
            not np.isnan(c_b).any(), "did not fill call price matrix (unexpected)"
        )
        qassert.require(
            not np.isnan(f_b).any(), "did not fill floor price matrix (unexpected)"
        )

        self._cf_maturity_times = [
            self.time_from_reference(self.cpi_option_date_from_tenor(m))
            for m in self._cf_maturities
        ]
        xs = np.asarray(self._cf_maturity_times, dtype=np.float64)
        ys = np.asarray(self._cf_strikes, dtype=np.float64)
        self._cap_price = self._interpolator2d(xs, ys, c_b)
        self._cap_price.enable_extrapolation()
        self._floor_price = self._interpolator2d(xs, ys, f_b)
        self._floor_price.enable_extrapolation()

    # ----- price interface (strike uses quoting convention) ---------------

    def _price_at_date(self, d: Date, k: float) -> float:
        atm = self.atm_rate(d)
        return self._cap_price_at_date(d, k) if k > atm else self._floor_price_at_date(d, k)

    def _cap_price_at_date(self, d: Date, k: float) -> float:
        assert self._cap_price is not None
        t = self.time_from_reference(d)
        return self._cap_price(t, k, allow_extrapolation=True)

    def _floor_price_at_date(self, d: Date, k: float) -> float:
        assert self._floor_price is not None
        t = self.time_from_reference(d)
        return self._floor_price(t, k, allow_extrapolation=True)


def _find_close(strikes: Sequence[float], k: float) -> int | None:
    """Index of the first strike within tolerance of ``k`` (else None).

    # C++ parity: ``std::find_if(..., close_enough(x, k))``.
    """
    for i, s in enumerate(strikes):
        if close(s, k):
            return i
    return None
