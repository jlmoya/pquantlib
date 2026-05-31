"""AbcdAtmVolCurve — Abcd-interpolated ATM (no-smile) IR vol curve.

# C++ parity: ql/experimental/volatility/abcdatmvolcurve.{hpp,cpp} (v1.42.1).

Concrete :class:`BlackAtmVolCurve` whose ATM volatilities are fitted by
the Rebonato (a, b, c, d) parametric form via
:class:`~pquantlib.math.interpolations.abcd_calibration.AbcdCalibration`.
The fit may *exclude* selected option tenors (``inclusion_in_interpolation``)
while still exposing their dates/times.

The ATM vol at time ``t`` is the *k-adjusted* abcd value:
``atm_vol(t) = k(t) * abcd(t)`` where ``abcd(t) = (a + b t) e^{-c t} + d``
and ``k(t)`` is the linear interpolation of the per-knot adjustment
factors ``k[i] = vol_market[i] / abcd(t_i)``. Because ``k(t_i) =
vol_market[i] / abcd(t_i)``, the fit reproduces the input ATM vols
*exactly* at the (included) option times — the abcd form only governs
the shape *between* and *beyond* the knots.

Divergences from C++:

* **No ``Handle`` wrapper.** Quotes are held directly and the curve
  registers as their observer.
* **LazyObject via an internal cache** rather than multiple-inheriting
  ``LazyObject``: ``_calculate()`` refits on first use after an
  ``update()`` invalidation. The public surface (``a/b/c/d``, ``k``,
  ``atm_vol*``) triggers the fit transparently.
* **Optimizer.** The underlying :class:`AbcdCalibration` uses
  ``scipy.optimize.least_squares`` (see its module docstring); recovered
  (a, b, c, d) match the C++ probe at LOOSE tier.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.volatility.black_atm_vol_curve import BlackAtmVolCurve
from pquantlib.math.interpolations.abcd_calibration import AbcdCalibration
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.quotes.quote import Quote
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period


class AbcdAtmVolCurve(BlackAtmVolCurve):
    """Abcd-parametric ATM (no-smile) interest-rate volatility curve."""

    def __init__(
        self,
        settlement_days: int,
        calendar: Calendar,
        option_tenors: Sequence[Period],
        vol_handles: Sequence[Quote],
        inclusion_in_interpolation: Sequence[bool] | None = None,
        business_day_convention: BusinessDayConvention = BusinessDayConvention.Following,
        day_counter: DayCounter | None = None,
    ) -> None:
        dc = day_counter if day_counter is not None else Actual365Fixed()
        super().__init__(
            business_day_convention=business_day_convention,
            calendar=calendar,
            day_counter=dc,
            settlement_days=settlement_days,
        )
        self._n_option_tenors: int = len(option_tenors)
        self._option_tenors: list[Period] = list(option_tenors)
        self._vol_handles: list[Quote] = list(vol_handles)
        flags = (
            [True] if inclusion_in_interpolation is None
            else list(inclusion_in_interpolation)
        )
        self._inclusion: list[bool] = flags

        self._check_inputs()
        # Cache populated by _calculate().
        self._calibration: AbcdCalibration | None = None
        self._k_interp: LinearInterpolation | None = None
        self._calculated: bool = False
        self._option_dates: list[Date] = []
        self._option_times: list[float] = []
        self._actual_option_times: list[float] = []
        self._actual_option_tenors: list[Period] = []

        self._initialize_option_dates_and_times()
        # Register with each quote so the curve refreshes on changes.
        for q in self._vol_handles:
            q.register_with(self)

    # --- input validation --------------------------------------------------

    def _check_inputs(self) -> None:
        qassert.require(self._n_option_tenors > 0, "empty option tenor vector")
        qassert.require(
            self._n_option_tenors == len(self._vol_handles),
            f"mismatch between number of option tenors ({self._n_option_tenors}) "
            f"and number of volatilities ({len(self._vol_handles)})",
        )
        zero = Period(0, self._option_tenors[0].units)
        qassert.require(
            self._option_tenors[0] > zero,
            f"negative first option tenor: {self._option_tenors[0]}",
        )
        for i in range(1, self._n_option_tenors):
            qassert.require(
                self._option_tenors[i] > self._option_tenors[i - 1],
                f"non increasing option tenor: {i} is {self._option_tenors[i - 1]}, "
                f"{i + 1} is {self._option_tenors[i]}",
            )
        if len(self._inclusion) == 1:
            self._inclusion = [self._inclusion[0]] * self._n_option_tenors
        else:
            qassert.require(
                self._n_option_tenors == len(self._inclusion),
                f"mismatch between number of option tenors ({self._n_option_tenors}) "
                f"and number of inclusion's flags ({len(self._inclusion)})",
            )

    def _initialize_option_dates_and_times(self) -> None:
        self._option_dates = []
        self._option_times = []
        self._actual_option_times = []
        self._actual_option_tenors = []
        ref = self.reference_date()
        for i in range(self._n_option_tenors):
            d = self.option_date_from_tenor(self._option_tenors[i])
            t = self.day_counter().year_fraction(ref, d)
            self._option_dates.append(d)
            self._option_times.append(t)
            if self._inclusion[i]:
                self._actual_option_times.append(t)
                self._actual_option_tenors.append(self._option_tenors[i])

    # --- lazy fit ----------------------------------------------------------

    def _calculate(self) -> None:
        if self._calculated:
            return
        actual_vols: list[float] = [
            self._vol_handles[i].value()
            for i in range(self._n_option_tenors)
            if self._inclusion[i]
        ]
        calib = AbcdCalibration(self._actual_option_times, actual_vols)
        calib.compute()
        self._calibration = calib
        # k(t) is the linear interpolation of the per-knot adjustment
        # factors ``vol[i] / abcd(t_i)``.
        k_values = calib.k(self._actual_option_times, actual_vols)
        self._k_interp = LinearInterpolation(
            np.asarray(self._actual_option_times, dtype=np.float64),
            np.asarray(k_values, dtype=np.float64),
        )
        self._calculated = True

    def update(self) -> None:
        """Observer.update — recompute dates if moving, invalidate fit cache.

        # C++ parity: AbcdAtmVolCurve::update + BlackAtmVolCurve::update +
        # LazyObject::update.
        """
        super().update()  # invalidates the moving reference-date cache.
        if self._moving:
            self._initialize_option_dates_and_times()
        self._calculated = False

    # --- fitted-parameter inspectors ---------------------------------------

    def _calib(self) -> AbcdCalibration:
        self._calculate()
        assert self._calibration is not None
        return self._calibration

    def a(self) -> float:
        return self._calib().a()

    def b(self) -> float:
        return self._calib().b()

    def c(self) -> float:
        return self._calib().c()

    def d(self) -> float:
        return self._calib().d()

    def rms_error(self) -> float:
        return self._calib().error()

    def max_error(self) -> float:
        return self._calib().max_error()

    def end_criteria(self) -> str:
        return self._calib().end_criteria()

    def k_values(self) -> list[float]:
        """Per-knot k adjustment factors (one per included option time)."""
        self._calculate()
        actual_vols = [
            self._vol_handles[i].value()
            for i in range(self._n_option_tenors)
            if self._inclusion[i]
        ]
        return self._calib().k(self._actual_option_times, actual_vols)

    def k_at_time(self, t: float) -> float:
        """k adjustment factor at time ``t`` (linear interp of the knots)."""
        self._calculate()
        assert self._k_interp is not None
        return self._k_interp(t, allow_extrapolation=True)

    # --- inspectors --------------------------------------------------------

    def option_tenors(self) -> list[Period]:
        return list(self._option_tenors)

    def option_tenors_in_interpolation(self) -> list[Period]:
        self._calculate()
        return list(self._actual_option_tenors)

    def option_dates(self) -> list[Date]:
        return list(self._option_dates)

    def option_times(self) -> list[float]:
        return list(self._option_times)

    # --- TermStructure / VolatilityTermStructure ---------------------------

    def max_date(self) -> Date:
        self._calculate()
        return self.option_date_from_tenor(self._option_tenors[-1])

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    # --- BlackAtmVolCurve hooks --------------------------------------------

    def _atm_vol_impl(self, t: float) -> float:
        self._calculate()
        assert self._calibration is not None
        return self.k_at_time(t) * self._calibration.value(t)

    def _atm_variance_impl(self, t: float) -> float:
        vol = self._atm_vol_impl(t)
        return vol * vol * t


__all__ = ["AbcdAtmVolCurve"]
