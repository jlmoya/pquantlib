"""SabrVolSurface — interpolated SABR vol surface from market vol spreads.

# C++ parity: ql/experimental/volatility/sabrvolsurface.{hpp,cpp} (v1.42.1).

An :class:`InterestRateVolSurface` whose per-expiry smile is a SABR fit
to market data expressed as *spreads*:

- ``atm_rate_spreads`` — strike offsets relative to the ATM forward
  (one per smile column).
- ``vol_spreads[tenor][col]`` — additive volatility spreads over the ATM
  vol (one row per reference option tenor, one column per
  ``atm_rate_spread``).

For a requested option time ``t`` (rounded to a date), the surface:

1. linearly interpolates each strike-column's vol spread across the
   reference option tenors;
2. forms the absolute smile slice
   ``strikes[i] = forward + atm_rate_spread[i]`` and
   ``vols[i] = atm_curve.atm_vol(d) + vol_spread[i]``;
3. fits a :class:`SabrInterpolatedSmileSection` to that slice.

The SABR fit uses the C++ hard-coded initial guesses (alpha=0.025,
beta=0.5, rho=0.3, nu=0.0) with all four parameters free and
vega-weighted residuals.

Divergences from C++:

* **No ``Handle`` wrapper.** The ATM curve and vol-spread quotes are
  held directly; the surface registers as their observer.
* **Spread -> absolute conversion.** C++ ``SabrInterpolatedSmileSection``
  has a ``hasFloatingStrikes`` mode that adds ``forward`` /
  ``atmVol`` internally; PQuantLib's port keeps absolute strikes, so we
  perform the ``forward + spread`` / ``atm_vol + vol_spread`` conversion
  here before fitting.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.volatility.black_atm_vol_curve import BlackAtmVolCurve
from pquantlib.experimental.volatility.interest_rate_vol_surface import (
    InterestRateVolSurface,
)
from pquantlib.indexes.interest_rate_index import InterestRateIndex
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.volatility.sabr_interpolated_smile_section import (
    SabrInterpolatedSmileSection,
)
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit

# Hard-coded SABR initial guesses (C++ SabrVolSurface ctor).
#
# C++ stores ``sabrGuesses_[i] = {0.025 /*alpha*/, 0.5 /*beta*/,
# 0.3 /*rho*/, 0.0 /*nu*/}`` but passes them *positionally* to the
# SabrInterpolatedSmileSection ``(alpha, beta, nu, rho)`` ctor — so the
# 0.3 ("rho"-labelled) guess actually seeds ``nu`` and the 0.0
# ("nu"-labelled) guess seeds ``rho``. Since all four parameters are
# free in the fit, the labelling is immaterial to the result; we keep
# the C++ positional mapping faithfully: guesses = [alpha, beta, nu, rho]
# = [0.025, 0.5, 0.3, 0.0].
_ALPHA_GUESS = 0.025
_BETA_GUESS = 0.5
_NU_GUESS = 0.3
_RHO_GUESS = 0.0


class SabrVolSurface(InterestRateVolSurface):
    """SABR volatility (smile) surface interpolated from market vol spreads."""

    def __init__(
        self,
        index: InterestRateIndex,
        atm_curve: BlackAtmVolCurve,
        option_tenors: Sequence[Period],
        atm_rate_spreads: Sequence[float],
        vol_spreads: Sequence[Sequence[Quote]],
    ) -> None:
        super().__init__(index, business_day_convention=BusinessDayConvention.Following)
        self._atm_curve: BlackAtmVolCurve = atm_curve
        self._option_tenors: list[Period] = list(option_tenors)
        self._atm_rate_spreads: list[float] = [float(s) for s in atm_rate_spreads]
        self._vol_spreads: list[list[Quote]] = [list(row) for row in vol_spreads]

        self._check_inputs()

        # Hard-coded fit configuration (C++).
        self._is_alpha_fixed: bool = False
        self._is_beta_fixed: bool = False
        self._is_nu_fixed: bool = False
        self._is_rho_fixed: bool = False
        self._vega_weighted: bool = True

        self._option_dates: list[Date] = []
        self._option_times: list[float] = []
        self._sabr_guesses: list[list[float]] = []
        for p in self._option_tenors:
            d = self.option_date_from_tenor(p)
            self._option_dates.append(d)
            self._option_times.append(self.time_from_reference(d))
            # guesses = [alpha, beta, nu, rho] (C++ positional order).
            self._sabr_guesses.append(
                [_ALPHA_GUESS, _BETA_GUESS, _NU_GUESS, _RHO_GUESS]
            )

        self._register_with_market_data()

    # --- delegated TermStructure surface (the ATM curve is the anchor) -----

    def day_counter(self) -> DayCounter:
        return self._atm_curve.day_counter()

    def reference_date(self) -> Date:
        return self._atm_curve.reference_date()

    def calendar(self) -> Calendar:
        return self._atm_curve.calendar()

    def settlement_days(self) -> int:
        return self._atm_curve.settlement_days()

    def max_date(self) -> Date:
        return self._atm_curve.max_date()

    def max_time(self) -> float:
        return self._atm_curve.max_time()

    def min_strike(self) -> float:
        return -math.inf

    def max_strike(self) -> float:
        return math.inf

    def atm_curve(self) -> BlackAtmVolCurve:
        return self._atm_curve

    # --- input validation --------------------------------------------------

    def _check_inputs(self) -> None:
        n_strikes = len(self._atm_rate_spreads)
        qassert.require(n_strikes > 1, f"too few strikes ({n_strikes})")
        for i in range(1, n_strikes):
            qassert.require(
                self._atm_rate_spreads[i - 1] < self._atm_rate_spreads[i],
                f"non increasing strike spreads: {i} is "
                f"{self._atm_rate_spreads[i - 1]}, {i + 1} is "
                f"{self._atm_rate_spreads[i]}",
            )
        for i, row in enumerate(self._vol_spreads):
            qassert.require(
                len(self._atm_rate_spreads) == len(row),
                f"mismatch between number of strikes ({len(self._atm_rate_spreads)}) "
                f"and number of columns ({len(row)}) in the {i + 1} row",
            )

    def _register_with_market_data(self) -> None:
        self._atm_curve.register_with(self)
        for row in self._vol_spreads:
            for q in row:
                q.register_with(self)

    # --- smile spreads -----------------------------------------------------

    def volatility_spreads_at_date(self, d: Date) -> list[float]:
        """Vol spreads at date ``d``, linearly interpolated across tenors.

        # C++ parity: SabrVolSurface::volatilitySpreads(const Date&).
        """
        n_option_times = len(self._option_times)
        n_atm_rate_spreads = len(self._atm_rate_spreads)
        interpolated: list[float] = [0.0] * n_atm_rate_spreads
        target_t = self.time_from_reference(d)
        times_arr = np.asarray(self._option_times, dtype=np.float64)
        for i in range(n_atm_rate_spreads):
            col = np.asarray(
                [self._vol_spreads[j][i].value() for j in range(n_option_times)],
                dtype=np.float64,
            )
            interp = LinearInterpolation(times_arr, col)
            interpolated[i] = interp(target_t, allow_extrapolation=True)
        return interpolated

    def volatility_spreads(self, p: Period) -> list[float]:
        return self.volatility_spreads_at_date(self.option_date_from_tenor(p))

    def _sabr_guesses_at_date(self, d: Date) -> list[float]:
        # piecewise-constant guesses (C++ SabrVolSurface::sabrGuesses).
        if d <= self._option_dates[0]:
            return self._sabr_guesses[0]
        i = 0
        while i < len(self._option_dates) - 1 and d < self._option_dates[i]:
            i += 1
        return self._sabr_guesses[i]

    # --- update ------------------------------------------------------------

    def update(self) -> None:
        """Observer.update — recompute option dates/times, propagate.

        # C++ parity: SabrVolSurface::update.
        """
        for i, p in enumerate(self._option_tenors):
            d = self.option_date_from_tenor(p)
            self._option_dates[i] = d
            self._option_times[i] = self.time_from_reference(d)
        self.notify_observers()

    # --- BlackVolSurface hook ----------------------------------------------

    def _smile_section_impl(self, t: float) -> SmileSection:
        # C++ rounds the time to a whole-day date offset from the
        # reference date: ``Date d = referenceDate() + int(t*365)*Days``.
        n = int(t * 365.0)
        d = self.reference_date() + Period(n, TimeUnit.Days)

        vol_spreads = self.volatility_spreads_at_date(d)
        forward = self._index.fixing(d, True)
        atm_vol = self._atm_curve.atm_vol_at_date(d, True)

        # Convert ATM-relative spreads to an absolute smile slice.
        strikes = [forward + s for s in self._atm_rate_spreads]
        vols = [atm_vol + vs for vs in vol_spreads]

        # guesses = [alpha, beta, nu, rho] (C++ positional order).
        guesses = self._sabr_guesses_at_date(d)
        return SabrInterpolatedSmileSection(
            option_date=d,
            forward=forward,
            strikes=strikes,
            vols=vols,
            alpha=guesses[0],
            beta=guesses[1],
            nu=guesses[2],
            rho=guesses[3],
            alpha_is_fixed=self._is_alpha_fixed,
            beta_is_fixed=self._is_beta_fixed,
            nu_is_fixed=self._is_nu_fixed,
            rho_is_fixed=self._is_rho_fixed,
            vega_weighted=self._vega_weighted,
            day_counter=self.day_counter(),
        )


__all__ = ["SabrVolSurface"]
