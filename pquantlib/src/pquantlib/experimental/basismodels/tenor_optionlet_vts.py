"""TenorOptionletVTS — tenor-rescaled optionlet (caplet) volatility term structure.

# C++ parity: ql/experimental/basismodels/tenoroptionletvts.hpp + .cpp (v1.42.1,
# 099987f0).

Transforms a *base*-tenor caplet normal-vol surface into a *target*-tenor
surface: a target-tenor caplet is decomposed into a strip of base-tenor FRAs
(``baseFreq % targFreq == 0``), and the target vol is the correlation-weighted
aggregation of the base-tenor vols. Designed for normal volatilities.

# C++ parity divergence (smile section): the C++ ``TenorOptionletSmileSection``
# queries the base structure's *smile section* via
# ``volatility(strike, Normal, 0.0)``. PQuantLib's OptionletVolatilityStructure
# does not expose a per-date smile section, so the (Normal-typed) base vol is
# queried directly via ``baseVTS.volatility(fixingDate, strike)`` — identical to
# the C++ same-type smile-section short-circuit.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from pquantlib import qassert
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.termstructures.volatility.optionlet.optionlet_volatility_structure import (
    OptionletVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


class CorrelationStructure(ABC):
    """Functor: correlation between two FRA rates starting at ``start1``/``start2``.

    # C++ parity: ``TenorOptionletVTS::CorrelationStructure``.
    """

    @abstractmethod
    def __call__(self, start1: float, start2: float) -> float: ...


class TwoParameterCorrelation(CorrelationStructure):
    """rho = rhoInf + (1 - rhoInf) * exp(-beta * |start2 - start1|).

    # C++ parity: ``TenorOptionletVTS::TwoParameterCorrelation``.
    """

    def __init__(self, rho_inf: Interpolation, beta: Interpolation) -> None:
        self._rho_inf: Interpolation = rho_inf
        self._beta: Interpolation = beta

    def __call__(self, start1: float, start2: float) -> float:
        rho_inf = self._rho_inf(start1, allow_extrapolation=True)
        beta = self._beta(start1, allow_extrapolation=True)
        return rho_inf + (1.0 - rho_inf) * math.exp(-beta * abs(start2 - start1))


class TenorOptionletVTS(OptionletVolatilityStructure):
    """Tenor-rescaled optionlet volatility term structure (normal vols)."""

    def __init__(
        self,
        base_vts: OptionletVolatilityStructure,
        base_index: IborIndex,
        targ_index: IborIndex,
        correlation: CorrelationStructure,
    ) -> None:
        super().__init__(
            business_day_convention=base_vts.business_day_convention(),
            reference_date=base_vts.reference_date(),
            calendar=base_vts.calendar(),
            day_counter=base_vts.day_counter(),
        )
        # # C++ parity: baseFreq % targFreq == 0 (tenoroptionletvts.cpp:46-47).
        qassert.require(
            base_index.tenor().frequency() % targ_index.tenor().frequency() == 0,
            "Base index frequency must be a multiple of target tenor frequency",
        )
        self._base_vts: OptionletVolatilityStructure = base_vts
        self._base_index: IborIndex = base_index
        self._targ_index: IborIndex = targ_index
        self._correlation: CorrelationStructure = correlation

    # --- TermStructure interface ----------------------------------------------

    def max_date(self) -> Date:
        return self._base_vts.max_date()

    def min_strike(self) -> float:
        return self._base_vts.min_strike()

    def max_strike(self) -> float:
        return self._base_vts.max_strike()

    def volatility_type(self) -> VolatilityType:
        return VolatilityType.Normal

    # --- transformation --------------------------------------------------------

    def _volatility_impl(self, t: float, strike: float) -> float:
        """# C++ parity: TenorOptionletVTS::volatilityImpl via the smile section."""
        return self._section_volatility(t, strike)

    def _section_volatility(self, option_time: float, strike: float) -> float:
        """Build the base-tenor FRA strip and aggregate its vols.

        # C++ parity: TenorOptionletSmileSection ctor + volatilityImpl
        # (tenoroptionletvts.cpp:52-91).
        """
        base_index = self._base_index
        targ_index = self._targ_index

        one_day_as_year = self.day_counter().year_fraction(
            self.reference_date(), self.reference_date() + 1
        )
        exercise_date = self.reference_date() + round(option_time / one_day_as_year)
        effective_date = base_index.fixing_calendar().advance(
            exercise_date, base_index.fixing_days(), TimeUnit.Days
        )
        maturity_date = base_index.fixing_calendar().advance_period(
            effective_date, targ_index.tenor(), BusinessDayConvention.Unadjusted, False
        )
        base_float_schedule = Schedule.from_rule(
            effective_date,
            maturity_date,
            base_index.tenor(),
            base_index.fixing_calendar(),
            BusinessDayConvention.ModifiedFollowing,
            BusinessDayConvention.Unadjusted,
            DateGeneration.Backward,
            False,
        )

        fra_rate_targ = targ_index.fixing(exercise_date)
        yf_targ = targ_index.day_counter().year_fraction(effective_date, maturity_date)

        dates = base_float_schedule.dates
        start_time_base: list[float] = []
        fra_rate_base: list[float] = []
        v: list[float] = []
        fixing_dates: list[Date] = []
        for k in range(len(dates) - 1):
            start_date = dates[k]
            fixing_date = base_index.fixing_calendar().advance(
                start_date, -base_index.fixing_days(), TimeUnit.Days
            )
            year_frac = base_index.day_counter().year_fraction(dates[k], dates[k + 1])
            start_time_base.append(
                self.day_counter().year_fraction(self.reference_date(), start_date)
            )
            fr_base = base_index.fixing(fixing_date)
            fra_rate_base.append(fr_base)
            fixing_dates.append(fixing_date)
            v.append(
                year_frac / yf_targ * (1.0 + yf_targ * fra_rate_targ) / (1.0 + year_frac * fr_base)
            )

        sum_v = sum(v)
        # base vols at the transformed strikes.
        vol_base: list[float] = []
        for k in range(len(fra_rate_base)):
            strike_k = (strike - (fra_rate_targ - sum_v * fra_rate_base[k])) / sum_v
            vol_base.append(self._base_vts.volatility(fixing_dates[k], strike_k, True))

        var = 0.0
        for i in range(len(vol_base)):
            var += v[i] * v[i] * vol_base[i] * vol_base[i]
            for j in range(i + 1, len(vol_base)):
                corr = self._correlation(start_time_base[i], start_time_base[j])
                var += 2.0 * corr * v[i] * v[j] * vol_base[i] * vol_base[j]
        return math.sqrt(var)


__all__ = [
    "CorrelationStructure",
    "TenorOptionletVTS",
    "TwoParameterCorrelation",
]
