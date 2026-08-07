"""TenorOptionletVTS — tenor-rescaled optionlet (caplet) volatility term structure.

# C++ parity: ql/experimental/basismodels/tenoroptionletvts.hpp:37-133 +
# tenoroptionletvts.cpp:36-107 (v1.43).

Transforms a *base*-tenor caplet normal-vol surface into a *target*-tenor
surface: a target-tenor caplet is decomposed into a strip of base-tenor FRAs
(``baseFreq % targFreq == 0``), and the target vol is the correlation-weighted
aggregation of the base-tenor vols. Designed for normal volatilities.

# C++ parity divergence (smile section): the C++ ``TenorOptionletSmileSection``
# holds ``std::vector<ext::shared_ptr<SmileSection>> baseSmileSection_``, one
# per base-tenor FRA, obtained from ``baseVTS_->smileSection(fixingDate, true)``,
# and reads it back with ``volatility(strike, Normal, 0.0)``. PQuantLib's
# ``OptionletVolatilityStructure`` has no ``smile_section`` accessor and
# ``SmileSection`` has no volatility-type-converting overload, so this class
# keeps the base structure plus the per-FRA fixing dates and reads the base vol
# as ``baseVTS.volatility(fixingDate, strike, True)``.
#
# The substitution is exact whenever the base structure's ``smileSectionImpl``
# and ``volatilityImpl`` agree (which is the normal case: every
# smile-section-backed surface in QuantLib defines
# ``volatilityImpl(t, k) = smileSectionImpl(t)->volatility(k)``). It is NOT
# exact for ``StrippedOptionletAdapter``, whose ``smileSectionImpl`` re-fits a
# cubic spline through the strike axis while its ``volatilityImpl`` uses the
# stripper's own interpolation, nor for a base whose smile section reports a
# different ``volatilityType`` than the surface — see
# ``ConstantOptionletVolatility``, whose C++ ``smileSectionImpl``
# (constantoptionletvol.cpp:74-84) forgets to forward ``volatilityType()``, so
# in C++ v1.43 a ``TenorOptionletVTS`` layered on it always throws
# ("smile section must provide atm level"), while PQuantLib returns a number.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from pquantlib import qassert
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.rounding import ClosestRounding
from pquantlib.termstructures.volatility.optionlet.optionlet_volatility_structure import (
    OptionletVolatilityStructure,
)
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


class CorrelationStructure(ABC):
    """Functor: correlation between two FRA rates starting at ``start1``/``start2``.

    # C++ parity: ``TenorOptionletVTS::CorrelationStructure``
    # (tenoroptionletvts.hpp:75-81).
    """

    @abstractmethod
    def __call__(self, start1: float, start2: float) -> float: ...


class TwoParameterCorrelation(CorrelationStructure):
    """rho = rhoInf + (1 - rhoInf) * exp(-beta * |start2 - start1|).

    # C++ parity: ``TenorOptionletVTS::TwoParameterCorrelation``
    # (tenoroptionletvts.hpp:84-99).
    """

    def __init__(self, rho_inf: Interpolation, beta: Interpolation) -> None:
        self._rho_inf: Interpolation = rho_inf
        self._beta: Interpolation = beta

    def __call__(self, start1: float, start2: float) -> float:
        # # C++ parity note: ``(*rhoInf_)(start1)`` uses
        # ``Interpolation::operator()``'s DEFAULT ``allowExtrapolation=false``
        # (interpolation.hpp:122-125), so a start time outside the parameter
        # grid is an error, not a flat extrapolation. Only ``start1`` is
        # looked up; ``start2`` enters solely through ``|start2 - start1|``
        # and may be arbitrarily far away.
        rho_inf = self._rho_inf(start1)
        beta = self._beta(start1)
        return rho_inf + (1.0 - rho_inf) * math.exp(-beta * abs(start2 - start1))


class TenorOptionletSmileSection(SmileSection):
    """Tenor-rescaled optionlet smile at one expiry.

    # C++ parity: ``TenorOptionletVTS::TenorOptionletSmileSection``
    # (tenoroptionletvts.hpp:43-66, tenoroptionletvts.cpp:51-107).

    The target-tenor caplet is written as a strip of base-tenor FRAs; each FRA
    vol is read off the base surface at a shifted/scaled strike, and the
    strip is aggregated through the correlation structure:

        vol(K)^2 = sum_ij rho(t_i, t_j) v_i v_j volBase_i(K_i) volBase_j(K_j)
        K_i      = (K - (F_targ - (sum v) F_base_i)) / (sum v)
    """

    def __init__(self, vol_ts: TenorOptionletVTS, option_time: float) -> None:
        super().__init__(
            exercise_time=option_time,
            day_counter=vol_ts.base_vts().day_counter(),
            volatility_type=VolatilityType.Normal,
            shift=0.0,
        )
        self._correlation: CorrelationStructure = vol_ts.correlation()
        self._base_vts: OptionletVolatilityStructure = vol_ts.base_vts()

        base_index = vol_ts.base_index()
        targ_index = vol_ts.targ_index()

        # We assume the long (target) tenor is a multiple of the short (base)
        # tenor; first we need the long-tenor start and end date.
        one_day_as_year = vol_ts.day_counter().year_fraction(
            vol_ts.reference_date(), vol_ts.reference_date() + 1
        )
        # # C++ parity note: the exercise date is raw day arithmetic off the
        # reference date, with NO calendar adjustment, so an optionTime that
        # lands on a weekend makes ``targIndex_->fixing`` reject the date.
        # Reproduced verbatim (tenoroptionletvts.cpp:57-60).
        exercise_date = vol_ts.reference_date() + int(
            ClosestRounding(0)(option_time / one_day_as_year)
        )
        effective_date = base_index.fixing_calendar().advance(
            exercise_date, base_index.fixing_days(), TimeUnit.Days
        )
        maturity_date = base_index.fixing_calendar().advance_period(
            effective_date, targ_index.tenor(), BusinessDayConvention.Unadjusted, False
        )
        # Now we can set up the short-tenor schedule.
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

        # Scalar attributes.
        self._fra_rate_targ: float = targ_index.fixing(exercise_date)
        yf_targ = targ_index.day_counter().year_fraction(effective_date, maturity_date)

        # Vector attributes.
        self._base_fixing_dates: list[Date] = []
        self._start_time_base: list[float] = []
        self._fra_rate_base: list[float] = []
        self._v: list[float] = []

        dates = base_float_schedule.dates
        for k in range(len(dates) - 1):
            start_date = dates[k]
            fixing_date = base_index.fixing_calendar().advance(
                start_date, -base_index.fixing_days(), TimeUnit.Days
            )
            year_frac = base_index.day_counter().year_fraction(dates[k], dates[k + 1])
            self._base_fixing_dates.append(fixing_date)
            self._start_time_base.append(
                vol_ts.day_counter().year_fraction(vol_ts.reference_date(), start_date)
            )
            fr_base = base_index.fixing(fixing_date)
            self._fra_rate_base.append(fr_base)
            self._v.append(
                year_frac
                / yf_targ
                * (1.0 + yf_targ * self._fra_rate_targ)
                / (1.0 + year_frac * fr_base)
            )

    # --- inspectors -------------------------------------------------------

    def base_fixing_dates(self) -> list[Date]:
        """Per-FRA base fixing dates (the C++ ``baseSmileSection_`` anchors)."""
        return self._base_fixing_dates

    def start_time_base(self) -> list[float]:
        """# C++ parity: ``startTimeBase_`` — FRA start times for the correlation."""
        return self._start_time_base

    def fra_rate_base(self) -> list[float]:
        """# C++ parity: ``fraRateBase_``."""
        return self._fra_rate_base

    def fra_rate_targ(self) -> float:
        """# C++ parity: ``fraRateTarg_``."""
        return self._fra_rate_targ

    def v(self) -> list[float]:
        """# C++ parity: ``v_`` — the per-FRA transformation weights."""
        return self._v

    # --- SmileSection interface -------------------------------------------

    def _volatility_impl(self, strike: float) -> float:
        """# C++ parity: TenorOptionletSmileSection::volatilityImpl (.cpp:88-107)."""
        sum_v = 0.0
        for k in self._v:
            sum_v += k

        vol_base: list[float] = []
        for k in range(len(self._fra_rate_base)):
            strike_k = (strike - (self._fra_rate_targ - sum_v * self._fra_rate_base[k])) / sum_v
            vol_base.append(
                self._base_vts.volatility(self._base_fixing_dates[k], strike_k, True)
            )

        var = 0.0
        for i in range(len(vol_base)):
            var += self._v[i] * self._v[i] * vol_base[i] * vol_base[i]
            for j in range(i + 1, len(vol_base)):
                corr = self._correlation(self._start_time_base[i], self._start_time_base[j])
                var += 2.0 * corr * self._v[i] * self._v[j] * vol_base[i] * vol_base[j]
        return math.sqrt(var)

    def min_strike(self) -> float:
        """# C++ parity: tenoroptionletvts.hpp:59-61.

        C++ reads ``baseSmileSection_[0]->minStrike()``; PQuantLib uses the base
        structure's ``min_strike()`` (see the module-level divergence note). The
        two agree for every base whose smile section inherits the surface's
        strike range.
        """
        return self._base_vts.min_strike() + self._fra_rate_targ - self._fra_rate_base[0]

    def max_strike(self) -> float:
        """# C++ parity: tenoroptionletvts.hpp:62-64."""
        return self._base_vts.max_strike() + self._fra_rate_targ - self._fra_rate_base[0]

    def atm_level(self) -> float:
        """# C++ parity: tenoroptionletvts.hpp:65 — the target-tenor FRA rate."""
        return self._fra_rate_targ


class TenorOptionletVTS(OptionletVolatilityStructure):
    """Tenor-rescaled optionlet volatility term structure (normal vols).

    # C++ parity: ``TenorOptionletVTS`` (tenoroptionletvts.hpp:37-131).
    """

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

    # --- inspectors (used by the smile section) -------------------------------

    def base_vts(self) -> OptionletVolatilityStructure:
        return self._base_vts

    def base_index(self) -> IborIndex:
        return self._base_index

    def targ_index(self) -> IborIndex:
        return self._targ_index

    def correlation(self) -> CorrelationStructure:
        return self._correlation

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

    def smile_section(
        self, option_date: Period | Date | float, extrapolate: bool = False
    ) -> TenorOptionletSmileSection:
        """Smile section at ``option_date`` (Period / Date / Time).

        # C++ parity: ``OptionletVolatilityStructure::smileSection`` dispatch
        # (optionletvolatilitystructure.hpp:142-205) onto
        # ``TenorOptionletVTS::smileSectionImpl`` (tenoroptionletvts.hpp:115-117).
        """
        if isinstance(option_date, Period):
            return self.smile_section(self.option_date_from_tenor(option_date), extrapolate)
        if isinstance(option_date, Date):
            self.check_range(option_date, extrapolate)
            return TenorOptionletSmileSection(self, self.time_from_reference(option_date))
        self.check_time_range(option_date, extrapolate)
        return TenorOptionletSmileSection(self, option_date)

    def _volatility_impl(self, t: float, strike: float) -> float:
        """# C++ parity: TenorOptionletVTS::volatilityImpl (hpp:119-121).

        C++ routes through ``smileSection(optionTime)`` — note the *default*
        ``extrapolate=false`` on that inner call, reproduced here.
        """
        return self.smile_section(t).volatility(strike)


__all__ = [
    "CorrelationStructure",
    "TenorOptionletSmileSection",
    "TenorOptionletVTS",
    "TwoParameterCorrelation",
]
