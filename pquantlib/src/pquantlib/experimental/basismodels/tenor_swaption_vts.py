"""TenorSwaptionVTS — tenor-rescaled swaption volatility term structure.

# C++ parity: ql/experimental/basismodels/tenorswaptionvts.hpp:37-125 +
# tenorswaptionvts.cpp:37-135 (v1.43).

Transforms a *base*-tenor swaption normal-vol surface into a *target*-tenor
surface by an affine-TSR model: each smile section rebuilds base/target/final
vanilla swaps, decomposes them via SwaptionCashFlows, and rescales strikes +
vols by the resulting (lambda, annuity-scaling) factors. Designed for normal
volatilities.
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exercise import EuropeanExercise
from pquantlib.experimental.basismodels.swaption_cfs import SwaptionCashFlows
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import Swaption
from pquantlib.instruments.vanilla_swap import VanillaSwap
from pquantlib.math.rounding import ClosestRounding
from pquantlib.pricingengines.swap.discounting_swap_engine import DiscountingSwapEngine
from pquantlib.termstructures.protocols import YieldTermStructureProtocol
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.swaption.swaption_volatility_structure import (
    SwaptionVolatilityStructure,
)
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.date import Date
from pquantlib.time.date_generation import DateGeneration
from pquantlib.time.period import Period
from pquantlib.time.schedule import Schedule
from pquantlib.time.time_unit import TimeUnit


class TenorSwaptionSmileSection(SmileSection):
    """Affine-TSR-rescaled smile section for a target tenor.

    # C++ parity: ``TenorSwaptionVTS::TenorSwaptionSmileSection``
    # (tenorswaptionvts.hpp:39-64, tenorswaptionvts.cpp:37-135).

        vol(K) = A * volBase((K - C) / A)
        A      = annuityScaling * (1 + lambda)
        C      = swapRateTarg - (1 + lambda) * swapRateBase
    """

    def __init__(
        self, vol_ts: TenorSwaptionVTS, option_time: float, swap_length: float
    ) -> None:
        super().__init__(
            exercise_time=option_time,
            day_counter=vol_ts.base_vts().day_counter(),
            volatility_type=VolatilityType.Normal,
            shift=0.0,
        )
        self._base_smile_section: SmileSection = vol_ts.base_vts().smile_section(
            option_time, swap_length, True
        )

        # swap start / end dates.
        one_day_as_year = vol_ts.day_counter().year_fraction(
            vol_ts.reference_date(), vol_ts.reference_date() + 1
        )
        # # C++ parity note: raw day arithmetic off the reference date, with NO
        # calendar adjustment (tenorswaptionvts.cpp:42-45).
        exercise_date = vol_ts.reference_date() + int(
            ClosestRounding(0)(option_time / one_day_as_year)
        )
        base_index = vol_ts.base_index()
        targ_index = vol_ts.targ_index()
        effective_date = base_index.fixing_calendar().advance(
            exercise_date, base_index.fixing_days(), TimeUnit.Days
        )
        # # C++ parity note: ``((BigInteger)swapLength * 12.0) * Months``
        # (tenorswaptionvts.cpp:48-49) casts swapLength to an integer number of
        # YEARS *first*, then multiplies by 12 — it does NOT round the swap
        # length to the nearest month. A 7.5y swap length therefore builds an
        # 84-month (7y) swap, not a 90-month one. Reproduced verbatim.
        maturity_date = base_index.fixing_calendar().advance(
            effective_date,
            int(swap_length) * 12,
            TimeUnit.Months,
            BusinessDayConvention.Unadjusted,
            False,
        )

        # schedules.
        base_fixed_schedule = _sched(
            effective_date, maturity_date, vol_ts.base_fixed_freq(), base_index
        )
        finl_fixed_schedule = _sched(
            effective_date, maturity_date, vol_ts.targ_fixed_freq(), targ_index
        )
        base_float_schedule = _sched(
            effective_date, maturity_date, base_index.tenor(), base_index
        )
        targ_float_schedule = _sched(
            effective_date, maturity_date, targ_index.tenor(), base_index
        )

        # swaps (Payer, unit nominal, zero rate/spread).
        base_swap = VanillaSwap(
            SwapType.Payer, 1.0, base_fixed_schedule, 1.0, vol_ts.base_fixed_dc(),
            base_float_schedule, base_index, 0.0, base_index.day_counter(),
        )
        targ_swap = VanillaSwap(
            SwapType.Payer, 1.0, base_fixed_schedule, 1.0, vol_ts.base_fixed_dc(),
            targ_float_schedule, targ_index, 0.0, targ_index.day_counter(),
        )
        finl_swap = VanillaSwap(
            SwapType.Payer, 1.0, finl_fixed_schedule, 1.0, vol_ts.targ_fixed_dc(),
            targ_float_schedule, targ_index, 0.0, targ_index.day_counter(),
        )

        engine = DiscountingSwapEngine(vol_ts.discount_curve())
        base_swap.set_pricing_engine(engine)
        targ_swap.set_pricing_engine(engine)
        finl_swap.set_pricing_engine(engine)

        self._swap_rate_base: float = base_swap.fair_rate()
        self._swap_rate_targ: float = targ_swap.fair_rate()
        self._swap_rate_finl: float = finl_swap.fair_rate()

        cfs = SwaptionCashFlows(
            Swaption(base_swap, EuropeanExercise(exercise_date)),
            vol_ts.discount_curve(),
        )
        cf2 = SwaptionCashFlows(
            Swaption(targ_swap, EuropeanExercise(exercise_date)),
            vol_ts.discount_curve(),
        )

        # affine-TSR model u, v.
        sum_tauj = sum(cfs.annuity_weights())
        sum_tauj_delta_t = sum(
            cfs.annuity_weights()[k] * (cfs.fixed_times()[-1] - cfs.fixed_times()[k])
            for k in range(len(cfs.annuity_weights()))
        )
        sum_wi = sum(cfs.float_weights())
        sum_wi_delta_t = sum(
            cfs.float_weights()[k] * (cfs.float_times()[-1] - cfs.float_times()[k])
            for k in range(len(cfs.float_weights()))
        )
        den = sum_tauj_delta_t * sum_wi - sum_wi_delta_t * sum_tauj
        u = -sum_tauj / den
        v = sum_tauj_delta_t / den

        t_n = cfs.fixed_times()[-1]
        # skip first and last weights (notional flows).
        sum_base = sum(
            cfs.float_weights()[k] * (u * (t_n - cfs.float_times()[k]) + v)
            for k in range(1, len(cfs.float_weights()) - 1)
        )
        sum_targ = sum(
            cf2.float_weights()[k] * (u * (t_n - cf2.float_times()[k]) + v)
            for k in range(1, len(cf2.float_weights()) - 1)
        )
        self._lambda: float = sum_targ - sum_base
        self._annuity_scaling: float = targ_swap.fixed_leg_bps() / finl_swap.fixed_leg_bps()

    # --- inspectors ------------------------------------------------------------

    def base_smile_section(self) -> SmileSection:
        """# C++ parity: ``baseSmileSection_``."""
        return self._base_smile_section

    def swap_rate_base(self) -> float:
        """# C++ parity: ``swapRateBase_`` — fair rate of the base-tenor swap."""
        return self._swap_rate_base

    def swap_rate_targ(self) -> float:
        """# C++ parity: ``swapRateTarg_`` — base fixed leg vs target float leg."""
        return self._swap_rate_targ

    def swap_rate_finl(self) -> float:
        """# C++ parity: ``swapRateFinl_`` — the fully target-tenor swap."""
        return self._swap_rate_finl

    def lambda_(self) -> float:
        """# C++ parity: ``lambda_`` — affine-TSR float-leg weight difference."""
        return self._lambda

    def annuity_scaling(self) -> float:
        """# C++ parity: ``annuityScaling_`` = targSwap BPS / finlSwap BPS."""
        return self._annuity_scaling

    # --- SmileSection interface ------------------------------------------------

    def _volatility_impl(self, strike: float) -> float:
        """# C++ parity: TenorSwaptionSmileSection::volatilityImpl (.cpp:129-135).

        C++ reads the base section with ``volatility(strikeBase, Normal, 0.0)``;
        that overload short-circuits to plain ``volatility(strikeBase)`` when
        the base section is already Normal with zero shift
        (smilesection.cpp:119-122), which is the designed configuration.
        """
        strike_base = (
            (strike - (self._swap_rate_targ - (1.0 + self._lambda) * self._swap_rate_base))
            / (1.0 + self._lambda)
            / self._annuity_scaling
        )
        vol_base = self._base_smile_section.volatility(strike_base)
        return self._annuity_scaling * (1.0 + self._lambda) * vol_base

    def min_strike(self) -> float:
        """# C++ parity: tenorswaptionvts.hpp:57-59."""
        return self._base_smile_section.min_strike() + self._swap_rate_targ - self._swap_rate_base

    def max_strike(self) -> float:
        """# C++ parity: tenorswaptionvts.hpp:60-62."""
        return self._base_smile_section.max_strike() + self._swap_rate_targ - self._swap_rate_base

    def atm_level(self) -> float:
        """# C++ parity: tenorswaptionvts.hpp:63 — the target-tenor swap rate."""
        return self._swap_rate_finl


class TenorSwaptionVTS(SwaptionVolatilityStructure):
    """Tenor-rescaled swaption volatility term structure (normal vols)."""

    def __init__(
        self,
        base_vts: SwaptionVolatilityStructure,
        discount_curve: YieldTermStructureProtocol,
        base_index: IborIndex,
        targ_index: IborIndex,
        base_fixed_freq: Period,
        targ_fixed_freq: Period,
        base_fixed_dc: DayCounter,
        targ_fixed_dc: DayCounter,
    ) -> None:
        super().__init__(
            business_day_convention=base_vts.business_day_convention(),
            reference_date=base_vts.reference_date(),
            calendar=base_vts.calendar(),
            day_counter=base_vts.day_counter(),
        )
        self._base_vts: SwaptionVolatilityStructure = base_vts
        self._discount_curve: YieldTermStructureProtocol = discount_curve
        self._base_index: IborIndex = base_index
        self._targ_index: IborIndex = targ_index
        self._base_fixed_freq: Period = base_fixed_freq
        self._targ_fixed_freq: Period = targ_fixed_freq
        self._base_fixed_dc: DayCounter = base_fixed_dc
        self._targ_fixed_dc: DayCounter = targ_fixed_dc

    # --- inspectors (used by the smile section) -------------------------------

    def base_vts(self) -> SwaptionVolatilityStructure:
        return self._base_vts

    def discount_curve(self) -> YieldTermStructureProtocol:
        return self._discount_curve

    def base_index(self) -> IborIndex:
        return self._base_index

    def targ_index(self) -> IborIndex:
        return self._targ_index

    def base_fixed_freq(self) -> Period:
        return self._base_fixed_freq

    def targ_fixed_freq(self) -> Period:
        return self._targ_fixed_freq

    def base_fixed_dc(self) -> DayCounter:
        return self._base_fixed_dc

    def targ_fixed_dc(self) -> DayCounter:
        return self._targ_fixed_dc

    # --- TermStructure interface ----------------------------------------------

    def max_date(self) -> Date:
        return self._base_vts.max_date()

    def min_strike(self) -> float:
        return self._base_vts.min_strike()

    def max_strike(self) -> float:
        return self._base_vts.max_strike()

    def max_swap_tenor(self) -> Period:
        return self._base_vts.max_swap_tenor()

    def volatility_type(self) -> VolatilityType:
        return VolatilityType.Normal

    # --- SwaptionVolatilityStructure interface --------------------------------

    def smile_section(
        self,
        option_expiry: Period | Date | float,
        swap_tenor: Period | float,
        extrapolate: bool = False,
    ) -> TenorSwaptionSmileSection:
        """# C++ parity: ``SwaptionVolatilityStructure::smileSection`` dispatch
        # (swaptionvolstructure.hpp:400-460) onto
        # ``TenorSwaptionVTS::smileSectionImpl`` (tenorswaptionvts.hpp:113-117).
        """
        self.check_swap_tenor(swap_tenor, extrapolate)
        if isinstance(option_expiry, Period):
            option_date = self.option_date_from_tenor(option_expiry)
            self.check_range(option_date, extrapolate)
            option_time = self.time_from_reference(option_date)
        elif isinstance(option_expiry, Date):
            self.check_range(option_expiry, extrapolate)
            option_time = self.time_from_reference(option_expiry)
        else:
            self.check_time_range(option_expiry, extrapolate)
            option_time = float(option_expiry)
        swap_length = (
            self.swap_length(swap_tenor) if isinstance(swap_tenor, Period) else swap_tenor
        )
        return TenorSwaptionSmileSection(self, option_time, swap_length)

    def _volatility_impl(
        self, option_time: float, swap_length: float, strike: float
    ) -> float:
        """# C++ parity: TenorSwaptionVTS::volatilityImpl (hpp:119-121).

        C++ calls ``smileSectionImpl(...)`` (no range check — the public
        ``volatility`` already did one) and reads it with
        ``volatility(strike, Normal, 0.0)``, which short-circuits.
        """
        return TenorSwaptionSmileSection(self, option_time, swap_length).volatility(strike)


def _sched(
    effective_date: Date, maturity_date: Date, tenor: Period, index: IborIndex
) -> Schedule:
    """Backward-rule, ModifiedFollowing/Unadjusted schedule on the index calendar.

    # C++ parity: the four ``Schedule(...)`` constructions in tenorswaptionvts.cpp
    # (lines 51-62) all use ModifiedFollowing + Unadjusted termination + Backward.
    """
    return Schedule.from_rule(
        effective_date,
        maturity_date,
        tenor,
        index.fixing_calendar(),
        BusinessDayConvention.ModifiedFollowing,
        BusinessDayConvention.Unadjusted,
        DateGeneration.Backward,
        False,
    )


__all__ = [
    "TenorSwaptionSmileSection",
    "TenorSwaptionVTS",
]
