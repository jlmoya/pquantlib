"""BlackVolatilitySurfaceDelta — Black vol surface quoted in delta space.

# C++ parity: ql/termstructures/volatility/equityfx/blackvolsurfacedelta.hpp +
#             blackvolsurfacedelta.cpp (v1.43).

FX vol is quoted by *delta*, not by strike: a row of the market matrix is
``[25d put, 10d put, ATM, 10d call, 25d call]`` at one expiry. This surface
stores one :class:`BlackVarianceCurve` per delta column — so time
interpolation happens in delta space, where the quotes live — and only
converts delta to strike at query time, with a
:class:`~pquantlib.pricingengines.vanilla.black_delta_calculator.BlackDeltaCalculator`
under the configured delta / ATM conventions.

Three details carry the behaviour and are easy to get wrong:

* **Strike de-duplication.** C++ collects ``(strike, vol)`` into a
  ``std::map`` whose comparator treats two strikes as the same key when
  ``close(a, b)`` holds (blackvolsurfacedelta.cpp:115-116). Insertion order is
  puts, then ATM, then calls, and the *first* delta to produce a given strike
  wins — a later one is dropped, not overwritten. The map then yields its
  pillars in ascending strike order.

* **The switch tenor.** Below ``switch_time`` the short-term delta / ATM
  conventions apply; at or above it (``close_enough`` counts as "at") the
  long-term ones do (cpp:103-111). ``switch_tenor == 0 * Days`` means "never
  switch" and is spelled as ``switch_time = QL_MAX_REAL``.

* **Strike 0 short-circuits the smile.** ``black_vol(t, 0)`` reads the ATM
  variance curve directly when there is an ATM column, and otherwise
  re-enters at the forward (cpp:219-227).

The smile itself is a :class:`SmileSection`: an
:class:`InterpolatedSmileSection` under one of four interpolation methods, or
a :class:`FlatSmileSection` when only one strike survives de-duplication (which
happens whenever the surface has a single quote column).

Divergence, inherited and already documented at its source: C++'s
``FlatSmileSection`` reports ``QL_MIN_REAL - shift()`` / ``QL_MAX_REAL`` as its
strike bounds and ``Null<Rate>()`` as its ATM level, where PQuantLib's reports
``-inf`` / ``+inf`` and ``nan`` — see flat_smile_section.py:51-62. That only
affects the single-column case, and only the sentinel, never a volatility.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import IntEnum

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.closeness import close, close_enough
from pquantlib.math.constants import QL_MAX_REAL
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition,
    Cubic,
    DerivativeApprox,
)
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import Linear
from pquantlib.math.matrix import Matrix
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.vanilla.black_delta_calculator import BlackDeltaCalculator
from pquantlib.quotes.quote import Quote
from pquantlib.quotes.simple_quote import SimpleQuote
from pquantlib.termstructures.volatility.equity_fx.black_variance_curve import (
    BlackVarianceCurve,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_term_structure import (
    BlackVolatilityTermStructure,
)
from pquantlib.termstructures.volatility.equity_fx.black_vol_time_extrapolation import (
    BlackVolTimeExtrapolation,
)
from pquantlib.termstructures.volatility.equity_fx.delta_vol_quote import (
    AtmType,
    DeltaType,
)
from pquantlib.termstructures.volatility.flat_smile_section import FlatSmileSection
from pquantlib.termstructures.volatility.interpolated_smile_section import (
    InterpolatedSmileSection,
)
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.business_day_convention import BusinessDayConvention
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date
from pquantlib.time.period import Period
from pquantlib.time.time_unit import TimeUnit


class SmileInterpolationMethod(IntEnum):
    """Strike-dimension interpolation for the per-expiry smile.

    # C++ parity: ``BlackVolatilitySurfaceDelta::SmileInterpolationMethod``
    # (blackvolsurfacedelta.hpp:57). ``NaturalCubic`` is C++'s name for
    # ``Cubic(CubicInterpolation::Kruger)`` — a Kruger-slope cubic with natural
    # end conditions, *not* a natural cubic spline; ``CubicSpline`` is the
    # spline-slope variant (cpp:181-199).
    """

    Linear = 0
    NaturalCubic = 1
    FinancialCubic = 2
    CubicSpline = 3


def _interpolator_policy(method: SmileInterpolationMethod) -> Linear | Cubic:
    """Return the C++ interpolator policy object for ``method``.

    # C++ parity: blackvolsurfacedelta.cpp:178-202.
    """
    if method == SmileInterpolationMethod.Linear:
        return Linear()
    if method == SmileInterpolationMethod.NaturalCubic:
        return Cubic(DerivativeApprox.Kruger)
    if method == SmileInterpolationMethod.FinancialCubic:
        # Cubic(Kruger, true, SecondDerivative, 0.0, FirstDerivative) — the
        # right-hand condition value falls back to C++'s 0.0 default.
        return Cubic(
            DerivativeApprox.Kruger,
            True,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.FirstDerivative,
            0.0,
        )
    if method == SmileInterpolationMethod.CubicSpline:
        return Cubic(DerivativeApprox.Spline)
    qassert.fail(f"Invalid method {int(method)}")
    raise AssertionError  # unreachable


class BlackVolatilitySurfaceDelta(BlackVolatilityTermStructure):
    """Black volatility surface parameterized by market deltas.

    # C++ parity: ``class BlackVolatilitySurfaceDelta``
    # (blackvolsurfacedelta.hpp:54).

    Args:
        reference_date: valuation date of the surface.
        dates: option expiry dates, strictly increasing and all after
            ``reference_date``.
        put_deltas: put-side deltas (negative), one per leading matrix column.
        call_deltas: call-side deltas (positive), one per trailing column.
        has_atm: whether an ATM column sits between the put and call blocks.
        black_vol_matrix: rows = expiries, columns = deltas (puts, [ATM],
            calls).
        day_counter: converts dates to times.
        calendar: used by ``option_date_from_tenor`` for the switch tenor.
        spot: spot quote used for the delta-to-strike conversion.
        domestic_ts: domestic (discounting) curve.
        foreign_ts: foreign curve; ``spot * fDiscount / dDiscount`` is the
            forward.
        delta_type: short-term delta convention.
        atm_type: short-term ATM convention.
        atm_delta_type: delta convention for the ATM column; defaults to
            ``delta_type``.
        interpolation_method: strike-dimension interpolation of the smile.
        flat_strike_extrapolation: clamp to the outermost pillar vol instead
            of extrapolating in strike.
        time_extrapolation_type: handed to each per-delta
            :class:`BlackVarianceCurve`.
        switch_tenor: expiries at or beyond this tenor use the long-term
            conventions; ``0 * Days`` disables the switch.
        long_term_delta_type / long_term_atm_type / long_term_atm_delta_type:
            the conventions used beyond ``switch_tenor``.
    """

    # The signature is one-for-one with the C++ ctor (blackvolsurfacedelta.hpp:87).
    def __init__(
        self,
        *,
        reference_date: Date,
        dates: Sequence[Date],
        put_deltas: Sequence[float],
        call_deltas: Sequence[float],
        has_atm: bool,
        black_vol_matrix: Matrix | Sequence[Sequence[float]],
        day_counter: DayCounter,
        calendar: Calendar,
        spot: Quote | float,
        domestic_ts: YieldTermStructure,
        foreign_ts: YieldTermStructure,
        delta_type: DeltaType = DeltaType.Spot,
        atm_type: AtmType = AtmType.AtmDeltaNeutral,
        atm_delta_type: DeltaType | None = None,
        interpolation_method: SmileInterpolationMethod = SmileInterpolationMethod.Linear,
        flat_strike_extrapolation: bool = False,
        time_extrapolation_type: BlackVolTimeExtrapolation.Type = (
            BlackVolTimeExtrapolation.Type.FlatVolatility
        ),
        switch_tenor: Period | None = None,
        long_term_delta_type: DeltaType = DeltaType.Fwd,
        long_term_atm_type: AtmType = AtmType.AtmDeltaNeutral,
        long_term_atm_delta_type: DeltaType | None = None,
    ) -> None:
        super().__init__(
            business_day_convention=BusinessDayConvention.Following,
            reference_date=reference_date,
            calendar=calendar,
            day_counter=day_counter,
        )
        self._dates: list[Date] = list(dates)
        self._put_deltas: list[float] = [float(d) for d in put_deltas]
        self._call_deltas: list[float] = [float(d) for d in call_deltas]
        self._has_atm: bool = has_atm
        self._spot: Quote = spot if isinstance(spot, Quote) else SimpleQuote(float(spot))
        self._domestic_ts: YieldTermStructure = domestic_ts
        self._foreign_ts: YieldTermStructure = foreign_ts
        self._delta_type: DeltaType = delta_type
        self._atm_type: AtmType = atm_type
        # C++ parity: cpp:45 — `atmDeltaType ? *atmDeltaType : deltaType`.
        self._atm_delta_type: DeltaType = (
            atm_delta_type if atm_delta_type is not None else delta_type
        )
        self._interpolation_method: SmileInterpolationMethod = interpolation_method
        self._flat_strike_extrapolation: bool = flat_strike_extrapolation
        self._time_extrapolation_type: BlackVolTimeExtrapolation.Type = time_extrapolation_type
        self._switch_tenor: Period = (
            switch_tenor if switch_tenor is not None else Period(0, TimeUnit.Days)
        )
        self._long_term_delta_type: DeltaType = long_term_delta_type
        self._long_term_atm_type: AtmType = long_term_atm_type
        self._long_term_atm_delta_type: DeltaType = (
            long_term_atm_delta_type
            if long_term_atm_delta_type is not None
            else long_term_delta_type
        )

        # C++ parity: cpp:51 — a zero switch tenor means "never switch".
        # The test is on the length, not on ``Period(0, Days)`` equality:
        # C++'s ``operator==`` treats every zero-length period as equal
        # regardless of unit (period.cpp ``operator<`` zero branch), whereas
        # this port's frozen-dataclass ``__eq__`` compares the unit too.
        self._switch_time: float = (
            QL_MAX_REAL
            if self._switch_tenor.length == 0
            else self.time_from_reference(self.option_date_from_tenor(self._switch_tenor))
        )

        # C++ parity: cpp:53-61.
        qassert.require(len(self._dates) > 1, "at least 1 date required")
        times: list[float] = [0.0] * len(self._dates)
        for i, d in enumerate(self._dates):
            qassert.require(reference_date < d, "Dates must be greater than reference date")
            times[i] = self.time_from_reference(d)
            if i > 0:
                qassert.require(times[i] > times[i - 1], "dates must be sorted unique!")
        self._times: list[float] = times

        matrix = np.asarray(black_vol_matrix, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(len(self._dates), -1)

        # C++ parity: cpp:64-70.
        n = len(self._put_deltas) + (1 if has_atm else 0) + len(self._call_deltas)
        qassert.require(n > 0, "Need at least one delta")
        qassert.require(
            int(matrix.shape[1]) == n,
            f"Invalid number of columns in blackVolMatrix, got {matrix.shape[1]} "
            f"but have {n} deltas",
        )
        qassert.require(
            int(matrix.shape[0]) == len(self._dates),
            f"Invalid number of rows in blackVolMatrix, got {matrix.shape[0]} "
            f"but have {len(self._dates)} dates",
        )

        # C++ parity: cpp:72-85 — one BlackVarianceCurve per delta column.
        # `forceMonotoneVariance` is hard-coded false in C++ (cpp:74).
        self._interpolators: list[BlackVarianceCurve] = [
            BlackVarianceCurve(
                reference_date=reference_date,
                dates=self._dates,
                black_vol_curve=[float(matrix[j][i]) for j in range(len(self._dates))],
                day_counter=day_counter,
                force_monotone_variance=False,
                time_extrapolation_type=time_extrapolation_type,
            )
            for i in range(n)
        ]

        # C++ parity: cpp:87-90 — registerWith(spot_/domesticTS_/foreignTS_).
        self._spot.register_with(self)
        self._domestic_ts.register_with(self)
        self._foreign_ts.register_with(self)

    # --- TermStructure interface -------------------------------------------

    def max_date(self) -> Date:
        """# C++ parity: blackvolsurfacedelta.hpp:108 — ``Date::maxDate()``."""
        return Date.max_date()

    # --- VolatilityTermStructure interface ---------------------------------

    def min_strike(self) -> float:
        """# C++ parity: blackvolsurfacedelta.hpp:112."""
        return 0.0

    def max_strike(self) -> float:
        """# C++ parity: blackvolsurfacedelta.hpp:113 — ``QL_MAX_REAL``."""
        return QL_MAX_REAL

    # --- Inspectors ---------------------------------------------------------

    def dates(self) -> list[Date]:
        """# C++ parity: blackvolsurfacedelta.hpp:127."""
        return list(self._dates)

    # --- BlackVolTermStructure interface -----------------------------------

    def atm_level(self, t: float) -> float:
        """Forward at time ``t``.

        # C++ parity: ``BlackVolatilitySurfaceDelta::atmLevel`` (cpp:210-212).
        """
        return self._spot.value() * self._foreign_ts.discount(t) / self._domestic_ts.discount(t)

    # --- the smile ----------------------------------------------------------

    def black_vol_smile(self, t: float) -> SmileSection:
        """Smile section at time ``t``.

        # C++ parity: ``BlackVolatilitySurfaceDelta::blackVolSmile(Time)``
        # (cpp:93-204).
        """
        spot = self._spot.value()
        d_discount = self._domestic_ts.discount(t)
        f_discount = self._foreign_ts.discount(t)
        sqrt_t = math.sqrt(t)

        # C++ parity: cpp:103-111 — `close_enough(t, switchTime_)` counts as
        # "at or past the switch", so the boundary uses the long-term set.
        if t < self._switch_time and not close_enough(t, self._switch_time):
            at = self._atm_type
            dt = self._delta_type
            atm_dt = self._atm_delta_type
        else:
            at = self._long_term_atm_type
            dt = self._long_term_delta_type
            atm_dt = self._long_term_atm_delta_type

        # C++ parity: cpp:113-116 — a std::map keyed on strike whose
        # comparator collapses `close()` strikes onto one key, so the first
        # delta to reach a strike keeps it. Held here as an insertion-ordered
        # list and sorted on the way out; every surviving key is pairwise
        # non-close, so the map's ordering is plain ascending strike.
        pillars: list[tuple[float, float]] = []

        def insert(strike: float, vol: float) -> None:
            if not any(close(k, strike) for k, _ in pillars):
                pillars.append((strike, vol))

        i = 0
        atm_level = 1.0  # C++ parity: cpp:118 — stays 1.0 when hasAtm_ is false.

        for delta in self._put_deltas:
            vol = self._interpolators[i].black_vol_at_time(t, 1.0, True)
            try:
                bdc = BlackDeltaCalculator(
                    OptionType.Put, dt, spot, d_discount, f_discount, vol * sqrt_t
                )
                insert(bdc.strike_from_delta(delta), vol)
            # C++ re-wraps any std::exception here (cpp:127-130).
            except Exception as e:
                qassert.fail(
                    "BlackVolatilitySurfaceDelta: Error during calculating put strike at "
                    f"delta {delta}: {e}"
                )
            i += 1

        if self._has_atm:
            vol = self._interpolators[i].black_vol_at_time(t, 1.0, True)
            atm_level = vol
            try:
                bdc = BlackDeltaCalculator(
                    OptionType.Put, atm_dt, spot, d_discount, f_discount, vol * sqrt_t
                )
                insert(bdc.atm_strike(at), vol)
            except Exception as e:
                qassert.fail(
                    f"BlackVolatilitySurfaceDelta: Error during calculating atm strike: {e}"
                )
            i += 1

        for delta in self._call_deltas:
            vol = self._interpolators[i].black_vol_at_time(t, 1.0, True)
            try:
                bdc = BlackDeltaCalculator(
                    OptionType.Call, dt, spot, d_discount, f_discount, vol * sqrt_t
                )
                insert(bdc.strike_from_delta(delta), vol)
            except Exception as e:
                qassert.fail(
                    "BlackVolatilitySurfaceDelta: Error during calculating call strike at "
                    f"delta {delta}: {e}"
                )
            i += 1

        pillars.sort(key=lambda kv: kv[0])
        strikes = [k for k, _ in pillars]
        # C++ parity: cpp:167 — the map's vols become std-devs here, and
        # InterpolatedSmileSection divides them back by the same sqrt(t)
        # (interpolatedsmilesection.hpp:213). The round trip is reproduced
        # rather than short-circuited so the last bit matches.
        std_devs = [v * sqrt_t for _, v in pillars]

        qassert.require(
            len(std_devs) > 0,
            f"BlackVolatilitySurfaceDelta::blackVolSmile({t}): no strikes given, "
            "this is unexpected.",
        )

        if len(std_devs) == 1:
            # C++ parity: cpp:173-175 — one strike (e.g. a single quote
            # column, or t=0) collapses to a flat section, with no ATM level.
            return FlatSmileSection(
                exercise_time=t,
                volatility=std_devs[0] / sqrt_t,
                day_counter=self.day_counter(),
            )

        # C++ parity: cpp:177-202.
        policy = _interpolator_policy(self._interpolation_method)

        def factory(xs: Matrix, ys: Matrix) -> Interpolation:
            return policy.interpolate(xs, ys)

        return InterpolatedSmileSection(
            exercise_time=t,
            strikes=strikes,
            volatilities=[sd / sqrt_t for sd in std_devs],
            atm_level=atm_level,
            interpolator=factory,
            day_counter=self.day_counter(),
            volatility_type=VolatilityType.ShiftedLognormal,
            shift=0.0,
            flat_strike_extrapolation=self._flat_strike_extrapolation,
        )

    def black_vol_smile_at_date(self, d: Date) -> SmileSection:
        """Smile section at expiry date ``d``.

        # C++ parity: ``blackVolSmile(const Date&)`` (cpp:206-208).
        """
        return self.black_vol_smile(self.time_from_reference(d))

    def _smile_section_impl(self, t: float) -> SmileSection:
        """# C++ parity: none — v1.43's ``BlackVolTermStructure`` smile hook.

        The base class would wrap this surface in a read-back adapter; this
        surface has a native smile, so it hands back the real section.
        """
        return self.black_vol_smile(t)

    # --- BlackVolatilityTermStructure hook ---------------------------------

    def _black_vol_impl(self, t: float, strike: float) -> float:
        """# C++ parity: ``blackVolImpl`` (cpp:214-229)."""
        # C++ parity: cpp:216-217 — under FlatVolatility, queries past the last
        # pillar are answered at the last pillar rather than by the curve's own
        # extrapolation, so the *smile* is flat in time too.
        tme = (
            self._times[-1]
            if (
                t > self._times[-1]
                and self._time_extrapolation_type
                == BlackVolTimeExtrapolation.Type.FlatVolatility
            )
            else t
        )

        # C++ parity: cpp:219-227 — `strike == 0 || strike == Null<Real>()`.
        # PQuantLib's null-Real analogue is NaN (see BlackVolTermStructure.
        # atm_level), so both sentinels route here.
        if strike == 0 or math.isnan(strike):
            if self._has_atm:
                # Ask the ATM variance curve directly; it ignores the strike.
                return self._interpolators[len(self._put_deltas)].black_vol_at_time(
                    tme, math.nan, True
                )
            strike = self.atm_level(tme)

        return self.black_vol_smile(tme).volatility(strike)


__all__ = ["BlackVolatilitySurfaceDelta", "SmileInterpolationMethod"]
