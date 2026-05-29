"""InterpolatedSmileSection — strike-vol slice with cubic interpolation.

# C++ parity: ql/termstructures/volatility/interpolatedsmilesection.hpp
# (v1.42.1).

The C++ template takes an ``Interpolator`` policy (typically ``Linear``
in the swaption-vol-cube wiring, but free to be ``Cubic``, ``Pchip``, etc.).
PQuantLib defaults to **``CubicNaturalSpline``** from L9-A — the choice
required by the Phase 9 plan for InterpolatedSmileSection. Other
interpolators can be passed via ``interpolator=Linear`` etc. The
constructor stores the raw vols (not std-devs as in C++) and wraps them
in the interpolator directly; we re-evaluate the interpolation each
call so that the section can be lazily refreshed after a strike update.

The C++ class also exposes a ``flatStrikeExtrapolation`` flag — at
out-of-pillar strikes the interpolated vol is clamped to the nearest
pillar's vol rather than extrapolated. We preserve this flag.

Both constructor overloads from C++ (Time and Date anchors) are folded
into a single Python constructor with the standard keyword scheme.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.actual_365_fixed import Actual365Fixed
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import CubicNaturalSpline
from pquantlib.termstructures.volatility.smile_section import SmileSection
from pquantlib.termstructures.volatility.volatility_type import VolatilityType
from pquantlib.time.date import Date

# Factory type: ``(strikes, vols) -> interpolation callable``.
# Each interpolator must be callable as ``f(strike) -> vol`` and must
# accept a single float argument. We accept the type loosely (Any)
# because pquantlib's Interpolation base implements `__call__` via a
# `_value(x)` hook + bound checks.
InterpolatorFactory = Callable[[Array, Array], Any]


def _default_interpolator(strikes: Array, vols: Array) -> CubicNaturalSpline:
    """Default factory — natural cubic spline from L9-A."""
    return CubicNaturalSpline(strikes, vols)


class InterpolatedSmileSection(SmileSection):
    """Strike-vol slice with interpolation (cubic-natural by default).

    Args:
        strikes: x-axis (strikes); must be sorted ascending.
        volatilities: y-axis (raw vols, *not* std-devs — diverges from
            the C++ template that stores std-dev quotes).
        atm_level: ATM forward.
        exercise_time / exercise_date / day_counter / reference_date:
            same construction modes as :class:`SmileSection`. When
            ``exercise_date`` is given without a day-counter, defaults
            to C++ Actual365Fixed.
        interpolator: factory function ``(strikes, vols) -> interp``.
            Default is :func:`_default_interpolator` (CubicNaturalSpline).
        volatility_type: ShiftedLognormal (default) or Normal.
        shift: shifted-lognormal shift; default 0.
        flat_strike_extrapolation: if True, return the nearest-pillar
            vol when strike is outside ``[min_strike, max_strike]``.
            Mirrors the C++ flag.
    """

    def __init__(
        self,
        *,
        strikes: Sequence[float],
        volatilities: Sequence[float],
        atm_level: float,
        exercise_time: float | None = None,
        exercise_date: Date | None = None,
        day_counter: DayCounter | None = None,
        reference_date: Date | None = None,
        interpolator: InterpolatorFactory | None = None,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        shift: float = 0.0,
        flat_strike_extrapolation: bool = False,
    ) -> None:
        if exercise_date is not None and day_counter is None:
            day_counter = Actual365Fixed()
        super().__init__(
            exercise_date=exercise_date,
            exercise_time=exercise_time,
            day_counter=day_counter,
            reference_date=reference_date,
            volatility_type=volatility_type,
            shift=shift,
        )

        ks = np.ascontiguousarray(strikes, dtype=np.float64)
        vs = np.ascontiguousarray(volatilities, dtype=np.float64)
        qassert.require(
            ks.size >= 2,
            "InterpolatedSmileSection needs at least 2 strike pillars",
        )
        qassert.require(
            ks.size == vs.size,
            "strikes and volatilities must have the same length",
        )
        qassert.require(
            bool(np.all(np.diff(ks) > 0.0)),
            "strikes must be strictly sorted in ascending order",
        )
        self._strikes: Array = ks
        self._vols: Array = vs
        self._atm_level: float = atm_level
        self._flat_strike_extrapolation: bool = flat_strike_extrapolation
        factory = interpolator if interpolator is not None else _default_interpolator
        self._interp: Any = factory(ks, vs)

    # --- SmileSection overrides ---------------------------------------

    def min_strike(self) -> float:
        return float(self._strikes[0])

    def max_strike(self) -> float:
        return float(self._strikes[-1])

    def atm_level(self) -> float:
        return self._atm_level

    def _volatility_impl(self, strike: float) -> float:
        if self._flat_strike_extrapolation:
            if strike < self.min_strike():
                strike = self.min_strike()
            elif strike > self.max_strike():
                strike = self.max_strike()
        # Allow modest off-pillar extrapolation by passing through the
        # interpolator. Negative vols are clamped to zero to match C++.
        return max(float(self._interp(strike, allow_extrapolation=True)), 0.0)
