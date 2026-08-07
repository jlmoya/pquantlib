"""InterpolatedAffineHazardRateCurve + the AffineHazardRate bootstrap trait.

# C++ parity: ql/experimental/credit/interpolatedaffinehazardratecurve.hpp
# :61-234 (v1.43).

A default-probability structure whose intensity is the sum of a *deterministic*
interpolated hazard rate (the thing this curve tabulates and that a bootstrap
solves for) and a *stochastic* one-factor affine short-rate model — the credit
analogue of a CIR++ setup. The total probabilities are the model's:

    S(t) = exp(-INT_0^t h_det(u) du) * P_model(0, t, r_init)

so ``hazard_rate(t)`` returns only the deterministic component, which is
confusing but is what the bootstrap needs (and what C++ documents).

# C++ parity note: ``survivalProbabilityImpl`` computes its initial intensity
# as ``pow(model_->dynamics()->process()->x0(), 2)``
# (interpolatedaffinehazardratecurve.hpp:297) — squared — while the base class
# ``OneFactorAffineSurvivalStructure`` uses
# ``dynamics()->shortRate(0., process()->x0())``, which for every v1.43
# one-factor affine model is ``x0`` itself. The two therefore disagree
# (0.0004 vs 0.02 for a CIR with r0 = 0.02). The comment in the C++ source
# ("the way x0 is defined") points at an older CoxIngersollRoss::Dynamics whose
# ``shortRate`` was ``y*y``; v1.43's is the identity, so the square is now a
# live discrepancy. It is reproduced verbatim, and it is *visible*: it moves
# every survival probability this class returns.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Final

import numpy as np

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.experimental.credit.one_factor_affine_survival import (
    OneFactorAffineSurvivalStructure,
)
from pquantlib.math.array import Array
from pquantlib.math.interpolations.backward_flat import BackwardFlatInterpolation
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.models.shortrate.onefactor.one_factor_affine_model import (
    OneFactorAffineModel,
)
from pquantlib.quotes.quote import Quote
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date

InterpolationFactory = Callable[[Array, Array], Interpolation]

# C++ parity: ql/termstructures/credit/probabilitytraits.hpp anon namespace.
_AVG_HAZARD_RATE: Final[float] = 0.01
_MAX_HAZARD_RATE: Final[float] = 1.0
# C++ parity: interpolatedaffinehazardratecurve.hpp:146 — this one lives in
# the *affine* header's ``detail`` namespace, and unlike the plain HazardRate
# trait's ``QL_EPSILON`` floor it allows a negative deterministic component.
_MIN_HAZARD_RATE_COMP: Final[float] = -1.0


class InterpolatedAffineHazardRateCurve(OneFactorAffineSurvivalStructure):
    """Deterministic interpolated hazard rate on top of an affine model.

    # C++ parity: interpolatedaffinehazardratecurve.hpp:61-141 + the template
    # bodies at :239-439.

    The C++ class is a template over the interpolator; Python takes an
    ``InterpolationFactory`` callable, defaulting to ``BackwardFlat`` (the
    interpolator every credit-curve traits pairing in QuantLib uses).
    """

    def __init__(
        self,
        dates: Sequence[Date],
        hazard_rates: Sequence[float],
        day_counter: DayCounter,
        model: OneFactorAffineModel,
        calendar: Calendar | None = None,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
        interpolator: InterpolationFactory = BackwardFlatInterpolation,
    ) -> None:
        qassert.require(len(dates) >= 1, "no input dates given")
        # C++ parity: interpolatedaffinehazardratecurve.hpp:389-402 — the
        # reference date is ``dates.at(0)`` and the calendar is forwarded to
        # the OneFactorAffineSurvivalStructure fixed-date ctor (which then
        # drops it; see that class's note).
        super().__init__(
            model,
            reference_date=dates[0],
            calendar=calendar,
            day_counter=day_counter,
            jumps=jumps,
            jump_dates=jump_dates,
        )
        self._dates: list[Date] = list(dates)
        self._data: list[float] = list(hazard_rates)
        self._interpolator: InterpolationFactory = interpolator
        self._times: list[float] = []
        self._interpolation: Interpolation | None = None
        self._initialize()

    def _initialize(self) -> None:
        # C++ parity: interpolatedaffinehazardratecurve.hpp:429-439. Note the
        # absence of a "negative hazard rate" check — unlike
        # InterpolatedHazardRateCurve, this curve deliberately permits a
        # negative deterministic component (see _MIN_HAZARD_RATE_COMP).
        qassert.require(len(self._dates) >= 2, "not enough input dates given")
        qassert.require(
            len(self._data) == len(self._dates), "dates/data count mismatch"
        )
        ref = self._dates[0]
        dc = self.day_counter()
        self._times = [dc.year_fraction(ref, d) for d in self._dates]
        self._interpolation = self._interpolator(
            np.asarray(self._times, dtype=np.float64),
            np.asarray(self._data, dtype=np.float64),
        )

    # ---- TermStructure interface -------------------------------------------

    def max_date(self) -> Date:
        # C++ parity: interpolatedaffinehazardratecurve.hpp:240-242.
        return self._dates[-1]

    # ---- other inspectors ---------------------------------------------------

    def times(self) -> list[float]:
        return list(self._times)

    def dates(self) -> list[Date]:
        return list(self._dates)

    def data(self) -> list[float]:
        return list(self._data)

    def hazard_rates(self) -> list[float]:
        return list(self._data)

    def nodes(self) -> list[tuple[Date, float]]:
        # C++ parity: interpolatedaffinehazardratecurve.hpp:268-275.
        return list(zip(self._dates, self._data, strict=True))

    # ---- implementation -----------------------------------------------------

    def _hazard_rate_impl(self, t: float) -> float:
        """The *deterministic* hazard-rate component only.

        # C++ parity: interpolatedaffinehazardratecurve.hpp:281-288 — flat
        # extrapolation of the last node past the last time.
        """
        assert self._interpolation is not None
        max_time = self._times[-1]
        if t <= max_time:
            return self._interpolation(t, allow_extrapolation=True)
        return self._data[-1]

    def _deterministic_integral(self, t: float) -> float:
        """INT_0^t h_det(u) du, with flat extrapolation past the last node."""
        assert self._interpolation is not None
        max_time = self._times[-1]
        if t <= max_time:
            return self._interpolation.primitive(t, allow_extrapolation=True)
        return (
            self._interpolation.primitive(max_time, allow_extrapolation=True)
            + self._data[-1] * (t - max_time)
        )

    def _curve_initial_intensity(self) -> float:
        """``pow(process.x0(), 2)`` — see the module-level parity note."""
        # C++ parity: interpolatedaffinehazardratecurve.hpp:297.
        return math.pow(self._model.dynamics().process.x0(), 2)

    def _survival_probability_impl(self, t: float) -> float:
        # C++ parity: interpolatedaffinehazardratecurve.hpp:291-312. Rewritten
        # in C++ rather than delegating to hazardRateImpl; reproduced as-is.
        init_val_hr = self._curve_initial_intensity()
        if t == 0.0:
            return self._model.discount_bond_scalar(0.0, t, init_val_hr)
        integral = self._deterministic_integral(t)
        return math.exp(-integral) * self._model.discount_bond_scalar(
            0.0, t, init_val_hr
        )

    def _conditional_survival_probability_impl(
        self, t_fwd: float, t_tgt: float, y_val: float
    ) -> float:
        # C++ parity: interpolatedaffinehazardratecurve.hpp:314-352.
        qassert.require(t_fwd <= t_tgt, "Probability time in the past.")
        if t_fwd == 0.0:
            return self._survival_probability_impl(t_tgt)
        if t_fwd - t_tgt == 0.0:
            return 1.0
        integral_t_fwd = self._deterministic_integral(t_fwd)
        integral_tp = self._deterministic_integral(t_tgt)
        return math.exp(-(integral_tp - integral_t_fwd)) * (
            self._model.discount_bond_scalar(t_fwd, t_tgt, y_val)
        )


class AffineHazardRate:
    """Bootstrap trait for :class:`InterpolatedAffineHazardRateCurve`.

    # C++ parity: ``struct AffineHazardRate`` at
    # interpolatedaffinehazardratecurve.hpp:152-234 (v1.43).

    Same shape as the ``HazardRate`` / ``DefaultDensity`` / ``SurvivalProbability``
    traits in ``pquantlib.termstructures.credit.probability_traits``: all
    members are static, and the class is consumed without instantiation.

    Two things differ from the plain ``HazardRate`` trait and both matter:

    * ``guess`` seeds the first pillar with ``0.0001``, not ``avgHazardRate``
      (the ``avgHazardRate`` line is commented out in the C++ source), and
      extrapolates later pillars with ``curve.hazard_rate(date, True)`` rather
      than copying ``data[i-1]``;
    * ``min_value_after`` floors at ``-1.0`` (``detail::minHazardRateComp``),
      not at ``QL_EPSILON`` — the deterministic component of a ++ model is
      allowed to be negative.
    """

    max_iterations_value: int = 30

    @staticmethod
    def initial_date(ts: Any) -> Any:
        # C++ parity: interpolatedaffinehazardratecurve.hpp:162-164.
        return ts.reference_date()

    @staticmethod
    def initial_value(ts: Any = None) -> float:
        # C++ parity: interpolatedaffinehazardratecurve.hpp:166-168 —
        # ``detail::avgHazardRate`` from probabilitytraits.hpp.
        del ts
        return _AVG_HAZARD_RATE

    @staticmethod
    def guess(i: int, curve: Any, valid_data: bool, first_alive_helper: int = 0) -> float:
        # C++ parity: interpolatedaffinehazardratecurve.hpp:171-192.
        del first_alive_helper
        if valid_data:
            return float(curve.data()[i])
        if i == 1:
            return 0.0001
        # Extrapolate off the curve itself, not off the previous datum.
        return float(curve.hazard_rate(curve.dates()[i], True))

    @staticmethod
    def min_value_after(
        i: int, curve: Any, valid_data: bool, first_alive_helper: int = 0
    ) -> float:
        # C++ parity: interpolatedaffinehazardratecurve.hpp:195-208.
        del i, first_alive_helper
        if valid_data:
            return min(curve.data()) / 2.0
        return _MIN_HAZARD_RATE_COMP

    @staticmethod
    def max_value_after(
        i: int, curve: Any, valid_data: bool, first_alive_helper: int = 0
    ) -> float:
        # C++ parity: interpolatedaffinehazardratecurve.hpp:209-223.
        del i, first_alive_helper
        if valid_data:
            return max(curve.data()) * 2.0
        return _MAX_HAZARD_RATE

    @staticmethod
    def update_guess(data: list[float], rate: float, i: int) -> None:
        # C++ parity: interpolatedaffinehazardratecurve.hpp:225-231.
        data[i] = rate
        if i == 1:
            data[0] = rate

    @staticmethod
    def max_iterations() -> int:
        # C++ parity: interpolatedaffinehazardratecurve.hpp:233.
        return AffineHazardRate.max_iterations_value


__all__ = ["AffineHazardRate", "InterpolatedAffineHazardRateCurve"]
