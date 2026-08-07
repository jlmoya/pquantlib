"""OneFactorAffineSurvivalStructure — survival curve driven by an affine model.

# C++ parity: ql/experimental/credit/onefactoraffinesurvival.hpp:41-173 (v1.43).

The structure derives from :class:`HazardRateStructure` but the *probabilities*
do not come from the hazard rate: they come from a one-factor affine short-rate
model reinterpreted as a stochastic default intensity, so that

    S(t) = P_model(0, t, r0),   r0 = dynamics.short_rate(0, process.x0())

and, conditional on a realisation ``y`` of the intensity at ``t_fwd``,

    S(t_fwd -> t_tgt | y) = P_model(t_fwd, t_tgt, y).

``hazard_rate_impl`` is the *deterministic* component only and is zero on this
base class; subclasses (see
:class:`~pquantlib.experimental.credit.interpolated_affine_hazard_rate_curve.InterpolatedAffineHazardRateCurve`)
override it and rewrite the probabilities accordingly.

# C++ parity note: because ``hazardRateImpl`` returns 0 here,
# ``defaultDensityImpl`` — written as ``hazardRateImpl(t) * survivalProbabilityImpl(t)
# / discountBond(0, t, r0)`` — is identically 0 for this class. That is what
# v1.43 does; it is reproduced rather than "fixed".
"""

from __future__ import annotations

from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.models.shortrate.onefactor.one_factor_affine_model import (
    OneFactorAffineModel,
)
from pquantlib.quotes.quote import Quote
from pquantlib.termstructures.credit.hazard_rate_structure import HazardRateStructure
from pquantlib.time.calendar import Calendar
from pquantlib.time.date import Date


class OneFactorAffineSurvivalStructure(HazardRateStructure):
    """Survival-probability term structure based on a one-factor affine model.

    Three construction modes mirror the three C++ constructors:

    * ``OneFactorAffineSurvivalStructure(model, day_counter=...)`` — moving
      reference date taken from the global evaluation date;
    * ``... (model, reference_date=..., calendar=..., day_counter=...)`` — fixed
      reference date;
    * ``... (model, settlement_days=..., calendar=..., day_counter=...)`` —
      settlement-days mode.

    # C++ parity note: the fixed-reference-date C++ ctor drops the ``cal``
    # argument on the floor — it forwards ``Calendar()`` to the base, not
    # ``cal`` (onefactoraffinesurvival.hpp:59). The Python port reproduces
    # that: passing ``calendar=`` together with ``reference_date=`` has no
    # effect on the curve's calendar.
    """

    def __init__(
        self,
        model: OneFactorAffineModel,
        reference_date: Date | None = None,
        calendar: Calendar | None = None,
        day_counter: DayCounter | None = None,
        settlement_days: int | None = None,
        jumps: list[Quote] | None = None,
        jump_dates: list[Date] | None = None,
    ) -> None:
        if reference_date is not None:
            # C++ parity: onefactoraffinesurvival.hpp:52-60 — ``cal`` is
            # accepted and then *not* forwarded.
            super().__init__(
                reference_date=reference_date,
                calendar=None,
                day_counter=day_counter,
                jumps=jumps,
                jump_dates=jump_dates,
            )
        elif settlement_days is not None:
            # C++ parity: onefactoraffinesurvival.hpp:62-70.
            super().__init__(
                settlement_days=settlement_days,
                calendar=calendar,
                day_counter=day_counter,
                jumps=jumps,
                jump_dates=jump_dates,
            )
        else:
            # C++ parity: onefactoraffinesurvival.hpp:45-50.
            super().__init__(
                day_counter=day_counter,
                jumps=jumps,
                jump_dates=jump_dates,
            )
        self._model: OneFactorAffineModel = model

    # ---- TermStructure interface -------------------------------------------

    def max_date(self) -> Date:
        """C++ parity: onefactoraffinesurvival.hpp:75 — ``Date::maxDate()``."""
        return Date.max_date()

    def model(self) -> OneFactorAffineModel:
        """The affine model driving the stochastic intensity."""
        return self._model

    # ---- conditional survival ----------------------------------------------

    def conditional_survival_probability(
        self,
        d_fwd: float | Date,
        d_tgt: float | Date,
        y_val: float,
        extrapolate: bool = False,
    ) -> float:
        """P(tau > tgt | F_fwd), given the realised intensity ``y_val`` at fwd.

        # C++ parity: onefactoraffinesurvival.hpp:101-121 — the Date overload
        # converts both dates to times and re-enters the Time overload, which
        # range-checks *both* arguments and then defers to
        # ``conditionalSurvivalProbabilityImpl``.
        """
        if isinstance(d_fwd, Date) or isinstance(d_tgt, Date):
            t_fwd = (
                self.time_from_reference(d_fwd) if isinstance(d_fwd, Date) else d_fwd
            )
            t_tgt = (
                self.time_from_reference(d_tgt) if isinstance(d_tgt, Date) else d_tgt
            )
            return self.conditional_survival_probability(
                t_fwd, t_tgt, y_val, extrapolate
            )
        self.check_time_range(d_fwd, extrapolate)
        self.check_time_range(d_tgt, extrapolate)
        # C++ TODO: "ADD JUMPS TREATMENT" — jumps are not applied here.
        return self._conditional_survival_probability_impl(d_fwd, d_tgt, y_val)

    def hazard_rate(self, t: float | Date, extrapolate: bool = False) -> float:
        """Deterministic hazard-rate component only.

        # C++ parity: onefactoraffinesurvival.hpp:125-128 — this shadows the
        # base-class implementation (which would divide default density by
        # survival probability) and returns ``hazardRateImpl`` directly.
        """
        if isinstance(t, Date):
            return self.hazard_rate(self.time_from_reference(t), extrapolate)
        self.check_time_range(t, extrapolate)
        return self._hazard_rate_impl(t)

    # ---- DefaultProbabilityTermStructure implementation ---------------------

    def _initial_intensity(self) -> float:
        """r0 as the C++ base class computes it.

        # C++ parity: onefactoraffinesurvival.hpp:152-154 —
        # ``model_->dynamics()->shortRate(0., model_->dynamics()->process()->x0())``.
        """
        dyn = self._model.dynamics()
        return dyn.short_rate(0.0, dyn.process.x0())

    def _survival_probability_impl(self, t: float) -> float:
        # C++ parity: onefactoraffinesurvival.hpp:148-157.
        return self._model.discount_bond_scalar(0.0, t, self._initial_intensity())

    def _default_density_impl(self, t: float) -> float:
        # C++ parity: onefactoraffinesurvival.hpp:165-173.
        init_hr = self._initial_intensity()
        return (
            self._hazard_rate_impl(t)
            * self._survival_probability_impl(t)
            / self._model.discount_bond_scalar(0.0, t, init_hr)
        )

    def _conditional_survival_probability_impl(
        self, t_fwd: float, t_tgt: float, y_val: float
    ) -> float:
        # C++ parity: onefactoraffinesurvival.hpp:159-163.
        return self._model.discount_bond_scalar(t_fwd, t_tgt, y_val)

    def _hazard_rate_impl(self, t: float) -> float:
        """No deterministic component on this class.

        # C++ parity: onefactoraffinesurvival.hpp:140-143.
        """
        del t
        return 0.0


__all__ = ["OneFactorAffineSurvivalStructure"]
