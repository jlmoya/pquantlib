"""One-factor Gaussian model swaption engine (numerical integration).

# C++ parity: ql/pricingengines/swaption/gaussian1dswaptionengine.{hpp,cpp}
# @ v1.43 (6b57206e0).

Backward induction on the model's standardized-state grid. Walking the
exercise dates in reverse:

1. Roll the continuation value back from the previous (later) exercise
   date by interpolating the stored NPV grid at the conditional
   ``y``-grid, re-interpolating in the unconditional grid and
   integrating the resulting piecewise cubic against the standard-normal
   density in closed form.
2. At an exercise date compute the exercise value — the remaining
   floating leg minus the remaining fixed leg, discounted with the
   model's zero bonds and divided by the numeraire — and take the max.
3. The value at ``y = 0``, ``t = 0`` times ``N(0, 0)`` is the swaption
   NPV.

All fixed coupons whose period start is on or after the exercise date
are part of the exercise-into right (the C++ selection is
``upper_bound(schedule.dates(), expiry0 - 1)``, i.e. inclusive of
``expiry0`` itself).

Cash-settled (ParYieldCurve) swaptions are rejected. Non-constant
nominals are rejected.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import (
    SettlementMethod,
    SwaptionArguments,
    SwaptionResults,
)
from pquantlib.models.shortrate.gaussian1d_model import (
    EXTRAPOLATION_BOUND,
    Gaussian1dModel,
    payoff_interpolation,
)
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.generic_engine import GenericEngine

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.date import Date


class Probabilities(IntEnum):
    """Which exercise-probability flavour to accumulate, if any.

    # C++ parity: ``Gaussian1dSwaptionEngine::Probabilities`` in
    # gaussian1dswaptionengine.hpp:48-52 (v1.43).
    """

    None_ = 0
    Naive = 1
    Digital = 2


class Gaussian1dSwaptionEngine(GenericEngine[SwaptionArguments, SwaptionResults]):
    """Swaption engine under a one-factor Gaussian short-rate model.

    # C++ parity: ``class Gaussian1dSwaptionEngine`` in
    # gaussian1dswaptionengine.hpp:44-96 (v1.43).
    """

    Probabilities = Probabilities

    def __init__(
        self,
        model: Gaussian1dModel,
        integration_points: int = 64,
        stddevs: float = 7.0,
        extrapolate_payoff: bool = True,
        flat_payoff_extrapolation: bool = False,
        discount_curve: YieldTermStructureProtocol | None = None,
        probabilities: Probabilities = Probabilities.None_,
    ) -> None:
        # # C++ parity: gaussian1dswaptionengine.hpp:54-86.
        super().__init__(SwaptionArguments(), SwaptionResults())
        self._model: Gaussian1dModel = model
        self._integration_points: int = int(integration_points)
        self._stddevs: float = float(stddevs)
        self._extrapolate_payoff: bool = bool(extrapolate_payoff)
        self._flat_payoff_extrapolation: bool = bool(flat_payoff_extrapolation)
        self._discount_curve: YieldTermStructureProtocol | None = discount_curve
        self._probabilities: Probabilities = probabilities

    # --- helpers ------------------------------------------------------

    def _rollback(
        self,
        z: np.ndarray,
        npv1: np.ndarray,
        expiry1_time: float,
        expiry0_time: float,
        y: float,
        option_type: OptionType,
    ) -> float:
        """Continuation value at state ``y``, rolled back from ``expiry1_time``.

        # C++ parity: gaussian1dswaptionengine.cpp:126-176 (and the
        # verbatim copy at :184-248 used for the probability grids).
        """
        z_size = int(z.size)
        yg = self._model.y_grid(
            self._stddevs, self._integration_points, expiry1_time, expiry0_time, y
        )
        payoff0 = payoff_interpolation(z, npv1)
        p = np.empty(int(yg.size), dtype=np.float64)
        for i in range(int(yg.size)):
            # C++ passes allowExtrapolation = true.
            p[i] = payoff0(float(yg[i]), allow_extrapolation=True)
        payoff1 = payoff_interpolation(z, p)
        a_c = payoff1.a_coefficients()
        b_c = payoff1.b_coefficients()
        c_c = payoff1.c_coefficients()

        price = 0.0
        for i in range(z_size - 1):
            price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                0.0, c_c[i], b_c[i], a_c[i], float(p[i]), float(z[i]),
                float(z[i]), float(z[i + 1]),
            )
        if self._extrapolate_payoff:
            last = z_size - 2
            if self._flat_payoff_extrapolation:
                price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                    0.0, 0.0, 0.0, 0.0, float(p[last]), float(z[last]),
                    float(z[z_size - 1]), EXTRAPOLATION_BOUND,
                )
                price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                    0.0, 0.0, 0.0, 0.0, float(p[0]), float(z[0]),
                    -EXTRAPOLATION_BOUND, float(z[0]),
                )
            elif option_type == OptionType.Call:
                price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                    0.0, c_c[last], b_c[last], a_c[last],
                    float(p[last]), float(z[last]),
                    float(z[z_size - 1]), EXTRAPOLATION_BOUND,
                )
            else:
                price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                    0.0, c_c[0], b_c[0], a_c[0], float(p[0]), float(z[0]),
                    -EXTRAPOLATION_BOUND, float(z[0]),
                )
        return price

    @staticmethod
    def _upper_bound(dates: Sequence[Date], value: Date) -> int:
        """Index of the first element strictly greater than ``value``.

        # C++ parity: ``std::upper_bound`` on a sorted date vector.
        """
        lo, hi = 0, len(dates)
        while lo < hi:
            mid = (lo + hi) // 2
            if dates[mid] > value:
                hi = mid
            else:
                lo = mid + 1
        return lo

    # --- engine -------------------------------------------------------

    def calculate(self) -> None:  # noqa: PLR0915 (one-shot port of the C++ body)
        # # C++ parity: gaussian1dswaptionengine.cpp:26-345 (v1.43).
        args = self._arguments
        results = self._results
        results.reset()

        qassert.require(
            args.settlement_method != SettlementMethod.ParYieldCurve,
            "cash settled (ParYieldCurve) swaptions not priced with "
            "Gaussian1dSwaptionEngine",
        )
        qassert.require(
            args.nominal is not None, "non-constant nominals are not supported yet"
        )
        nominal = args.nominal
        assert nominal is not None

        qassert.require(args.exercise is not None, "no exercise given")
        exercise = args.exercise
        assert isinstance(exercise, Exercise)
        exercise_dates = exercise.dates()

        settlement = self._model.term_structure.reference_date()
        if exercise_dates[-1] <= settlement:
            # Swaption is expired; the generated swap is not even valued.
            results.value = 0.0
            return

        idx = len(exercise_dates) - 1
        min_idx_alive = self._upper_bound(exercise_dates, settlement)

        swap = args.swap
        assert swap is not None
        option_type = (
            OptionType.Call if args.swap_type == SwapType.Payer else OptionType.Put
        )
        fixed_dates = swap.fixed_schedule().dates
        float_dates = swap.floating_schedule().dates

        n_grid = 2 * self._integration_points + 1
        npv0 = np.zeros(n_grid, dtype=np.float64)
        npv1 = np.zeros(n_grid, dtype=np.float64)
        z = self._model.y_grid(self._stddevs, self._integration_points)

        # Probability grids: one per alive exercise date, plus one for the
        # "never exercised" bucket.
        npvp0: list[np.ndarray] = []
        npvp1: list[np.ndarray] = []
        if self._probabilities != Probabilities.None_:
            for _ in range(idx - min_idx_alive + 2):
                npvp0.append(np.zeros(n_grid, dtype=np.float64))
                npvp1.append(np.zeros(n_grid, dtype=np.float64))

        expiry1_time: float | None = None

        while True:
            expiry0 = settlement if idx == min_idx_alive - 1 else exercise_dates[idx]
            expiry0_time = max(
                self._model.term_structure.time_from_reference(expiry0), 0.0
            )

            j1 = self._upper_bound(fixed_dates, expiry0 - 1)
            k1 = self._upper_bound(float_dates, expiry0 - 1)

            alive = expiry0 > settlement
            for k in range(n_grid if alive else 1):
                price = 0.0
                if expiry1_time is not None:
                    price = self._rollback(
                        z, npv1, expiry1_time, expiry0_time,
                        float(z[k]) if alive else 0.0, option_type,
                    )
                npv0[k] = price

                if self._probabilities != Probabilities.None_:
                    for m in range(len(npvp0)):
                        pprice = 0.0
                        if expiry1_time is not None:
                            pprice = self._rollback(
                                z, npvp1[m], expiry1_time, expiry0_time,
                                float(z[k]) if alive else 0.0, option_type,
                            )
                        npvp0[m][k] = pprice

                if alive:
                    zk = float(z[k])
                    floating_leg_npv = 0.0
                    for ell in range(k1, len(args.floating_coupons)):
                        floating_leg_npv += (
                            nominal
                            * args.floating_accrual_times[ell]
                            * (
                                args.floating_spreads[ell]
                                + self._model.forward_rate(
                                    args.floating_fixing_dates[ell],  # type: ignore[arg-type]
                                    expiry0,
                                    zk,
                                    swap.ibor_index(),  # type: ignore[arg-type]
                                )
                            )
                            * self._model.zerobond_date(
                                args.floating_pay_dates[ell],  # type: ignore[arg-type]
                                expiry0,
                                zk,
                                self._discount_curve,
                            )
                        )
                    fixed_leg_npv = 0.0
                    for ell in range(j1, len(args.fixed_coupons)):
                        fixed_leg_npv += args.fixed_coupons[ell] * self._model.zerobond_date(
                            args.fixed_pay_dates[ell],  # type: ignore[arg-type]
                            expiry0,
                            zk,
                            self._discount_curve,
                        )
                    exercise_value = (
                        (1.0 if option_type == OptionType.Call else -1.0)
                        * (floating_leg_npv - fixed_leg_npv)
                        / self._model.numeraire(expiry0_time, zk, self._discount_curve)
                    )

                    if self._probabilities != Probabilities.None_:
                        if idx == len(exercise_dates) - 1:
                            # Latest date: initialise the no-call probability.
                            # NOTE the C++ asymmetry reproduced verbatim here —
                            # the Digital branch calls
                            # ``numeraire(expiry0, ...)`` with the DATE-valued
                            # overload while the exercise branch below calls
                            # ``numeraire(expiry0Time, ...)``. See
                            # gaussian1dswaptionengine.cpp:288-295.
                            npvp0[-1][k] = (
                                1.0
                                if self._probabilities == Probabilities.Naive
                                else 1.0
                                / (
                                    self._model.zerobond(
                                        expiry0_time, 0.0, 0.0, self._discount_curve
                                    )
                                    * self._model.numeraire_date(
                                        expiry0, zk, self._discount_curve
                                    )
                                )
                            )
                        if exercise_value >= npv0[k]:
                            npvp0[idx - min_idx_alive][k] = (
                                1.0
                                if self._probabilities == Probabilities.Naive
                                else 1.0
                                / (
                                    self._model.zerobond(
                                        expiry0_time, 0.0, 0.0, self._discount_curve
                                    )
                                    * self._model.numeraire(
                                        expiry0_time, zk, self._discount_curve
                                    )
                                )
                            )
                            for ii in range(idx - min_idx_alive + 1, len(npvp0)):
                                npvp0[ii][k] = 0.0

                    npv0[k] = max(npv0[k], exercise_value)

            npv0, npv1 = npv1, npv0
            if self._probabilities != Probabilities.None_:
                npvp0, npvp1 = npvp1, npvp0

            expiry1_time = expiry0_time

            idx -= 1
            if idx < min_idx_alive - 1:
                break

        results.value = float(npv1[0]) * self._model.numeraire(
            0.0, 0.0, self._discount_curve
        )

        if self._probabilities != Probabilities.None_:
            scale = (
                1.0
                if self._probabilities == Probabilities.Naive
                else self._model.numeraire(0.0, 0.0, self._discount_curve)
            )
            results.additional_results["probabilities"] = [
                float(npvp1[i][0]) * scale for i in range(len(npvp1))
            ]


__all__ = ["Gaussian1dSwaptionEngine", "Probabilities"]
