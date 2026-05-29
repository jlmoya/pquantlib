"""Gaussian1d non-standard swaption engine.

# C++ parity: ql/pricingengines/swaption/gaussian1dnonstandardswaptionengine.{hpp,cpp}
# @ v1.42.1 (099987f0).

Backward-induction Bermudan engine for a :class:`NonstandardSwaption`
(option on a :class:`NonstandardSwap` with per-coupon amortizing
notional / step-up fixed rate / floating gearing & spread).

Algorithm (matches gaussian1dnonstandardswaptionengine.cpp:30-496):

1. Build the standardized state grid via ``Gaussian1dModel.y_grid``.
2. Walk exercise dates in reverse, with one event per exercise date.
3. At each exercise date, compute the discounted continuation NPV
   (state-grid cubic-spline interpolation + Gaussian-shifted-polynomial
   integration), the exercise value (NPV of remaining swap from the
   current state), and update via ``max(continuation, exercise)``.
4. The final result at ``y=0, t=0`` times ``N(0,0)`` is the swaption
   NPV.

Carve-outs (Phase 11 W1-B):

- BasketGeneratingEngine.calibrationBasket method (deferred).
- The ``Probabilities`` enum / additional ``probabilities`` result.
- OAS Z-spread Handle<Quote>.
- RebatedExercise rebate flows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.instruments.nonstandard_swaption import (
    NonstandardSwaptionArguments,
    NonstandardSwaptionResults,
)
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import SettlementMethod
from pquantlib.models.shortrate.gaussian1d_model import Gaussian1dModel
from pquantlib.pricingengines.generic_engine import GenericEngine

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


_EXTRAPOLATION_BOUND: float = 100.0
_OPTION_CALL: int = 1
_OPTION_PUT: int = -1


class Gaussian1dNonstandardSwaptionEngine(
    GenericEngine[NonstandardSwaptionArguments, NonstandardSwaptionResults]
):
    """Gaussian1d engine for NonstandardSwaption.

    # C++ parity: ``class Gaussian1dNonstandardSwaptionEngine``
    # (gaussian1dnonstandardswaptionengine.hpp:53-136).
    """

    def __init__(
        self,
        model: Gaussian1dModel,
        integration_points: int = 64,
        stddevs: float = 7.0,
        extrapolate_payoff: bool = True,
        flat_payoff_extrapolation: bool = False,
        discount_curve: YieldTermStructureProtocol | None = None,
    ) -> None:
        super().__init__(
            NonstandardSwaptionArguments(), NonstandardSwaptionResults()
        )
        self._model: Gaussian1dModel = model
        self._integration_points: int = int(integration_points)
        self._stddevs: float = float(stddevs)
        self._extrapolate_payoff: bool = bool(extrapolate_payoff)
        self._flat_payoff_extrapolation: bool = bool(flat_payoff_extrapolation)
        self._discount_curve: YieldTermStructureProtocol | None = discount_curve

    def calculate(self) -> None:  # noqa: PLR0915
        # # C++ parity: gaussian1dnonstandardswaptionengine.cpp:134-496.
        args = self._arguments
        results = self._results
        results.reset()

        qassert.require(
            args.settlement_method != int(SettlementMethod.ParYieldCurve),
            "cash settled (ParYieldCurve) swaptions not priced with "
            "Gaussian1dNonstandardSwaptionEngine",
        )

        settlement = self._model.term_structure.reference_date()
        from pquantlib.exercise import Exercise  # noqa: PLC0415
        exercise = args.exercise
        assert isinstance(exercise, Exercise)

        # Last exercise date <= settlement → expired.
        if exercise.dates()[-1] <= settlement:
            results.value = 0.0
            return

        idx = len(exercise.dates()) - 1
        # min_idx_alive = index of first exercise date strictly after
        # settlement.
        min_idx_alive = 0
        for d_check in exercise.dates():
            if d_check > settlement:
                break
            min_idx_alive += 1

        option_call_put = (
            _OPTION_CALL if args.type == SwapType.Payer else _OPTION_PUT
        )

        npv0 = np.zeros(2 * self._integration_points + 1, dtype=np.float64)
        npv1 = np.zeros(2 * self._integration_points + 1, dtype=np.float64)
        z = self._model.y_grid(self._stddevs, self._integration_points)
        z_size = int(z.size)
        p = np.zeros(z_size, dtype=np.float64)

        expiry1_time: float | None = None

        while idx >= min_idx_alive - 1:
            expiry0 = (
                settlement if idx == min_idx_alive - 1
                else exercise.dates()[idx]
            )
            expiry0_time = max(
                self._model.term_structure.time_from_reference(expiry0), 0.0
            )

            # Find first fixed / floating coupon with reset_date >= expiry0.
            j1 = 0
            for d_resf in args.fixed_reset_dates:
                if d_resf > (expiry0 - 1):
                    break
                j1 += 1
            k1 = 0
            for d_resfl in args.floating_reset_dates:
                if d_resfl > (expiry0 - 1):
                    break
                k1 += 1

            outer_k_count = npv0.size if expiry0 > settlement else 1
            for k in range(outer_k_count):
                # Roll back (continuation NPV).
                price = 0.0
                if expiry1_time is not None:
                    yg = self._model.y_grid(
                        self._stddevs, self._integration_points,
                        expiry1_time, expiry0_time,
                        float(z[k]) if expiry0 > settlement else 0.0,
                    )
                    payoff0 = CubicSpline(z, npv1, bc_type="natural", extrapolate=True)
                    for i in range(int(yg.size)):
                        p[i] = float(payoff0(float(yg[i])))
                    payoff1 = CubicSpline(z, p, bc_type="natural", extrapolate=True)
                    coef = payoff1.c
                    for i in range(z_size - 1):
                        price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                            0.0,
                            float(coef[0, i]),
                            float(coef[1, i]),
                            float(coef[2, i]),
                            float(p[i]),
                            float(z[i]),
                            float(z[i]),
                            float(z[i + 1]),
                        )
                    # Tail extrapolation.
                    if self._extrapolate_payoff:
                        last_idx = z_size - 2
                        if self._flat_payoff_extrapolation:
                            price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0, 0.0, 0.0, 0.0,
                                float(p[last_idx]), float(z[last_idx]),
                                float(z[z_size - 1]), _EXTRAPOLATION_BOUND,
                            )
                            price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0, 0.0, 0.0, 0.0,
                                float(p[0]), float(z[0]),
                                -_EXTRAPOLATION_BOUND, float(z[0]),
                            )
                        else:
                            # The C++ engine only extends the right tail
                            # for non-standard swaption (cpp:122-128).
                            price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0,
                                float(coef[0, last_idx]),
                                float(coef[1, last_idx]),
                                float(coef[2, last_idx]),
                                float(p[last_idx]),
                                float(z[last_idx]),
                                float(z[z_size - 1]),
                                _EXTRAPOLATION_BOUND,
                            )

                npv0[k] = price

                # At an exercise date, compute exercise value and
                # take the max.
                if expiry0 > settlement:
                    zk = float(z[k])
                    floating_leg_npv = 0.0
                    for ell in range(k1, len(args.floating_coupons)):
                        if args.floating_is_redemption_flow[ell]:
                            amount = args.floating_coupons[ell]
                        else:
                            est_fixing = self._model.forward_rate(
                                args.floating_fixing_dates[ell],
                                expiry0,
                                zk,
                                args.ibor_index,
                            )
                            rate = (
                                args.floating_gearings[ell] * est_fixing
                                + args.floating_spreads[ell]
                            )
                            amount = (
                                rate
                                * args.floating_nominal[ell]
                                * args.floating_accrual_times[ell]
                            )
                        floating_leg_npv += amount * self._model.zerobond_date(
                            args.floating_pay_dates[ell],
                            expiry0,
                            zk,
                            self._discount_curve,
                        )

                    fixed_leg_npv = 0.0
                    for ell2 in range(j1, len(args.fixed_coupons)):
                        fixed_leg_npv += args.fixed_coupons[ell2] * self._model.zerobond_date(
                            args.fixed_pay_dates[ell2],
                            expiry0,
                            zk,
                            self._discount_curve,
                        )

                    num = self._model.numeraire(
                        expiry0_time, zk, self._discount_curve
                    )
                    exercise_value = (
                        (1.0 if option_call_put == _OPTION_CALL else -1.0)
                        * (floating_leg_npv - fixed_leg_npv)
                    ) / num

                    npv0[k] = max(npv0[k], exercise_value)

            # Swap buffers.
            npv1 = npv0.copy()
            npv0[:] = 0.0
            expiry1_time = expiry0_time
            idx -= 1

        results.value = float(npv1[0]) * self._model.numeraire(
            0.0, 0.0, self._discount_curve
        )


__all__ = ["Gaussian1dNonstandardSwaptionEngine"]
