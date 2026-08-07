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

The engine also derives from
:class:`~pquantlib.pricingengines.swaption.basket_generating_engine.BasketGeneratingEngine`
(matching the C++ class hierarchy) and supplies its four hooks —
``underlying_npv``, ``underlying_type``, ``underlying_last_date`` and
``initial_guess`` — so that ``calibration_basket(...)`` works.

Carve-outs (Phase 11 W1-B):

- The ``Probabilities`` enum / additional ``probabilities`` result.
- RebatedExercise rebate flows (``RebatedExercise`` is not ported; C++
  ``dynamic_pointer_cast``s to it and falls back to rebate = 0 when the
  cast fails, which is the branch every non-rebated exercise takes).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.instruments.nonstandard_swaption import (
    NonstandardSwaptionArguments,
    NonstandardSwaptionResults,
)
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import SettlementMethod
from pquantlib.math.closeness import close
from pquantlib.models.shortrate.gaussian1d_model import (
    Gaussian1dModel,
    payoff_interpolation,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.swaption.basket_generating_engine import (
    BasketGeneratingEngine,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from pquantlib.quotes.quote import Quote
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol
    from pquantlib.time.date import Date


EXTRAPOLATION_BOUND: float = 100.0
_ZERO_NOMINAL_TOLERANCE: float = 1e-8
"""# C++ parity: gaussian1dnonstandardswaptionengine.cpp:115 — periods with a
nominal at or below this are excluded from the initial-guess average."""
_OPTION_CALL: int = 1
_OPTION_PUT: int = -1


class Gaussian1dNonstandardSwaptionEngine(
    GenericEngine[NonstandardSwaptionArguments, NonstandardSwaptionResults],
    BasketGeneratingEngine,
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
        oas: Quote | None = None,
    ) -> None:
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, NonstandardSwaptionArguments(), NonstandardSwaptionResults()
        )
        BasketGeneratingEngine.__init__(self, model, oas, discount_curve)
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
                    # C++ uses CubicInterpolation(Spline, monotonic=True,
                    # Lagrange BC), NOT a natural cubic spline — see
                    # gaussian1dnonstandardswaptionengine.cpp:98-104.
                    payoff0 = payoff_interpolation(z, npv1)
                    for i in range(int(yg.size)):
                        p[i] = payoff0(float(yg[i]), allow_extrapolation=True)
                    payoff1 = payoff_interpolation(z, p)
                    a_c = payoff1.a_coefficients()
                    b_c = payoff1.b_coefficients()
                    c_c = payoff1.c_coefficients()
                    for i in range(z_size - 1):
                        price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                            0.0,
                            c_c[i],
                            b_c[i],
                            a_c[i],
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
                                float(z[z_size - 1]), EXTRAPOLATION_BOUND,
                            )
                            price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0, 0.0, 0.0, 0.0,
                                float(p[0]), float(z[0]),
                                -EXTRAPOLATION_BOUND, float(z[0]),
                            )
                        else:
                            # The C++ engine only extends the right tail
                            # for non-standard swaption (cpp:122-128).
                            price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0,
                                c_c[last_idx],
                                b_c[last_idx],
                                a_c[last_idx],
                                float(p[last_idx]),
                                float(z[last_idx]),
                                float(z[z_size - 1]),
                                EXTRAPOLATION_BOUND,
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

    # --- BasketGeneratingEngine hooks ---------------------------------

    def underlying_npv(self, expiry: Date, y: float) -> float:
        """NPV at ``expiry``, state ``y``, of the flows exercised into.

        # C++ parity: ``Gaussian1dNonstandardSwaptionEngine::underlyingNpv``
        # (gaussian1dnonstandardswaptionengine.cpp:31-88, v1.43).
        """
        args = self._arguments
        cutoff = expiry - 1
        fixed_idx = 0
        while (
            fixed_idx < len(args.fixed_reset_dates)
            and args.fixed_reset_dates[fixed_idx] <= cutoff
        ):
            fixed_idx += 1
        floating_idx = 0
        while (
            floating_idx < len(args.floating_reset_dates)
            and args.floating_reset_dates[floating_idx] <= cutoff
        ):
            floating_idx += 1

        oas = self._oas
        dc = self._model.term_structure.day_counter()

        npv = 0.0
        for i in range(fixed_idx, len(args.fixed_reset_dates)):
            z_spread = (
                1.0
                if oas is None
                else math.exp(
                    -oas.value() * dc.year_fraction(expiry, args.fixed_pay_dates[i])
                )
            )
            npv -= (
                args.fixed_coupons[i]
                * self._model.zerobond_date(
                    args.fixed_pay_dates[i], expiry, y, self._discount_curve
                )
                * z_spread
            )

        for i in range(floating_idx, len(args.floating_reset_dates)):
            if args.floating_is_redemption_flow[i]:
                amount = args.floating_coupons[i]
            else:
                amount = (
                    args.floating_gearings[i]
                    * self._model.forward_rate(
                        args.floating_fixing_dates[i], expiry, y, args.ibor_index
                    )
                    + args.floating_spreads[i]
                ) * args.floating_nominal[i] * args.floating_accrual_times[i]
            z_spread = (
                1.0
                if oas is None
                else math.exp(
                    -oas.value() * dc.year_fraction(expiry, args.floating_pay_dates[i])
                )
            )
            npv += (
                amount
                * self._model.zerobond_date(
                    args.floating_pay_dates[i], expiry, y, self._discount_curve
                )
                * z_spread
            )

        return float(args.type) * npv

    def underlying_type(self) -> SwapType:
        """# C++ parity: gaussian1dnonstandardswaptionengine.cpp:90-92."""
        return self._arguments.type

    def underlying_last_date(self) -> Date:
        """# C++ parity: gaussian1dnonstandardswaptionengine.cpp:95-97."""
        return self._arguments.fixed_pay_dates[-1]

    def initial_guess(self, expiry: Date) -> npt.NDArray[np.float64]:
        """``(average nominal, remaining maturity, weighted fixed rate)``.

        # C++ parity: gaussian1dnonstandardswaptionengine.cpp:100-132.
        """
        args = self._arguments
        cutoff = expiry - 1
        fixed_idx = 0
        while (
            fixed_idx < len(args.fixed_reset_dates)
            and args.fixed_reset_dates[fixed_idx] <= cutoff
        ):
            fixed_idx += 1

        nominal_sum = 0.0
        weighted_rate = 0.0
        ind = 0.0
        for i in range(fixed_idx, len(args.fixed_reset_dates)):
            nominal_sum += args.fixed_nominal[i]
            rate = args.fixed_rate[i]
            if close(rate, 0.0):
                rate = 0.03  # this value is at least better than zero
            weighted_rate += args.fixed_nominal[i] * rate
            if args.fixed_nominal[i] > _ZERO_NOMINAL_TOLERANCE:
                ind += 1.0

        nominal_avg = nominal_sum / ind
        qassert.require(
            nominal_sum > 0.0,
            f"sum of nominals on fixed leg must be positive ({nominal_sum})",
        )
        weighted_rate /= nominal_sum

        ts = self._model.term_structure
        return np.array(
            [
                nominal_avg,
                ts.time_from_reference(self.underlying_last_date())
                - ts.time_from_reference(expiry),
                weighted_rate,
            ],
            dtype=np.float64,
        )


__all__ = ["Gaussian1dNonstandardSwaptionEngine"]
