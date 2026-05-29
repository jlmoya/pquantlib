"""Gaussian1d float-float swaption engine.

# C++ parity: ql/pricingengines/swaption/gaussian1dfloatfloatswaptionengine.{hpp,cpp}
# @ v1.42.1 (099987f0).

Backward-induction state-grid engine for a :class:`FloatFloatSwaption`
under a :class:`Gaussian1dModel`. The algorithm walks events (coupon
fixing dates + exercise dates) in reverse, propagating the option and
underlying NPV through a cubic-spline-interpolated state grid.

Algorithm (matches gaussian1dfloatfloatswaptionengine.cpp:25-691):

1. Aggregate event dates from both legs' fixing dates + the exercise
   schedule.
2. Build the standardized state grid ``z = model.y_grid(stddevs,
   integration_points)``.
3. Walk events in reverse:
   a. Roll back the option NPV and underlying NPV via state-grid
      interpolation + Gaussian-shifted-polynomial-integral integration.
   b. At a coupon fixing date, add the coupon contribution to the
      underlying NPV.
   c. At an exercise date, take ``max(continuation, exercise)``.
4. The result is the option NPV at ``y=0, t=0`` times the model's
   ``N(0, 0)``.

Carve-outs (Phase 11 W1-B):

- ``BasketGeneratingEngine`` calibration-basket method is **not** ported
  (depends on BlackCalibrationHelper + SwapIndex.clone(Period); both
  scheduled for later waves).
- The ``probabilities`` enum / additional ``probabilities`` result are
  not surfaced.
- The ``oas`` Handle<Quote> Z-spread is not surfaced (always 1.0).
- CMS / SwapSpreadIndex dispatch is replaced by IborIndex-only
  evaluation (matches FloatFloatSwap's W1-B carve-out scope).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.instruments.float_float_swaption import (
    FloatFloatSwaptionArguments,
    FloatFloatSwaptionResults,
)
from pquantlib.instruments.swap import SwapType
from pquantlib.instruments.swaption import SettlementMethod
from pquantlib.models.shortrate.gaussian1d_model import Gaussian1dModel
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.time.date import Date

if TYPE_CHECKING:
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


_EXTRAPOLATION_BOUND: float = 100.0
# # C++ parity: gaussian1dfloatfloatswaptionengine.cpp:140 — Option::Call vs Put.
# We use ints (1 = Call/Payer, -1 = Put/Receiver).
_OPTION_CALL: int = 1
_OPTION_PUT: int = -1


class Gaussian1dFloatFloatSwaptionEngine(
    GenericEngine[FloatFloatSwaptionArguments, FloatFloatSwaptionResults]
):
    """Gaussian1d engine for FloatFloatSwaption.

    # C++ parity: ``class Gaussian1dFloatFloatSwaptionEngine``
    # (gaussian1dfloatfloatswaptionengine.hpp:57-153).
    """

    def __init__(
        self,
        model: Gaussian1dModel,
        integration_points: int = 64,
        stddevs: float = 7.0,
        extrapolate_payoff: bool = True,
        flat_payoff_extrapolation: bool = False,
        discount_curve: YieldTermStructureProtocol | None = None,
        include_todays_exercise: bool = False,
    ) -> None:
        super().__init__(FloatFloatSwaptionArguments(), FloatFloatSwaptionResults())
        self._model: Gaussian1dModel = model
        self._integration_points: int = int(integration_points)
        self._stddevs: float = float(stddevs)
        self._extrapolate_payoff: bool = bool(extrapolate_payoff)
        self._flat_payoff_extrapolation: bool = bool(flat_payoff_extrapolation)
        self._discount_curve: YieldTermStructureProtocol | None = discount_curve
        self._include_todays_exercise: bool = bool(include_todays_exercise)

    def calculate(self) -> None:
        # # C++ parity: gaussian1dfloatfloatswaptionengine.cpp:27-52.
        args = self._arguments
        results = self._results
        results.reset()

        qassert.require(
            args.settlement_method != int(SettlementMethod.ParYieldCurve),
            "cash settled (ParYieldCurve) swaptions not priced with "
            "Gaussian1dFloatFloatSwaptionEngine",
        )

        settlement = self._model.term_structure.reference_date()
        qassert.require(args.exercise is not None, "exercise not set")
        # Help pyright narrow.
        from pquantlib.exercise import Exercise  # noqa: PLC0415
        exercise = args.exercise
        assert isinstance(exercise, Exercise)

        # Last exercise date <= settlement → expired.
        if exercise.dates()[-1] <= settlement:
            results.value = 0.0
            return

        option_npv, underlying_npv = self._npvs(
            settlement, 0.0, self._include_todays_exercise
        )
        results.value = option_npv
        results.additional_results["underlyingValue"] = underlying_npv

    # ------------------------------------------------------------------
    # Internal: backward-induction pair (option NPV, underlying NPV)
    # ------------------------------------------------------------------
    def _npvs(  # noqa: PLR0915
        self,
        expiry: Date,
        y: float,
        include_exercise_on_expiry: bool,
    ) -> tuple[float, float]:
        # # C++ parity: gaussian1dfloatfloatswaptionengine.cpp:104-691.
        args = self._arguments
        from pquantlib.exercise import Exercise  # noqa: PLC0415
        exercise = args.exercise
        assert isinstance(exercise, Exercise)

        # Aggregate event dates.
        events_set: set[Date] = set(exercise.dates())
        for d in args.leg1_fixing_dates:
            events_set.add(d)
        for d2 in args.leg2_fixing_dates:
            events_set.add(d2)
        events = sorted(events_set)
        # Filter events with date > expiry - (1 if include_today else 0).
        # In QuantLib, comparing a Date with Date() (the past) means:
        # we want events strictly after ``expiry`` minus 1 day if
        # include_today, else after ``expiry``.
        cutoff = expiry - 1 if include_exercise_on_expiry else expiry
        events = [d for d in events if d > cutoff]
        idx = len(events) - 1

        type_payer_receiver = args.type
        option_call_put = (
            _OPTION_CALL if type_payer_receiver == SwapType.Payer else _OPTION_PUT
        )

        z = self._model.y_grid(self._stddevs, self._integration_points)
        z_size = int(z.size)

        npv0 = np.zeros(z_size, dtype=np.float64)
        npv1 = np.zeros(z_size, dtype=np.float64)
        npv0a = np.zeros(z_size, dtype=np.float64)
        npv1a = np.zeros(z_size, dtype=np.float64)
        p = np.zeros(z_size, dtype=np.float64)
        pa = np.zeros(z_size, dtype=np.float64)

        event1_time: float | None = None

        while idx >= -1:
            if idx == -1:
                event0 = expiry
                is_event_date = False
            else:
                event0 = events[idx]
                is_event_date = True
                if event0 == expiry:
                    idx = -1  # avoid double roll back

            is_exercise = event0 in set(exercise.dates())
            is_leg1_fixing = event0 in set(args.leg1_fixing_dates)
            is_leg2_fixing = event0 in set(args.leg2_fixing_dates)

            event0_time = max(
                self._model.term_structure.time_from_reference(event0), 0.0
            )

            outer_k_count = z_size if event0 > expiry else 1
            for k in range(outer_k_count):
                # Roll back.
                price = 0.0
                pricea = 0.0
                if event1_time is not None:
                    z_state_k = z[k] if event0 > expiry else y
                    yg = self._model.y_grid(
                        self._stddevs, self._integration_points,
                        event1_time, event0_time, float(z_state_k),
                    )
                    # Build the per-state payoff interpolants.
                    payoff0 = CubicSpline(z, npv1, bc_type="natural", extrapolate=True)
                    payoff0a = CubicSpline(z, npv1a, bc_type="natural", extrapolate=True)
                    for i in range(int(yg.size)):
                        p[i] = float(payoff0(float(yg[i])))
                        pa[i] = float(payoff0a(float(yg[i])))
                    payoff1 = CubicSpline(z, p, bc_type="natural", extrapolate=True)
                    payoff1a = CubicSpline(z, pa, bc_type="natural", extrapolate=True)
                    coef = payoff1.c
                    coef_a = payoff1a.c
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
                        pricea += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                            0.0,
                            float(coef_a[0, i]),
                            float(coef_a[1, i]),
                            float(coef_a[2, i]),
                            float(pa[i]),
                            float(z[i]),
                            float(z[i]),
                            float(z[i + 1]),
                        )
                    # Tail extrapolation.
                    if self._extrapolate_payoff:
                        last_idx = z_size - 2
                        if self._flat_payoff_extrapolation:
                            price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0, 0.0, 0.0, 0.0, float(p[last_idx]),
                                float(z[last_idx]), float(z[z_size - 1]),
                                _EXTRAPOLATION_BOUND,
                            )
                            price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0, 0.0, 0.0, 0.0, float(p[0]),
                                float(z[0]), -_EXTRAPOLATION_BOUND, float(z[0]),
                            )
                            pricea += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0, 0.0, 0.0, 0.0, float(pa[last_idx]),
                                float(z[last_idx]), float(z[z_size - 1]),
                                _EXTRAPOLATION_BOUND,
                            )
                            pricea += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0, 0.0, 0.0, 0.0, float(pa[0]),
                                float(z[0]), -_EXTRAPOLATION_BOUND, float(z[0]),
                            )
                        # Cubic extension: type-direction-aware.
                        elif option_call_put == _OPTION_CALL:
                            price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0,
                                float(coef[0, last_idx]),
                                float(coef[1, last_idx]),
                                float(coef[2, last_idx]),
                                float(p[last_idx]),
                                float(z[last_idx]), float(z[z_size - 1]),
                                _EXTRAPOLATION_BOUND,
                            )
                            pricea += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0,
                                float(coef_a[0, last_idx]),
                                float(coef_a[1, last_idx]),
                                float(coef_a[2, last_idx]),
                                float(pa[last_idx]),
                                float(z[last_idx]), float(z[z_size - 1]),
                                _EXTRAPOLATION_BOUND,
                            )
                        else:
                            price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0,
                                float(coef[0, 0]),
                                float(coef[1, 0]),
                                float(coef[2, 0]),
                                float(p[0]),
                                float(z[0]),
                                -_EXTRAPOLATION_BOUND, float(z[0]),
                            )
                            pricea += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                                0.0,
                                float(coef_a[0, 0]),
                                float(coef_a[1, 0]),
                                float(coef_a[2, 0]),
                                float(pa[0]),
                                float(z[0]),
                                -_EXTRAPOLATION_BOUND, float(z[0]),
                            )

                npv0[k] = price
                npv0a[k] = pricea

                # Event-date updates.
                if is_event_date:
                    zk = float(z[k]) if event0 > expiry else y

                    if is_leg1_fixing:
                        self._add_leg_contribution(
                            event0, event0_time, zk, k, npv0a,
                            args.leg1_fixing_dates,
                            args.leg1_pay_dates,
                            args.leg1_accrual_times,
                            args.leg1_spreads,
                            args.leg1_gearings,
                            args.leg1_capped_rates,
                            args.leg1_floored_rates,
                            args.nominal1,
                            args.leg1_is_redemption_flow,
                            args.leg1_coupons,
                            args.index1,
                            sign=-1.0,
                        )

                    if is_leg2_fixing:
                        self._add_leg_contribution(
                            event0, event0_time, zk, k, npv0a,
                            args.leg2_fixing_dates,
                            args.leg2_pay_dates,
                            args.leg2_accrual_times,
                            args.leg2_spreads,
                            args.leg2_gearings,
                            args.leg2_capped_rates,
                            args.leg2_floored_rates,
                            args.nominal2,
                            args.leg2_is_redemption_flow,
                            args.leg2_coupons,
                            args.index2,
                            sign=+1.0,
                        )

                    if is_exercise:
                        # # C++ parity: cpp:591-649 (rebate skipped).
                        exercise_value = (
                            float(option_call_put) * npv0a[k]
                            # No rebated exercise / OAS in W1-B port.
                        )
                        npv0[k] = max(npv0[k], exercise_value)

            # Swap the buffers.
            npv1 = npv0.copy()
            npv1a = npv0a.copy()
            npv0[:] = 0.0
            npv0a[:] = 0.0

            event1_time = event0_time
            idx -= 1

        assert event1_time is not None
        numeraire_factor = self._model.numeraire(event1_time, y, self._discount_curve)
        return (
            float(npv1[0]) * numeraire_factor,
            float(npv1a[0]) * numeraire_factor * float(option_call_put),
        )

    # ------------------------------------------------------------------
    # Internal helper: add per-leg coupon contribution at an event date.
    # ------------------------------------------------------------------
    def _add_leg_contribution(
        self,
        event0: Date,
        event0_time: float,
        zk: float,
        k: int,
        npv_underlying: np.ndarray,
        fixing_dates: list[Date],
        pay_dates: list[Date],
        accrual_times: list[float],
        spreads: list[float],
        gearings: list[float],
        capped: list[float],
        floored: list[float],
        nominal: list[float],
        is_redemption: list[bool],
        coupons: list[float],
        ibor_index: object | None,
        sign: float,
    ) -> None:
        """# C++ parity: gaussian1dfloatfloatswaptionengine.cpp:430-589.

        Handles the IborIndex-only case (CMS / CMS-spread are W1-B
        carve-outs). Multiple coupons can share the same fixing date
        (rare but allowed by the schedule); the inner ``done`` loop
        consumes all of them.
        """
        # Locate the first coupon whose fixing date matches ``event0``.
        j = fixing_dates.index(event0)
        done = False
        while not done:
            amount: float
            if is_redemption[j]:
                amount = coupons[j]
            else:
                est_fixing = 0.0
                if ibor_index is not None:
                    # # C++ parity: cpp:455-458.
                    from pquantlib.indexes.ibor_index import IborIndex  # noqa: PLC0415
                    assert isinstance(ibor_index, IborIndex)
                    est_fixing = self._model.forward_rate(
                        fixing_dates[j], event0, zk, ibor_index
                    )
                rate = spreads[j] + gearings[j] * est_fixing
                cap = capped[j]
                floor = floored[j]
                import math  # noqa: PLC0415
                if not math.isnan(cap):
                    rate = min(cap, rate)
                if not math.isnan(floor):
                    rate = max(floor, rate)
                amount = rate * nominal[j] * accrual_times[j]
            disc_pay = self._model.zerobond_date(
                pay_dates[j], event0, zk, self._discount_curve
            )
            num = self._model.numeraire(event0_time, zk, self._discount_curve)
            npv_underlying[k] += sign * amount * disc_pay / num
            if j < len(fixing_dates) - 1:
                j += 1
                done = event0 != fixing_dates[j]
            else:
                done = True


__all__ = ["Gaussian1dFloatFloatSwaptionEngine"]
