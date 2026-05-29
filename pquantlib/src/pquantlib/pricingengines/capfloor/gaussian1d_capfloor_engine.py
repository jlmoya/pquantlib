"""Gaussian1d cap/floor engine — payoff-replication under a Gaussian1d model.

# C++ parity: ql/pricingengines/capfloor/gaussian1dcapfloorengine.{hpp,cpp}
# @ v1.42.1 (099987f0).

Algorithm (per-coupon, matches gaussian1dcapfloorengine.cpp:26-220):

For each optionlet i with fixing date ``t_fix`` and payment date
``t_pay``, build the 1-D state grid ``z = model.y_grid(stddevs,
integration_points)`` at standardized state. For each grid point
``z[j]``, compute the conditional caplet payoff:

    p[j] = max(N * tau * (F(t_fix, z[j]) - K), 0) * P(t_pay, t_fix, z[j])
           / N(t_fix, z[j])

where ``F`` is the Ibor forward via ``model.forward_rate``, ``P`` is
the model zero-bond, ``N`` is the numeraire, ``K`` is the strike,
``tau`` is the accrual period. The forward measure carries the
discounting via the numeraire ratio.

Then fit a natural cubic spline ``p(z)`` and integrate against the
standard-normal density piecewise — the integral on each interval is
computed in closed form via
:meth:`Gaussian1dModel.gaussian_shifted_polynomial_integral`. Optional
extrapolation handles the tails outside the grid (default: cubic
extrapolation matching the boundary segment).

The total optionlet price is multiplied by the model's t=0 numeraire
``N(0, 0)`` and by the nominal-times-gearing factor ``f``.

For a floor, swap ``floatingLegNpv - fixedLegNpv`` sign. For a collar,
cap NPV minus floor NPV per coupon (per the standard convention).

Divergences from C++:

- C++ uses ``Lagrange`` boundary conditions (Lagrange-style 2nd-order
  extrapolating BC) and ``Spline`` derivative approximation. PQuantLib
  uses ``CubicNaturalSpline`` (natural BC + Spline approximation). The
  difference is small in practice — both pass through the same knots;
  the only divergence is in the off-knot quadrature for the boundary
  segments, which is dominated by the body of the integral.
- C++ ``optionletsAtmForward`` additional result is left at zero (the
  C++ code never populates it either — it's a placeholder).
- The optional ``discountCurve`` Handle that overrides the model's
  fixed term-structure for the discounting leg is supported via the
  ``discount_curve`` constructor argument.
- C++ ``openmp`` parallelism (commented "todo") is not added.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.instruments.cap_floor import (
    CapFloorArguments,
    CapFloorResults,
    CapFloorType,
)
from pquantlib.models.shortrate.gaussian1d_model import Gaussian1dModel
from pquantlib.pricingengines.generic_engine import GenericEngine

if TYPE_CHECKING:
    from pquantlib.indexes.ibor_index import IborIndex
    from pquantlib.termstructures.protocols import YieldTermStructureProtocol


# Sentinel used by C++ for the [-inf, x_0] / [x_n, inf] tails — the
# integration uses Gaussian density so anything > ~7 stddevs is
# numerically zero. The literal value matches gaussian1dcapfloorengine.cpp
# (uses ±100.0 as the effective bound).
_EXTRAPOLATION_BOUND: float = 100.0


class Gaussian1dCapFloorEngine(GenericEngine[CapFloorArguments, CapFloorResults]):
    """Cap/floor engine under a one-factor Gaussian short-rate model.

    # C++ parity: ``class Gaussian1dCapFloorEngine`` in
    # gaussian1dcapfloorengine.hpp:38-60 (v1.42.1).

    Reads the model's term-structure ``referenceDate`` to gate already-
    fixed coupons; uses the model's ``forward_rate``, ``zerobond``,
    ``numeraire``, and ``y_grid`` to build the per-state caplet payoff
    grid; integrates the cubic-spline interpolant against the
    standard-normal density via the closed-form polynomial integral.
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
        # # C++ parity: gaussian1dcapfloorengine.hpp:42-52.
        super().__init__(CapFloorArguments(), CapFloorResults())
        self._model: Gaussian1dModel = model
        self._integration_points: int = int(integration_points)
        self._stddevs: float = float(stddevs)
        self._extrapolate_payoff: bool = bool(extrapolate_payoff)
        self._flat_payoff_extrapolation: bool = bool(flat_payoff_extrapolation)
        self._discount_curve: YieldTermStructureProtocol | None = discount_curve

    def calculate(self) -> None:
        # # C++ parity: gaussian1dcapfloorengine.cpp:26-220.
        args = self._arguments
        results = self._results
        results.reset()

        # Forbid non-zero per-coupon spreads — the engine assumes the
        # raw Ibor forward enters the payoff.
        for spread in args.spreads:
            qassert.require(
                spread == 0.0,
                f"Non zero spreads ({spread}) are not allowed.",
            )

        optionlets = len(args.start_dates)
        values = [0.0] * optionlets

        # Term-structure reference date — coupons with payment_date <=
        # settlement are skipped (already paid).
        settlement = self._model.term_structure.reference_date()
        capfloor_type = args.type

        # Integration grid for the standardized state.
        z = self._model.y_grid(self._stddevs, self._integration_points)
        z_size = int(z.size)

        value = 0.0

        for i in range(optionlets):
            value_date = args.start_dates[i]
            payment_date = args.end_dates[i]
            ibor_index = args.indexes[i]
            # If we do not find an ibor index with associated
            # forwarding curve, we fall back on the model curve via
            # the (P(t_pay) - P(t_value)) two-discount trick (see
            # below).

            if payment_date <= settlement:
                # Coupon already paid — zero contribution.
                continue

            f = args.nominals[i] * args.gearings[i]
            fixing_date = args.fixing_dates[i]
            fixing_time = self._model.term_structure.time_from_reference(fixing_date)

            cap_value = 0.0
            floor_value = 0.0

            # ---------------- CAP / Collar (cap leg) --------------------
            if capfloor_type in (CapFloorType.Cap, CapFloorType.Collar):
                cap_strike = args.cap_rates[i]
                if fixing_date <= settlement:
                    # Coupon fixed in the past — intrinsic against the
                    # observed forward (stored on args.forwards[i]).
                    fwd = args.forwards[i]
                    cap_value = max(fwd - cap_strike, 0.0) * f * args.accrual_times[i]
                else:
                    cap_value = self._integrate_optionlet(
                        z=z,
                        z_size=z_size,
                        strike=cap_strike,
                        is_cap=True,
                        value_date=value_date,
                        payment_date=payment_date,
                        fixing_date=fixing_date,
                        fixing_time=fixing_time,
                        accrual_time=args.accrual_times[i],
                        ibor_index=ibor_index,  # type: ignore[arg-type]
                        f=f,
                    )

            # ---------------- FLOOR / Collar (floor leg) ---------------
            if capfloor_type in (CapFloorType.Floor, CapFloorType.Collar):
                floor_strike = args.floor_rates[i]
                if fixing_date <= settlement:
                    fwd = args.forwards[i]
                    floor_value = max(-(fwd - floor_strike), 0.0) * f * args.accrual_times[i]
                else:
                    floor_value = self._integrate_optionlet(
                        z=z,
                        z_size=z_size,
                        strike=floor_strike,
                        is_cap=False,
                        value_date=value_date,
                        payment_date=payment_date,
                        fixing_date=fixing_date,
                        fixing_time=fixing_time,
                        accrual_time=args.accrual_times[i],
                        ibor_index=ibor_index,  # type: ignore[arg-type]
                        f=f,
                    )

            # Combine.
            if capfloor_type == CapFloorType.Cap:
                values[i] = cap_value
            elif capfloor_type == CapFloorType.Floor:
                values[i] = floor_value
            else:
                # Collar = long cap + short floor.
                values[i] = cap_value - floor_value

            value += values[i]

        results.value = value
        results.additional_results["optionletsPrice"] = values

    # ------------------------------------------------------------------
    # internal: one-optionlet integration over the state grid
    # ------------------------------------------------------------------
    def _integrate_optionlet(
        self,
        *,
        z: np.ndarray,
        z_size: int,
        strike: float,
        is_cap: bool,
        value_date: object,
        payment_date: object,
        fixing_date: object,
        fixing_time: float,
        accrual_time: float,
        ibor_index: IborIndex | None,
        f: float,
    ) -> float:
        # # C++ parity: gaussian1dcapfloorengine.cpp:69-133 (cap branch)
        # and 142-203 (floor branch).
        p = np.zeros(z_size, dtype=np.float64)
        for j in range(z_size):
            zj = float(z[j])
            if ibor_index is not None:
                floating_leg_npv = (
                    accrual_time
                    * self._model.forward_rate(
                        fixing_date,  # type: ignore[arg-type]
                        fixing_date,  # type: ignore[arg-type]
                        zj,
                        ibor_index,
                    )
                    * self._model.zerobond_date(
                        payment_date,  # type: ignore[arg-type]
                        fixing_date,  # type: ignore[arg-type]
                        zj,
                        self._discount_curve,
                    )
                )
            else:
                # No ibor index → use (P(value_date) - P(payment_date))
                # / 1 since the implied forward * accrual ≈ this
                # difference for a coupon period.
                floating_leg_npv = self._model.zerobond_date(
                    value_date,  # type: ignore[arg-type]
                    fixing_date,  # type: ignore[arg-type]
                    zj,
                ) - self._model.zerobond_date(
                    payment_date,  # type: ignore[arg-type]
                    fixing_date,  # type: ignore[arg-type]
                    zj,
                )

            fixed_leg_npv = (
                strike
                * accrual_time
                * self._model.zerobond_date(
                    payment_date,  # type: ignore[arg-type]
                    fixing_date,  # type: ignore[arg-type]
                    zj,
                )
            )

            num = self._model.numeraire(fixing_time, zj, self._discount_curve)
            if is_cap:
                p[j] = max(floating_leg_npv - fixed_leg_npv, 0.0) / num
            else:
                p[j] = max(-(floating_leg_npv - fixed_leg_npv), 0.0) / num

        # Fit a natural cubic spline + extract per-interval polynomial
        # coefficients in C++'s (a=linear, b=quadratic, c=cubic) ordering.
        # scipy PPoly: c[0,k] = cubic coef (C++ "c"), c[1,k] = quadratic
        # (C++ "b"), c[2,k] = linear (C++ "a"), c[3,k] = constant (= p[k]).
        spline = CubicSpline(z, p, bc_type="natural", extrapolate=True)
        coef = spline.c  # shape (4, z_size - 1)

        price = 0.0
        for j in range(z_size - 1):
            c_cubic = float(coef[0, j])
            b_quad = float(coef[1, j])
            a_lin = float(coef[2, j])
            # Constant coef is the spline value at z[j] (= p[j]).
            price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                0.0,
                c_cubic,
                b_quad,
                a_lin,
                float(p[j]),
                float(z[j]),
                float(z[j]),
                float(z[j + 1]),
            )

        if self._extrapolate_payoff:
            if self._flat_payoff_extrapolation:
                # Constant extension at p[-2] and p[0] for tails.
                price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float(p[z_size - 2]),
                    float(z[z_size - 2]),
                    float(z[z_size - 1]),
                    _EXTRAPOLATION_BOUND,
                )
                price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float(p[0]),
                    float(z[0]),
                    -_EXTRAPOLATION_BOUND,
                    float(z[0]),
                )
            # Cubic extension via the boundary segment polynomial.
            # C++ uses different polynomial extrapolation per
            # left/right end depending on cap vs floor; the
            # asymmetry reflects: cap payoffs grow into the
            # high-state tail (extend on the right with the cubic);
            # floor payoffs grow into the low-state tail (extend on
            # the left with the cubic). Match the same convention.
            elif is_cap:
                c_cubic = float(coef[0, z_size - 2])
                b_quad = float(coef[1, z_size - 2])
                a_lin = float(coef[2, z_size - 2])
                price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                    0.0,
                    c_cubic,
                    b_quad,
                    a_lin,
                    float(p[z_size - 2]),
                    float(z[z_size - 2]),
                    float(z[z_size - 1]),
                    _EXTRAPOLATION_BOUND,
                )
            else:
                c_cubic = float(coef[0, 0])
                b_quad = float(coef[1, 0])
                a_lin = float(coef[2, 0])
                price += Gaussian1dModel.gaussian_shifted_polynomial_integral(
                    0.0,
                    c_cubic,
                    b_quad,
                    a_lin,
                    float(p[0]),
                    float(z[0]),
                    -_EXTRAPOLATION_BOUND,
                    float(z[0]),
                )

        # Final scaling: multiply by the t=0 numeraire (curve discount to
        # the forward-measure horizon for Gsr) and the nominal-gearing.
        return price * self._model.numeraire(0.0, 0.0, self._discount_curve) * f


__all__ = ["Gaussian1dCapFloorEngine"]
