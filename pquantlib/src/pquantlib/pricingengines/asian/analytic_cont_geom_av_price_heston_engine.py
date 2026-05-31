"""AnalyticContinuousGeometricAveragePriceAsianHestonEngine.

# C++ parity: ql/experimental/asian/analytic_cont_geom_av_price_heston.{hpp,cpp}
#             (v1.42.1).

Closed-form (semi-analytic) pricing of a European continuous geometric
average-price Asian option under the Heston stochastic-volatility model,
following Kim & Wee, "Pricing of geometric Asian options under Heston's
stochastic volatility model", Quantitative Finance 14:10 (2014).

The value is assembled from two terms (eq. 29 of the paper): a real
"asian forward minus strike" term and a characteristic-function integral
``term2`` computed by Gauss-Legendre quadrature over the inversion
contour. The characteristic function ``Phi`` (eq. 25) is built from a
highly recursive coefficient function ``f`` (eq. 21) evaluated with
memoization.

Tolerance note: like the L4-C ``AnalyticHestonEngine`` this is a
numerical CF integration; agreement with C++ is at the LOOSE tier.
"""

from __future__ import annotations

import cmath
import math

import numpy as np

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.asian_option import (
    AverageType,
    ContinuousAveragingAsianOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.heston_process import HestonProcess

# Gauss-Legendre nodes/weights on [-1, 1]. C++ uses GaussLegendreIntegration(128);
# numpy.leggauss gives the identical nodes/weights so ``sum(w_i f(x_i))`` matches.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(128)
_I = complex(0.0, 1.0)


class AnalyticContinuousGeometricAveragePriceAsianHestonEngine(
    GenericEngine[ContinuousAveragingAsianOptionArguments, OneAssetOptionResults]
):
    """Kim-Wee continuous geometric-average Asian engine under Heston.

    # C++ parity: ``AnalyticContinuousGeometricAveragePriceAsianHestonEngine``.

    Args:
        process: the Heston process.
        summation_cutoff: terms kept in the F / F-tilde series (eq. 19/20).
        xi_right_limit: upper bound of the inversion integral (eq. 29).
    """

    def __init__(
        self,
        process: HestonProcess,
        summation_cutoff: int = 50,
        xi_right_limit: float = 100.0,
    ) -> None:
        super().__init__(
            ContinuousAveragingAsianOptionArguments(), OneAssetOptionResults()
        )
        self._process = process
        process.register_with(self)

        self._v0 = process.v0
        self._rho = process.rho
        self._kappa = process.kappa
        self._theta = process.theta
        self._sigma = process.sigma
        self._s0 = process.s0()
        self._risk_free_rate = process.risk_free_rate()
        self._dividend_yield = process.dividend_yield()

        self._summation_cutoff = summation_cutoff
        self._xi_right_limit = xi_right_limit

        # Constant intermediate variables (T-independent part).
        self._a1 = 2.0 * self._v0 / (self._sigma * self._sigma)
        self._a2 = 2.0 * self._kappa * self._theta / (self._sigma * self._sigma)
        # T-dependent; reset per pricing.
        self._a3 = 0.0
        self._a4 = 0.0
        self._a5 = 0.0
        # Memoization table for f().
        self._f_lookup: dict[int, complex] = {}

    # --- equations (13): z1..z4 -----------------------------------------

    def _z1_f(self, s: complex, w: complex, t_total: float) -> complex:
        del w  # C++ z1_f keeps w in the signature but doesn't use it.
        return s * s * (1 - self._rho * self._rho) / (2 * t_total * t_total)

    def _z2_f(self, s: complex, w: complex, t_total: float) -> complex:
        return s * (2 * self._rho * self._kappa - self._sigma) / (
            2 * self._sigma * t_total
        ) + s * w * (1 - self._rho * self._rho) / t_total

    def _z3_f(self, s: complex, w: complex, t_total: float) -> complex:
        return (
            s * self._rho / (self._sigma * t_total)
            + 0.5 * w * (2 * self._rho * self._kappa - self._sigma) / self._sigma
            + 0.5 * w * w * (1 - self._rho * self._rho)
        )

    def _z4_f(self, w: complex) -> complex:
        return w * self._rho / self._sigma

    # --- equation (21): recursive f with memoization --------------------

    def _f(self, z1: complex, z2: complex, z3: complex, z4: complex, n: int, tau: float) -> complex:
        if n < 2:
            if n < 0:
                result: complex = 0.0
            elif n == 0:
                result = 1.0
            else:
                result = 0.5 * (self._kappa - z4 * self._sigma * self._sigma) * tau
        else:
            prefactor = -0.5 * self._sigma * self._sigma * tau * tau / (n * (n - 1))
            f_minus = [complex(0.0)] * 4
            for offset in range(1, 5):
                location = n - offset
                cached = self._f_lookup.get(location)
                if cached is not None:
                    f_minus[offset - 1] = cached
                else:
                    f_minus[offset - 1] = self._f(z1, z2, z3, z4, location, tau)
            result = prefactor * (
                z1 * tau * tau * f_minus[3]
                + z2 * tau * f_minus[2]
                + (z3 - 0.5 * self._kappa * self._kappa / (self._sigma * self._sigma)) * f_minus[1]
            )
        self._f_lookup[n] = result
        return result

    # --- equations (19)/(20): F, F_tilde --------------------------------

    def _f_f_tilde(
        self, z1: complex, z2: complex, z3: complex, z4: complex, tau: float, cutoff: int
    ) -> tuple[complex, complex]:
        running_sum1 = complex(0.0)
        running_sum2 = complex(0.0)
        for i in range(cutoff):
            temp = self._f(z1, z2, z3, z4, i, tau)
            running_sum1 += temp
            running_sum2 += temp * i / tau
        return running_sum1, running_sum2

    # --- equation (25): Phi ---------------------------------------------

    def phi(self, s: complex, w: complex, t_total: float, t: float = 0.0, cutoff: int = 50) -> complex:
        """Characteristic function Phi (eq. 25). Public for the integrand."""
        tau = t_total - t
        z1 = self._z1_f(s, w, t_total)
        z2 = self._z2_f(s, w, t_total)
        z3 = self._z3_f(s, w, t_total)
        z4 = self._z4_f(w)

        # Clear the memo table before the series sum.
        self._f_lookup = {}
        f_val, f_tilde = self._f_f_tilde(z1, z2, z3, z4, tau, cutoff)

        return cmath.exp(
            -self._a1 * f_tilde / f_val
            - self._a2 * cmath.log(f_val)
            + self._a3 * s
            + self._a4 * w
            + self._a5
        )

    # --- integrands (Gauss-Legendre over [-1, 1]) -----------------------

    def _integrand(self, xi: float, t_total: float, k: float, log_k: float, cutoff: int) -> float:
        xi_dash = (0.5 + 1e-8 + 0.5 * xi) * self._xi_right_limit  # map xi to full range
        inner1 = self.phi(1.0 + xi_dash * _I, 0.0, t_total, 0.0, cutoff)
        inner2 = -k * self.phi(xi_dash * _I, 0.0, t_total, 0.0, cutoff)
        return (
            0.5
            * self._xi_right_limit
            * ((inner1 + inner2) * cmath.exp(-xi_dash * log_k * _I) / (xi_dash * _I)).real
        )

    def _integrated_dcf(self, t: float, t_total: float) -> float:
        denominator = math.log(self._risk_free_rate.discount(t)) - math.log(
            self._dividend_yield.discount(t)
        )
        total = 0.0
        for node, weight in zip(_GL_NODES, _GL_WEIGHTS, strict=True):
            u_dash = (0.5 + 1e-8 + 0.5 * node) * (t_total - t) + t  # map u to full range
            val = (
                0.5
                * (t_total - t)
                * (
                    -math.log(self._risk_free_rate.discount(u_dash))
                    + math.log(self._dividend_yield.discount(u_dash))
                    + denominator
                )
            )
            total += weight * val
        return total

    def calculate(self) -> None:
        args = self._arguments
        results = self._results
        qassert.require(
            args.average_type == AverageType.Geometric, "not a geometric average option"
        )
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European, "not an European Option"
        )
        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)

        strike = payoff.strike()
        exercise_date = args.exercise.last_date()
        expiry_time = self._process.time(exercise_date)
        qassert.require(expiry_time >= 0.0, "Expiry Date cannot be in the past")

        expiry_dcf = self._risk_free_rate.discount(expiry_time)
        expiry_dividend_discount = self._dividend_yield.discount(expiry_time)

        # Unseasoned only (TODO in C++ too).
        t = 0.0
        t_total = expiry_time
        tau = t_total - t
        log_s0 = math.log(self._s0.value())

        dcf = self._risk_free_rate.discount(t_total) / self._risk_free_rate.discount(t)
        qdcf = self._dividend_yield.discount(t_total) / self._dividend_yield.discount(t)
        integrated_dcf = self._integrated_dcf(t, t_total)

        self._a3 = (
            (tau * log_s0 + integrated_dcf) / t_total
            - self._kappa * self._theta * self._rho * tau * tau / (2 * self._sigma * t_total)
            - self._rho * tau * self._v0 / (self._sigma * t_total)
        )
        self._a4 = (
            log_s0 * qdcf / dcf
            - self._rho * self._v0 / self._sigma
            + self._rho * self._kappa * self._theta * tau / self._sigma
        )
        self._a5 = (
            self._kappa * self._v0 + self._kappa * self._kappa * self._theta * tau
        ) / (self._sigma * self._sigma)

        # term1 = 0.5*(Re Phi(1,0) - strike) (Phi(1,0) is the Asian forward).
        term1 = 0.5 * (
            self.phi(complex(1.0), complex(0.0), t_total, t, self._summation_cutoff).real - strike
        )

        log_k = math.log(strike)
        term2_raw = 0.0
        for node, weight in zip(_GL_NODES, _GL_WEIGHTS, strict=True):
            term2_raw += weight * self._integrand(
                node, t_total, strike, log_k, self._summation_cutoff
            )
        term2 = term2_raw / math.pi

        if payoff.option_type() == OptionType.Call:
            value = expiry_dcf * (term1 + term2)
        elif payoff.option_type() == OptionType.Put:
            value = expiry_dcf * (-term1 + term2)
        else:
            qassert.fail("unknown option type")
            raise LibraryException("unknown option type")

        results.value = value
        results.additional_results["dcf"] = expiry_dcf
        results.additional_results["qf"] = expiry_dividend_discount
        results.additional_results["s0"] = self._s0.value()
        results.additional_results["strike"] = strike
        results.additional_results["expiryTime"] = expiry_time
        results.additional_results["term1"] = term1
        results.additional_results["term2"] = term2
        results.additional_results["a1"] = self._a1
        results.additional_results["a2"] = self._a2
        results.additional_results["a3"] = self._a3
        results.additional_results["a4"] = self._a4
        results.additional_results["a5"] = self._a5


__all__ = ["AnalyticContinuousGeometricAveragePriceAsianHestonEngine"]
