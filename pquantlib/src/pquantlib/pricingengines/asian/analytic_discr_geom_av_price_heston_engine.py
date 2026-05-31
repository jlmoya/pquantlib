"""AnalyticDiscreteGeometricAveragePriceAsianHestonEngine.

# C++ parity: ql/experimental/asian/analytic_discr_geom_av_price_heston.{hpp,cpp}
#             (v1.42.1).

Semi-analytic pricing of a European discrete geometric average-price Asian
option under the Heston model (Kim & Wee 2014, discrete variant). The
value (eq. 23) is the sum of a real term1 and a CF-inversion ``term2``
computed by 128-point Gauss-Legendre quadrature. The characteristic
function ``Phi`` (eq. 21) is built from F / F_tilde (eq. 11), z (eq. 14),
omega (eq. 15), a (eq. 16) and the recursive omega_tilde (eq. 19, memoized
by fixing index).

Tolerance note: numerical CF integration — LOOSE-tier agreement with C++
(empirically machine precision when the GL nodes match).
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
    DiscreteAveragingAsianOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.heston_process import HestonProcess

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(128)
_I = complex(0.0, 1.0)


class AnalyticDiscreteGeometricAveragePriceAsianHestonEngine(
    GenericEngine[DiscreteAveragingAsianOptionArguments, OneAssetOptionResults]
):
    """Kim-Wee discrete geometric-average Asian engine under Heston.

    # C++ parity: ``AnalyticDiscreteGeometricAveragePriceAsianHestonEngine``.

    Args:
        process: the Heston process.
        xi_right_limit: upper bound of the inversion integral.
    """

    def __init__(self, process: HestonProcess, xi_right_limit: float = 100.0) -> None:
        super().__init__(
            DiscreteAveragingAsianOptionArguments(), OneAssetOptionResults()
        )
        self._process = process
        process.register_with(self)

        self._v0 = process.v0
        self._rho = process.rho
        self._kappa = process.kappa
        self._theta = process.theta
        self._sigma = process.sigma
        self._s0 = process.s0()
        self._log_s0 = math.log(self._s0.value())
        self._risk_free_rate = process.risk_free_rate()
        self._dividend_yield = process.dividend_yield()

        self._xi_right_limit = xi_right_limit

        # Set in calculate(); read by a().
        self._tr_t = 0.0
        self._t_r_t = 0.0
        self._tkr_tk: list[float] = []
        # omega_tilde memo (per Phi call), keyed by fixing index.
        self._omega_tilde_lookup: dict[int, complex] = {}

    # --- equation (11): F, F_tilde --------------------------------------

    def _f(self, z1: complex, z2: complex, tau: float) -> complex:
        temp = cmath.sqrt(self._kappa * self._kappa - 2.0 * z1 * self._sigma * self._sigma)
        if abs(self._kappa * self._kappa - 2.0 * self._sigma * self._sigma) < 1e-8:
            return 1.0 + 0.5 * (self._kappa - z2 * self._sigma * self._sigma)
        return cmath.cosh(0.5 * tau * temp) + (
            self._kappa - z2 * self._sigma * self._sigma
        ) * cmath.sinh(0.5 * tau * temp) / temp

    def _f_tilde(self, z1: complex, z2: complex, tau: float) -> complex:
        temp = cmath.sqrt(self._kappa * self._kappa - 2.0 * z1 * self._sigma * self._sigma)
        return 0.5 * temp * cmath.sinh(0.5 * tau * temp) + 0.5 * (
            self._kappa - z2 * self._sigma * self._sigma
        ) * cmath.cosh(0.5 * tau * temp)

    # --- equation (14): z -----------------------------------------------

    def _z(self, s: complex, w: complex, k: int, n: int) -> complex:
        k_ = float(k)
        n_ = float(n)
        term1 = (2 * self._rho * self._kappa - self._sigma) * (
            (n_ - k_ + 1) * s + n_ * w
        ) / (2 * self._sigma * n_)
        term2 = (1 - self._rho * self._rho) * ((n_ - k_ + 1) * s + n_ * w) ** 2 / (2 * n_ * n_)
        return term1 + term2

    # --- equation (15): omega -------------------------------------------

    def _omega(self, s: complex, w: complex, k: int, k_star: int, n: int) -> complex:
        if k == k_star:
            return complex(0.0)
        if k == n + 1:
            return self._rho * w / self._sigma
        return self._rho * s / (self._sigma * n)

    # --- equation (16): a -----------------------------------------------

    def _a(
        self,
        s: complex,
        w: complex,
        t: float,
        t_total: float,
        k_star: int,
        t_n: list[float],
    ) -> complex:
        k_star_ = float(k_star)
        n_ = float(len(t_n))
        temp = -self._rho * self._kappa * self._theta / self._sigma

        summation = 0.0
        summation2 = 0.0
        for i in range(k_star + 1, len(t_n) + 1):
            summation += t_n[i - 1]
            summation2 += self._tkr_tk[i - 1]
        # Eq (16) modified for non-constant rates.
        term1 = (s * (n_ - k_star_) / n_ + w) * (
            self._log_s0 - self._rho * self._v0 / self._sigma - t * temp - self._tr_t
        )
        term2 = (
            temp * (s * summation / n_ + w * t_total) + w * self._t_r_t + summation2 * s / n_
        )
        return term1 + term2

    # --- equation (19): omega_tilde (recursive, memoized) ---------------

    def _omega_tilde(
        self, s: complex, w: complex, k: int, k_star: int, n: int, tau_k: list[float]
    ) -> complex:
        omega_k = self._omega(s, w, k, k_star, n)
        if k == n + 1:
            return omega_k
        d_tau_k = tau_k[k + 1] - tau_k[k]
        z_kp1 = self._z(s, w, k + 1, n)

        cached = self._omega_tilde_lookup.get(k + 1)
        omega_kp1 = (
            cached if cached is not None else self._omega_tilde(s, w, k + 1, k_star, n, tau_k)
        )

        ratio = self._f_tilde(z_kp1, omega_kp1, d_tau_k) / self._f(z_kp1, omega_kp1, d_tau_k)
        result = omega_k + self._kappa / self._sigma**2 - 2.0 * ratio / self._sigma**2
        self._omega_tilde_lookup[k] = result
        return result

    # --- equation (21): Phi ---------------------------------------------

    def phi(
        self,
        s: complex,
        w: complex,
        t: float,
        t_total: float,
        k_star: int,
        t_n: list[float],
        tau_k: list[float],
    ) -> complex:
        """Characteristic function Phi (eq. 21). Public for the integrand."""
        self._omega_tilde_lookup = {}

        n = len(t_n)
        a_term = self._a(s, w, t, t_total, k_star, t_n)
        omega_term = self._v0 * self._omega_tilde(s, w, k_star, k_star, n, tau_k)
        term3 = self._kappa * self._kappa * self._theta * (t_total - t) / self._sigma**2

        summation = complex(0.0)
        for i in range(k_star + 1, n + 2):
            d_tau = tau_k[i] - tau_k[i - 1]
            z_k = self._z(s, w, i, n)
            omega_tilde_k = self._omega_tilde(s, w, i, k_star, n, tau_k)
            summation += cmath.log(self._f(z_k, omega_tilde_k, d_tau))
        term4 = 2 * self._kappa * self._theta * summation / self._sigma**2

        return cmath.exp(a_term + omega_term + term3 - term4)

    # --- integrand ------------------------------------------------------

    def _integrand(
        self,
        xi: float,
        t: float,
        t_total: float,
        k_star: int,
        t_n: list[float],
        tau_k: list[float],
        k: float,
        log_k: float,
    ) -> float:
        xi_dash = (0.5 + 1e-8 + 0.5 * xi) * self._xi_right_limit
        inner1 = self.phi(1.0 + xi_dash * _I, complex(0.0), t, t_total, k_star, t_n, tau_k)
        inner2 = -k * self.phi(xi_dash * _I, complex(0.0), t, t_total, k_star, t_n, tau_k)
        return (
            0.5
            * self._xi_right_limit
            * ((inner1 + inner2) * cmath.exp(-xi_dash * log_k * _I) / (xi_dash * _I)).real
        )

    def calculate(self) -> None:  # noqa: PLR0915 (faithful C++ port; one long method)
        args = self._arguments
        results = self._results
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European, "not an European Option"
        )

        if args.average_type == AverageType.Geometric:
            assert args.running_accumulator is not None
            qassert.require(
                args.running_accumulator > 0.0,
                f"positive running product required: {args.running_accumulator} not allowed",
            )
            running_log = math.log(args.running_accumulator)
            assert args.past_fixings is not None
            past_fixings = args.past_fixings
        else:
            # Being used as a control variate for the arithmetic engine.
            running_log = 0.0
            past_fixings = 0

        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)
        strike = payoff.strike()
        exercise_date = args.exercise.last_date()
        expiry_time = self._process.time(exercise_date)
        qassert.require(expiry_time >= 0.0, "Expiry Date cannot be in the past")
        expiry_dcf = self._risk_free_rate.discount(expiry_time)

        start_time = 0.0
        fixing_times = sorted(self._process.time(d) for d in args.fixing_dates)
        tau_k = list(fixing_times)
        tau_k.insert(0, start_time)
        tau_k.append(expiry_time)

        # Dummy past fixings at t=-1 (front).
        for _ in range(past_fixings):
            fixing_times.insert(0, -1.0)
            tau_k.insert(0, -1.0)

        k_star = past_fixings

        # log discount factors for the r-adjusted a factor (eq 16).
        self._tr_t = -math.log(
            self._risk_free_rate.discount(start_time) / self._dividend_yield.discount(start_time)
        )
        self._t_r_t = -math.log(
            self._risk_free_rate.discount(expiry_time) / self._dividend_yield.discount(expiry_time)
        )
        self._tkr_tk = []
        for fixing_time in fixing_times:
            if fixing_time < 0:
                self._tkr_tk.append(1.0)
            else:
                self._tkr_tk.append(
                    -math.log(
                        self._risk_free_rate.discount(fixing_time)
                        / self._dividend_yield.discount(fixing_time)
                    )
                )

        # Adjusted strike for seasoning (eq 6).
        prefactor = math.exp(running_log / len(fixing_times))
        adjusted_strike = strike / prefactor

        term1 = 0.5 * (
            self.phi(
                complex(1.0), complex(0.0), start_time, expiry_time, k_star, fixing_times, tau_k
            ).real
            - adjusted_strike
        )

        log_k = math.log(adjusted_strike)
        term2_raw = 0.0
        for node, weight in zip(_GL_NODES, _GL_WEIGHTS, strict=True):
            term2_raw += weight * self._integrand(
                node, start_time, expiry_time, k_star, fixing_times, tau_k, adjusted_strike, log_k
            )
        term2 = term2_raw / math.pi

        if payoff.option_type() == OptionType.Call:
            value = expiry_dcf * prefactor * (term1 + term2)
        elif payoff.option_type() == OptionType.Put:
            value = expiry_dcf * prefactor * (-term1 + term2)
        else:
            qassert.fail("unknown option type")
            raise LibraryException("unknown option type")

        results.value = value
        results.additional_results["dcf"] = expiry_dcf
        results.additional_results["s0"] = self._s0.value()
        results.additional_results["strike"] = strike
        results.additional_results["expiryTime"] = expiry_time
        results.additional_results["term1"] = term1
        results.additional_results["term2"] = term2
        results.additional_results["adjustedStrike"] = adjusted_strike
        results.additional_results["prefactor"] = prefactor
        results.additional_results["kStar"] = k_star


__all__ = ["AnalyticDiscreteGeometricAveragePriceAsianHestonEngine"]
