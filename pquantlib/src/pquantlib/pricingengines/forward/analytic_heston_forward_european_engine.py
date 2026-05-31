"""AnalyticHestonForwardEuropeanEngine — Heston forward-starting European.

# C++ parity: ql/experimental/forward/analytichestonforwardeuropeanengine.{hpp,cpp}
#             (v1.42.1).

Analytic pricing of a forward-starting (strike-reset) European option
under Heston, following Kruse (2003), "On the Pricing of Forward Starting
Options under Stochastic Volatility". The price requires a nested 2D
integration: an outer integral over the reset-time variance ``nu``
(weighted by the non-central chi-square propagator, eq. 18) of an inner
characteristic-function inversion for the conditional P1/P2 probabilities.

The Heston characteristic function ``chF`` is reproduced inline from the
Andersen-Lake-stabilized ``AnalyticHestonEngine::lnChF``; the propagator's
modified Bessel function ``I_nu`` delegates to ``scipy.special.iv``.

Both nested integrals use 128-point Gauss-Legendre quadrature, matching
the C++ ``GaussLegendreIntegration(128)``. Tolerance: LOOSE (numerical
double integration).
"""

from __future__ import annotations

import cmath
import math

import numpy as np
from scipy.special import iv  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.forward_vanilla_option import ForwardOptionArguments
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.heston_process import HestonProcess

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(128)
_I = complex(0.0, 1.0)


class AnalyticHestonForwardEuropeanEngine(
    GenericEngine[ForwardOptionArguments, OneAssetOptionResults]
):
    """Kruse forward-starting European engine under Heston.

    # C++ parity: ``AnalyticHestonForwardEuropeanEngine``.

    Args:
        process: the Heston process (requires sigma > 0.1 — see C++ note).
        integration_order: order of the (unused, kept for API parity)
            sub-engine integrator; the actual quadratures are GL(128).
    """

    def __init__(self, process: HestonProcess, integration_order: int = 144) -> None:
        super().__init__(ForwardOptionArguments(), OneAssetOptionResults())
        self._process = process
        process.register_with(self)

        self._v0 = process.v0
        self._rho = process.rho
        self._kappa = process.kappa
        self._theta = process.theta
        self._sigma = process.sigma
        self._s0 = process.s0()
        self._integration_order = integration_order

        qassert.require(
            self._sigma > 0.1,
            "Very low values (<~10%) for Heston Vol-of-Vol cause numerical issues "
            "in this implementation of the propagator function, try using "
            "MCForwardEuropeanHestonEngine Monte-Carlo engine instead",
        )

        self._risk_free_rate = process.risk_free_rate()
        self._dividend_yield = process.dividend_yield()

        # Constant intermediate variables.
        self._kappa_hat = self._kappa - self._rho * self._sigma
        self._theta_hat = self._kappa * self._theta / self._kappa_hat
        self._r = 4 * self._kappa_hat * self._theta_hat / (self._sigma * self._sigma)

    # --- Heston characteristic function (Andersen-Lake lnChF) -----------

    def _ln_chf(self, z: complex, t: float, v_reset: float) -> complex:
        kappa = self._kappa
        sigma = self._sigma
        theta = self._theta
        rho = self._rho
        sigma2 = sigma * sigma

        # g = kappa + rho*sigma*(z.imag - i z.real)
        g = kappa + rho * sigma * complex(z.imag, -z.real)
        d = cmath.sqrt(g * g + (z * z + complex(-z.imag, z.real)) * sigma2)

        r = g - d
        if g.real * d.real + g.imag * d.imag > 0.0:
            r = -sigma2 * z * complex(z.real, z.imag + 1) / (g + d)

        y = np.expm1(-d * t) / (2.0 * d) if (d.real != 0.0 or d.imag != 0.0) else complex(-0.5 * t)

        a = kappa * theta / sigma2 * (r * t - 2.0 * np.log1p(-r * y))
        b = z * complex(z.real, z.imag + 1) * y / (1.0 - r * y)
        return a + v_reset * b

    def _chf(self, z: complex, t: float, v_reset: float) -> complex:
        # sigma > 0.1 (enforced in ctor) => sigma > 1e-6, so chF = exp(lnChF).
        return cmath.exp(self._ln_chf(z, t, v_reset))

    # --- inner P1/P2 integrand ------------------------------------------

    def _p12_integral(
        self, log_k: float, tenor: float, v_reset: float, p1: bool, phi_right_limit: float
    ) -> float:
        adj = complex(0.0, -1.0) if p1 else complex(0.0, 0.0)
        total = 0.0
        for node, weight in zip(_GL_NODES, _GL_WEIGHTS, strict=True):
            phi_dash = (0.5 + 1e-8 + 0.5 * node) * phi_right_limit
            val = (
                0.5
                * phi_right_limit
                * (
                    (cmath.exp(-phi_dash * log_k * _I) / (phi_dash * _I))
                    * self._chf(phi_dash + adj, tenor, v_reset)
                ).real
            )
            total += weight * val
        return total

    # --- propagator (eq. 18, non-central chi-square density) ------------

    def propagator(self, reset_time: float, var_reset: float) -> float:
        """Variance-evolution density from t0 to reset_time (eq. 18)."""
        b = 4 * self._kappa_hat / (self._sigma * self._sigma * (1 - math.exp(-self._kappa_hat * reset_time)))
        lam = b * math.exp(-self._kappa_hat * reset_time) * self._v0

        term1 = math.exp(-0.5 * (b * var_reset + lam)) * b / 2
        term2 = (b * var_reset / lam) ** (0.5 * (self._r / 2 - 1))
        term3 = float(iv(self._r / 2 - 1, math.sqrt(lam * b * var_reset)))
        return term1 * term2 * term3

    # --- outer P1Hat/P2Hat integration ----------------------------------

    def _calculate_p1p2_hat(
        self,
        tenor: float,
        reset_time: float,
        moneyness: float,
        ratio: float,
        phi_right_limit: float,
        nu_right_limit: float,
    ) -> tuple[float, float]:
        # Moneyness re-expressed in terms of the forward at expiry.
        log_moneyness = math.log(moneyness * ratio)

        def outer(p1: bool) -> float:
            total = 0.0
            for node, weight in zip(_GL_NODES, _GL_WEIGHTS, strict=True):
                nu_dash = nu_right_limit * (0.5 * node + 0.5 + 1e-8)
                p_integral = self._p12_integral(
                    log_moneyness, tenor, nu_dash, p1, phi_right_limit
                )
                propagator = self.propagator(reset_time, nu_dash)
                total += weight * (propagator * (0.5 + p_integral / math.pi))
            return 0.5 * nu_right_limit * total

        return outer(True), outer(False)

    # --- fallback for very short reset times ----------------------------

    def _calculate_p1p2(
        self, tenor: float, st: float, k: float, ratio: float, phi_right_limit: float
    ) -> tuple[float, float]:
        log_k = math.log(k * ratio / st)
        p1_integral = self._p12_integral(log_k, tenor, self._v0, True, phi_right_limit)
        p2_integral = self._p12_integral(log_k, tenor, self._v0, False, phi_right_limit)
        return 0.5 + p1_integral / math.pi, 0.5 + p2_integral / math.pi

    def calculate(self) -> None:
        args = self._arguments
        results = self._results
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European, "not an European option"
        )
        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non plain vanilla payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)
        assert args.reset_date is not None
        assert args.moneyness is not None

        reset_time = self._process.time(args.reset_date)
        expiry_time = self._process.time(args.exercise.last_date())
        tenor = expiry_time - reset_time
        moneyness = args.moneyness

        expiry_dcf = self._risk_free_rate.discount(expiry_time)
        reset_dcf = self._risk_free_rate.discount(reset_time)
        expiry_dividend_discount = self._dividend_yield.discount(expiry_time)
        reset_dividend_discount = self._dividend_yield.discount(reset_time)
        expiry_ratio = expiry_dcf / expiry_dividend_discount
        reset_ratio = reset_dcf / reset_dividend_discount

        qassert.require(reset_time >= 0.0, "Reset Date cannot be in the past")
        qassert.require(expiry_time >= 0.0, "Expiry Date cannot be in the past")

        phi_right_limit = 100.0
        nu_right_limit = max(
            2.0,
            10.0
            * (1 + max(0.0, self._rho))
            * self._sigma
            * math.sqrt(reset_time * max(self._v0, self._theta)),
        )

        if reset_time <= 1e-3:
            p1_hat, p2_hat = self._calculate_p1p2(
                tenor, self._s0.value(), moneyness * self._s0.value(), expiry_ratio, phi_right_limit
            )
        else:
            p1_hat, p2_hat = self._calculate_p1p2_hat(
                tenor,
                reset_time,
                moneyness,
                expiry_ratio / reset_ratio,
                phi_right_limit,
                nu_right_limit,
            )

        s0 = self._s0.value()
        fwd = s0 / expiry_ratio
        if payoff.option_type() == OptionType.Call:
            value = expiry_dcf * (fwd * p1_hat - moneyness * s0 * p2_hat / reset_ratio)
        elif payoff.option_type() == OptionType.Put:
            value = expiry_dcf * (
                moneyness * s0 * (1 - p2_hat) / reset_ratio - fwd * (1 - p1_hat)
            )
        else:
            qassert.fail("unknown option type")
            raise LibraryException("unknown option type")

        results.value = value
        results.additional_results["dcf"] = expiry_dcf
        results.additional_results["qf"] = expiry_dividend_discount
        results.additional_results["expiryRatio"] = expiry_ratio
        results.additional_results["resetRatio"] = reset_ratio
        results.additional_results["moneyness"] = moneyness
        results.additional_results["s0"] = s0
        results.additional_results["fwd"] = fwd
        results.additional_results["resetTime"] = reset_time
        results.additional_results["expiryTime"] = expiry_time
        results.additional_results["P1Hat"] = p1_hat
        results.additional_results["P2Hat"] = p2_hat


__all__ = ["AnalyticHestonForwardEuropeanEngine"]
