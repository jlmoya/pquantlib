"""AnalyticPDFHestonEngine — PDF-based pricing under Heston.

# C++ parity:
# ql/pricingengines/vanilla/analyticpdfhestonengine.{hpp,cpp} (v1.42.1)
# + the ``HestonRNDCalculator`` machinery in
# ql/methods/finitedifferences/utilities/hestonrndcalculator.{hpp,cpp}.

Dragulescu-Yakovenko 2002 PDF-based engine for arbitrary European
payoffs under the Heston model:

  price = discount(T) * integral of payoff(s_t) * p(x_t = log(s_t), T) dx_t

The Heston log-spot PDF ``p(x, T)`` is reconstructed from its
characteristic function via numerical inversion. We integrate over
the truncation interval ``[-x_max + drift, x_max + drift]`` chosen
so the truncation error is below the integration accuracy.

Reference: A. Dragulescu, V. Yakovenko, "Probability distribution
of returns in the Heston model with stochastic volatility",
arxiv cond-mat/0203046.

Divergences from C++:

* The C++ engine uses ``QuantLib::GaussLobattoIntegral``; the
  Python port uses ``scipy.integrate.quad`` (QUADPACK adaptive).
  Both converge to ~1e-8 absolute accuracy on the standard
  parameter range; the quadrature noise difference is at the
  7th-8th significant figure, which is well below the LOOSE
  tolerance tier used in tests.
* The Heston RND inversion uses a transformed integration
  variable ``u_x = -log(p_x) / c_inf`` to handle the
  exponential decay of the characteristic function at infinity
  on a fixed ``[0, 1]`` interval. Same transformation as C++.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from scipy.integrate import quad  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.option import OptionArguments
from pquantlib.pricingengines.generic_engine import GenericEngine


def _heston_pdf(
    x: float,
    t: float,
    *,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    integration_eps: float,
    max_iterations: int,
) -> float:
    """Heston log-spot PDF ``p(x, T)`` via characteristic-function inversion.

    # C++ parity: ``HestonRNDCalculator::pdf(x, t)`` in
    # methods/finitedifferences/utilities/hestonrndcalculator.cpp:123-128.
    """
    sigma2 = sigma * sigma
    c_inf = (
        min(10.0, max(0.0001, math.sqrt(1.0 - rho * rho) / sigma))
        * (v0 + kappa * theta * t)
    )

    def transform_phi(u: float) -> complex:
        if u < 1e-15:
            return 0 + 0j
        u_x = -math.log(u) / c_inf
        # phi at u_x.
        gamma_val = complex(kappa, rho * sigma * u_x)
        omega_val = np.lib.scimath.sqrt(
            gamma_val * gamma_val
            + sigma2 * complex(u_x * u_x, -u_x)
        )
        gamma_ratio = (gamma_val - omega_val) / (gamma_val + omega_val)
        phi = 2.0 * np.exp(
            complex(0.0, u_x * x)
            - v0 * complex(u_x * u_x, -u_x)
              / (
                gamma_val
                + omega_val
                * (1.0 + np.exp(-omega_val * t))
                / (1.0 - np.exp(-omega_val * t))
            )
            + kappa
            * theta
            / sigma2
            * (
                (gamma_val - omega_val) * t
                - 2.0
                * np.lib.scimath.log(
                    (1.0 - gamma_ratio * np.exp(-omega_val * t))
                    / (1.0 - gamma_ratio)
                )
            )
        )
        return phi / (u * c_inf)

    def integrand(u: float) -> float:
        return float(transform_phi(u).real)

    quad_result = cast(
        "tuple[float, float, dict[str, Any]]",
        quad(
            integrand,
            0.0,
            1.0,
            epsabs=0.1 * integration_eps,
            epsrel=0.1 * integration_eps,
            limit=max_iterations,
            full_output=1,
        ),
    )
    result, _abs_err, _info = quad_result
    return float(result) / (2.0 * math.pi)


class AnalyticPDFHestonEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Heston PDF-based pricing for any European payoff.

    # C++ parity: ``AnalyticPDFHestonEngine``.
    """

    def __init__(
        self,
        model: HestonModel,
        gauss_lobatto_eps: float = 1e-6,
        gauss_lobatto_integration_order: int = 10000,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._model: HestonModel = model
        self._integration_eps: float = gauss_lobatto_eps
        self._max_iterations: int = gauss_lobatto_integration_order
        model.register_with(self)

    def model(self) -> HestonModel:
        return self._model

    def pv(self, x: float, t: float) -> float:
        """Heston log-spot PDF ``p(x, t)``.

        # C++ parity: ``AnalyticPDFHestonEngine::Pv`` →
        # ``HestonRNDCalculator::pdf(x, t)``. The C++ rnd
        # internally transforms ``x`` to the centred coordinate
        # ``x_t = x - x0 + log(dr/dq)`` before invoking the
        # characteristic-function integrator; we mirror that.
        """
        process = self._model.process()
        x0 = math.log(process.s0().value())
        dr = process.risk_free_rate().discount(t)
        dq = process.dividend_yield().discount(t)
        x_centred = x - x0 + math.log(dr / dq)
        return _heston_pdf(
            x_centred,
            t,
            v0=process.v0,
            kappa=process.kappa,
            theta=process.theta,
            sigma=process.sigma,
            rho=process.rho,
            integration_eps=self._integration_eps,
            max_iterations=self._max_iterations,
        )

    def cdf(self, s: float, t: float) -> float:
        """Heston spot CDF ``P(S_T <= s)``.

        # C++ parity: ``AnalyticPDFHestonEngine::cdf``. Uses the
        # ``p0`` transformed-integration form to integrate the PDF
        # from -infinity to ``log(s)`` over a finite [0, 1] grid.

        Note: this method is exposed for parity with C++ but is
        not exercised by the standard NPV path (``calculate``
        integrates the weighted payoff directly).
        """
        process = self._model.process()
        x_t = math.log(s)
        return _heston_cdf(
            x_t,
            t,
            v0=process.v0,
            kappa=process.kappa,
            theta=process.theta,
            sigma=process.sigma,
            rho=process.rho,
            integration_eps=self._integration_eps,
            max_iterations=self._max_iterations,
        )

    def _weighted_payoff(self, x_t: float, t: float) -> float:
        """``payoff(exp(x_t)) * p(x_t, t) * discount``."""
        process = self._model.process()
        rd = process.risk_free_rate().discount(t)
        s_t = math.exp(x_t)
        assert self._arguments.payoff is not None
        payoff_value = self._arguments.payoff(s_t)
        if payoff_value == 0.0:
            return 0.0
        return payoff_value * self.pv(x_t, t) * rd

    def calculate(self) -> None:
        """Run the PDF-weighted integral.

        # C++ parity: ``AnalyticPDFHestonEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European option",
        )

        process = self._model.process()
        t = process.time(args.exercise.last_date())

        # x_max: outer integration limit. Chosen as 8 standard
        # deviations of the integrated variance.
        x_max = 8.0 * math.sqrt(
            process.theta * t
            + (process.v0 - process.theta)
              * (1.0 - math.exp(-process.kappa * t))
              / process.kappa
        )
        x0 = math.log(process.s0().value())
        r_d = process.risk_free_rate().discount(t)
        q_d = process.dividend_yield().discount(t)

        drift = x0 + math.log(r_d / q_d)

        def integrand(y: float) -> float:
            return self._weighted_payoff(y, t)

        quad_result = cast(
            "tuple[float, float, dict[str, Any]]",
            quad(
                integrand,
                -x_max + drift,
                x_max + drift,
                epsabs=self._integration_eps,
                epsrel=self._integration_eps,
                limit=self._max_iterations,
                full_output=1,
            ),
        )

        results.reset()
        results.value = float(quad_result[0])

    def update(self) -> None:
        self.notify_observers()


def _heston_cdf(
    x: float,
    t: float,
    *,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    integration_eps: float,
    max_iterations: int,
) -> float:
    """Heston log-spot CDF — ``HestonRNDCalculator::cdf`` form.

    Integrates the ``p0`` transformed integrand and adds 0.5 for the
    centred CDF baseline.
    """
    sigma2 = sigma * sigma
    c_inf = (
        min(10.0, max(0.0001, math.sqrt(1.0 - rho * rho) / sigma))
        * (v0 + kappa * theta * t)
    )

    def p0_integrand(p_x: float) -> float:
        if p_x < 1e-15:
            return 0.0
        u_x = max(1e-15, -math.log(p_x) / c_inf)
        gamma_val = complex(kappa, rho * sigma * u_x)
        omega_val = np.lib.scimath.sqrt(
            gamma_val * gamma_val
            + sigma2 * complex(u_x * u_x, -u_x)
        )
        gamma_ratio = (gamma_val - omega_val) / (gamma_val + omega_val)
        phi = 2.0 * np.exp(
            complex(0.0, u_x * x)
            - v0 * complex(u_x * u_x, -u_x)
              / (
                gamma_val
                + omega_val
                * (1.0 + np.exp(-omega_val * t))
                / (1.0 - np.exp(-omega_val * t))
            )
            + kappa
            * theta
            / sigma2
            * (
                (gamma_val - omega_val) * t
                - 2.0
                * np.lib.scimath.log(
                    (1.0 - gamma_ratio * np.exp(-omega_val * t))
                    / (1.0 - gamma_ratio)
                )
            )
        )
        return float(
            (phi / ((p_x * c_inf) * complex(0.0, u_x))).real
        )

    quad_result = cast(
        "tuple[float, float, dict[str, Any]]",
        quad(
            p0_integrand,
            0.0,
            1.0,
            epsabs=0.1 * integration_eps,
            epsrel=0.1 * integration_eps,
            limit=max_iterations,
            full_output=1,
        ),
    )
    result, _abs_err, _info = quad_result
    return float(result) / (2.0 * math.pi) + 0.5


__all__ = ["AnalyticPDFHestonEngine"]
