"""PearsonSpreadEngine — Pearson (1995) spread-option engine.

# C++ parity: ql/pricingengines/basket/pearsonspreadengine.{hpp,cpp}
# (new in v1.43).

Reference: Neil D. Pearson, "An Efficient Approach for Pricing Spread
Options", Journal of Derivatives 3 (1995), 76-91.

The two-dimensional expectation collapses to a one-dimensional integral
over the Brownian factor driving the second asset: conditional on it the
first asset is lognormal and the spread payoff is a plain Black
call/put struck at ``F2(z) + K``. The outer integral is adaptive
Gauss-Lobatto over ``z`` in ``[-n_std, n_std]``.

The engine validates nothing at construction: ``|rho| > 1`` is accepted
and only ``sqrt(max(1 - rho**2, 0))`` clamps it downstream. That is
upstream behaviour and is reproduced.
"""

from __future__ import annotations

import math

from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.math.integrals.lobatto import GaussLobattoIntegral
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.basket.spread_black_scholes_vanilla_engine import (
    SpreadBlackScholesVanillaEngine,
)
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class PearsonSpreadEngine(SpreadBlackScholesVanillaEngine):
    """Spread-option engine following Pearson (1995).

    # C++ parity: ``class PearsonSpreadEngine``.

    Args:
        process1: first leg.
        process2: second leg.
        correlation: correlation between the two Brownian drivers. Not
            validated — see the module docstring.
        integration_tolerance: absolute accuracy demanded of the
            Gauss-Lobatto quadrature.
        max_integration_iterations: evaluation cap for that quadrature.
        n_std: half-width of the integration range, in standard
            deviations of the second asset's driving Brownian motion.
    """

    def __init__(
        self,
        process1: GeneralizedBlackScholesProcess,
        process2: GeneralizedBlackScholesProcess,
        correlation: float,
        integration_tolerance: float = 1.0e-10,
        max_integration_iterations: int = 10000,
        n_std: float = 8.0,
    ) -> None:
        super().__init__(process1, process2, correlation)
        self._integration_tolerance: float = integration_tolerance
        self._max_integration_iterations: int = max_integration_iterations
        self._n_std: float = n_std

    def _calculate(
        self,
        f1: float,
        f2: float,
        strike: float,
        option_type: OptionType,
        variance1: float,
        variance2: float,
        df: float,
    ) -> float:
        """Pearson's conditional-Black integral.

        # C++ parity: ``PearsonSpreadEngine::calculate``.

        Under the forward measure ``ln F1`` and ``ln F2`` are jointly normal
        with correlation ``rho``. Conditioning on the standard normal ``z``
        driving asset 2 gives ``F2(z) = f2 exp(-var2/2 + sigma2 z)`` and leaves
        asset 1 lognormal with mean ``ln f1 - var1/2 + rho sigma1 z`` and
        variance ``var1 (1 - rho**2)``, so the conditional spread option is a
        Black call/put on ``F1 | z`` struck at ``K + F2(z)``.
        """
        sigma1 = math.sqrt(variance1)
        sigma2 = math.sqrt(variance2)
        rho = self._rho
        sigma1_cond = sigma1 * math.sqrt(max(1.0 - rho * rho, 0.0))
        phi = NormalDistribution()

        def integrand(z: float) -> float:
            f2z = f2 * math.exp(-0.5 * variance2 + sigma2 * z)
            effective_strike = f2z + strike

            if effective_strike <= 0.0:
                # Reproduced verbatim from C++ v1.43, *including* the fact that
                # this branch returns the CALL intrinsic whatever option_type
                # is. For a sufficiently negative strike a put therefore picks
                # up a non-zero value here where zero is correct, and put-call
                # parity breaks. Upstream's own test strikes at 5 and never
                # reaches the branch. This is upstream behaviour, not a porting
                # slip; diverging would make the two libraries disagree.
                return phi(z) * max(
                    0.0,
                    f1 * math.exp(rho * sigma1 * z - 0.5 * rho * rho * variance1)
                    - effective_strike,
                )

            f1_cond = f1 * math.exp(rho * sigma1 * z - 0.5 * rho * rho * variance1)
            black = BlackCalculator(
                PlainVanillaPayoff(option_type, effective_strike),
                f1_cond,
                sigma1_cond,
                1.0,
            )
            return phi(z) * black.value()

        integrator = GaussLobattoIntegral(
            self._max_integration_iterations, self._integration_tolerance
        )
        undiscounted = integrator(integrand, -self._n_std, self._n_std)
        return df * undiscounted


__all__ = ["PearsonSpreadEngine"]
