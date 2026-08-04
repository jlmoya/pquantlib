"""GaussianCopulaSpreadEngine — smile-aware spread-option engine.

# C++ parity: ql/pricingengines/basket/gaussiancopulaspreadengine.{hpp,cpp}
# (new in v1.43).

Prices ``max(S1 - S2 - K, 0)`` by nested Gauss-Hermite quadrature over a
Gaussian copula whose marginals come from
:class:`~pquantlib.methods.finitedifferences.utilities.smile_section_rnd_calculator.SmileSectionRNDCalculator`.
Decoupling the marginals from the dependence structure is the point:
each leg keeps its whole smile instead of collapsing to one effective
volatility, which is what distinguishes this engine from
:class:`~pquantlib.pricingengines.basket.pearson_spread_engine.PearsonSpreadEngine`
on the same market.

Three details a port has to get right:

* The constructor compares the two processes' risk-free term structures
  by **identity**, not value — two curves holding the same numbers are
  rejected. The payoff is discounted on process 1's curve, so sharing it
  is the only way the result is well defined. Correlation must be in
  ``[-1, 1]`` **inclusive**.
* QuantLib's Gauss-Hermite weights divide out the weight function, so
  ``sum_i w_i f(x_i)`` approximates the *unweighted* integral. The
  engine therefore re-applies ``exp(-x**2)`` explicitly and normalises by
  ``1/pi``. Classical Gauss-Hermite weights would be wrong by
  ``exp(x**2)`` at every node.
* Each smile is re-anchored at the engine's own forward through
  :class:`~pquantlib.termstructures.volatility.atm_smile_section.AtmSmileSection`,
  which **overrides** whatever ATM level the underlying section reports.

Only ``value`` is populated: no greeks, no additional results.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise, Exercise
from pquantlib.instruments.basket_option import (
    BasketOptionResults,
    SpreadBasketPayoff,
)
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.integrals.gaussian_quadrature import GaussHermiteIntegration
from pquantlib.methods.finitedifferences.utilities.smile_section_rnd_calculator import (
    SmileSectionRNDCalculator,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.volatility.atm_smile_section import AtmSmileSection

_SQRT2 = math.sqrt(2.0)


class GaussianCopulaSpreadEngine(GenericEngine[OptionArguments, BasketOptionResults]):
    """Gaussian-copula spread engine with smile-implied marginals.

    # C++ parity: ``class GaussianCopulaSpreadEngine``.

    Args:
        process1: first leg; its risk-free curve discounts the payoff.
        process2: second leg; must share that same curve object.
        correlation: copula correlation, in ``[-1, 1]`` inclusive.
        n_points: order of the Gauss-Hermite rule on each axis, so the
            work is quadratic in it.
    """

    def __init__(
        self,
        process1: GeneralizedBlackScholesProcess,
        process2: GeneralizedBlackScholesProcess,
        correlation: float,
        n_points: int = 64,
    ) -> None:
        super().__init__(OptionArguments(), BasketOptionResults())
        qassert.require(
            -1.0 <= correlation <= 1.0,
            f"correlation must be in [-1, 1], got {correlation}",
        )
        # Identity, not equality: C++ compares the two handles' current links
        # by pointer, so two curves that merely hold the same numbers are
        # rejected.
        qassert.require(
            process1.risk_free_rate() is process2.risk_free_rate(),
            "process1 and process2 must share the risk-free term structure "
            "(used for discounting the spread payoff)",
        )
        self._process1: GeneralizedBlackScholesProcess = process1
        self._process2: GeneralizedBlackScholesProcess = process2
        self._rho: float = correlation
        self._n_points: int = n_points
        process1.register_with(self)
        process2.register_with(self)

    def calculate(self) -> None:
        """Nested Gauss-Hermite quadrature over the copula.

        # C++ parity: ``GaussianCopulaSpreadEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European, "not a European exercise"
        )
        assert isinstance(args.exercise, EuropeanExercise)

        spread_payoff = args.payoff
        qassert.require(
            isinstance(spread_payoff, SpreadBasketPayoff), "spread payoff expected"
        )
        assert isinstance(spread_payoff, SpreadBasketPayoff)
        base_payoff = spread_payoff.base_payoff()
        qassert.require(
            isinstance(base_payoff, PlainVanillaPayoff), "non-plain payoff given"
        )
        assert isinstance(base_payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = base_payoff

        # 1. Forwards, discount, and smile sections.
        maturity_date = args.exercise.last_date()

        fwd1 = (
            self._process1.state_variable().value()
            * self._process1.dividend_yield().discount(maturity_date)
            / self._process1.risk_free_rate().discount(maturity_date)
        )
        fwd2 = (
            self._process2.state_variable().value()
            * self._process2.dividend_yield().discount(maturity_date)
            / self._process2.risk_free_rate().discount(maturity_date)
        )
        df = self._process1.risk_free_rate().discount(maturity_date)

        t1 = self._process1.black_volatility().time_from_reference(maturity_date)
        t2 = self._process2.black_volatility().time_from_reference(maturity_date)

        # AtmSmileSection re-anchors each smile at the engine's own forward,
        # overriding whatever ATM level the underlying section reports.
        smile1 = AtmSmileSection(
            base=self._process1.black_volatility().smile_section_at_time(t1), atm=fwd1
        )
        smile2 = AtmSmileSection(
            base=self._process2.black_volatility().smile_section_at_time(t2), atm=fwd2
        )

        # 2. Risk-neutral marginals.
        rnd1 = SmileSectionRNDCalculator(smile1)
        rnd2 = SmileSectionRNDCalculator(smile2)

        # 3. Nested Gauss-Hermite quadrature. GaussHermiteIntegration
        # approximates int f(x) dx — QuantLib's weights already divide out
        # exp(-x^2) — so integrating against phi(z) means substituting
        # z = sqrt(2) x and re-applying the (1/pi) exp(-x_i^2 - x_j^2) kernel
        # by hand.
        gh = GaussHermiteIntegration(self._n_points)
        x = gh.x()
        w = gh.weights()

        phi = CumulativeNormalDistribution()
        rho_comp = math.sqrt(max(1.0 - self._rho * self._rho, 0.0))
        norm_factor = 1.0 / math.pi

        total = 0.0
        for i in range(gh.order()):
            xi = float(x[i])
            z1 = _SQRT2 * xi
            # At extreme GH nodes Phi(z) saturates to exactly 0 or 1; nudge it
            # inside the open interval invcdf requires.
            u1 = min(max(phi(z1), QL_EPSILON), 1.0 - QL_EPSILON)
            s1 = math.exp(rnd1.invcdf(u1))
            exp_x1_sq = math.exp(-xi * xi)

            inner = 0.0
            for j in range(gh.order()):
                xj = float(x[j])
                z2perp = _SQRT2 * xj
                z2 = self._rho * z1 + rho_comp * z2perp
                u2 = min(max(phi(z2), QL_EPSILON), 1.0 - QL_EPSILON)
                s2 = math.exp(rnd2.invcdf(u2))

                payoff_val = payoff(s1 - s2)
                kernel = exp_x1_sq * math.exp(-xj * xj)
                inner += float(w[j]) * kernel * payoff_val
            total += float(w[i]) * inner

        results.reset()
        results.value = df * norm_factor * total

    def update(self) -> None:
        self.notify_observers()


__all__ = ["GaussianCopulaSpreadEngine"]
