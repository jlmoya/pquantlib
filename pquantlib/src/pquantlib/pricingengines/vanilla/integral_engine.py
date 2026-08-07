"""IntegralEngine — European vanilla pricing by direct numerical integration.

# C++ parity: ql/pricingengines/vanilla/integralengine.{hpp,cpp} (v1.43) —
# ``class IntegralEngine : public VanillaOption::engine``.

Prices by integrating the payoff against the risk-neutral lognormal density
rather than by a closed form:

    NPV = rDF / sqrt(2 pi var) * INT payoff(S0 e^x) exp(-(x - drift)^2 / 2var) dx

over ``[drift - 10 sqrt(var), drift + 10 sqrt(var)]``, with
``drift = ln(qDF / rDF) - 0.5 var``.

Two things a port must not "improve"
------------------------------------
1. The quadrature is a **fixed** :class:`SegmentIntegral` with **5000**
   intervals.  That number is part of the answer — an adaptive integrator
   converges to a slightly different value and will not reproduce the
   reference.  (The C++ header's ``\\todo define tolerance for calculate()``
   is still open in v1.43.)
2. The integrand applies ``arguments_.payoff`` — the *raw* payoff object,
   not the ``StrikedTypePayoff`` the engine just validated.  Binary payoffs
   therefore work, and are covered by the cross-validation.

The engine fills **only** ``value``; every greek stays unset.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.integrals.segment import SegmentIntegral
from pquantlib.option import OptionArguments
from pquantlib.payoffs import Payoff, StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class _Integrand:
    """Payoff times the (unnormalised) lognormal density.

    # C++ parity: ``class Integrand`` in the anonymous namespace of
    # integralengine.cpp — a translation-unit-local functor, not part of the
    # public surface. (The public ``Integrand`` name in QuantLib belongs to
    # ql/pricingengines/forward/mcvarianceswapengine.hpp and is unrelated.)
    """

    def __init__(self, payoff: Payoff, s0: float, drift: float, variance: float) -> None:
        self._payoff: Payoff = payoff
        self._s0: float = s0
        self._drift: float = drift
        self._variance: float = variance

    def __call__(self, x: float, /) -> float:
        temp = self._s0 * math.exp(x)
        result = self._payoff(temp)
        return result * math.exp(
            -(x - self._drift) * (x - self._drift) / (2.0 * self._variance)
        )


class IntegralEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """European vanilla option engine using a fixed-segment integral.

    # C++ parity: ``class IntegralEngine``.
    """

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        process.register_with(self)

    def calculate(self) -> None:
        """Integrate the payoff against the risk-neutral density.

        # C++ parity: ``IntegralEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "not an European Option",
        )

        qassert.require(args.payoff is not None, "no payoff given")
        assert args.payoff is not None
        qassert.require(isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given")
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff

        process = self._process
        last_date = args.exercise.last_date()

        variance = process.black_volatility().black_variance(
            last_date, payoff.strike(), extrapolate=True
        )
        dividend_discount = process.dividend_yield().discount(last_date)
        risk_free_discount = process.risk_free_rate().discount(last_date)
        drift = math.log(dividend_discount / risk_free_discount) - 0.5 * variance

        f = _Integrand(args.payoff, process.state_variable().value(), drift, variance)
        integrator = SegmentIntegral(5000)

        infinity = 10.0 * math.sqrt(variance)
        results.value = (
            process.risk_free_rate().discount(last_date)
            / math.sqrt(2.0 * math.pi * variance)
            * integrator(f, drift - infinity, drift + infinity)
        )


__all__ = ["IntegralEngine"]
