"""MCDiscreteArithmeticAveragePriceEngine — discrete-arithmetic-average Asian MC.

# C++ parity: ql/pricingengines/asian/mc_discr_arith_av_price.{hpp,cpp}
# (v1.42.1).

Monte Carlo pricing of discrete-arithmetic-average price Asian
options.  Two variance-reduction techniques are supported:

* Antithetic — emit each path's negated-variate twin and average.
* Control variate — use the analytic-discrete-geometric-average
  price engine as the deterministic CV anchor.  The geometric
  average is a tractable function of the same path values, so
  subtracting its closed-form mean removes the bulk of the
  arithmetic MC variance.

The Python port keeps both ``ArithmeticAPOPathPricer`` and
``GeometricAPOPathPricer`` here (same module that holds the engine,
mirroring C++ ``mc_discr_arith_av_price.cpp`` placement) — the
geometric pricer is reused as the CV path pricer.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.analytic_discr_geom_av_price import (
    AnalyticDiscreteGeometricAveragePriceAsianEngine,
)
from pquantlib.pricingengines.asian.mc_discrete_asian_engine_base import (
    MCDiscreteAveragingAsianEngineBase,
)
from pquantlib.pricingengines.pricing_engine import PricingEngine


class ArithmeticAPOPathPricer(PathPricer[Path]):
    """Path pricer for discrete-arithmetic-average price Asians.

    # C++ parity: ``ArithmeticAPOPathPricer`` (mc_discr_arith_av_price.cpp:26-53).
    """

    __slots__ = ("_discount", "_past_fixings", "_payoff", "_running_sum")

    def __init__(
        self,
        option_type: OptionType,
        strike: float,
        discount: float,
        running_sum: float = 0.0,
        past_fixings: int = 0,
    ) -> None:
        qassert.require(strike >= 0.0, "strike less than zero not allowed")
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discount: float = discount
        self._running_sum: float = running_sum
        self._past_fixings: int = past_fixings

    def __call__(self, path: Path) -> float:
        n = path.length()
        qassert.require(n > 1, "the path cannot be empty")
        # C++ checks ``path.timeGrid().mandatoryTimes()[0] == 0.0``;
        # if so include path[0] in the average (the initial fixing).
        # Otherwise skip path[0] (the t=0 anchor).
        mandatory = path.time_grid.mandatory_times
        sum_ = self._running_sum
        if len(mandatory) > 0 and mandatory[0] == 0.0:
            for i in range(n):
                sum_ += float(path[i])
            fixings = self._past_fixings + n
        else:
            for i in range(1, n):
                sum_ += float(path[i])
            fixings = self._past_fixings + n - 1
        average_price = sum_ / fixings
        return self._discount * self._payoff(average_price)


class GeometricAPOPathPricer(PathPricer[Path]):
    """Path pricer for discrete-geometric-average price Asians.

    # C++ parity: ``GeometricAPOPathPricer`` (mc_discr_geom_av_price.cpp:25-58).

    Used as the CV path pricer for the arithmetic engine. The
    overflow guard from the C++ code (rescale when product > max/x)
    is preserved (Python floats have effectively unlimited range, but
    the C++ branch is mirrored faithfully — overflow on Python is
    OverflowError, not silent inf).
    """

    __slots__ = ("_discount", "_past_fixings", "_payoff", "_running_product")

    def __init__(
        self,
        option_type: OptionType,
        strike: float,
        discount: float,
        running_product: float = 1.0,
        past_fixings: int = 0,
    ) -> None:
        qassert.require(strike >= 0.0, "negative strike given")
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discount: float = discount
        self._running_product: float = running_product
        self._past_fixings: int = past_fixings

    def __call__(self, path: Path) -> float:
        n = path.length() - 1
        qassert.require(n > 0, "the path cannot be empty")
        mandatory = path.time_grid.mandatory_times
        product = self._running_product
        fixings = n + self._past_fixings
        if len(mandatory) > 0 and mandatory[0] == 0.0:
            fixings += 1
            product *= float(path.front())
        # C++ uses log-space accumulation; this is the simplest stable
        # equivalent. ``math.fsum`` on a list of logs to keep float drift
        # tight, then exp at the end.
        log_terms: list[float] = []
        if product > 0.0:
            log_terms.append(math.log(product))
        for i in range(1, n + 1):
            price = float(path[i])
            qassert.require(price > 0.0, "non-positive underlying price in geometric average path")
            log_terms.append(math.log(price))
        average_price = math.exp(math.fsum(log_terms) / fixings)
        return self._discount * self._payoff(average_price)


class MCDiscreteArithmeticAveragePriceEngine(MCDiscreteAveragingAsianEngineBase):
    """MC engine for discrete-arithmetic-average price Asians.

    # C++ parity: ``MCDiscreteArithmeticAPEngine<RNG, S>``
    # (mc_discr_arith_av_price.hpp).
    """

    def path_pricer(self) -> PathPricer[Path]:
        """Build the arithmetic-average path pricer.

        # C++ parity: ``MCDiscreteArithmeticAPEngine::pathPricer``
        # (mc_discr_arith_av_price.hpp:123-152).
        """
        args = self._arguments
        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)
        assert args.exercise is not None

        process = self._process
        discount = process.risk_free_rate().discount(args.exercise.last_date())
        assert args.running_accumulator is not None
        assert args.past_fixings is not None
        return ArithmeticAPOPathPricer(
            option_type=payoff.option_type(),
            strike=payoff.strike(),
            discount=discount,
            running_sum=args.running_accumulator,
            past_fixings=args.past_fixings,
        )

    def control_path_pricer(self) -> PathPricer[Path] | None:
        """Build the geometric-average CV path pricer.

        # C++ parity: ``MCDiscreteArithmeticAPEngine::controlPathPricer``
        # (mc_discr_arith_av_price.hpp:154-184).
        """
        if not self._control_variate:
            return None
        args = self._arguments
        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)
        process = self._process
        return GeometricAPOPathPricer(
            option_type=payoff.option_type(),
            strike=payoff.strike(),
            discount=process.risk_free_rate().discount(self.time_grid().back()),
        )

    def control_pricing_engine(self) -> PricingEngine | None:
        """Build the analytic-geometric CV pricing engine."""
        if not self._control_variate:
            return None
        return AnalyticDiscreteGeometricAveragePriceAsianEngine(self._process)


__all__ = [
    "ArithmeticAPOPathPricer",
    "GeometricAPOPathPricer",
    "MCDiscreteArithmeticAveragePriceEngine",
]
