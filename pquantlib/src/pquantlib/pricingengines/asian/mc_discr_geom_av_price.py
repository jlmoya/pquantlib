"""MCDiscreteGeometricAPEngine — discrete-geometric-average price Asian MC.

# C++ parity: ql/pricingengines/asian/mc_discr_geom_av_price.{hpp,cpp} (v1.43) —
# ``template <class RNG = PseudoRandom, class S = Statistics>
#  class MCDiscreteGeometricAPEngine
#      : public MCDiscreteAveragingAsianEngineBase<SingleVariate,RNG,S>``,
# ``class GeometricAPOPathPricer : public PathPricer<Path>`` and
# ``template <class RNG, class S> class MakeMCDiscreteGeometricAPEngine``.

The geometric average of the fixings has a closed form under Black-Scholes
(:class:`~pquantlib.pricingengines.asian.analytic_discr_geom_av_price.AnalyticDiscreteGeometricAveragePriceAsianEngine`),
so this engine exists mainly to be *compared* with it — and
:class:`GeometricAPOPathPricer` doubles as the control-variate path pricer of
the arithmetic engine, which is why it lives here rather than next to the
engine that uses it (same split as C++).

The engine hard-wires ``controlVariate = false`` (mc_discr_geom_av_price.hpp:99-106)
and never supplies a control pricing engine.
"""

from __future__ import annotations

import sys

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.mc_discrete_asian_engine_base import (
    MCDiscreteAveragingAsianEngineBase,
)
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)

#: # C++ parity: ``QL_MAX_REAL`` = ``std::numeric_limits<Real>::max()``.
_QL_MAX_REAL = sys.float_info.max


class GeometricAPOPathPricer(PathPricer[Path]):
    """Path pricer for discrete-geometric-average price Asians.

    # C++ parity: ``GeometricAPOPathPricer``
    # (mc_discr_geom_av_price.hpp:70-84, mc_discr_geom_av_price.cpp:25-59).
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
        # C++ parity: mc_discr_geom_av_price.cpp:31 — note the message differs
        # from ArithmeticAPOPathPricer's for the same condition.
        qassert.require(strike >= 0.0, "negative strike given")
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discount: float = discount
        self._running_product: float = running_product
        self._past_fixings: int = past_fixings

    def __call__(self, path: Path) -> float:
        """# C++ parity: ``GeometricAPOPathPricer::operator()``
        # (mc_discr_geom_av_price.cpp:34-59).

        Transcribed literally, including the running-product accumulation and
        the overflow guard: the C++ code multiplies the fixings together and
        takes ``pow(product, 1/fixings)``, rescaling only when the next
        multiplication would overflow. A log-space rewrite gives a *different*
        answer in the last bits, so it is not used.
        """
        n = path.length() - 1
        qassert.require(n > 0, "the path cannot be empty")

        # C++ reads ``path.timeGrid().mandatoryTimes()[0]``: a fixing on the
        # evaluation date makes path[0] part of the average.
        mandatory = path.time_grid.mandatory_times
        product = self._running_product
        fixings = n + self._past_fixings
        if mandatory[0] == 0.0:
            fixings += 1
            product *= path.front()

        # care must be taken not to overflow product
        average_price = 1.0
        for i in range(1, n + 1):
            price = path[i]
            if product < _QL_MAX_REAL / price:
                product *= price
            else:
                average_price *= product ** (1.0 / fixings)
                product = price
        average_price *= product ** (1.0 / fixings)
        return self._discount * self._payoff(average_price)


class MCDiscreteGeometricAPEngine(MCDiscreteAveragingAsianEngineBase[Path]):
    """MC engine for discrete-geometric-average price Asians.

    # C++ parity: ``MCDiscreteGeometricAPEngine<RNG,S>``
    # (mc_discr_geom_av_price.hpp:44-67, 89-139).

    C++ hard-wires ``controlVariate = false``; there is no control pricing
    engine for this one.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        *,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        super().__init__(
            process,
            brownian_bridge=brownian_bridge,
            antithetic_variate=antithetic_variate,
            control_variate=False,
            required_samples=required_samples,
            required_tolerance=required_tolerance,
            max_samples=max_samples,
            seed=seed,
            rng_traits=rng_traits,
            multi_variate=False,
        )

    def path_pricer(self) -> PathPricer[Path]:
        """Build the geometric-average path pricer.

        # C++ parity: ``MCDiscreteGeometricAPEngine::pathPricer``
        # (mc_discr_geom_av_price.hpp:110-139).
        """
        args = self._arguments
        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)

        # C++ ``ext::dynamic_pointer_cast<EuropeanExercise>``: American and
        # Bermudan derive from EarlyExercise, so isinstance is the faithful
        # spelling.
        exercise = args.exercise
        qassert.require(isinstance(exercise, EuropeanExercise), "wrong exercise given")
        assert isinstance(exercise, EuropeanExercise)

        process = self._process
        qassert.require(
            isinstance(process, GeneralizedBlackScholesProcess),
            "Black-Scholes process required",
        )
        assert isinstance(process, GeneralizedBlackScholesProcess)

        assert args.running_accumulator is not None
        assert args.past_fixings is not None
        return GeometricAPOPathPricer(
            payoff.option_type(),
            payoff.strike(),
            process.risk_free_rate().discount(exercise.last_date()),
            args.running_accumulator,
            args.past_fixings,
        )


class MakeMCDiscreteGeometricAPEngine:
    """Fluent builder for :class:`MCDiscreteGeometricAPEngine`.

    # C++ parity: ``MakeMCDiscreteGeometricAPEngine<RNG,S>``
    # (mc_discr_geom_av_price.hpp:142-232).

    ``brownianBridge`` defaults to **True** here (C++ member initialiser
    ``bool brownianBridge_ = true``), unlike the engine constructor's default.
    """

    __slots__ = (
        "_antithetic",
        "_brownian_bridge",
        "_max_samples",
        "_process",
        "_rng_traits",
        "_samples",
        "_seed",
        "_tolerance",
    )

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        self._process: GeneralizedBlackScholesProcess = process
        self._rng_traits: RngTraits = rng_traits
        self._antithetic: bool = False
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._brownian_bridge: bool = True
        self._seed: int = 0

    def with_samples(self, samples: int) -> MakeMCDiscreteGeometricAPEngine:
        """# C++ parity: ``withSamples`` (mc_discr_geom_av_price.hpp:171-178)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(
        self, tolerance: float
    ) -> MakeMCDiscreteGeometricAPEngine:
        """# C++ parity: ``withAbsoluteTolerance``
        # (mc_discr_geom_av_price.hpp:180-191)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCDiscreteGeometricAPEngine:
        """# C++ parity: ``withMaxSamples`` (mc_discr_geom_av_price.hpp:193-198)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCDiscreteGeometricAPEngine:
        """# C++ parity: ``withSeed`` (mc_discr_geom_av_price.hpp:200-205)."""
        self._seed = seed
        return self

    def with_brownian_bridge(
        self, b: bool = True
    ) -> MakeMCDiscreteGeometricAPEngine:
        """# C++ parity: ``withBrownianBridge`` (mc_discr_geom_av_price.hpp:207-212)."""
        self._brownian_bridge = b
        return self

    def with_antithetic_variate(
        self, b: bool = True
    ) -> MakeMCDiscreteGeometricAPEngine:
        """# C++ parity: ``withAntitheticVariate``
        # (mc_discr_geom_av_price.hpp:214-219)."""
        self._antithetic = b
        return self

    def engine(self) -> MCDiscreteGeometricAPEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (mc_discr_geom_av_price.hpp:221-232) — no validation happens here."""
        return MCDiscreteGeometricAPEngine(
            self._process,
            brownian_bridge=self._brownian_bridge,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            rng_traits=self._rng_traits,
        )


__all__ = [
    "GeometricAPOPathPricer",
    "MCDiscreteGeometricAPEngine",
    "MakeMCDiscreteGeometricAPEngine",
]
