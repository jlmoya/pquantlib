"""MCDiscreteArithmeticAPEngine — discrete-arithmetic-average price Asian MC.

# C++ parity: ql/pricingengines/asian/mc_discr_arith_av_price.{hpp,cpp} (v1.43) —
# ``template <class RNG = PseudoRandom, class S = Statistics>
#  class MCDiscreteArithmeticAPEngine
#      : public MCDiscreteAveragingAsianEngineBase<SingleVariate,RNG,S>``,
# ``class ArithmeticAPOPathPricer : public PathPricer<Path>`` and
# ``template <class RNG, class S> class MakeMCDiscreteArithmeticAPEngine``.

Monte Carlo pricing of discrete-arithmetic-average price Asian options. Two
variance-reduction techniques are supported:

* Antithetic — emit each path's negated-variate twin and average.
* Control variate — the geometric average of the same fixings has a closed
  form under Black-Scholes, so
  :class:`~pquantlib.pricingengines.asian.analytic_discr_geom_av_price.AnalyticDiscreteGeometricAveragePriceAsianEngine`
  supplies the deterministic anchor while
  :class:`~pquantlib.pricingengines.asian.mc_discr_geom_av_price.GeometricAPOPathPricer`
  supplies the pathwise control.

Two asymmetries in the control-variate wiring are C++'s, and are reproduced
rather than tidied:

* the control path pricer is built WITHOUT the seasoning
  (``runningAccumulator`` / ``pastFixings``), even for a seasoned option
  (mc_discr_arith_av_price.hpp:175-184), and the analytic control engine's
  non-Geometric branch does the same;
* the control path pricer discounts at ``timeGrid().back()`` while the pricing
  path pricer discounts at ``exercise->lastDate()``. Those differ whenever the
  exercise date is later than the last fixing.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.analytic_discr_geom_av_price import (
    AnalyticDiscreteGeometricAveragePriceAsianEngine,
)
from pquantlib.pricingengines.asian.mc_discr_geom_av_price import (
    GeometricAPOPathPricer,
)
from pquantlib.pricingengines.asian.mc_discrete_asian_engine_base import (
    MCDiscreteAveragingAsianEngineBase,
)
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class ArithmeticAPOPathPricer(PathPricer[Path]):
    """Path pricer for discrete-arithmetic-average price Asians.

    # C++ parity: ``ArithmeticAPOPathPricer``
    # (mc_discr_arith_av_price.hpp:84-98, mc_discr_arith_av_price.cpp:26-52).
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
        # C++ parity: mc_discr_arith_av_price.cpp:32-33 — note the message
        # differs from GeometricAPOPathPricer's for the same condition.
        qassert.require(strike >= 0.0, "strike less than zero not allowed")
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discount: float = discount
        self._running_sum: float = running_sum
        self._past_fixings: int = past_fixings

    def __call__(self, path: Path) -> float:
        """# C++ parity: ``ArithmeticAPOPathPricer::operator()``
        # (mc_discr_arith_av_price.cpp:36-52).

        NB ``n`` is ``path.length()`` here but ``path.length() - 1`` in
        ``GeometricAPOPathPricer``; the two spell the same fixing count
        differently.
        """
        n = path.length()
        qassert.require(n > 1, "the path cannot be empty")

        # C++ reads ``path.timeGrid().mandatoryTimes()[0]``: a fixing on the
        # evaluation date makes path[0] part of the average; otherwise path[0]
        # is the t=0 anchor and is skipped.
        mandatory = path.time_grid.mandatory_times
        total = self._running_sum
        if mandatory[0] == 0.0:
            # include initial fixing
            for i in range(n):
                total += path[i]
            fixings = self._past_fixings + n
        else:
            for i in range(1, n):
                total += path[i]
            fixings = self._past_fixings + n - 1
        average_price = total / fixings
        return self._discount * self._payoff(average_price)


class MCDiscreteArithmeticAPEngine(MCDiscreteAveragingAsianEngineBase[Path]):
    """MC engine for discrete-arithmetic-average price Asians.

    # C++ parity: ``MCDiscreteArithmeticAPEngine<RNG,S>``
    # (mc_discr_arith_av_price.hpp:48-81, 103-184).
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        *,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        control_variate: bool = False,
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
            control_variate=control_variate,
            required_samples=required_samples,
            required_tolerance=required_tolerance,
            max_samples=max_samples,
            seed=seed,
            rng_traits=rng_traits,
            multi_variate=False,
        )

    # --- helpers ----------------------------------------------------------

    def _payoff_exercise_process(
        self,
    ) -> tuple[PlainVanillaPayoff, EuropeanExercise, GeneralizedBlackScholesProcess]:
        """The three ``dynamic_pointer_cast`` guards both path pricers share.

        # C++ parity: mc_discr_arith_av_price.hpp:129-142 (and the identical
        # block at 160-173).
        """
        args = self._arguments
        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)

        exercise = args.exercise
        qassert.require(isinstance(exercise, EuropeanExercise), "wrong exercise given")
        assert isinstance(exercise, EuropeanExercise)

        process = self._process
        qassert.require(
            isinstance(process, GeneralizedBlackScholesProcess),
            "Black-Scholes process required",
        )
        assert isinstance(process, GeneralizedBlackScholesProcess)
        return payoff, exercise, process

    # --- McSimulation hooks -----------------------------------------------

    def path_pricer(self) -> PathPricer[Path]:
        """Build the arithmetic-average path pricer.

        # C++ parity: ``MCDiscreteArithmeticAPEngine::pathPricer``
        # (mc_discr_arith_av_price.hpp:123-152).
        """
        payoff, exercise, process = self._payoff_exercise_process()
        args = self._arguments
        assert args.running_accumulator is not None
        assert args.past_fixings is not None
        return ArithmeticAPOPathPricer(
            payoff.option_type(),
            payoff.strike(),
            process.risk_free_rate().discount(exercise.last_date()),
            args.running_accumulator,
            args.past_fixings,
        )

    def control_path_pricer(self) -> PathPricer[Path] | None:
        """Build the geometric-average CV path pricer.

        # C++ parity: ``MCDiscreteArithmeticAPEngine::controlPathPricer``
        # (mc_discr_arith_av_price.hpp:154-184).

        For a seasoned option the geometric strike would have to be rescaled to
        obtain an equivalent arithmetic strike; C++ does not do that and passes
        no seasoning at all here. Any change applied here MUST be applied to
        the analytic engine too.
        """
        payoff, _exercise, process = self._payoff_exercise_process()
        # C++ discounts at ``this->timeGrid().back()`` -- NOT at the exercise
        # date, unlike ``pathPricer`` above.
        return GeometricAPOPathPricer(
            payoff.option_type(),
            payoff.strike(),
            process.risk_free_rate().discount(self.time_grid().back()),
        )

    def control_pricing_engine(self) -> PricingEngine | None:
        """Build the analytic-geometric CV pricing engine.

        # C++ parity: ``MCDiscreteArithmeticAPEngine::controlPricingEngine``
        # (mc_discr_arith_av_price.hpp:73-80).
        """
        process = self._process
        qassert.require(
            isinstance(process, GeneralizedBlackScholesProcess),
            "Black-Scholes process required",
        )
        assert isinstance(process, GeneralizedBlackScholesProcess)
        return AnalyticDiscreteGeometricAveragePriceAsianEngine(process)


class MakeMCDiscreteArithmeticAPEngine:
    """Fluent builder for :class:`MCDiscreteArithmeticAPEngine`.

    # C++ parity: ``MakeMCDiscreteArithmeticAPEngine<RNG,S>``
    # (mc_discr_arith_av_price.hpp:186-284).

    ``brownianBridge`` defaults to **True** here (C++ member initialiser
    ``bool brownianBridge_ = true``), unlike the engine constructor's default.
    """

    __slots__ = (
        "_antithetic",
        "_brownian_bridge",
        "_control_variate",
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
        self._control_variate: bool = False
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._brownian_bridge: bool = True
        self._seed: int = 0

    def with_samples(self, samples: int) -> MakeMCDiscreteArithmeticAPEngine:
        """# C++ parity: ``withSamples`` (mc_discr_arith_av_price.hpp:216-223)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(
        self, tolerance: float
    ) -> MakeMCDiscreteArithmeticAPEngine:
        """# C++ parity: ``withAbsoluteTolerance``
        # (mc_discr_arith_av_price.hpp:225-236)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCDiscreteArithmeticAPEngine:
        """# C++ parity: ``withMaxSamples`` (mc_discr_arith_av_price.hpp:238-243)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCDiscreteArithmeticAPEngine:
        """# C++ parity: ``withSeed`` (mc_discr_arith_av_price.hpp:245-250)."""
        self._seed = seed
        return self

    def with_brownian_bridge(self, b: bool = True) -> MakeMCDiscreteArithmeticAPEngine:
        """# C++ parity: ``withBrownianBridge``
        # (mc_discr_arith_av_price.hpp:252-257)."""
        self._brownian_bridge = b
        return self

    def with_antithetic_variate(
        self, b: bool = True
    ) -> MakeMCDiscreteArithmeticAPEngine:
        """# C++ parity: ``withAntitheticVariate``
        # (mc_discr_arith_av_price.hpp:259-264)."""
        self._antithetic = b
        return self

    def with_control_variate(self, b: bool = True) -> MakeMCDiscreteArithmeticAPEngine:
        """# C++ parity: ``withControlVariate``
        # (mc_discr_arith_av_price.hpp:266-271)."""
        self._control_variate = b
        return self

    def engine(self) -> MCDiscreteArithmeticAPEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (mc_discr_arith_av_price.hpp:273-284) — no validation happens here."""
        return MCDiscreteArithmeticAPEngine(
            self._process,
            brownian_bridge=self._brownian_bridge,
            antithetic_variate=self._antithetic,
            control_variate=self._control_variate,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            rng_traits=self._rng_traits,
        )


__all__ = [
    "ArithmeticAPOPathPricer",
    "MCDiscreteArithmeticAPEngine",
    "MakeMCDiscreteArithmeticAPEngine",
]
