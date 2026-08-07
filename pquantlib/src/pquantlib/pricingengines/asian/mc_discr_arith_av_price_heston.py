"""MCDiscreteArithmeticAPHestonEngine — Heston MC for discrete arithmetic Asians.

# C++ parity: ql/pricingengines/asian/mc_discr_arith_av_price_heston.{hpp,cpp}
# (v1.43) —
# ``template <class RNG = PseudoRandom, class S = Statistics,
#            class P = HestonProcess>
#  class MCDiscreteArithmeticAPHestonEngine
#      : public MCDiscreteAveragingAsianEngineBase<MultiVariate,RNG,S>``,
# ``class ArithmeticAPOHestonPathPricer : public PathPricer<MultiPath>`` and
# ``template <...> class MakeMCDiscreteArithmeticAPHestonEngine``.

By default the MC discretization uses one time step per fixing date, but this
can be controlled via ``time_steps`` / ``time_steps_per_year``, which provide
additional steps. The realised grid is published in
``results.additional_results["TimeGrid"]``, and the path pricers average over
the fixing indices only, so the extra points never enter the average.

The control variate is the experimental analytic discrete-geometric Heston
engine plus
:class:`~pquantlib.pricingengines.asian.mc_discr_geom_av_price_heston.GeometricAPOHestonPathPricer`.
Neither carries seasoning: the analytic pricer does not support seasoned Asian
options yet, so C++ deliberately passes none to the control path pricer either
(mc_discr_arith_av_price_heston.hpp:222-233).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.asian.analytic_discr_geom_av_price_heston_engine import (
    AnalyticDiscreteGeometricAveragePriceAsianHestonEngine,
)
from pquantlib.pricingengines.asian.mc_discr_geom_av_price_heston import (
    GeometricAPOHestonPathPricer,
    fixing_indices,
)
from pquantlib.pricingengines.asian.mc_discrete_asian_engine_base import (
    MCDiscreteAveragingAsianEngineBase,
)
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.heston_process import HestonProcess


class ArithmeticAPOHestonPathPricer(PathPricer[MultiPath]):
    """Arithmetic-average path pricer reading the spot leg of a ``MultiPath``.

    # C++ parity: ``ArithmeticAPOHestonPathPricer``
    # (mc_discr_arith_av_price_heston.hpp:108-124,
    #  mc_discr_arith_av_price_heston.cpp:23-49).
    """

    __slots__ = (
        "_discount",
        "_fixing_indices",
        "_past_fixings",
        "_payoff",
        "_running_sum",
    )

    def __init__(
        self,
        option_type: OptionType,
        strike: float,
        discount: float,
        fixing_index_list: list[int],
        running_sum: float = 0.0,
        past_fixings: int = 0,
    ) -> None:
        # C++ parity: mc_discr_arith_av_price_heston.cpp:31-32.
        qassert.require(strike >= 0.0, "strike less than zero not allowed")
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discount: float = discount
        self._fixing_indices: list[int] = fixing_index_list
        self._running_sum: float = running_sum
        self._past_fixings: int = past_fixings

    def __call__(self, path: MultiPath) -> float:
        """# C++ parity: ``ArithmeticAPOHestonPathPricer::operator()``
        # (mc_discr_arith_av_price_heston.cpp:35-49).

        Only ``multiPath[0]`` (the spot leg) is read; the variance leg is
        ignored. Unlike the single-variate pricer, the t=0 point enters the
        average iff index 0 is one of the fixing indices — there is no
        ``mandatoryTimes()[0] == 0`` special case here.
        """
        spot = path[0]
        qassert.require(path.path_size() > 0, "the path cannot be empty")

        total = self._running_sum
        fixings = self._past_fixings + len(self._fixing_indices)

        for index in self._fixing_indices:
            total += spot[index]

        average_price = total / fixings
        return self._discount * self._payoff(average_price)


class MCDiscreteArithmeticAPHestonEngine(MCDiscreteAveragingAsianEngineBase[MultiPath]):
    """Heston MC engine for discrete-arithmetic-average price Asians.

    # C++ parity: ``MCDiscreteArithmeticAPHestonEngine<RNG,S,P>``
    # (mc_discr_arith_av_price_heston.hpp:49-80, 129-234).
    """

    def __init__(
        self,
        process: HestonProcess,
        *,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        control_variate: bool = False,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        super().__init__(
            process,
            brownian_bridge=False,
            antithetic_variate=antithetic_variate,
            control_variate=control_variate,
            required_samples=required_samples,
            required_tolerance=required_tolerance,
            max_samples=max_samples,
            seed=seed,
            time_steps=time_steps,
            time_steps_per_year=time_steps_per_year,
            rng_traits=rng_traits,
            multi_variate=True,
        )
        # C++ parity: mc_discr_arith_av_price_heston.hpp:151-152.
        qassert.require(
            time_steps is None or time_steps_per_year is None,
            "both time steps and time steps per year were provided",
        )

    # --- helpers ----------------------------------------------------------

    def _payoff_exercise_process(
        self,
    ) -> tuple[PlainVanillaPayoff, EuropeanExercise, HestonProcess]:
        """The three ``dynamic_pointer_cast`` guards both path pricers share.

        # C++ parity: mc_discr_arith_av_price_heston.hpp:169-181 (and the
        # identical block at 208-220).
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
            isinstance(process, HestonProcess), "Heston like process required"
        )
        assert isinstance(process, HestonProcess)
        return payoff, exercise, process

    # --- McSimulation hooks -----------------------------------------------

    def path_pricer(self) -> PathPricer[MultiPath]:
        """Build the arithmetic-average Heston path pricer.

        # C++ parity: ``MCDiscreteArithmeticAPHestonEngine::pathPricer``
        # (mc_discr_arith_av_price_heston.hpp:155-192).
        """
        # Keep track of the fixing indices, the path pricer will need to sum
        # only these.
        indices = fixing_indices(self.time_grid())
        payoff, exercise, process = self._payoff_exercise_process()
        args = self._arguments
        assert args.running_accumulator is not None
        assert args.past_fixings is not None
        return ArithmeticAPOHestonPathPricer(
            payoff.option_type(),
            payoff.strike(),
            process.risk_free_rate().discount(exercise.last_date()),
            indices,
            args.running_accumulator,
            args.past_fixings,
        )

    def control_path_pricer(self) -> PathPricer[MultiPath] | None:
        """Build the geometric-average Heston CV path pricer.

        # C++ parity: ``MCDiscreteArithmeticAPHestonEngine::controlPathPricer``
        # (mc_discr_arith_av_price_heston.hpp:194-234).

        The analytic pricer does not support seasoned Asian options, so no
        seasoning is passed here either.
        """
        indices = fixing_indices(self.time_grid())
        payoff, exercise, process = self._payoff_exercise_process()
        return GeometricAPOHestonPathPricer(
            payoff.option_type(),
            payoff.strike(),
            process.risk_free_rate().discount(exercise.last_date()),
            indices,
        )

    def control_pricing_engine(self) -> PricingEngine | None:
        """Build the analytic discrete-geometric Heston CV pricing engine.

        # C++ parity: ``MCDiscreteArithmeticAPHestonEngine::controlPricingEngine``
        # (mc_discr_arith_av_price_heston.hpp:73-79).
        """
        process = self._process
        qassert.require(
            isinstance(process, HestonProcess), "Heston-like process required"
        )
        assert isinstance(process, HestonProcess)
        return AnalyticDiscreteGeometricAveragePriceAsianHestonEngine(process)


class MakeMCDiscreteArithmeticAPHestonEngine:
    """Fluent builder for :class:`MCDiscreteArithmeticAPHestonEngine`.

    # C++ parity: ``MakeMCDiscreteArithmeticAPHestonEngine<RNG,S,P>``
    # (mc_discr_arith_av_price_heston.hpp:83-105, 236-322).

    Like its geometric twin this builder rejects over-specified steps EARLY,
    inside :meth:`with_steps` / :meth:`with_steps_per_year`; :meth:`engine`
    performs NO validation at all.
    """

    __slots__ = (
        "_antithetic",
        "_control_variate",
        "_max_samples",
        "_process",
        "_rng_traits",
        "_samples",
        "_seed",
        "_steps",
        "_steps_per_year",
        "_tolerance",
    )

    def __init__(
        self, process: HestonProcess, rng_traits: RngTraits = PseudoRandom
    ) -> None:
        self._process: HestonProcess = process
        self._rng_traits: RngTraits = rng_traits
        self._antithetic: bool = False
        self._control_variate: bool = False
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    def with_samples(self, samples: int) -> MakeMCDiscreteArithmeticAPHestonEngine:
        """# C++ parity: ``withSamples``
        # (mc_discr_arith_av_price_heston.hpp:242-249)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(
        self, tolerance: float
    ) -> MakeMCDiscreteArithmeticAPHestonEngine:
        """# C++ parity: ``withAbsoluteTolerance``
        # (mc_discr_arith_av_price_heston.hpp:251-262)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCDiscreteArithmeticAPHestonEngine:
        """# C++ parity: ``withMaxSamples``
        # (mc_discr_arith_av_price_heston.hpp:264-269)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCDiscreteArithmeticAPHestonEngine:
        """# C++ parity: ``withSeed``
        # (mc_discr_arith_av_price_heston.hpp:271-276)."""
        self._seed = seed
        return self

    def with_antithetic_variate(
        self, b: bool = True
    ) -> MakeMCDiscreteArithmeticAPHestonEngine:
        """# C++ parity: ``withAntitheticVariate``
        # (mc_discr_arith_av_price_heston.hpp:278-283)."""
        self._antithetic = b
        return self

    def with_steps(self, steps: int) -> MakeMCDiscreteArithmeticAPHestonEngine:
        """# C++ parity: ``withSteps``
        # (mc_discr_arith_av_price_heston.hpp:285-292)."""
        qassert.require(
            self._steps_per_year is None, "number of steps per year already set"
        )
        self._steps = steps
        return self

    def with_steps_per_year(
        self, steps: int
    ) -> MakeMCDiscreteArithmeticAPHestonEngine:
        """# C++ parity: ``withStepsPerYear``
        # (mc_discr_arith_av_price_heston.hpp:294-301)."""
        qassert.require(self._steps is None, "number of steps already set")
        self._steps_per_year = steps
        return self

    def with_control_variate(
        self, b: bool = False
    ) -> MakeMCDiscreteArithmeticAPHestonEngine:
        """# C++ parity: ``withControlVariate``
        # (mc_discr_arith_av_price_heston.hpp:303-308).

        NB the C++ default argument is ``false``, not ``true`` as on every
        other ``with*Variate`` in the library; a bare ``with_control_variate()``
        therefore switches the control variate OFF.
        """
        self._control_variate = b
        return self

    def engine(self) -> MCDiscreteArithmeticAPHestonEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (mc_discr_arith_av_price_heston.hpp:310-322)."""
        return MCDiscreteArithmeticAPHestonEngine(
            self._process,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            time_steps=self._steps,
            time_steps_per_year=self._steps_per_year,
            control_variate=self._control_variate,
            rng_traits=self._rng_traits,
        )


__all__ = [
    "ArithmeticAPOHestonPathPricer",
    "MCDiscreteArithmeticAPHestonEngine",
    "MakeMCDiscreteArithmeticAPHestonEngine",
]
