"""MCDiscreteArithmeticASEngine — discrete-arithmetic-average *strike* Asian MC.

# C++ parity: ql/pricingengines/asian/mc_discr_arith_av_strike.{hpp,cpp} (v1.43) —
# ``template <class RNG = PseudoRandom, class S = Statistics>
#  class MCDiscreteArithmeticASEngine
#      : public MCDiscreteAveragingAsianEngineBase<SingleVariate,RNG,S>``,
# ``class ArithmeticASOPathPricer : public PathPricer<Path>`` and
# ``template <class RNG, class S> class MakeMCDiscreteArithmeticASEngine``.

An average-*strike* Asian pays ``max(omega * (S_T - A_n), 0)``: the arithmetic
average of the fixings becomes the strike and the terminal spot is the
underlying. The payoff's own strike is therefore ignored.

This is the only engine in the cluster constructed with
``includeExerciseDate = true`` (mc_discr_arith_av_strike.hpp:92-102), because
it needs the spot AT EXERCISE, which may fall after the last fixing. When that
happens the base class appends the exercise time to the grid, and
:class:`ArithmeticASOPathPricer` must average over the fixing points only —
hence the ``fixingCount`` parameter, computed in
:meth:`MCDiscreteArithmeticASEngine.path_pricer`.
"""

from __future__ import annotations

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


class ArithmeticASOPathPricer(PathPricer[Path]):
    """Path pricer for discrete-arithmetic-average *strike* Asians.

    # C++ parity: ``ArithmeticASOPathPricer``
    # (mc_discr_arith_av_strike.hpp:61-76, mc_discr_arith_av_strike.cpp:24-65).

    ``fixing_count`` is ``None`` for C++'s ``Null<Size>()``.
    """

    __slots__ = (
        "_discount",
        "_fixing_count",
        "_past_fixings",
        "_running_sum",
        "_type",
    )

    def __init__(
        self,
        option_type: OptionType,
        discount: float,
        running_sum: float = 0.0,
        past_fixings: int = 0,
        fixing_count: int | None = None,
    ) -> None:
        # C++ parity: mc_discr_arith_av_strike.cpp:24-31 — no strike guard
        # here; there is no strike.
        self._type: OptionType = option_type
        self._discount: float = discount
        self._running_sum: float = running_sum
        self._past_fixings: int = past_fixings
        self._fixing_count: int | None = fixing_count

    def __call__(self, path: Path) -> float:
        """# C++ parity: ``ArithmeticASOPathPricer::operator()``
        # (mc_discr_arith_av_strike.cpp:34-65)."""
        n = path.length()
        qassert.require(n > 1, "the path cannot be empty")

        # When fixing_count is set, the path may include an extra point at the
        # exercise date beyond the last fixing date. Average only over the
        # fixing points; use path.back() for the spot at exercise.
        n_fixings = self._fixing_count if self._fixing_count is not None else n
        qassert.require(
            n_fixings <= n,
            f"fixingCount ({n_fixings}) exceeds path length ({n})",
        )

        mandatory = path.time_grid.mandatory_times
        if mandatory[0] == 0.0:
            # include initial fixing (T=0 is a fixing date)
            total = self._running_sum
            for i in range(n_fixings):
                total += path[i]
            average_strike = total / (self._past_fixings + n_fixings)
        else:
            # first path point is T=0 (not a fixing), skip it
            total = self._running_sum
            for i in range(1, n_fixings):
                total += path[i]
            average_strike = total / (self._past_fixings + n_fixings - 1)

        return self._discount * PlainVanillaPayoff(self._type, average_strike)(
            path.back()
        )


class MCDiscreteArithmeticASEngine(MCDiscreteAveragingAsianEngineBase[Path]):
    """MC engine for discrete-arithmetic-average strike Asians.

    # C++ parity: ``MCDiscreteArithmeticASEngine<RNG,S>``
    # (mc_discr_arith_av_strike.hpp:36-58, 82-154).

    C++ hard-wires ``controlVariate = false``, ``timeSteps = Null``,
    ``timeStepsPerYear = Null`` and ``includeExerciseDate = true``.
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
            time_steps=None,
            time_steps_per_year=None,
            include_exercise_date=True,
            rng_traits=rng_traits,
            multi_variate=False,
        )

    def path_pricer(self) -> PathPricer[Path]:
        """Build the average-strike path pricer.

        # C++ parity: ``MCDiscreteArithmeticASEngine::pathPricer``
        # (mc_discr_arith_av_strike.hpp:104-154).
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

        # When the exercise date was added to the time grid (i.e., it falls
        # after the last fixing), tell the path pricer how many grid points are
        # fixings so it can exclude the exercise point from the average.
        fixing_count: int | None = None
        if self._include_exercise_date:
            qassert.require(
                self._time_steps is None and self._time_steps_per_year is None,
                "extra time steps are not supported when "
                "includeExerciseDate is enabled",
            )
            grid = self.time_grid()
            last_fixing = process.time(args.fixing_dates[-1])
            exercise_time = process.time(exercise.last_date())
            if exercise_time > last_fixing:
                # exercise date was added to the grid; path has one extra point
                # at the end that is NOT a fixing
                fixing_count = len(grid) - 1

        assert args.running_accumulator is not None
        assert args.past_fixings is not None
        return ArithmeticASOPathPricer(
            payoff.option_type(),
            process.risk_free_rate().discount(exercise.last_date()),
            args.running_accumulator,
            args.past_fixings,
            fixing_count,
        )


class MakeMCDiscreteArithmeticASEngine:
    """Fluent builder for :class:`MCDiscreteArithmeticASEngine`.

    # C++ parity: ``MakeMCDiscreteArithmeticASEngine<RNG,S>``
    # (mc_discr_arith_av_strike.hpp:158-248).

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

    def with_samples(self, samples: int) -> MakeMCDiscreteArithmeticASEngine:
        """# C++ parity: ``withSamples`` (mc_discr_arith_av_strike.hpp:187-194)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(
        self, tolerance: float
    ) -> MakeMCDiscreteArithmeticASEngine:
        """# C++ parity: ``withAbsoluteTolerance``
        # (mc_discr_arith_av_strike.hpp:196-207)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCDiscreteArithmeticASEngine:
        """# C++ parity: ``withMaxSamples`` (mc_discr_arith_av_strike.hpp:209-214)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCDiscreteArithmeticASEngine:
        """# C++ parity: ``withSeed`` (mc_discr_arith_av_strike.hpp:216-221)."""
        self._seed = seed
        return self

    def with_brownian_bridge(self, b: bool = True) -> MakeMCDiscreteArithmeticASEngine:
        """# C++ parity: ``withBrownianBridge``
        # (mc_discr_arith_av_strike.hpp:223-228)."""
        self._brownian_bridge = b
        return self

    def with_antithetic_variate(
        self, b: bool = True
    ) -> MakeMCDiscreteArithmeticASEngine:
        """# C++ parity: ``withAntitheticVariate``
        # (mc_discr_arith_av_strike.hpp:230-235)."""
        self._antithetic = b
        return self

    def engine(self) -> MCDiscreteArithmeticASEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (mc_discr_arith_av_strike.hpp:237-248) — no validation happens here."""
        return MCDiscreteArithmeticASEngine(
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
    "ArithmeticASOPathPricer",
    "MCDiscreteArithmeticASEngine",
    "MakeMCDiscreteArithmeticASEngine",
]
