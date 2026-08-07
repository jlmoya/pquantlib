"""MCVarianceSwapEngine — Monte Carlo variance-swap engine.

# C++ parity: ql/pricingengines/forward/mcvarianceswapengine.hpp (v1.43) —
# ``template <class RNG = PseudoRandom, class S = Statistics>
#  class MCVarianceSwapEngine : public VarianceSwap::engine,
#                               public McSimulation<SingleVariate, RNG, S>``
# and ``template <class RNG, class S> class MakeMCVarianceSwapEngine``.

Demeterfi, Derman, Kamal & Zou, "A Guide to Volatility and Variance Swaps"
(1999). The per-path realised variance lives in
:class:`~pquantlib.pricingengines.forward.variance_path_pricer.VariancePathPricer`
(C++ declares it in this same header; the Python port gave it its own module and
its own cross-validation test, so it is imported here rather than redefined).

What this engine adds on top of the path pricer, and what a port gets wrong
--------------------------------------------------------------------------

* **The grid is plain uniform.** ``TimeGrid(t, timeSteps)``, or
  ``TimeGrid(t, max(int(timeStepsPerYear * t), 1))`` — *unlike* the forward-start
  engines in this package, which use mandatory points. The ``max(..., 1)`` clamp
  really fires: a sub-year swap with ``timeStepsPerYear = 1`` collapses to a
  single step.

* **The discount is taken at the maturity DATE**, ``riskFreeRate()->discount(
  arguments_.maturityDate)``, not at ``timeGrid().back()``. That is the opposite
  convention from ``MCEuropeanEngine`` and from the forward-start engines in this
  same directory, and the difference is observable on a non-flat curve.

* **The position multiplier reaches the error estimate too.** C++ computes
  ``multiplier = ±1 * riskFreeDiscount * notional`` and then assigns
  ``results_.errorEstimate = multiplier * varianceError``. For a *short*
  position that error estimate is **negative** — an odd quantity, but it is what
  v1.43 produces and the reference pins it
  (``vs_short_steps52_pr``: ``errorEstimate = -1.38...e-11``). A port that takes
  an absolute value passes the long tests and fails the short one.

* **`results.variance` is the raw sample mean**, and is filled whichever RNG is
  used; only ``errorEstimate`` is gated on ``allowsErrorEstimate``.

* **No control variate.** The C++ constructor passes ``false`` to
  ``McSimulation`` unconditionally and ``MakeMCVarianceSwapEngine`` has no
  ``withControlVariate``.

* **The engine does not register with the process.** Every other MC engine in
  this wave ends its constructor with ``registerWith(process_)``; this one does
  not (mcvarianceswapengine.hpp:177-193). So a quote bump does not invalidate a
  cached NPV here. Reproduced deliberately — it is v1.43 behaviour, not a port
  oversight.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.variance_swap import VarianceSwapArguments, VarianceSwapResults
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.monte_carlo_model import PathGeneratorTypeProtocol
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.position import PositionType
from pquantlib.pricingengines.forward.variance_path_pricer import VariancePathPricer
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.time_grid import TimeGrid


class MCVarianceSwapEngine(
    GenericEngine[VarianceSwapArguments, VarianceSwapResults],
    McSimulation[Path],
):
    """Monte Carlo variance-swap engine.

    # C++ parity: ``MCVarianceSwapEngine<RNG, S>``
    # (mcvarianceswapengine.hpp:48-126, 166-221).
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        *,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, VarianceSwapArguments(), VarianceSwapResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            antithetic_variate=antithetic_variate,
            # C++ passes a literal ``false`` (mcvarianceswapengine.hpp:177).
            control_variate=False,
        )

        # C++ parity: mcvarianceswapengine.hpp:181-192 — the four guards.
        qassert.require(
            (time_steps is not None) or (time_steps_per_year is not None),
            "no time steps provided",
        )
        qassert.require(
            (time_steps is None) or (time_steps_per_year is None),
            "both time steps and time steps per year were provided",
        )
        if time_steps is not None:
            qassert.require(
                time_steps != 0, f"timeSteps must be positive, {time_steps} not allowed"
            )
        if time_steps_per_year is not None:
            qassert.require(
                time_steps_per_year != 0,
                f"timeStepsPerYear must be positive, {time_steps_per_year} not allowed",
            )

        self._process: GeneralizedBlackScholesProcess = process
        self._time_steps: int | None = time_steps
        self._time_steps_per_year: int | None = time_steps_per_year
        self._required_samples: int | None = required_samples
        self._max_samples: int | None = max_samples
        self._required_tolerance: float | None = required_tolerance
        self._brownian_bridge: bool = brownian_bridge
        self._seed: int = seed
        self._rng_traits: RngTraits = rng_traits
        # NOTE: no ``process.register_with(self)`` here -- see the module
        # docstring. C++ omits it too.

    # --- engine entry-point -------------------------------------------------

    def calculate(self) -> None:
        """Run the MC, then scale the fair variance into an NPV.

        # C++ parity: ``MCVarianceSwapEngine::calculate``
        # (mcvarianceswapengine.hpp:70-100).
        """
        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )
        assert self._mc_model is not None
        accumulator = self._mc_model.sample_accumulator()

        args = self._arguments
        results = self._results
        results.variance = accumulator.mean()

        risk_free_discount = self._process.risk_free_rate().discount(args.maturity_date)
        if args.position == PositionType.Long:
            multiplier = 1.0
        elif args.position == PositionType.Short:
            multiplier = -1.0
        else:  # pragma: no cover - PositionType has exactly two members
            raise LibraryException("Unknown position")
        assert args.notional is not None
        assert args.strike is not None
        multiplier *= risk_free_discount * args.notional

        results.value = multiplier * (results.variance - args.strike)

        # C++ parity: ``if constexpr (RNG::allowsErrorEstimate)``. The
        # multiplier is applied to the error too, so a SHORT position reports a
        # negative error estimate.
        if self._rng_traits.allows_error_estimate:
            results.error_estimate = multiplier * accumulator.error_estimate()

    # --- McSimulation hooks -------------------------------------------------

    def time_grid(self) -> TimeGrid:
        """Uniform grid over ``[0, t(maturity)]``.

        # C++ parity: ``MCVarianceSwapEngine::timeGrid``
        # (mcvarianceswapengine.hpp:196-209).
        """
        t = self._process.time(self._arguments.maturity_date)
        if self._time_steps is not None:
            return TimeGrid.regular(t, self._time_steps)
        qassert.require(self._time_steps_per_year is not None, "time steps not specified")
        assert self._time_steps_per_year is not None
        # C++ ``Size(timeStepsPerYear_*t)`` truncates; ``max(steps, 1)`` rescues
        # the zero case.
        steps = int(self._time_steps_per_year * t)
        return TimeGrid.regular(t, max(steps, 1))

    def path_generator(self) -> PathGeneratorTypeProtocol[Path]:
        """# C++ parity: ``MCVarianceSwapEngine::pathGenerator``
        # (mcvarianceswapengine.hpp:107-118)."""
        grid = self.time_grid()
        dimensions = self._process.factors() * (len(grid) - 1)
        generator = self._rng_traits.make_sequence_generator(dimensions, self._seed)
        return PathGenerator.with_time_grid(
            self._process, grid, generator, brownian_bridge=self._brownian_bridge
        )

    def path_pricer(self) -> PathPricer[Path]:
        """# C++ parity: ``MCVarianceSwapEngine::pathPricer``
        # (mcvarianceswapengine.hpp:212-221)."""
        return VariancePathPricer(self._process)


class MakeMCVarianceSwapEngine:
    """Fluent builder for :class:`MCVarianceSwapEngine`.

    # C++ parity: ``MakeMCVarianceSwapEngine<RNG, S>``
    # (mcvarianceswapengine.hpp:131-152, 224-309).

    Note the conversion-time message: "number of steps overspecified", with no
    trailing explanation — unlike the forward-start builders, which append
    " - set EITHER steps OR stepsPerYear". Copied verbatim, differences included.
    """

    __slots__ = (
        "_antithetic",
        "_brownian_bridge",
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
        self,
        process: GeneralizedBlackScholesProcess,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        self._process: GeneralizedBlackScholesProcess = process
        self._rng_traits: RngTraits = rng_traits
        self._antithetic: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._brownian_bridge: bool = False
        self._seed: int = 0

    def with_steps(self, steps: int) -> MakeMCVarianceSwapEngine:
        """# C++ parity: ``withSteps`` (mcvarianceswapengine.hpp:230-235)."""
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCVarianceSwapEngine:
        """# C++ parity: ``withStepsPerYear`` (hpp:237-242)."""
        self._steps_per_year = steps
        return self

    def with_brownian_bridge(self, b: bool = True) -> MakeMCVarianceSwapEngine:
        """# C++ parity: ``withBrownianBridge(bool b = true)`` (hpp:279-284)."""
        self._brownian_bridge = b
        return self

    def with_samples(self, samples: int) -> MakeMCVarianceSwapEngine:
        """# C++ parity: ``withSamples`` (hpp:244-251)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCVarianceSwapEngine:
        """# C++ parity: ``withAbsoluteTolerance`` (hpp:253-264)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCVarianceSwapEngine:
        """# C++ parity: ``withMaxSamples`` (hpp:266-271)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCVarianceSwapEngine:
        """# C++ parity: ``withSeed`` (hpp:273-277)."""
        self._seed = seed
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCVarianceSwapEngine:
        """# C++ parity: ``withAntitheticVariate(bool b = true)`` (hpp:286-291)."""
        self._antithetic = b
        return self

    def engine(self) -> MCVarianceSwapEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (hpp:293-309)."""
        qassert.require(
            (self._steps is not None) or (self._steps_per_year is not None),
            "number of steps not given",
        )
        qassert.require(
            (self._steps is None) or (self._steps_per_year is None),
            "number of steps overspecified",
        )
        return MCVarianceSwapEngine(
            self._process,
            time_steps=self._steps,
            time_steps_per_year=self._steps_per_year,
            brownian_bridge=self._brownian_bridge,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            rng_traits=self._rng_traits,
        )


__all__ = ["MCVarianceSwapEngine", "MakeMCVarianceSwapEngine"]
