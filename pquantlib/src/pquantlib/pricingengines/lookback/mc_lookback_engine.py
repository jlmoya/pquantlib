"""MCLookbackEngine — Monte Carlo engine for continuous lookback options.

# C++ parity: ql/pricingengines/lookback/mclookbackengine.{hpp,cpp} (v1.43) —
# ``template <class I, class RNG = PseudoRandom, class S = Statistics>
#  class MCLookbackEngine : public I::engine,
#                           public McSimulation<SingleVariate, RNG, S>``,
# ``template <class I, class RNG, class S> class MakeMCLookbackEngine``,
# the four ``detail::mc_lookback_path_pricer`` overloads, and the four path
# pricers ``LookbackFixedPathPricer`` / ``LookbackPartialFixedPathPricer`` /
# ``LookbackFloatingPathPricer`` / ``LookbackPartialFloatingPathPricer``
# (mclookbackengine.cpp:25-254).

The C++ template parameter ``I`` is the *instrument* class, and it selects which
``detail::mc_lookback_path_pricer`` overload is compiled in. Python has no
template argument, so ``I`` becomes a constructor parameter (the instrument
class), used for exactly two things: choosing which ``arguments`` object the
engine owns, and — indirectly — which path pricer :func:`mc_lookback_path_pricer`
builds. The overload set is reproduced as a runtime dispatch that tests the
*partial* argument classes before their bases, because in Python
``ContinuousPartialFixedLookbackOptionArguments`` **is a**
``ContinuousFixedLookbackOptionArguments`` while in C++ overload resolution the
two are distinct static types.

What a port gets wrong here
---------------------------

1. **The extremum scan excludes ``path[0]``.** Every one of the four pricers
   starts at ``path.begin()+1``, so the spot at ``t = 0`` never enters the
   running maximum or minimum. Including it would systematically change a
   floating-strike price and would silently cheapen a fixed-strike call.

2. **The two partial variants are not mirror images.**
   ``LookbackPartialFixedPathPricer`` scans ``[start_index+1, end)`` — exclusive
   of the window-start node — while ``LookbackPartialFloatingPathPricer`` scans
   ``[1, end_index+1)`` — *inclusive* of the window-end node. A port that makes
   them symmetric gets one of the two wrong.

3. **The window boundary index is ``closest_index``**, not a floor. The grid is
   uniform with no mandatory points (unlike the forward-start engines), so the
   boundary genuinely lands between nodes.

4. **The MC engines ignore ``minmax`` and ``lambda``.** The analytic lookback
   engines use both; these path pricers never see either. That is v1.43
   behaviour, pinned by the ``lb_*_ignored`` cases in ``v143/pe/mcfwdlb``: two
   different ``minmax`` values (and two different ``lambda`` values) must give
   bit-identical MC prices. A port that "helpfully" seeds the running extremum
   with ``minmax`` is caught there.

5. **The Gaussian dimension is ``grid.size() - 1``, with no ``factors()``
   multiplier** (mclookbackengine.hpp:71-72). The engine is single-variate so a
   1-D process makes the two spellings identical, but the C++ text is the one
   reproduced.

6. **``calculate()`` checks the spot before simulating**, not after.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.instruments.lookback_option import (
    ContinuousFixedLookbackOption,
    ContinuousFixedLookbackOptionArguments,
    ContinuousFloatingLookbackOption,
    ContinuousFloatingLookbackOptionArguments,
    ContinuousPartialFixedLookbackOption,
    ContinuousPartialFixedLookbackOptionArguments,
    ContinuousPartialFloatingLookbackOption,
    ContinuousPartialFloatingLookbackOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.monte_carlo_model import PathGeneratorTypeProtocol
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.option import OptionArguments
from pquantlib.payoffs import FloatingTypePayoff, OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.time_grid import TimeGrid

#: The instrument classes C++ instantiates ``MCLookbackEngine<I, ...>`` with.
#: The two partial variants are subclasses of the two full-window ones, so the
#: union of the two bases already covers all four.
type LookbackInstrument = (
    type[ContinuousFixedLookbackOption] | type[ContinuousFloatingLookbackOption]
)


class LookbackFixedPathPricer(PathPricer[Path]):
    """Fixed-strike lookback: payoff on the running extremum of the whole path.

    # C++ parity: ``LookbackFixedPathPricer``
    # (mclookbackengine.cpp:25-35, 144-169).
    """

    __slots__ = ("_discount", "_payoff")

    def __init__(self, option_type: OptionType, strike: float, discount: float) -> None:
        # C++ parity: mclookbackengine.cpp:149-150.
        qassert.require(strike >= 0.0, "strike less than zero not allowed")
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discount: float = discount

    def __call__(self, path: Path) -> float:
        """# C++ parity: ``LookbackFixedPathPricer::operator()``
        # (mclookbackengine.cpp:153-169)."""
        qassert.require(not path.empty(), "the path cannot be empty")
        # C++ scans [begin+1, end): the t=0 spot is NOT part of the extremum.
        window = path.values[1:]
        if self._payoff.option_type() == OptionType.Put:
            underlying = float(window.min())
        else:
            underlying = float(window.max())
        return self._payoff(underlying) * self._discount


class LookbackPartialFixedPathPricer(PathPricer[Path]):
    """Partial-time fixed-strike lookback: extremum over ``[start, expiry]``.

    # C++ parity: ``LookbackPartialFixedPathPricer``
    # (mclookbackengine.cpp:37-49, 172-200).
    """

    __slots__ = ("_discount", "_lookback_start", "_payoff")

    def __init__(
        self,
        lookback_start: float,
        option_type: OptionType,
        strike: float,
        discount: float,
    ) -> None:
        # C++ parity: mclookbackengine.cpp:178-179.
        qassert.require(strike >= 0.0, "strike less than zero not allowed")
        self._lookback_start: float = lookback_start
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discount: float = discount

    def __call__(self, path: Path) -> float:
        """# C++ parity: ``LookbackPartialFixedPathPricer::operator()``
        # (mclookbackengine.cpp:182-200)."""
        qassert.require(not path.empty(), "the path cannot be empty")
        start_index = path.time_grid.closest_index(self._lookback_start)
        # C++ scans [begin+startIndex+1, end) -- EXCLUSIVE of the window-start
        # node, unlike the partial-floating pricer below.
        window = path.values[start_index + 1 :]
        if self._payoff.option_type() == OptionType.Put:
            underlying = float(window.min())
        else:
            underlying = float(window.max())
        return self._payoff(underlying) * self._discount


class LookbackFloatingPathPricer(PathPricer[Path]):
    """Floating-strike lookback: strike struck at the whole path's extremum.

    # C++ parity: ``LookbackFloatingPathPricer``
    # (mclookbackengine.cpp:51-60, 203-225).
    """

    __slots__ = ("_discount", "_payoff")

    def __init__(self, option_type: OptionType, discount: float) -> None:
        self._payoff: FloatingTypePayoff = FloatingTypePayoff(option_type)
        self._discount: float = discount

    def __call__(self, path: Path) -> float:
        """# C++ parity: ``LookbackFloatingPathPricer::operator()``
        # (mclookbackengine.cpp:208-225)."""
        qassert.require(not path.empty(), "the path cannot be empty")
        terminal_price = path.back()
        window = path.values[1:]
        # NOTE the inversion relative to the fixed pricer: a floating CALL takes
        # the MINIMUM as its strike, a floating PUT the maximum.
        strike = (
            float(window.min())
            if self._payoff.option_type() == OptionType.Call
            else float(window.max())
        )
        return self._payoff.call(terminal_price, strike) * self._discount


class LookbackPartialFloatingPathPricer(PathPricer[Path]):
    """Partial-time floating-strike lookback: extremum over ``[0, end]``.

    # C++ parity: ``LookbackPartialFloatingPathPricer``
    # (mclookbackengine.cpp:62-73, 228-254).
    """

    __slots__ = ("_discount", "_lookback_end", "_payoff")

    def __init__(
        self, lookback_end: float, option_type: OptionType, discount: float
    ) -> None:
        self._lookback_end: float = lookback_end
        self._payoff: FloatingTypePayoff = FloatingTypePayoff(option_type)
        self._discount: float = discount

    def __call__(self, path: Path) -> float:
        """# C++ parity: ``LookbackPartialFloatingPathPricer::operator()``
        # (mclookbackengine.cpp:234-254)."""
        qassert.require(not path.empty(), "the path cannot be empty")
        end_index = path.time_grid.closest_index(self._lookback_end)
        terminal_price = path.back()
        # C++ scans [begin+1, begin+endIndex+1) -- INCLUSIVE of the window-end
        # node. An end index of 0 makes that range empty, which is undefined
        # behaviour in C++ (min_element returns `end`, then gets dereferenced);
        # here numpy raises instead of reading past the array.
        window = path.values[1 : end_index + 1]
        strike = (
            float(window.min())
            if self._payoff.option_type() == OptionType.Call
            else float(window.max())
        )
        return self._payoff.call(terminal_price, strike) * self._discount


def mc_lookback_path_pricer(
    arguments: OptionArguments,
    process: GeneralizedBlackScholesProcess,
    discount: float,
) -> PathPricer[Path]:
    """Build the path pricer matching the lookback flavour of ``arguments``.

    # C++ parity: the four ``detail::mc_lookback_path_pricer`` overloads
    # (mclookbackengine.hpp:162-190 declarations, .cpp:75-141 definitions).

    C++ resolves the overload statically on the argument type. Python's argument
    classes form an inheritance chain (``...PartialFixed...`` derives from
    ``...Fixed...``), so the partial cases must be tested first for the dispatch
    to reproduce C++'s specificity.
    """
    if isinstance(arguments, ContinuousPartialFixedLookbackOptionArguments):
        payoff = arguments.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)
        lookback_start = process.time(arguments.lookback_period_start)
        return LookbackPartialFixedPathPricer(
            lookback_start, payoff.option_type(), payoff.strike(), discount
        )

    if isinstance(arguments, ContinuousPartialFloatingLookbackOptionArguments):
        payoff = arguments.payoff
        qassert.require(
            isinstance(payoff, FloatingTypePayoff), "non-floating payoff given"
        )
        assert isinstance(payoff, FloatingTypePayoff)
        lookback_end = process.time(arguments.lookback_period_end)
        return LookbackPartialFloatingPathPricer(
            lookback_end, payoff.option_type(), discount
        )

    if isinstance(arguments, ContinuousFixedLookbackOptionArguments):
        payoff = arguments.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)
        return LookbackFixedPathPricer(payoff.option_type(), payoff.strike(), discount)

    if isinstance(arguments, ContinuousFloatingLookbackOptionArguments):
        payoff = arguments.payoff
        qassert.require(
            isinstance(payoff, FloatingTypePayoff), "non-floating payoff given"
        )
        assert isinstance(payoff, FloatingTypePayoff)
        return LookbackFloatingPathPricer(payoff.option_type(), discount)

    # No C++ counterpart: there the absence of a matching overload is a compile
    # error, so this branch stands in for "would not have compiled".
    qassert.require(False, "unsupported lookback arguments type")
    raise AssertionError  # pragma: no cover - qassert.require always raises


def _arguments_for(instrument: LookbackInstrument) -> OptionArguments:
    """Map the C++ template parameter ``I`` onto its ``I::arguments``.

    Tested most-derived-first for the same reason
    :func:`mc_lookback_path_pricer` is.
    """
    if instrument is ContinuousPartialFixedLookbackOption:
        return ContinuousPartialFixedLookbackOptionArguments()
    if instrument is ContinuousPartialFloatingLookbackOption:
        return ContinuousPartialFloatingLookbackOptionArguments()
    if instrument is ContinuousFixedLookbackOption:
        return ContinuousFixedLookbackOptionArguments()
    if instrument is ContinuousFloatingLookbackOption:
        return ContinuousFloatingLookbackOptionArguments()
    qassert.require(False, "unsupported lookback instrument type")
    raise AssertionError  # pragma: no cover - qassert.require always raises


class MCLookbackEngine(
    GenericEngine[OptionArguments, OneAssetOptionResults],
    McSimulation[Path],
):
    """Monte Carlo engine for continuous lookback options.

    # C++ parity: ``MCLookbackEngine<I, RNG, S>``
    # (mclookbackengine.hpp:37-86, 116-202).

    Args:
        process: the Black-Scholes process of the underlying.
        instrument: the lookback class being priced — the Python stand-in for
            the C++ template parameter ``I``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        instrument: LookbackInstrument,
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
            self, _arguments_for(instrument), OneAssetOptionResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            antithetic_variate=antithetic_variate,
            # C++ passes a literal ``false`` (mclookbackengine.hpp:127).
            control_variate=False,
        )

        # C++ parity: mclookbackengine.hpp:131-142 — the four guards.
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
        self._instrument: LookbackInstrument = instrument
        self._time_steps: int | None = time_steps
        self._time_steps_per_year: int | None = time_steps_per_year
        self._required_samples: int | None = required_samples
        self._max_samples: int | None = max_samples
        self._required_tolerance: float | None = required_tolerance
        self._brownian_bridge: bool = brownian_bridge
        self._seed: int = seed
        self._rng_traits: RngTraits = rng_traits
        process.register_with(self)

    # --- engine entry-point -------------------------------------------------

    def calculate(self) -> None:
        """# C++ parity: ``MCLookbackEngine::calculate``
        # (mclookbackengine.hpp:54-64)."""
        spot = self._process.x0()
        # C++ checks this BEFORE running the simulation.
        qassert.require(spot > 0.0, "negative or null underlying given")
        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )
        assert self._mc_model is not None
        accumulator = self._mc_model.sample_accumulator()
        self._results.value = accumulator.mean()
        if self._rng_traits.allows_error_estimate:
            self._results.error_estimate = accumulator.error_estimate()

    # --- McSimulation hooks -------------------------------------------------

    def time_grid(self) -> TimeGrid:
        """Uniform grid over ``[0, t(last exercise date)]``.

        # C++ parity: ``MCLookbackEngine::timeGrid``
        # (mclookbackengine.hpp:147-159).
        """
        exercise = self._arguments.exercise
        qassert.require(exercise is not None, "no exercise given")
        assert exercise is not None
        residual_time = self._process.time(exercise.last_date())
        if self._time_steps is not None:
            return TimeGrid.regular(residual_time, self._time_steps)
        qassert.require(self._time_steps_per_year is not None, "time steps not specified")
        assert self._time_steps_per_year is not None
        steps = int(self._time_steps_per_year * residual_time)
        return TimeGrid.regular(residual_time, max(steps, 1))

    def path_generator(self) -> PathGeneratorTypeProtocol[Path]:
        """# C++ parity: ``MCLookbackEngine::pathGenerator``
        # (mclookbackengine.hpp:69-76).

        The dimension is ``grid.size() - 1`` with NO ``factors()`` multiplier —
        the C++ text, reproduced. Single-variate engines only ever see a 1-D
        process, so the two spellings agree.
        """
        grid = self.time_grid()
        generator = self._rng_traits.make_sequence_generator(len(grid) - 1, self._seed)
        return PathGenerator.with_time_grid(
            self._process, grid, generator, brownian_bridge=self._brownian_bridge
        )

    def path_pricer(self) -> PathPricer[Path]:
        """# C++ parity: ``MCLookbackEngine::pathPricer``
        # (mclookbackengine.hpp:193-202)."""
        grid = self.time_grid()
        discount = self._process.risk_free_rate().discount(grid.back())
        return mc_lookback_path_pricer(self._arguments, self._process, discount)


class MakeMCLookbackEngine:
    """Fluent builder for :class:`MCLookbackEngine`.

    # C++ parity: ``MakeMCLookbackEngine<I, RNG, S>``
    # (mclookbackengine.hpp:91-111, 205-291).

    ``instrument`` is the Python stand-in for the C++ template parameter ``I``.
    """

    __slots__ = (
        "_antithetic",
        "_brownian_bridge",
        "_instrument",
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
        instrument: LookbackInstrument,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        self._process: GeneralizedBlackScholesProcess = process
        self._instrument: LookbackInstrument = instrument
        self._rng_traits: RngTraits = rng_traits
        self._brownian_bridge: bool = False
        self._antithetic: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    def with_steps(self, steps: int) -> MakeMCLookbackEngine:
        """# C++ parity: ``withSteps`` (mclookbackengine.hpp:211-216)."""
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCLookbackEngine:
        """# C++ parity: ``withStepsPerYear`` (hpp:218-223)."""
        self._steps_per_year = steps
        return self

    def with_brownian_bridge(self, b: bool = True) -> MakeMCLookbackEngine:
        """# C++ parity: ``withBrownianBridge(bool b = true)`` (hpp:225-230)."""
        self._brownian_bridge = b
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCLookbackEngine:
        """# C++ parity: ``withAntitheticVariate(bool b = true)`` (hpp:232-237)."""
        self._antithetic = b
        return self

    def with_samples(self, samples: int) -> MakeMCLookbackEngine:
        """# C++ parity: ``withSamples`` (hpp:239-246)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCLookbackEngine:
        """# C++ parity: ``withAbsoluteTolerance`` (hpp:248-258)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCLookbackEngine:
        """# C++ parity: ``withMaxSamples`` (hpp:260-265)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCLookbackEngine:
        """# C++ parity: ``withSeed`` (hpp:267-272)."""
        self._seed = seed
        return self

    def engine(self) -> MCLookbackEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (hpp:274-291)."""
        qassert.require(
            (self._steps is not None) or (self._steps_per_year is not None),
            "number of steps not given",
        )
        qassert.require(
            (self._steps is None) or (self._steps_per_year is None),
            "number of steps overspecified",
        )
        return MCLookbackEngine(
            self._process,
            self._instrument,
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


__all__ = [
    "LookbackFixedPathPricer",
    "LookbackFloatingPathPricer",
    "LookbackInstrument",
    "LookbackPartialFixedPathPricer",
    "LookbackPartialFloatingPathPricer",
    "MCLookbackEngine",
    "MakeMCLookbackEngine",
    "mc_lookback_path_pricer",
]
