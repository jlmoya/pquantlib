"""MCForwardEuropeanBSEngine — MC forward-start European under Black-Scholes.

# C++ parity: ql/pricingengines/forward/mcforwardeuropeanbsengine.{hpp,cpp}
# (v1.43) — ``template <class RNG = PseudoRandom, class S = Statistics>
#  class MCForwardEuropeanBSEngine
#      : public MCForwardVanillaEngine<SingleVariate, RNG, S>``,
# ``class ForwardEuropeanBSPathPricer : public PathPricer<Path>``, and
# ``template <class RNG, class S> class MakeMCForwardEuropeanBSEngine``.

The whole difference from a plain European MC engine is two lines of the path
pricer::

    const Real resetLevel = path[resetIndex_];
    const Real strike     = resetLevel * moneyness_;

i.e. the strike is *read off the path* at the reset index rather than taken from
the payoff. The payoff carried by the instrument is a placeholder whose strike is
never used — only its option type is.

``resetIndex`` comes from ``timeGrid.closestIndex(resetTime)``: **closest**, not
floor and not ``lower_bound``. The base engine's mandatory-point grid puts the
reset time exactly on a node, so on an unperturbed grid the distinction does not
bite — but the C++ code does not rely on that, and neither does this port.

The discount is ``riskFreeRate()->discount(timeGrid.back())`` — the grid's last
*time*, not the exercise *date*. Those differ in the last bits under a flat curve
and by an interpolation step under a non-flat one.

Two moneyness guards, and the stricter one wins
-----------------------------------------------
:class:`ForwardEuropeanBSPathPricer` requires ``moneyness >= 0``, but
``ForwardOptionArguments.validate`` requires ``moneyness > 0`` and runs first, so
through the instrument a zero moneyness is *rejected* and the path pricer's
boundary is unreachable. Constructed directly the pricer does accept zero. Both
behaviours are pinned in ``v143/pe/mcfwdlb`` and ``v143/pe/mcforward``.

Control variate
---------------
C++ ``MCForwardEuropeanBSEngine``'s constructor does not forward a
``controlVariate`` flag (it takes the base-class default ``false``) and
``MakeMCForwardEuropeanBSEngine`` has no ``withControlVariate``. Reproduced: this
engine never runs a control variate. Only the Heston sibling exposes one.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.forward.mc_forward_vanilla_engine import (
    MCForwardVanillaEngine,
)
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class ForwardEuropeanBSPathPricer(PathPricer[Path]):
    """Terminal payoff struck at ``path[reset_index] * moneyness``.

    # C++ parity: ``ForwardEuropeanBSPathPricer``
    # (mcforwardeuropeanbsengine.hpp:89-102, .cpp:20-41).
    """

    __slots__ = ("_discount", "_moneyness", "_option_type", "_reset_index")

    def __init__(
        self,
        option_type: OptionType,
        moneyness: float,
        reset_index: int,
        discount: float,
    ) -> None:
        # C++ parity: mcforwardeuropeanbsengine.cpp:28-29.
        qassert.require(moneyness >= 0.0, "moneyness less than zero not allowed")
        self._option_type: OptionType = option_type
        self._moneyness: float = moneyness
        self._reset_index: int = reset_index
        self._discount: float = discount

    def __call__(self, path: Path) -> float:
        """# C++ parity: ``ForwardEuropeanBSPathPricer::operator()``
        # (mcforwardeuropeanbsengine.cpp:32-41)."""
        # C++ computes ``n = path.length() - 1`` purely for this guard.
        n = path.length() - 1
        qassert.require(n > 0, "the path cannot be empty")

        reset_level = path[self._reset_index]
        strike = reset_level * self._moneyness
        payoff = PlainVanillaPayoff(self._option_type, strike)
        return payoff(path.back()) * self._discount


class MCForwardEuropeanBSEngine(MCForwardVanillaEngine[Path]):
    """MC engine for a forward-starting European under a BSM process.

    # C++ parity: ``MCForwardEuropeanBSEngine<RNG, S>``
    # (mcforwardeuropeanbsengine.hpp:36-60, 107-127).
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
        super().__init__(
            process,
            time_steps=time_steps,
            time_steps_per_year=time_steps_per_year,
            brownian_bridge=brownian_bridge,
            antithetic_variate=antithetic_variate,
            required_samples=required_samples,
            required_tolerance=required_tolerance,
            max_samples=max_samples,
            seed=seed,
            # C++ passes no controlVariate argument at all
            # (mcforwardeuropeanbsengine.hpp:119-127), so the base default
            # ``false`` applies.
            control_variate=False,
            rng_traits=rng_traits,
            multi_variate=False,
        )

    def path_pricer(self) -> PathPricer[Path]:
        """# C++ parity: ``MCForwardEuropeanBSEngine::pathPricer``
        # (mcforwardeuropeanbsengine.hpp:130-163)."""
        grid = self.time_grid()

        args = self._arguments
        assert args.reset_date is not None
        reset_time = self._process.time(args.reset_date)
        reset_index = grid.closest_index(reset_time)

        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)

        qassert.require(
            isinstance(args.exercise, EuropeanExercise), "wrong exercise given"
        )

        process = self._process
        qassert.require(
            isinstance(process, GeneralizedBlackScholesProcess),
            "Black-Scholes process required",
        )
        assert isinstance(process, GeneralizedBlackScholesProcess)

        assert args.moneyness is not None
        return ForwardEuropeanBSPathPricer(
            payoff.option_type(),
            args.moneyness,
            reset_index,
            process.risk_free_rate().discount(grid.back()),
        )


class MakeMCForwardEuropeanBSEngine:
    """Fluent builder for :class:`MCForwardEuropeanBSEngine`.

    # C++ parity: ``MakeMCForwardEuropeanBSEngine<RNG, S>``
    # (mcforwardeuropeanbsengine.hpp:64-86, 167-253).

    C++ ends the chain with an implicit conversion operator; Python ends it with
    an explicit :meth:`engine` call. Every guard fires at the same moment as in
    C++: ``with_steps`` / ``with_steps_per_year`` never validate, the mutual
    exclusion is checked only at conversion time, and the tolerance/samples pair
    rejects each other on the spot.
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

    def with_steps(self, steps: int) -> MakeMCForwardEuropeanBSEngine:
        """# C++ parity: ``withSteps`` (mcforwardeuropeanbsengine.hpp:172-177)."""
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCForwardEuropeanBSEngine:
        """# C++ parity: ``withStepsPerYear`` (hpp:179-184)."""
        self._steps_per_year = steps
        return self

    def with_brownian_bridge(self, b: bool = False) -> MakeMCForwardEuropeanBSEngine:
        """# C++ parity: ``withBrownianBridge(bool b = false)`` (hpp:222-227).

        Note the default is ``false`` here, unlike ``MakeMCEuropeanEngine``'s
        ``withBrownianBridge(bool b = true)``. Calling it with no argument is
        therefore a no-op, which is what C++ does.
        """
        self._brownian_bridge = b
        return self

    def with_samples(self, samples: int) -> MakeMCForwardEuropeanBSEngine:
        """# C++ parity: ``withSamples`` (hpp:186-193)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCForwardEuropeanBSEngine:
        """# C++ parity: ``withAbsoluteTolerance`` (hpp:195-206).

        ``QL_REQUIRE(RNG::allowsErrorEstimate)`` is a compile-time constant in
        C++; here it is a runtime check on the traits class.
        """
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCForwardEuropeanBSEngine:
        """# C++ parity: ``withMaxSamples`` (hpp:208-213)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCForwardEuropeanBSEngine:
        """# C++ parity: ``withSeed`` (hpp:215-220)."""
        self._seed = seed
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCForwardEuropeanBSEngine:
        """# C++ parity: ``withAntitheticVariate`` (hpp:229-234)."""
        self._antithetic = b
        return self

    def engine(self) -> MCForwardEuropeanBSEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (hpp:236-253)."""
        qassert.require(
            (self._steps is not None) or (self._steps_per_year is not None),
            "number of steps not given",
        )
        qassert.require(
            (self._steps is None) or (self._steps_per_year is None),
            "number of steps overspecified - set EITHER steps OR stepsPerYear",
        )
        return MCForwardEuropeanBSEngine(
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


__all__ = [
    "ForwardEuropeanBSPathPricer",
    "MCForwardEuropeanBSEngine",
    "MakeMCForwardEuropeanBSEngine",
]
