"""MCEuropeanEngine — Monte Carlo pricing for European vanilla options.

# C++ parity: ql/pricingengines/vanilla/mceuropeanengine.hpp (v1.43) —
# ``template <class RNG = PseudoRandom, class S = Statistics>
#  class MCEuropeanEngine : public MCVanillaEngine<SingleVariate, RNG, S>``,
# ``class EuropeanPathPricer``, and
# ``template <class RNG, class S> class MakeMCEuropeanEngine``.

Drives a single-asset MC simulation for a European vanilla option under a
``GeneralizedBlackScholesProcess`` and prices each path as the terminal
payoff times the maturity discount factor.

The fluent builder
------------------
C++ ends the chain with an implicit conversion operator::

    ext::shared_ptr<PricingEngine> e =
        MakeMCEuropeanEngine<PseudoRandom>(process).withSteps(4).withSamples(1023);

Python has no implicit conversion operator, so the chain ends with an
explicit terminal call::

    engine = MakeMCEuropeanEngine(process).with_steps(4).with_samples(1023).engine()

Every ``with_*`` returns ``self`` (C++ returns ``*this``), and
:meth:`MakeMCEuropeanEngine.engine` is the port of
``operator ext::shared_ptr<PricingEngine>() const`` — including its two
guards, which fire at *conversion* time and not at ``with_*`` time.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import MCVanillaEngine, RngTraits
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class EuropeanPathPricer(PathPricer[Path]):
    """Plain-vanilla terminal-payoff pricer.

    # C++ parity: ``EuropeanPathPricer`` (mceuropeanengine.hpp:93-103, 246-257).

    Stores a ``PlainVanillaPayoff`` (option type + strike) plus the maturity
    discount factor. Each path prices as ``payoff(path.back()) * discount``.
    """

    __slots__ = ("_discount", "_payoff")

    def __init__(
        self,
        option_type: OptionType,
        strike: float,
        discount: float,
    ) -> None:
        # C++ parity: ``QL_REQUIRE(strike>=0.0, ...)`` (mceuropeanengine.hpp:250).
        qassert.require(strike >= 0.0, "strike less than zero not allowed")
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discount: float = discount

    def __call__(self, path: Path) -> float:
        # C++ parity: mceuropeanengine.hpp:254-257.
        qassert.require(not path.empty(), "the path cannot be empty")
        return self._payoff(path.back()) * self._discount


class MCEuropeanEngine(MCVanillaEngine[Path]):
    """Monte Carlo pricing engine for European vanilla options.

    # C++ parity: ``MCEuropeanEngine<RNG, S>`` (mceuropeanengine.hpp:42-66).

    The C++ constructor hard-wires ``controlVariate = false``
    (mceuropeanengine.hpp:125), so this engine never runs a control variate.
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
            control_variate=False,
            required_samples=required_samples,
            required_tolerance=required_tolerance,
            max_samples=max_samples,
            seed=seed,
            rng_traits=rng_traits,
            multi_variate=False,
        )

    def path_pricer(self) -> PathPricer[Path]:
        """Build the terminal-payoff pricer with the maturity discount.

        # C++ parity: ``MCEuropeanEngine::pathPricer``
        # (mceuropeanengine.hpp:132-153).
        """
        payoff = self._arguments.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)

        process = self._process
        qassert.require(
            isinstance(process, GeneralizedBlackScholesProcess),
            "Black-Scholes process required",
        )
        assert isinstance(process, GeneralizedBlackScholesProcess)

        # C++ discounts at ``this->timeGrid().back()`` -- the *grid* end point
        # ``(t/steps)*steps``, which is not bit-identical to ``t`` and, on a
        # non-flat curve, is not interpolation-identical to
        # ``discount(exerciseDate)`` either. Discounting by date instead would
        # be a silent divergence.
        discount = process.risk_free_rate().discount(self.time_grid().back())
        return EuropeanPathPricer(
            option_type=payoff.option_type(),
            strike=payoff.strike(),
            discount=discount,
        )


class MakeMCEuropeanEngine:
    """Fluent builder for :class:`MCEuropeanEngine`.

    # C++ parity: ``MakeMCEuropeanEngine<RNG, S>``
    # (mceuropeanengine.hpp:69-91, 157-242).

    Defaults reproduce the C++ member initialisers exactly:
    ``antithetic_ = false``, ``brownianBridge_ = false``, ``seed_ = 0``, and
    ``steps_ / stepsPerYear_ / samples_ / maxSamples_ / tolerance_`` all
    ``Null`` (``None`` here).
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

    def with_steps(self, steps: int) -> MakeMCEuropeanEngine:
        """# C++ parity: ``withSteps`` (mceuropeanengine.hpp:163-167).

        Note there is deliberately no guard here: the mutual exclusion with
        ``withStepsPerYear`` is checked at conversion time, not here.
        """
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCEuropeanEngine:
        """# C++ parity: ``withStepsPerYear`` (mceuropeanengine.hpp:170-174)."""
        self._steps_per_year = steps
        return self

    def with_brownian_bridge(self, brownian_bridge: bool = True) -> MakeMCEuropeanEngine:
        """# C++ parity: ``withBrownianBridge`` (mceuropeanengine.hpp:212-216)."""
        self._brownian_bridge = brownian_bridge
        return self

    def with_samples(self, samples: int) -> MakeMCEuropeanEngine:
        """# C++ parity: ``withSamples`` (mceuropeanengine.hpp:177-183)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCEuropeanEngine:
        """# C++ parity: ``withAbsoluteTolerance`` (mceuropeanengine.hpp:186-195).

        The ``QL_REQUIRE(RNG::allowsErrorEstimate)`` is a compile-time constant
        in C++; here it is a runtime check on the traits class.
        """
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCEuropeanEngine:
        """# C++ parity: ``withMaxSamples`` (mceuropeanengine.hpp:198-202)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCEuropeanEngine:
        """# C++ parity: ``withSeed`` (mceuropeanengine.hpp:205-209)."""
        self._seed = seed
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCEuropeanEngine:
        """# C++ parity: ``withAntitheticVariate`` (mceuropeanengine.hpp:219-223)."""
        self._antithetic = b
        return self

    def engine(self) -> MCEuropeanEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (mceuropeanengine.hpp:226-242)."""
        qassert.require(
            (self._steps is not None) or (self._steps_per_year is not None),
            "number of steps not given",
        )
        qassert.require(
            (self._steps is None) or (self._steps_per_year is None),
            "number of steps overspecified",
        )
        return MCEuropeanEngine(
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


__all__ = ["EuropeanPathPricer", "MCEuropeanEngine", "MakeMCEuropeanEngine"]
