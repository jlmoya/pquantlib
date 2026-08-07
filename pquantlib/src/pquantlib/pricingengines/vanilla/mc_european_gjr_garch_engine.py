"""MCEuropeanGJRGARCHEngine — Monte Carlo European pricing under GJR-GARCH.

# C++ parity: ql/pricingengines/vanilla/mceuropeangjrgarchengine.hpp (v1.43) —
# ``template <class RNG, class S> class MCEuropeanGJRGARCHEngine``,
# ``class EuropeanGJRGARCHPathPricer``, and
# ``template <class RNG, class S> class MakeMCEuropeanGJRGARCHEngine``.

This is a ``MultiVariate`` engine: paths come from a ``MultiPathGenerator`` over
the 2-factor GJR-GARCH process, so the Gaussian sequence has dimension
``factors * (grid.size() - 1)`` and is consumed two at a time per step. Brownian
bridge is unavailable (``MultiPathGenerator`` fails on it), which is why
``MCEuropeanGJRGARCHEngine`` has no ``brownianBridge`` parameter and passes
``false`` to its base (mceuropeangjrgarchengine.hpp:105).

The answer depends on ``GJRGARCHProcess``'s discretization scheme, whose C++
default is ``FullTruncation`` -- and on ``GJRGARCHProcess::evolve``, which is a
genuine override, not the generic Euler ``apply(expectation,
stdDeviation * dw)``.

Builder note: unlike ``MakeMCEuropeanEngine``, this builder rejects
over-specified steps EARLY, inside ``with_steps`` / ``with_steps_per_year``,
and its terminal :meth:`MakeMCEuropeanGJRGARCHEngine.engine` checks only the
"not given" half. That asymmetry is C++'s, and is reproduced deliberately.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import MCVanillaEngine, RngTraits
from pquantlib.processes.gjr_garch_process import GJRGARCHProcess


class EuropeanGJRGARCHPathPricer(PathPricer[MultiPath]):
    """Terminal-payoff pricer reading the asset leg of a ``MultiPath``.

    # C++ parity: ``EuropeanGJRGARCHPathPricer``
    # (mceuropeangjrgarchengine.hpp:81-91, 211-230).

    Only ``multiPath[0]`` (the asset) is read; the variance leg is ignored.
    """

    __slots__ = ("_discount", "_payoff")

    def __init__(self, option_type: OptionType, strike: float, discount: float) -> None:
        # C++ parity: ``QL_REQUIRE(strike>=0.0, ...)``.
        qassert.require(strike >= 0.0, "strike less than zero not allowed")
        self._payoff: PlainVanillaPayoff = PlainVanillaPayoff(option_type, strike)
        self._discount: float = discount

    def __call__(self, path: MultiPath) -> float:
        # C++ parity: mceuropeangjrgarchengine.hpp:225-233.
        asset_path = path[0]
        qassert.require(path.path_size() > 0, "the path cannot be empty")
        return self._payoff(asset_path.back()) * self._discount


class MCEuropeanGJRGARCHEngine(MCVanillaEngine[MultiPath]):
    """Monte Carlo European engine on a GJR-GARCH process.

    # C++ parity: ``MCEuropeanGJRGARCHEngine<RNG, S>``
    # (mceuropeangjrgarchengine.hpp:39-57, 98-131).

    C++ hard-wires ``brownianBridge = false`` and ``controlVariate = false``.
    """

    def __init__(
        self,
        process: GJRGARCHProcess,
        *,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
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
            brownian_bridge=False,
            antithetic_variate=antithetic_variate,
            control_variate=False,
            required_samples=required_samples,
            required_tolerance=required_tolerance,
            max_samples=max_samples,
            seed=seed,
            rng_traits=rng_traits,
            multi_variate=True,
        )

    def path_pricer(self) -> PathPricer[MultiPath]:
        """# C++ parity: ``MCEuropeanGJRGARCHEngine::pathPricer``
        # (mceuropeangjrgarchengine.hpp:113-131)."""
        payoff = self._arguments.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)

        process = self._process
        qassert.require(isinstance(process, GJRGARCHProcess), "GJRGARCH process required")
        assert isinstance(process, GJRGARCHProcess)

        # C++ discounts at ``this->timeGrid().back()``.
        discount = process.risk_free_rate().discount(self.time_grid().back())
        return EuropeanGJRGARCHPathPricer(
            payoff.option_type(), payoff.strike(), discount
        )


class MakeMCEuropeanGJRGARCHEngine:
    """Fluent builder for :class:`MCEuropeanGJRGARCHEngine`.

    # C++ parity: ``MakeMCEuropeanGJRGARCHEngine<RNG, S>``
    # (mceuropeangjrgarchengine.hpp:60-81, 135-211).
    """

    __slots__ = (
        "_antithetic",
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
        self, process: GJRGARCHProcess, rng_traits: RngTraits = PseudoRandom
    ) -> None:
        self._process: GJRGARCHProcess = process
        self._rng_traits: RngTraits = rng_traits
        self._antithetic: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    def with_steps(self, steps: int) -> MakeMCEuropeanGJRGARCHEngine:
        """# C++ parity: ``withSteps`` (mceuropeangjrgarchengine.hpp:145-152).

        Unlike ``MakeMCEuropeanEngine::withSteps``, this one guards *here*.
        """
        qassert.require(
            self._steps_per_year is None, "number of steps per year already set"
        )
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCEuropeanGJRGARCHEngine:
        """# C++ parity: ``withStepsPerYear`` (mceuropeangjrgarchengine.hpp:154-161)."""
        qassert.require(self._steps is None, "number of steps already set")
        self._steps_per_year = steps
        return self

    def with_samples(self, samples: int) -> MakeMCEuropeanGJRGARCHEngine:
        """# C++ parity: ``withSamples`` (mceuropeangjrgarchengine.hpp:163-170)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCEuropeanGJRGARCHEngine:
        """# C++ parity: ``withAbsoluteTolerance``
        # (mceuropeangjrgarchengine.hpp:172-182)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCEuropeanGJRGARCHEngine:
        """# C++ parity: ``withMaxSamples`` (mceuropeangjrgarchengine.hpp:184-189)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCEuropeanGJRGARCHEngine:
        """# C++ parity: ``withSeed`` (mceuropeangjrgarchengine.hpp:191-196)."""
        self._seed = seed
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCEuropeanGJRGARCHEngine:
        """# C++ parity: ``withAntitheticVariate``
        # (mceuropeangjrgarchengine.hpp:198-203)."""
        self._antithetic = b
        return self

    def engine(self) -> MCEuropeanGJRGARCHEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (mceuropeangjrgarchengine.hpp:205-211).

        Only the "not given" guard lives here; the over-specification guard
        already fired inside the setters.
        """
        qassert.require(
            (self._steps is not None) or (self._steps_per_year is not None),
            "number of steps not given",
        )
        return MCEuropeanGJRGARCHEngine(
            self._process,
            time_steps=self._steps,
            time_steps_per_year=self._steps_per_year,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            rng_traits=self._rng_traits,
        )


__all__ = [
    "EuropeanGJRGARCHPathPricer",
    "MCEuropeanGJRGARCHEngine",
    "MakeMCEuropeanGJRGARCHEngine",
]
