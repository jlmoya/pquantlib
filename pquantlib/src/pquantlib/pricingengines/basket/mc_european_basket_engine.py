"""MCEuropeanBasketEngine — Monte Carlo engine for European basket options.

# C++ parity:
# ql/pricingengines/basket/mceuropeanbasketengine.{hpp,cpp} (v1.43),
# ``QuantLib::MCEuropeanBasketEngine``, ``QuantLib::EuropeanMultiPathPricer``
# and ``QuantLib::MakeMCEuropeanBasketEngine``.

Plain multi-variate MC: evolve a :class:`StochasticProcessArray` to maturity on
a :class:`TimeGrid`, apply the basket payoff to the terminal asset vector,
discount, average.

Details a port gets wrong easily, both pinned by the probe:

* the path pricer reads ``multiPath[j].back()`` — the **terminal** value of
  each asset — and calls ``(*payoff)(finalPrice)``, i.e. the *vector* overload
  of ``BasketPayoff``, which accumulates then applies the base payoff. It does
  not accumulate per time step;
* ``brownianBridge`` is passed straight into ``MultiPathGenerator``, which
  ``QL_FAIL``s with "Brownian bridge not supported" for multi-variate paths. So
  ``with_brownian_bridge(True)`` raises at pricing time rather than doing
  anything.

``MakeMCEuropeanBasketEngine`` reproduces C++'s named-parameter guards:
``with_samples`` and ``with_absolute_tolerance`` are mutually exclusive, as are
``with_steps`` and ``with_steps_per_year``, and at least one of the latter two
must be given.
"""

from __future__ import annotations

from typing import Self

from pquantlib import qassert
from pquantlib.instruments.basket_option import BasketOptionResults, BasketPayoff
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.monte_carlo_model import PathGeneratorTypeProtocol
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.multi_path_generator import MultiPathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.option import OptionArguments
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_array import StochasticProcessArray
from pquantlib.time.time_grid import TimeGrid


class EuropeanMultiPathPricer(PathPricer[MultiPath]):
    """Discounted basket payoff on the terminal asset vector.

    # C++ parity: ``EuropeanMultiPathPricer``
    # (mceuropeanbasketengine.hpp:124-132, mceuropeanbasketengine.cpp:27-45).
    """

    __slots__ = ("_discount", "_payoff")

    def __init__(self, payoff: BasketPayoff, discount: float) -> None:
        self._payoff: BasketPayoff = payoff
        self._discount: float = discount

    def __call__(self, multi_path: MultiPath) -> float:
        # C++ parity: mceuropeanbasketengine.cpp:31-45.
        n = multi_path.path_size()
        qassert.require(n > 0, "the path cannot be empty")
        num_assets = multi_path.asset_number()
        qassert.require(num_assets > 0, "there must be some paths")

        final_price = [float(multi_path[j].values[-1]) for j in range(num_assets)]
        return self._payoff.evaluate(final_price) * self._discount


class MCEuropeanBasketEngine(
    GenericEngine[OptionArguments, BasketOptionResults],
    McSimulation[MultiPath],
):
    """Monte Carlo engine for European basket options.

    # C++ parity: ``MCEuropeanBasketEngine<RNG, S>``
    # (mceuropeanbasketengine.hpp:43-96, 137-201).

    Args:
        processes: the correlated basket drivers.
        time_steps: fixed number of steps to maturity (xor
            ``time_steps_per_year``).
        time_steps_per_year: steps per year; total = ``int(this * residual)``,
            at least 1 (xor ``time_steps``).
        brownian_bridge: forwarded to ``MultiPathGenerator``, which rejects it.
        antithetic_variate / seed: standard MC knobs. ``seed`` must be nonzero
            (PQuantLib's Mersenne twister has no ``SeedGenerator`` fallback).
        required_samples / required_tolerance / max_samples: stopping criteria.
    """

    def __init__(
        self,
        processes: StochasticProcessArray,
        *,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, OptionArguments(), BasketOptionResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, antithetic_variate=antithetic_variate, control_variate=False
        )
        # C++ parity: mceuropeanbasketengine.hpp:152-164.
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
                time_steps > 0, f"timeSteps must be positive, {time_steps} not allowed"
            )
        if time_steps_per_year is not None:
            qassert.require(
                time_steps_per_year > 0,
                f"timeStepsPerYear must be positive, {time_steps_per_year} not allowed",
            )

        self._processes: StochasticProcessArray = processes
        self._time_steps: int | None = time_steps
        self._time_steps_per_year: int | None = time_steps_per_year
        self._brownian_bridge: bool = brownian_bridge
        self._required_samples: int | None = required_samples
        self._required_tolerance: float | None = required_tolerance
        self._max_samples: int | None = max_samples
        self._seed: int = seed
        processes.register_with(self)

    # --- engine entry-point -------------------------------------------------

    def calculate(self) -> None:
        """# C++ parity: ``calculate`` (mceuropeanbasketengine.hpp:63-71)."""
        self._results.reset()
        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )
        assert self._mc_model is not None
        self._results.value = self._mc_model.sample_accumulator().mean()
        if self._mc_model.sample_accumulator().samples() > 1:
            self._results.error_estimate = (
                self._mc_model.sample_accumulator().error_estimate()
            )

    # --- McSimulation hooks -------------------------------------------------

    def time_grid(self) -> TimeGrid:
        """# C++ parity: ``timeGrid`` (mceuropeanbasketengine.hpp:167-180)."""
        exercise = self._arguments.exercise
        qassert.require(exercise is not None, "no exercise given")
        assert exercise is not None
        residual = self._processes.time(exercise.last_date())
        if self._time_steps is not None:
            return TimeGrid.regular(residual, self._time_steps)
        assert self._time_steps_per_year is not None
        steps = int(self._time_steps_per_year * residual)
        return TimeGrid.regular(residual, max(steps, 1))

    def path_generator(self) -> PathGeneratorTypeProtocol[MultiPath]:
        """# C++ parity: ``pathGenerator`` (mceuropeanbasketengine.hpp:76-86)."""
        num_assets = self._processes.size()
        grid = self.time_grid()
        gsg = make_pseudo_random_rsg(num_assets * (len(grid) - 1), self._seed)
        return MultiPathGenerator(
            self._processes, grid, gsg, brownian_bridge=self._brownian_bridge
        )

    def path_pricer(self) -> PathPricer[MultiPath]:
        """# C++ parity: ``pathPricer`` (mceuropeanbasketengine.hpp:182-201)."""
        payoff = self._arguments.payoff
        qassert.require(isinstance(payoff, BasketPayoff), "non-basket payoff given")
        assert isinstance(payoff, BasketPayoff)

        process = self._processes.process(0)
        qassert.require(
            isinstance(process, GeneralizedBlackScholesProcess),
            "Black-Scholes process required",
        )
        assert isinstance(process, GeneralizedBlackScholesProcess)

        exercise = self._arguments.exercise
        assert exercise is not None
        return EuropeanMultiPathPricer(
            payoff, process.risk_free_rate().discount(exercise.last_date())
        )

    def update(self) -> None:
        self.notify_observers()


class MakeMCEuropeanBasketEngine:
    """Named-parameter factory for :class:`MCEuropeanBasketEngine`.

    # C++ parity: ``MakeMCEuropeanBasketEngine<RNG, S>``
    # (mceuropeanbasketengine.hpp:100-121, 204-290).

    Every setter returns ``self``, and :meth:`engine` performs the same
    validation as C++'s ``operator ext::shared_ptr<PricingEngine>()``.
    """

    __slots__ = (
        "_antithetic",
        "_brownian_bridge",
        "_max_samples",
        "_process",
        "_samples",
        "_seed",
        "_steps",
        "_steps_per_year",
        "_tolerance",
    )

    def __init__(self, process: StochasticProcessArray) -> None:
        self._process: StochasticProcessArray = process
        self._brownian_bridge: bool = False
        self._antithetic: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    def with_steps(self, steps: int) -> Self:
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> Self:
        self._steps_per_year = steps
        return self

    def with_brownian_bridge(self, b: bool = True) -> Self:
        self._brownian_bridge = b
        return self

    def with_antithetic_variate(self, b: bool = True) -> Self:
        self._antithetic = b
        return self

    def with_samples(self, samples: int) -> Self:
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> Self:
        qassert.require(self._samples is None, "number of samples already set")
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> Self:
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> Self:
        self._seed = seed
        return self

    def engine(self) -> PricingEngine:
        """Build the engine.

        # C++ parity: ``operator ext::shared_ptr<PricingEngine>()``
        # (mceuropeanbasketengine.hpp:273-290).
        """
        qassert.require(
            self._steps is not None or self._steps_per_year is not None,
            "number of steps not given",
        )
        qassert.require(
            self._steps is None or self._steps_per_year is None,
            "number of steps overspecified",
        )
        return MCEuropeanBasketEngine(
            self._process,
            time_steps=self._steps,
            time_steps_per_year=self._steps_per_year,
            brownian_bridge=self._brownian_bridge,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
        )


__all__ = [
    "EuropeanMultiPathPricer",
    "MCEuropeanBasketEngine",
    "MakeMCEuropeanBasketEngine",
]
