"""MCPathBasketEngine — European path-dependent basket MC engine.

# C++ parity: ql/experimental/mcbasket/mcpathbasketengine.{hpp,cpp}
#             (v1.43).

Monte Carlo engine for a :class:`~pquantlib.experimental.mcbasket.path_multi_asset_option.PathMultiAssetOption`
whose payoff is a :class:`~pquantlib.experimental.mcbasket.path_payoff.PathPayoff`.
Each sampled multi-asset path is sliced at the fixing times, handed to
the payoff to produce per-fixing payments, and dotted with the per-fixing
discount factors. Early exercise is ignored (European); the
Longstaff-Schwartz multi-path variant is a deferred follow-up.

The C++ ``McSimulation<MultiVariate, RNG, S>`` template specialisation is
rendered by multi-inheriting :class:`McSimulation[MultiPath]` and supplying
the ``path_generator`` / ``path_pricer`` / ``time_grid`` hooks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.mcbasket.path_multi_asset_option import (
    PathMultiAssetOptionArguments,
)
from pquantlib.experimental.mcbasket.path_payoff import PathPayoff
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.monte_carlo_model import PathGeneratorTypeProtocol
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.multi_path_generator import MultiPathGenerator
from pquantlib.methods.montecarlo.path_generator import (
    GaussianSequenceGeneratorProtocol,
)
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_rng_traits import RngTraits
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_array import StochasticProcessArray
from pquantlib.termstructures.yield_.implied_term_structure import (
    ImpliedTermStructure,
)
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.time_grid import TimeGrid


class EuropeanPathMultiPathPricer(PathPricer[MultiPath]):
    """Discounts a PathPayoff's per-fixing payments over a MultiPath.

    # C++ parity: ``EuropeanPathMultiPathPricer``.
    """

    __slots__ = ("_discounts", "_forward_term_structures", "_payoff", "_time_positions")

    def __init__(
        self,
        payoff: PathPayoff,
        time_positions: list[int],
        forward_term_structures: Sequence[YieldTermStructure],
        discounts: npt.NDArray[np.float64],
    ) -> None:
        self._payoff: PathPayoff = payoff
        self._time_positions: list[int] = time_positions
        self._forward_term_structures: Sequence[YieldTermStructure] = forward_term_structures
        self._discounts: npt.NDArray[np.float64] = discounts

    def __call__(self, path: MultiPath) -> float:
        n = path.path_size()
        qassert.require(n > 0, "the path cannot be empty")
        number_of_assets = path.asset_number()
        qassert.require(number_of_assets > 0, "there must be some paths")

        number_of_times = len(self._time_positions)
        # path matrix: (assets, times).
        path_matrix = np.empty((number_of_assets, number_of_times), dtype=np.float64)
        for i in range(number_of_times):
            pos = self._time_positions[i]
            for j in range(number_of_assets):
                path_matrix[j, i] = path[j].values[pos]

        values = np.zeros(number_of_times, dtype=np.float64)
        # Early exercise ignored in this engine.
        exercises = np.empty(0, dtype=np.float64)
        states: list[npt.NDArray[np.float64]] = []

        self._payoff.value(path_matrix, self._forward_term_structures, values, exercises, states)

        return float(np.dot(values, self._discounts))


class MCPathBasketEngine(
    GenericEngine[PathMultiAssetOptionArguments, InstrumentResults],
    McSimulation[MultiPath],
):
    """European path-dependent basket Monte Carlo engine.

    # C++ parity: ``MCPathBasketEngine<RNG, S>``.

    Args:
        process: the multi-asset ``StochasticProcessArray`` (each leaf a
            ``GeneralizedBlackScholesProcess``).
        time_steps: total time steps (mutually exclusive with
            ``time_steps_per_year``).
        time_steps_per_year: time steps per year.
        brownian_bridge: must be False (multi-D BB unsupported, as C++).
        antithetic_variate: enable antithetic sampling.
        control_variate: enable control variate (unused here).
        required_samples: target sample count.
        required_tolerance: target absolute tolerance.
        max_samples: cap on samples.
        seed: RNG seed.
        rng_traits: the random-number policy (C++ ``RNG`` template argument).
    """

    def __init__(
        self,
        process: StochasticProcessArray,
        time_steps: int | None,
        time_steps_per_year: int | None,
        brownian_bridge: bool,
        antithetic_variate: bool,
        control_variate: bool,
        required_samples: int | None,
        required_tolerance: float | None,
        max_samples: int | None,
        seed: int,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, PathMultiAssetOptionArguments(), InstrumentResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, antithetic_variate, control_variate
        )
        qassert.require(
            time_steps is not None or time_steps_per_year is not None,
            "no time steps provided",
        )
        qassert.require(
            time_steps is None or time_steps_per_year is None,
            "both time steps and time steps per year were provided",
        )
        qassert.require(time_steps != 0, f"timeSteps must be positive, {time_steps} not allowed")
        qassert.require(
            time_steps_per_year != 0,
            f"timeStepsPerYear must be positive, {time_steps_per_year} not allowed",
        )
        self._process: StochasticProcessArray = process
        self._time_steps: int | None = time_steps
        self._time_steps_per_year: int | None = time_steps_per_year
        self._required_samples: int | None = required_samples
        self._max_samples: int | None = max_samples
        self._required_tolerance: float | None = required_tolerance
        self._brownian_bridge: bool = brownian_bridge
        self._seed: int = seed
        self._rng_traits: RngTraits = rng_traits
        process.register_with(self)

    # --- McSimulation hooks ---------------------------------------------

    def time_grid(self) -> TimeGrid:
        # C++ parity: ``MCPathBasketEngine::timeGrid``.
        fixings = self._arguments.fixing_dates
        fixing_times = [self._process.time(d) for d in fixings]
        if self._time_steps is not None:
            number_of_time_steps = self._time_steps
        else:
            assert self._time_steps_per_year is not None
            number_of_time_steps = int(self._time_steps_per_year * fixing_times[-1])
        return TimeGrid.with_mandatory_and_steps(fixing_times, number_of_time_steps)

    def path_generator(self) -> PathGeneratorTypeProtocol[MultiPath]:
        # C++ parity: ``MCPathBasketEngine::pathGenerator``.
        payoff = self._arguments.path_payoff
        qassert.require(payoff is not None, "non-basket payoff given")
        num_assets = self._process.size()
        grid = self.time_grid()
        total_dim = num_assets * (len(grid) - 1)
        # C++ parity: ``RNG::make_sequence_generator(numAssets * (grid.size() - 1),
        # seed_)`` — seed 0 goes through SeedGenerator (clock-derived), exactly
        # as in C++.
        # ``RngTraits.make_sequence_generator`` is annotated with the
        # ``SequenceSample`` of ``math.randomnumbers.random_number_generator``;
        # ``MultiPathGenerator`` asks for the structurally identical one from
        # ``methods.montecarlo.gaussian_sequence_generator`` (same two fields,
        # ``value: NDArray[float64]`` and ``weight: float``). The cast is
        # nominal only — nothing about the object changes.
        gen = cast(
            "GaussianSequenceGeneratorProtocol",
            self._rng_traits.make_sequence_generator(total_dim, self._seed),
        )
        return MultiPathGenerator(
            self._process, grid, gen, brownian_bridge=self._brownian_bridge
        )

    def path_pricer(self) -> PathPricer[MultiPath]:
        # C++ parity: ``MCPathBasketEngine::pathPricer``.
        payoff = self._arguments.path_payoff
        qassert.require(payoff is not None, "non-basket payoff given")
        assert payoff is not None
        process = self._process.process(0)
        qassert.require(
            isinstance(process, GeneralizedBlackScholesProcess),
            "Black-Scholes process required",
        )
        assert isinstance(process, GeneralizedBlackScholesProcess)

        grid = self.time_grid()
        times = list(grid.mandatory_times)
        number_of_times = len(times)
        fixings = self._arguments.fixing_dates
        qassert.require(len(fixings) == number_of_times, "Invalid dates/times")

        time_positions = [0] * number_of_times
        discount_factors = np.empty(number_of_times, dtype=np.float64)
        forward_term_structures: list[YieldTermStructure] = []
        risk_free_rate = process.risk_free_rate()
        for i in range(number_of_times):
            time_positions[i] = grid.index(times[i])
            discount_factors[i] = risk_free_rate.discount(times[i])
            forward_term_structures.append(ImpliedTermStructure(risk_free_rate, fixings[i]))

        return EuropeanPathMultiPathPricer(
            payoff, time_positions, forward_term_structures, discount_factors
        )

    # --- engine ----------------------------------------------------------

    def calculate(self) -> None:
        # C++ parity: ``MCPathBasketEngine::calculate`` (mcpathbasketengine.hpp:62-70).
        self.run_mc(self._required_tolerance, self._required_samples, self._max_samples)
        if self._mc_model is None:
            raise LibraryException("MC model not initialized")
        self._results.value = self._mc_model.sample_accumulator().mean()
        # C++ parity: ``if constexpr (RNG::allowsErrorEstimate)`` — a
        # low-discrepancy point set is not i.i.d., so no error is published.
        if self._rng_traits.allows_error_estimate:
            self._results.error_estimate = self._mc_model.sample_accumulator().error_estimate()


class MakeMCPathBasketEngine:
    """Fluent factory for :class:`MCPathBasketEngine`.

    # C++ parity: ``MakeMCPathBasketEngine<RNG, S>``
    # (mcpathbasketengine.hpp:222-245, 247-338).

    # C++ parity note: unlike ``MakeMCEverestEngine`` and
    # ``MakeMCAmericanPathEngine``, this builder does *not* guard the step
    # count in its conversion operator. Converting without a step count
    # reaches the engine constructor, which raises "no time steps provided".

    Args:
        process: the basket process.
        rng_traits: the random-number policy (C++ ``RNG`` template argument).
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
        "_steps",
        "_steps_per_year",
        "_tolerance",
    )

    def __init__(
        self, process: StochasticProcessArray, rng_traits: RngTraits = PseudoRandom
    ) -> None:
        # C++ parity: mcpathbasketengine.hpp:247-251.
        self._process: StochasticProcessArray = process
        self._rng_traits: RngTraits = rng_traits
        self._antithetic: bool = False
        self._control_variate: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._brownian_bridge: bool = False
        self._seed: int = 0

    # ---- named parameters --------------------------------------------------

    def with_steps(self, steps: int) -> MakeMCPathBasketEngine:
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCPathBasketEngine:
        self._steps_per_year = steps
        return self

    def with_brownian_bridge(self, brownian_bridge: bool = True) -> MakeMCPathBasketEngine:
        self._brownian_bridge = brownian_bridge
        return self

    def with_samples(self, samples: int) -> MakeMCPathBasketEngine:
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCPathBasketEngine:
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCPathBasketEngine:
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCPathBasketEngine:
        self._seed = seed
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCPathBasketEngine:
        self._antithetic = b
        return self

    def with_control_variate(self, b: bool = True) -> MakeMCPathBasketEngine:
        self._control_variate = b
        return self

    # ---- terminal ----------------------------------------------------------

    def build(self) -> MCPathBasketEngine:
        """Construct the engine.

        # C++ parity: ``operator ext::shared_ptr<PricingEngine>()``
        # (mcpathbasketengine.hpp:323-338).
        """
        return MCPathBasketEngine(
            self._process,
            self._steps,
            self._steps_per_year,
            self._brownian_bridge,
            self._antithetic,
            self._control_variate,
            self._samples,
            self._tolerance,
            self._max_samples,
            self._seed,
            self._rng_traits,
        )

    def __call__(self) -> MCPathBasketEngine:
        """Mirror the C++ conversion operator to ``shared_ptr<PricingEngine>``."""
        return self.build()


__all__ = [
    "EuropeanPathMultiPathPricer",
    "MCPathBasketEngine",
    "MakeMCPathBasketEngine",
]
