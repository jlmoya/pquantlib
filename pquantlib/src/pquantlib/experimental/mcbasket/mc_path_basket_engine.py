"""MCPathBasketEngine — European path-dependent basket MC engine.

# C++ parity: ql/experimental/mcbasket/mcpathbasketengine.{hpp,cpp}
#             (v1.42.1).

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

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.mcbasket.path_multi_asset_option import (
    PathMultiAssetOptionArguments,
)
from pquantlib.experimental.mcbasket.path_payoff import PathPayoff
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.monte_carlo_model import PathGeneratorTypeProtocol
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.multi_path_generator import MultiPathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.pricingengines.generic_engine import GenericEngine
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
        seed = self._seed if self._seed != 0 else 1
        gen = make_pseudo_random_rsg(total_dim, seed)
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
        # C++ parity: ``MCPathBasketEngine::calculate``.
        self.run_mc(self._required_tolerance, self._required_samples, self._max_samples)
        if self._mc_model is None:
            raise LibraryException("MC model not initialized")
        self._results.value = self._mc_model.sample_accumulator().mean()
        self._results.error_estimate = self._mc_model.sample_accumulator().error_estimate()


__all__ = ["EuropeanPathMultiPathPricer", "MCPathBasketEngine"]
