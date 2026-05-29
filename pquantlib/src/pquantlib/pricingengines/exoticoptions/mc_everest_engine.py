"""MCEverestEngine — Monte Carlo pricing engine for Everest options.

# C++ parity: ql/experimental/exoticoptions/mceverestengine.{hpp,cpp} (v1.42.1).

The pricer accumulates a per-path

    (1 + min_yield + guarantee) * notional * end_discount

where ``min_yield = min_i S_i(T) / S_i(0) - 1`` is taken over all
basket assets at the terminal node.

The engine drives a multi-asset MC over a TimeGrid built either
from a fixed ``time_steps`` count or from ``time_steps_per_year *
residual_time`` (rounded down to at least 1 step) — matching C++.

After the MC, ``results.yield_`` is also filled:
``value / (notional * end_discount) - 1``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.experimental.exoticoptions.everest_option import (
    EverestOptionArguments,
    EverestOptionResults,
)
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.monte_carlo_model import (
    PathGeneratorTypeProtocol,
)
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.multi_path_generator import MultiPathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.processes.stochastic_process_array import StochasticProcessArray
from pquantlib.time.time_grid import TimeGrid


class EverestMultiPathPricer(PathPricer[MultiPath]):
    """Path pricer applying the Everest rule over a MultiPath.

    # C++ parity: ``EverestMultiPathPricer`` (mceverestengine.cpp:24-44).
    """

    __slots__ = ("_discount", "_guarantee", "_notional")

    def __init__(self, notional: float, guarantee: float, discount: float) -> None:
        self._notional: float = notional
        self._guarantee: float = guarantee
        self._discount: float = discount

    def __call__(self, multi_path: MultiPath) -> float:
        n = multi_path.path_size()
        qassert.require(n > 0, "the path cannot be empty")
        num_assets = multi_path.asset_number()
        qassert.require(num_assets > 0, "there must be some paths")

        # Search the min yield over assets.
        # C++ ``yield = back / front - 1.0``.
        first = multi_path[0]
        min_yield = first.values[-1] / first.values[0] - 1.0
        for j in range(1, num_assets):
            pj = multi_path[j]
            yield_ = pj.values[-1] / pj.values[0] - 1.0
            min_yield = min(min_yield, yield_)
        return (1.0 + min_yield + self._guarantee) * self._notional * self._discount


class MCEverestEngine(
    GenericEngine[EverestOptionArguments, EverestOptionResults],
    McSimulation[MultiPath],
):
    """Monte Carlo engine for ``EverestOption``.

    # C++ parity: ``MCEverestEngine<RNG, S>``.

    Args:
        processes: ``StochasticProcessArray`` driving the basket.
        time_steps: Fixed number of steps to maturity (xor
            ``time_steps_per_year``).
        time_steps_per_year: Yearly steps; total = round(year_frac *
            this), at least 1 (xor ``time_steps``).
        brownian_bridge / antithetic_variate / seed: standard MC knobs.
        required_samples / required_tolerance / max_samples: standard
            stopping criteria.
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
            self, EverestOptionArguments(), EverestOptionResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            antithetic_variate=antithetic_variate,
            control_variate=False,
        )
        qassert.require(
            (time_steps is not None) or (time_steps_per_year is not None),
            "no time steps provided",
        )
        qassert.require(
            (time_steps is None) or (time_steps_per_year is None),
            "both time steps and time steps per year were provided",
        )
        if time_steps is not None:
            qassert.require(time_steps > 0, f"timeSteps must be positive, {time_steps} not allowed")
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

    # --- helpers ---------------------------------------------------------

    def _end_discount(self) -> float:
        """Discount factor at the last exercise date.

        # C++ parity: ``MCEverestEngine::endDiscount``.
        """
        p0: StochasticProcess1D = self._processes.process(0)
        qassert.require(
            isinstance(p0, GeneralizedBlackScholesProcess),
            "Black-Scholes process required",
        )
        assert isinstance(p0, GeneralizedBlackScholesProcess)
        assert self._arguments.exercise is not None
        return p0.risk_free_rate().discount(self._arguments.exercise.last_date())

    # --- engine entry-point ----------------------------------------------

    def calculate(self) -> None:
        """Drive MC + fill value/error/yield.

        # C++ parity: ``MCEverestEngine::calculate`` (mceverestengine.hpp:55-70).
        """
        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )
        assert self._mc_model is not None
        self._results.value = self._mc_model.sample_accumulator().mean()
        if self._mc_model.sample_accumulator().samples() > 1:
            self._results.error_estimate = self._mc_model.sample_accumulator().error_estimate()
        # Yield = value / (notional * end_discount) - 1
        notional = self._arguments.notional
        assert notional is not None
        discount = self._end_discount()
        self._results.yield_ = self._results.value / (notional * discount) - 1.0

    # --- McSimulation hooks ----------------------------------------------

    def time_grid(self) -> TimeGrid:
        """Build TimeGrid: fixed steps to last exercise date.

        # C++ parity: ``MCEverestEngine::timeGrid`` (mceverestengine.hpp:172-184).
        """
        assert self._arguments.exercise is not None
        residual = self._processes.time(self._arguments.exercise.last_date())
        if self._time_steps is not None:
            return TimeGrid.regular(residual, self._time_steps)
        assert self._time_steps_per_year is not None
        steps = int(self._time_steps_per_year * residual)
        return TimeGrid.regular(residual, max(steps, 1))

    def path_generator(self) -> PathGeneratorTypeProtocol[MultiPath]:
        """Fresh ``MultiPathGenerator`` per ``calculate``."""
        n_assets = self._processes.size()
        grid = self.time_grid()
        total_dim = n_assets * (len(grid) - 1)
        seed = self._seed if self._seed != 0 else 1
        gsg = make_pseudo_random_rsg(total_dim, seed)
        return MultiPathGenerator(
            self._processes, grid, gsg, brownian_bridge=self._brownian_bridge
        )

    def path_pricer(self) -> PathPricer[MultiPath]:
        """Construct the Everest path pricer."""
        assert self._arguments.notional is not None
        assert self._arguments.guarantee is not None
        return EverestMultiPathPricer(
            notional=self._arguments.notional,
            guarantee=self._arguments.guarantee,
            discount=self._end_discount(),
        )


__all__ = ["EverestMultiPathPricer", "MCEverestEngine"]
