"""MCPagodaEngine — Monte Carlo pricing engine for Pagoda options.

# C++ parity: ql/experimental/exoticoptions/mcpagodaengine.{hpp,cpp} (v1.42.1).

The C++ ``PagodaMultiPathPricer`` accumulates over fixings and assets:

    avg = (sum_{i=1..M-1, j=0..N-1} S[j][0] * (S[j][i]/S[j][i-1] - 1)) / N
    payoff = discount * fraction * max(0, min(roof, avg))

NB: the multiplier ``S[j][0]`` is intentional — accumulates a
performance-weighted basket-relative move, multiplied by the initial
price of asset j (so units are "price").

The TimeGrid is one node per fixing date (mirrors C++).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.experimental.exoticoptions.pagoda_option import (
    PagodaOptionArguments,
)
from pquantlib.instruments.multi_asset_option import MultiAssetOptionResults
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


class PagodaMultiPathPricer(PathPricer[MultiPath]):
    """Path pricer applying the Pagoda rule over a MultiPath.

    # C++ parity: ``PagodaMultiPathPricer`` (mcpagodaengine.cpp:24-46).
    """

    __slots__ = ("_discount", "_fraction", "_roof")

    def __init__(self, roof: float, fraction: float, discount: float) -> None:
        self._roof: float = roof
        self._fraction: float = fraction
        self._discount: float = discount

    def __call__(self, multi_path: MultiPath) -> float:
        num_assets = multi_path.asset_number()
        num_steps = multi_path.path_size()

        average_perf = 0.0
        for i in range(1, num_steps):
            for j in range(num_assets):
                p = multi_path[j].values
                average_perf += p[0] * (p[i] / p[i - 1] - 1.0)
        average_perf /= float(num_assets)

        return self._discount * self._fraction * max(0.0, min(self._roof, average_perf))


class MCPagodaEngine(
    GenericEngine[PagodaOptionArguments, MultiAssetOptionResults],
    McSimulation[MultiPath],
):
    """Monte Carlo engine for ``PagodaOption``.

    # C++ parity: ``MCPagodaEngine<RNG, S>``.
    """

    def __init__(
        self,
        processes: StochasticProcessArray,
        *,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, PagodaOptionArguments(), MultiAssetOptionResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            antithetic_variate=antithetic_variate,
            control_variate=False,
        )
        self._processes: StochasticProcessArray = processes
        self._brownian_bridge: bool = brownian_bridge
        self._required_samples: int | None = required_samples
        self._required_tolerance: float | None = required_tolerance
        self._max_samples: int | None = max_samples
        self._seed: int = seed
        processes.register_with(self)

    # --- engine entry-point ----------------------------------------------

    def calculate(self) -> None:
        """Drive MC + fill value/error.

        # C++ parity: ``MCPagodaEngine::calculate`` (mcpagodaengine.hpp:55-63).
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

    # --- McSimulation hooks ----------------------------------------------

    def time_grid(self) -> TimeGrid:
        """Build TimeGrid from fixing dates (one node per fixing).

        # C++ parity: ``MCPagodaEngine::timeGrid`` (mcpagodaengine.hpp:143-157).
        """
        fixing_times: list[float] = []
        prev = float("-inf")
        for fd in self._arguments.fixing_dates:
            t = self._processes.time(fd)
            qassert.require(t >= 0.0, "seasoned options are not handled")
            if fixing_times:
                qassert.require(t > prev, "fixing dates not sorted")
            fixing_times.append(t)
            prev = t
        return TimeGrid.with_mandatory(fixing_times)

    def path_generator(self) -> PathGeneratorTypeProtocol[MultiPath]:
        n_assets = self._processes.size()
        grid = self.time_grid()
        total_dim = n_assets * (len(grid) - 1)
        seed = self._seed if self._seed != 0 else 1
        gsg = make_pseudo_random_rsg(total_dim, seed)
        return MultiPathGenerator(
            self._processes, grid, gsg, brownian_bridge=self._brownian_bridge
        )

    def path_pricer(self) -> PathPricer[MultiPath]:
        p0: StochasticProcess1D = self._processes.process(0)
        qassert.require(
            isinstance(p0, GeneralizedBlackScholesProcess),
            "Black-Scholes process required",
        )
        assert isinstance(p0, GeneralizedBlackScholesProcess)
        assert self._arguments.exercise is not None
        discount = p0.risk_free_rate().discount(self._arguments.exercise.last_date())
        assert self._arguments.roof is not None
        assert self._arguments.fraction is not None
        return PagodaMultiPathPricer(
            roof=self._arguments.roof,
            fraction=self._arguments.fraction,
            discount=discount,
        )


__all__ = ["MCPagodaEngine", "PagodaMultiPathPricer"]
