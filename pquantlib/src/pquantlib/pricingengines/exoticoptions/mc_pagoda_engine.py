"""MCPagodaEngine — Monte Carlo pricing engine for Pagoda options.

# C++ parity: ql/experimental/exoticoptions/mcpagodaengine.{hpp,cpp} (v1.43).

The C++ ``PagodaMultiPathPricer`` accumulates over fixings and assets:

    avg = (sum_{i=1..M-1, j=0..N-1} S[j][0] * (S[j][i]/S[j][i-1] - 1)) / N
    payoff = discount * fraction * max(0, min(roof, avg))

NB: the multiplier ``S[j][0]`` is intentional — accumulates a
performance-weighted basket-relative move, multiplied by the initial
price of asset j (so units are "price").

The TimeGrid is one node per fixing date (mirrors C++).
"""

from __future__ import annotations

from typing import cast

from pquantlib import qassert
from pquantlib.experimental.exoticoptions.pagoda_option import (
    PagodaOptionArguments,
)
from pquantlib.instruments.multi_asset_option import MultiAssetOptionResults
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.monte_carlo_model import (
    PathGeneratorTypeProtocol,
)
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
        rng_traits: RngTraits = PseudoRandom,
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
        self._rng_traits: RngTraits = rng_traits
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
        # C++ parity: ``if constexpr (RNG::allowsErrorEstimate)`` — a
        # low-discrepancy point set is not i.i.d., so no error is published.
        # There is deliberately no ``samples() > 1`` guard: with a single
        # pseudo-random sample C++ raises "sample number <=1, unsufficient"
        # out of GeneralStatistics::variance, and so must this.
        if self._rng_traits.allows_error_estimate:
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
        # C++ parity: ``RNG::make_sequence_generator(numAssets * (grid.size() - 1),
        # seed_)`` — seed 0 goes through SeedGenerator (clock-derived), exactly
        # as in C++.
        # ``RngTraits.make_sequence_generator`` is annotated with the
        # ``SequenceSample`` of ``math.randomnumbers.random_number_generator``;
        # ``MultiPathGenerator`` asks for the structurally identical one from
        # ``methods.montecarlo.gaussian_sequence_generator`` (same two fields,
        # ``value: NDArray[float64]`` and ``weight: float``). The cast is
        # nominal only — nothing about the object changes.
        gsg = cast(
            "GaussianSequenceGeneratorProtocol",
            self._rng_traits.make_sequence_generator(total_dim, self._seed),
        )
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


class MakeMCPagodaEngine:
    """Fluent factory for :class:`MCPagodaEngine`.

    # C++ parity: ``MakeMCPagodaEngine<RNG, S>``
    # (mcpagodaengine.hpp:93-112, 178-244).

    # C++ parity note: like the Himalaya builder and unlike the Everest one,
    # there is no step count to give — the Pagoda grid is the fixing dates.

    Args:
        process: the basket process.
        rng_traits: the random-number policy (C++ ``RNG`` template argument).
    """

    __slots__ = (
        "_antithetic",
        "_brownian_bridge",
        "_max_samples",
        "_process",
        "_rng_traits",
        "_samples",
        "_seed",
        "_tolerance",
    )

    def __init__(
        self, process: StochasticProcessArray, rng_traits: RngTraits = PseudoRandom
    ) -> None:
        # C++ parity: mcpagodaengine.hpp:178-182.
        self._process: StochasticProcessArray = process
        self._rng_traits: RngTraits = rng_traits
        self._brownian_bridge: bool = False
        self._antithetic: bool = False
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    # ---- named parameters --------------------------------------------------

    def with_brownian_bridge(self, brownian_bridge: bool = True) -> MakeMCPagodaEngine:
        self._brownian_bridge = brownian_bridge
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCPagodaEngine:
        self._antithetic = b
        return self

    def with_samples(self, samples: int) -> MakeMCPagodaEngine:
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCPagodaEngine:
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCPagodaEngine:
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCPagodaEngine:
        self._seed = seed
        return self

    # ---- terminal ----------------------------------------------------------

    def build(self) -> MCPagodaEngine:
        """Construct the engine.

        # C++ parity: ``operator ext::shared_ptr<PricingEngine>()``
        # (mcpagodaengine.hpp:233-244).
        """
        return MCPagodaEngine(
            self._process,
            brownian_bridge=self._brownian_bridge,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            rng_traits=self._rng_traits,
        )

    def __call__(self) -> MCPagodaEngine:
        """Mirror the C++ conversion operator to ``shared_ptr<PricingEngine>``."""
        return self.build()


__all__ = ["MCPagodaEngine", "MakeMCPagodaEngine", "PagodaMultiPathPricer"]
