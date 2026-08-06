"""MCHimalayaEngine — Monte Carlo pricing engine for Himalaya options.

# C++ parity: ql/experimental/exoticoptions/mchimalayaengine.{hpp,cpp} (v1.43).

Drives a multi-asset MC simulation with the basket processes packed
into a :class:`pquantlib.processes.stochastic_process_array.StochasticProcessArray`,
and a :class:`HimalayaMultiPathPricer` that consumes the resulting
MultiPath:

    On each fixing date i (1 <= i < N):
      pick the asset with the highest yield S[j][i] / S[j][0]
      among the still-active assets, add S[j][i] to the running sum,
      and remove that asset from the active set.

The terminal value is then ``payoff(avg_price) * discount`` with
``avg_price = sum / min(N_fixings, N_assets)``.

The C++ engine accepts a TimeGrid built from the option's fixing
dates (one node per fixing).  The Python port mirrors that:
``time_grid()`` returns ``TimeGrid(fixing_times)``.
"""

from __future__ import annotations

from typing import cast

from pquantlib import qassert
from pquantlib.experimental.exoticoptions.himalaya_option import (
    HimalayaOptionArguments,
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
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_rng_traits import RngTraits
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.processes.stochastic_process_array import StochasticProcessArray
from pquantlib.time.time_grid import TimeGrid


class HimalayaMultiPathPricer(PathPricer[MultiPath]):
    """Path pricer applying the Himalaya rule over a MultiPath.

    # C++ parity: ``HimalayaMultiPathPricer`` (mchimalayaengine.cpp:26-62).
    """

    __slots__ = ("_discount", "_payoff")

    def __init__(self, payoff: Payoff, discount: float) -> None:
        self._payoff: Payoff = payoff
        self._discount: float = discount

    def __call__(self, multi_path: MultiPath) -> float:
        n_assets = multi_path.asset_number()
        n_nodes = multi_path.path_size()
        qassert.require(n_assets > 0, "no asset given")

        remaining = [True] * n_assets
        average_price = 0.0
        fixings = n_nodes - 1
        # C++ uses QL_MIN_REAL = -DBL_MAX; mirror with -inf.
        for i in range(1, n_nodes):
            best_price = 0.0
            best_yield = float("-inf")
            remove_asset = 0
            for j in range(n_assets):
                if remaining[j]:
                    price = multi_path[j].values[i]
                    initial = multi_path[j].values[0]
                    yield_ = price / initial
                    if yield_ >= best_yield:
                        best_price = price
                        best_yield = yield_
                        remove_asset = j
            remaining[remove_asset] = False
            average_price += best_price
        # min(fixings, n_assets) — see C++.
        average_price /= float(min(fixings, n_assets))

        payoff_val = self._payoff(average_price)
        return payoff_val * self._discount


class MCHimalayaEngine(
    GenericEngine[HimalayaOptionArguments, MultiAssetOptionResults],
    McSimulation[MultiPath],
):
    """Monte Carlo engine for ``HimalayaOption``.

    # C++ parity: ``MCHimalayaEngine<RNG, S>``.

    Args:
        processes: ``StochasticProcessArray`` driving the basket.
        brownian_bridge: Use a Brownian bridge for the path generator
            (not supported in C++ multi-asset MC — will raise).
        antithetic_variate: Enable antithetic variance reduction.
        required_samples: Stop after exactly this many samples (xor
            ``required_tolerance``).
        required_tolerance: Stop when the 1-sigma error is below this
            (xor ``required_samples``).
        max_samples: Upper bound on total samples (only used when
            ``required_tolerance`` is set).
        seed: PRNG seed (zero falls back to a deterministic offset for
            MT compatibility — see ``MCVanillaEngine``).
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
            self, HimalayaOptionArguments(), MultiAssetOptionResults()
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
        """Drive ``run_mc`` and write ``results.value`` + error.

        # C++ parity: ``MCHimalayaEngine::calculate`` (mchimalayaengine.hpp:54-63).
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
        """Build TimeGrid from fixing dates.

        # C++ parity: ``MCHimalayaEngine::timeGrid`` (mchimalayaengine.hpp:142-156).
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
        """Fresh ``MultiPathGenerator`` per ``calculate``.

        # C++ parity: ``MCHimalayaEngine::pathGenerator`` (inline).
        """
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
        """Construct the Himalaya path pricer.

        # C++ parity: ``MCHimalayaEngine::pathPricer`` (inline).
        """
        # C++ casts processes[0] to GeneralizedBlackScholesProcess and
        # uses its riskFreeRate to compute the discount at the last
        # exercise date. We mirror that constraint.
        p0: StochasticProcess1D = self._processes.process(0)
        qassert.require(
            isinstance(p0, GeneralizedBlackScholesProcess),
            "Black-Scholes process required",
        )
        assert isinstance(p0, GeneralizedBlackScholesProcess)
        last_date = self._arguments.exercise
        assert last_date is not None
        discount = p0.risk_free_rate().discount(last_date.last_date())
        assert self._arguments.payoff is not None
        return HimalayaMultiPathPricer(self._arguments.payoff, discount)


class MakeMCHimalayaEngine:
    """Fluent factory for :class:`MCHimalayaEngine`.

    # C++ parity: ``MakeMCHimalayaEngine<RNG, S>``
    # (mchimalayaengine.hpp:93-112, 176-243).

    # C++ parity note: this builder has no ``with_steps``/``with_steps_per_year``
    # — the Himalaya grid is exactly the option's fixing dates — so converting
    # with nothing but a sample count is legal, unlike the Everest builder.

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
        # C++ parity: mchimalayaengine.hpp:176-180.
        self._process: StochasticProcessArray = process
        self._rng_traits: RngTraits = rng_traits
        self._brownian_bridge: bool = False
        self._antithetic: bool = False
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    # ---- named parameters --------------------------------------------------

    def with_brownian_bridge(self, brownian_bridge: bool = True) -> MakeMCHimalayaEngine:
        self._brownian_bridge = brownian_bridge
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCHimalayaEngine:
        self._antithetic = b
        return self

    def with_samples(self, samples: int) -> MakeMCHimalayaEngine:
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCHimalayaEngine:
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCHimalayaEngine:
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCHimalayaEngine:
        self._seed = seed
        return self

    # ---- terminal ----------------------------------------------------------

    def build(self) -> MCHimalayaEngine:
        """Construct the engine.

        # C++ parity: ``operator ext::shared_ptr<PricingEngine>()``
        # (mchimalayaengine.hpp:231-243).
        """
        return MCHimalayaEngine(
            self._process,
            brownian_bridge=self._brownian_bridge,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            rng_traits=self._rng_traits,
        )

    def __call__(self) -> MCHimalayaEngine:
        """Mirror the C++ conversion operator to ``shared_ptr<PricingEngine>``."""
        return self.build()


__all__ = ["HimalayaMultiPathPricer", "MCHimalayaEngine", "MakeMCHimalayaEngine"]
