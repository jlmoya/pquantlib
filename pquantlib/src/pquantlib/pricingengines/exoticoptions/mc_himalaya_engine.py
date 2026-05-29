"""MCHimalayaEngine — Monte Carlo pricing engine for Himalaya options.

# C++ parity: ql/experimental/exoticoptions/mchimalayaengine.{hpp,cpp} (v1.42.1).

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

from pquantlib import qassert
from pquantlib.experimental.exoticoptions.himalaya_option import (
    HimalayaOptionArguments,
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
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.generic_engine import GenericEngine
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
        if self._mc_model.sample_accumulator().samples() > 1:
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
        seed = self._seed if self._seed != 0 else 1
        gsg = make_pseudo_random_rsg(total_dim, seed)
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


__all__ = ["HimalayaMultiPathPricer", "MCHimalayaEngine"]
