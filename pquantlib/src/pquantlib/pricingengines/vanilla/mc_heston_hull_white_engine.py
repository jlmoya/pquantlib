"""MCHestonHullWhiteEngine — Monte Carlo European pricing under Heston + Hull-White.

# C++ parity: ql/pricingengines/vanilla/mchestonhullwhiteengine.{hpp,cpp} (v1.43) —
# ``template <class RNG, class S> class MCHestonHullWhiteEngine``,
# ``class HestonHullWhitePathPricer``, and
# ``template <class RNG, class S> class MakeMCHestonHullWhiteEngine``.

A ``MultiVariate`` engine over
:class:`~pquantlib.processes.hybrid_heston_hull_white_process.HybridHestonHullWhiteProcess`,
so the Gaussian sequence has dimension ``3 * (grid.size() - 1)``.

The discounting is *stochastic*: the path pricer divides the payoff by
``process.numeraire(exerciseTime, terminal_state)`` rather than multiplying by a
deterministic discount factor, so the terminal short rate on each path matters.

Control variate
---------------
C++ wires three pieces:

``controlPathPricer()``
    another :class:`HestonHullWhitePathPricer` on the *same* process
    (mchestonhullwhiteengine.hpp:159-183);

``controlPathGenerator()``
    a generator over a **second** ``HybridHestonHullWhiteProcess`` built with
    ``corrEquityShortRate = 0`` and the same discretization, driven by the
    **same seed** as the pricing generator (mchestonhullwhiteengine.hpp:203-222)
    -- so the control paths are a different process fed the same numbers;

``controlPricingEngine()``
    ``AnalyticHestonHullWhiteEngine(hestonModel, hwModel, 144)``
    (mchestonhullwhiteengine.hpp:187-201).

All three are ported. The control-variate reference value comes from
:class:`~pquantlib.pricingengines.vanilla.analytic_heston_hull_white_engine.AnalyticHestonHullWhiteEngine`
at integration order 144, exactly as C++ builds it, with the ``HestonModel``
taken from the joint process's Heston leg and the ``HullWhite`` model rebuilt
from the Hull-White leg's ``a`` and ``sigma``.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.monte_carlo_model import PathGeneratorTypeProtocol
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.payoffs import Payoff
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.vanilla.analytic_heston_hull_white_engine import (
    AnalyticHestonHullWhiteEngine,
)
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import MCVanillaEngine, RngTraits
from pquantlib.processes.hybrid_heston_hull_white_process import (
    HybridHestonHullWhiteProcess,
)


class HestonHullWhitePathPricer(PathPricer[MultiPath]):
    """Terminal payoff discounted by the stochastic numeraire.

    # C++ parity: ``HestonHullWhitePathPricer``
    # (mchestonhullwhiteengine.hpp:95-107 + mchestonhullwhiteengine.cpp:29-46).
    """

    __slots__ = ("_exercise_time", "_payoff", "_process")

    def __init__(
        self,
        exercise_time: float,
        payoff: Payoff,
        process: HybridHestonHullWhiteProcess,
    ) -> None:
        self._exercise_time: float = exercise_time
        self._payoff: Payoff = payoff
        self._process: HybridHestonHullWhiteProcess = process

    def __call__(self, path: MultiPath) -> float:
        # C++ parity: mchestonhullwhiteengine.cpp:35-46.
        qassert.require(path.path_size() > 0, "the path cannot be empty")
        n = path.path_size()
        states = np.array(
            [path[j][n - 1] for j in range(path.asset_number())], dtype=np.float64
        )
        df = 1.0 / self._process.numeraire(self._exercise_time, states)
        return self._payoff(float(states[0])) * df


class MCHestonHullWhiteEngine(MCVanillaEngine[MultiPath]):
    """Monte Carlo European engine on a hybrid Heston / Hull-White process.

    # C++ parity: ``MCHestonHullWhiteEngine<RNG, S>``
    # (mchestonhullwhiteengine.hpp:36-68, 111-222).
    """

    def __init__(
        self,
        process: HybridHestonHullWhiteProcess,
        *,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        antithetic_variate: bool = False,
        control_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        # C++ hard-wires ``brownianBridge = false``
        # (mchestonhullwhiteengine.hpp:124).
        super().__init__(
            process,
            time_steps=time_steps,
            time_steps_per_year=time_steps_per_year,
            brownian_bridge=False,
            antithetic_variate=antithetic_variate,
            control_variate=control_variate,
            required_samples=required_samples,
            required_tolerance=required_tolerance,
            max_samples=max_samples,
            seed=seed,
            rng_traits=rng_traits,
            multi_variate=True,
        )
        # C++ keeps its own narrowed ``process_`` member "just to avoid
        # upcasting" (mchestonhullwhiteengine.hpp:59-60).
        self._hybrid_process: HybridHestonHullWhiteProcess = process

    def calculate(self) -> None:
        """# C++ parity: ``MCHestonHullWhiteEngine::calculate``
        # (mchestonhullwhiteengine.hpp:130-138) — the control variate can push
        # a deep-OTM value slightly negative, so C++ clamps at zero, but only
        # when the control variate is on."""
        super().calculate()
        if self._control_variate:
            self._results.value = max(0.0, self._results.value or 0.0)

    def path_pricer(self) -> PathPricer[MultiPath]:
        """# C++ parity: ``pathPricer`` (mchestonhullwhiteengine.hpp:141-156)."""
        exercise = self._arguments.exercise
        qassert.require(exercise is not None, "no exercise given")
        assert exercise is not None
        qassert.require(
            exercise.type() == Exercise.Type.European,
            "only european exercise is supported",
        )
        payoff = self._arguments.payoff
        qassert.require(payoff is not None, "no payoff given")
        assert payoff is not None
        exercise_time = self._hybrid_process.time(exercise.last_date())
        return HestonHullWhitePathPricer(exercise_time, payoff, self._hybrid_process)

    # --- control variate --------------------------------------------------

    def control_path_pricer(self) -> PathPricer[MultiPath] | None:
        """# C++ parity: ``controlPathPricer``
        # (mchestonhullwhiteengine.hpp:159-183) — the same pricer shape, on the
        # same process; only the *generator* differs."""
        # C++ guards ``QL_REQUIRE(hestonProcess, "first constituent of the
        # joint stochastic process need to be of type HestonProcess")``
        # (mchestonhullwhiteengine.hpp:166-167). That is a null-check on a
        # ``dynamic_pointer_cast`` result; the Python process stores a typed
        # ``HestonProcess``, so the condition is unreachable rather than merely
        # unlikely and asserting it would be dead code.
        exercise = self._arguments.exercise
        qassert.require(exercise is not None, "no exercise given")
        assert exercise is not None
        qassert.require(
            exercise.type() == Exercise.Type.European,
            "only european exercise is supported",
        )
        payoff = self._arguments.payoff
        qassert.require(payoff is not None, "no payoff given")
        assert payoff is not None
        exercise_time = self._hybrid_process.time(exercise.last_date())
        return HestonHullWhitePathPricer(exercise_time, payoff, self._hybrid_process)

    def control_path_generator(self) -> PathGeneratorTypeProtocol[MultiPath] | None:
        """A generator over the zero-correlation twin of the pricing process.

        # C++ parity: ``controlPathGenerator``
        # (mchestonhullwhiteengine.hpp:203-222). Note it uses ``this->seed_``,
        # i.e. the *same* stream as the pricing generator, so the control paths
        # are the same numbers pushed through a decorrelated process.
        """
        grid = self.time_grid()
        cv_process = HybridHestonHullWhiteProcess(
            self._hybrid_process.heston_process(),
            self._hybrid_process.hull_white_process(),
            0.0,
            self._hybrid_process.discretization(),
        )
        dimensions = cv_process.factors() * (len(grid) - 1)
        generator = self._rng_traits.make_sequence_generator(dimensions, self._seed)
        from pquantlib.methods.montecarlo.multi_path_generator import (  # noqa: PLC0415
            MultiPathGenerator,
        )

        return MultiPathGenerator(cv_process, grid, generator, False)  # type: ignore[return-value]

    def control_pricing_engine(self) -> PricingEngine | None:
        """``AnalyticHestonHullWhiteEngine(hestonModel, hwModel, 144)``.

        # C++ parity: ``controlPricingEngine``
        # (mchestonhullwhiteengine.hpp:187-201). C++ builds the ``HullWhite``
        # model from the *process* parameters rather than reusing the one the
        # joint process holds, and pins the integration order at 144.
        """
        heston_process = self._hybrid_process.heston_process()
        hull_white_process = self._hybrid_process.hull_white_process()
        return AnalyticHestonHullWhiteEngine(
            HestonModel(heston_process),
            HullWhite(
                heston_process.risk_free_rate(),
                hull_white_process.a(),
                hull_white_process.sigma(),
            ),
            144,
        )


class MakeMCHestonHullWhiteEngine:
    """Fluent builder for :class:`MCHestonHullWhiteEngine`.

    # C++ parity: ``MakeMCHestonHullWhiteEngine<RNG, S>``
    # (mchestonhullwhiteengine.hpp:71-92, 226-314).
    """

    __slots__ = (
        "_antithetic",
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
        self,
        process: HybridHestonHullWhiteProcess,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        self._process: HybridHestonHullWhiteProcess = process
        self._rng_traits: RngTraits = rng_traits
        self._antithetic: bool = False
        self._control_variate: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    def with_steps(self, steps: int) -> MakeMCHestonHullWhiteEngine:
        """# C++ parity: ``withSteps`` (mchestonhullwhiteengine.hpp:232-236)."""
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCHestonHullWhiteEngine:
        """# C++ parity: ``withStepsPerYear`` (mchestonhullwhiteengine.hpp:238-242)."""
        self._steps_per_year = steps
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCHestonHullWhiteEngine:
        """# C++ parity: ``withAntitheticVariate``
        # (mchestonhullwhiteengine.hpp:244-248)."""
        self._antithetic = b
        return self

    def with_control_variate(self, b: bool = True) -> MakeMCHestonHullWhiteEngine:
        """# C++ parity: ``withControlVariate``
        # (mchestonhullwhiteengine.hpp:250-254)."""
        self._control_variate = b
        return self

    def with_samples(self, samples: int) -> MakeMCHestonHullWhiteEngine:
        """# C++ parity: ``withSamples`` (mchestonhullwhiteengine.hpp:256-262)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCHestonHullWhiteEngine:
        """# C++ parity: ``withAbsoluteTolerance``
        # (mchestonhullwhiteengine.hpp:264-273)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCHestonHullWhiteEngine:
        """# C++ parity: ``withMaxSamples`` (mchestonhullwhiteengine.hpp:275-279)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCHestonHullWhiteEngine:
        """# C++ parity: ``withSeed`` (mchestonhullwhiteengine.hpp:281-285)."""
        self._seed = seed
        return self

    def engine(self) -> MCHestonHullWhiteEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (mchestonhullwhiteengine.hpp:287-314)."""
        qassert.require(
            (self._steps is not None) or (self._steps_per_year is not None),
            "number of steps not given",
        )
        qassert.require(
            (self._steps is None) or (self._steps_per_year is None),
            "number of steps overspecified",
        )
        return MCHestonHullWhiteEngine(
            self._process,
            time_steps=self._steps,
            time_steps_per_year=self._steps_per_year,
            antithetic_variate=self._antithetic,
            control_variate=self._control_variate,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            rng_traits=self._rng_traits,
        )


__all__ = [
    "HestonHullWhitePathPricer",
    "MCHestonHullWhiteEngine",
    "MakeMCHestonHullWhiteEngine",
]
