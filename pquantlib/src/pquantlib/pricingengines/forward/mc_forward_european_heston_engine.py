"""MCForwardEuropeanHestonEngine — MC forward-start European under Heston.

# C++ parity: ql/pricingengines/forward/mcforwardeuropeanhestonengine.{hpp,cpp}
# (v1.43) — ``template <class RNG = PseudoRandom, class S = Statistics,
#  class P = HestonProcess> class MCForwardEuropeanHestonEngine
#      : public MCForwardVanillaEngine<MultiVariate, RNG, S>``,
# ``class ForwardEuropeanHestonPathPricer : public PathPricer<MultiPath>``, and
# ``template <class RNG, class S, class P>
#  class MakeMCForwardEuropeanHestonEngine``.

Same forward-start strike-reset arithmetic as the Black-Scholes sibling, but
``MultiVariate``: paths come from a ``MultiPathGenerator`` over the 2-factor
Heston process, the Gaussian sequence has dimension ``factors * (grid.size()-1)``
and is consumed two at a time per step, and Brownian bridge is unavailable
(``MultiPathGenerator`` fails on it) — which is why the C++ constructor takes no
``brownianBridge`` parameter and hard-wires ``false``.

The path pricer reads ``multiPath[0]`` only: the variance leg is never touched.

Control variate — the real difference from the BS engine
--------------------------------------------------------
This engine *does* expose ``controlVariate``, and the BS one does not. Three
pieces have to line up, and a port that gets any one of them wrong silently
returns the un-controlled number:

``control_pricing_engine()``
    builds ``AnalyticHestonEngine(HestonModel(process))`` — a fresh model wrapped
    around the *same* process object.

``control_variate_value()`` (inherited)
    prices a plain vanilla struck at ``moneyness * process.initial_values()[0]``
    on that engine. Note it is ``initialValues()[0]``, the asset leg of the
    2-vector, and not the variance.

``control_path_pricer()``
    is the same ``ForwardEuropeanHestonPathPricer`` **with reset index 0**, i.e.
    a vanilla struck at ``moneyness * S(0)`` evaluated along the *same* path. The
    C++ comment explains why index 0 works: the first entry of the time grid is
    always 0, so re-using the forward path pricer at index 0 is exactly the
    vanilla. Control quality therefore degrades as the reset time grows —
    documented in QuantLib PR #948 and reproduced here rather than "improved".

``McSimulation`` then books ``pathValue + controlVariateValue -
controlPathValue`` per sample.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.exercise import EuropeanExercise
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.forward.mc_forward_vanilla_engine import (
    MCForwardVanillaEngine,
)
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.vanilla.analytic_heston_engine import AnalyticHestonEngine
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.heston_process import HestonProcess


class ForwardEuropeanHestonPathPricer(PathPricer[MultiPath]):
    """Terminal payoff struck at ``multiPath[0][reset_index] * moneyness``.

    # C++ parity: ``ForwardEuropeanHestonPathPricer``
    # (mcforwardeuropeanhestonengine.hpp:112-125, .cpp:20-42).

    Only the asset leg is read; the variance leg is ignored entirely.
    """

    __slots__ = ("_discount", "_moneyness", "_option_type", "_reset_index")

    def __init__(
        self,
        option_type: OptionType,
        moneyness: float,
        reset_index: int,
        discount: float,
    ) -> None:
        # C++ parity: mcforwardeuropeanhestonengine.cpp:28-29.
        qassert.require(moneyness >= 0.0, "moneyness less than zero not allowed")
        self._option_type: OptionType = option_type
        self._moneyness: float = moneyness
        self._reset_index: int = reset_index
        self._discount: float = discount

    def __call__(self, path: MultiPath) -> float:
        """# C++ parity: ``ForwardEuropeanHestonPathPricer::operator()``
        # (mcforwardeuropeanhestonengine.cpp:32-42)."""
        asset_path = path[0]
        # C++ guards on ``multiPath.pathSize()``, i.e. the number of points on
        # the path, NOT on the number of assets.
        qassert.require(path.path_size() > 0, "the path cannot be empty")

        reset_level = asset_path[self._reset_index]
        strike = reset_level * self._moneyness
        payoff = PlainVanillaPayoff(self._option_type, strike)
        return payoff(asset_path.back()) * self._discount


class MCForwardEuropeanHestonEngine(MCForwardVanillaEngine[MultiPath]):
    """MC engine for a forward-starting European under a Heston-like process.

    # C++ parity: ``MCForwardEuropeanHestonEngine<RNG, S, P>``
    # (mcforwardeuropeanhestonengine.hpp:48-84, 130-217).
    """

    def __init__(
        self,
        process: HestonProcess,
        *,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        control_variate: bool = False,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        super().__init__(
            process,
            time_steps=time_steps,
            time_steps_per_year=time_steps_per_year,
            # C++ hard-wires brownianBridge = false
            # (mcforwardeuropeanhestonengine.hpp:144).
            brownian_bridge=False,
            antithetic_variate=antithetic_variate,
            required_samples=required_samples,
            required_tolerance=required_tolerance,
            max_samples=max_samples,
            seed=seed,
            control_variate=control_variate,
            rng_traits=rng_traits,
            multi_variate=True,
        )

    def _make_path_pricer(self, reset_index: int) -> PathPricer[MultiPath]:
        """Shared body of ``pathPricer`` and ``controlPathPricer``.

        C++ writes the two out separately (hpp:153-184 and 186-217); they differ
        only in the reset index, and every guard is identical.
        """
        grid = self.time_grid()
        args = self._arguments

        payoff = args.payoff
        qassert.require(isinstance(payoff, PlainVanillaPayoff), "non-plain payoff given")
        assert isinstance(payoff, PlainVanillaPayoff)

        qassert.require(
            isinstance(args.exercise, EuropeanExercise), "wrong exercise given"
        )

        process = self._process
        qassert.require(isinstance(process, HestonProcess), "Heston like process required")
        assert isinstance(process, HestonProcess)

        assert args.moneyness is not None
        return ForwardEuropeanHestonPathPricer(
            payoff.option_type(),
            args.moneyness,
            reset_index,
            process.risk_free_rate().discount(grid.back()),
        )

    def path_pricer(self) -> PathPricer[MultiPath]:
        """# C++ parity: ``MCForwardEuropeanHestonEngine::pathPricer``
        # (mcforwardeuropeanhestonengine.hpp:153-184)."""
        args = self._arguments
        assert args.reset_date is not None
        reset_time = self._process.time(args.reset_date)
        return self._make_path_pricer(self.time_grid().closest_index(reset_time))

    def control_path_pricer(self) -> PathPricer[MultiPath] | None:
        """The same pricer with reset index 0 — i.e. a plain vanilla.

        # C++ parity: ``MCForwardEuropeanHestonEngine::controlPathPricer``
        # (mcforwardeuropeanhestonengine.hpp:186-217) — "First entry in TimeGrid
        # is 0, so use the existing path pricer reset at 0".
        """
        return self._make_path_pricer(0)

    def control_pricing_engine(self) -> PricingEngine | None:
        """# C++ parity: ``MCForwardEuropeanHestonEngine::controlPricingEngine``
        # (mcforwardeuropeanhestonengine.hpp:76-83)."""
        process = self._process
        qassert.require(isinstance(process, HestonProcess), "Heston-like process required")
        assert isinstance(process, HestonProcess)
        return AnalyticHestonEngine(HestonModel(process))


class MakeMCForwardEuropeanHestonEngine:
    """Fluent builder for :class:`MCForwardEuropeanHestonEngine`.

    # C++ parity: ``MakeMCForwardEuropeanHestonEngine<RNG, S, P>``
    # (mcforwardeuropeanhestonengine.hpp:89-109, 219-306).

    Unlike ``MakeMCEuropeanHestonEngine`` in the vanilla wave — which rejects
    over-specified steps *early*, inside ``withSteps`` / ``withStepsPerYear`` —
    this builder defers both halves of the steps rule to conversion time, exactly
    like the BS one. That asymmetry between the two Heston builders is real and
    is pinned in ``v143/pe/mcfwdlb`` (``fwdheston_make_shape``).

    There is no ``with_brownian_bridge``: the engine is MultiVariate and C++
    offers no such knob.
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
        self, process: HestonProcess, rng_traits: RngTraits = PseudoRandom
    ) -> None:
        self._process: HestonProcess = process
        self._rng_traits: RngTraits = rng_traits
        self._antithetic: bool = False
        self._control_variate: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._tolerance: float | None = None
        self._seed: int = 0

    def with_steps(self, steps: int) -> MakeMCForwardEuropeanHestonEngine:
        """# C++ parity: ``withSteps`` (hpp:225-230) — no guard here."""
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCForwardEuropeanHestonEngine:
        """# C++ parity: ``withStepsPerYear`` (hpp:232-237) — no guard here."""
        self._steps_per_year = steps
        return self

    def with_samples(self, samples: int) -> MakeMCForwardEuropeanHestonEngine:
        """# C++ parity: ``withSamples`` (hpp:239-246)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(
        self, tolerance: float
    ) -> MakeMCForwardEuropeanHestonEngine:
        """# C++ parity: ``withAbsoluteTolerance`` (hpp:248-259)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCForwardEuropeanHestonEngine:
        """# C++ parity: ``withMaxSamples`` (hpp:261-266)."""
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCForwardEuropeanHestonEngine:
        """# C++ parity: ``withSeed`` (hpp:268-273)."""
        self._seed = seed
        return self

    def with_antithetic_variate(
        self, b: bool = True
    ) -> MakeMCForwardEuropeanHestonEngine:
        """# C++ parity: ``withAntitheticVariate(bool b = true)`` (hpp:275-280)."""
        self._antithetic = b
        return self

    def with_control_variate(self, b: bool = False) -> MakeMCForwardEuropeanHestonEngine:
        """# C++ parity: ``withControlVariate(bool b = false)`` (hpp:282-287).

        The default is ``false``, so calling it with no argument is a no-op —
        the opposite convention from ``withAntitheticVariate``. That is C++'s.
        """
        self._control_variate = b
        return self

    def engine(self) -> MCForwardEuropeanHestonEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (hpp:289-306)."""
        qassert.require(
            (self._steps is not None) or (self._steps_per_year is not None),
            "number of steps not given",
        )
        qassert.require(
            (self._steps is None) or (self._steps_per_year is None),
            "number of steps overspecified - set EITHER steps OR stepsPerYear",
        )
        return MCForwardEuropeanHestonEngine(
            self._process,
            time_steps=self._steps,
            time_steps_per_year=self._steps_per_year,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            control_variate=self._control_variate,
            rng_traits=self._rng_traits,
        )


__all__ = [
    "ForwardEuropeanHestonPathPricer",
    "MCForwardEuropeanHestonEngine",
    "MakeMCForwardEuropeanHestonEngine",
]
