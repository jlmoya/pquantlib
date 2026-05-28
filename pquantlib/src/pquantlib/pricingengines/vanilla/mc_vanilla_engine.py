"""MCVanillaEngine — abstract MC pricing engine for vanilla options.

# C++ parity: ql/pricingengines/vanilla/mcvanillaengine.hpp (v1.42.1) —
# ``template <template <class> class MC, class RNG, class S, class Inst>
#  class MCVanillaEngine``.

C++ folds the inheritance chain ``MCVanillaEngine : Inst::engine,
McSimulation<MC, RNG, S>`` so a single class supplies both the
pricing-engine interface (``calculate()`` filling ``results_.value``
+ optional ``errorEstimate``) and the MC orchestrator hooks.

The Python port keeps the same role-split but uses multiple
inheritance: ``MCVanillaEngine`` extends
:class:`pquantlib.pricingengines.generic_engine.GenericEngine`
(for the engine arguments/results pair) and
:class:`pquantlib.pricingengines.mc_simulation.McSimulation` (for
the MC machinery).

Concrete engines (e.g. ``MCEuropeanEngine``) supply
``path_pricer()`` and (typically) inherit the default
``path_generator()`` defined here, which wires a fresh
PseudoRandom GSG → PathGenerator on every ``calculate()``.

The Python port aligns its constructor with the simplified
``MakeMCEuropeanEngine`` API rather than the verbose 10-arg C++
constructor — exposing only the parameters mainstream callers use
in practice: ``process``, ``time_steps`` (xor ``time_steps_per_year``),
``brownian_bridge``, ``antithetic_variate``, ``control_variate``,
``required_samples`` (xor ``required_tolerance``), ``max_samples``,
``seed``.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.monte_carlo_model import (
    PathGeneratorTypeProtocol,
)
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_generator import PathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.option import OptionArguments
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.time.time_grid import TimeGrid


class MCVanillaEngine(
    GenericEngine[OptionArguments, OneAssetOptionResults],
    McSimulation[Path],
):
    """Abstract MC engine for single-asset vanilla options.

    # C++ parity: ``MCVanillaEngine<MC, RNG, S, Inst>``.

    The constructor accepts either ``time_steps`` or
    ``time_steps_per_year`` (exactly one — both ``None`` or both set is
    a configuration error).  Likewise, either ``required_samples`` or
    ``required_tolerance`` must be provided.
    """

    def __init__(
        self,
        process: StochasticProcess1D,
        *,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        control_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        # Initialize the GenericEngine slots (arguments + results).
        # NOTE: pyright struggles to track explicit base-class __init__
        # forwarding through PEP-695 generic bases. The
        # ``# pyright: ignore[reportUnknownMemberType]`` keeps the
        # checker quiet without polluting runtime behavior.
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, OptionArguments(), OneAssetOptionResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            antithetic_variate=antithetic_variate,
            control_variate=control_variate,
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

        self._process: StochasticProcess1D = process
        self._time_steps: int | None = time_steps
        self._time_steps_per_year: int | None = time_steps_per_year
        self._required_samples: int | None = required_samples
        self._max_samples: int | None = max_samples
        self._required_tolerance: float | None = required_tolerance
        self._brownian_bridge: bool = brownian_bridge
        self._seed: int = seed
        process.register_with(self)

    # --- engine entry-point ----------------------------------------------

    def calculate(self) -> None:
        """Run the MC and fill ``self._results``.

        # C++ parity: ``MCVanillaEngine::calculate`` (mcvanillaengine.hpp:40-48).
        """
        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )
        assert self._mc_model is not None
        self._results.value = self._mc_model.sample_accumulator().mean()
        # PseudoRandom allows error estimate (C++ allowsErrorEstimate = 1).
        # SobolRsg-driven LowDiscrepancy would not (allowsErrorEstimate = 0)
        # — when we add a low-discrepancy variant in a future cluster we'll
        # gate this assignment behind an introspection of the rsg factory.
        if self._mc_model.sample_accumulator().samples() > 1:
            self._results.error_estimate = self._mc_model.sample_accumulator().error_estimate()

    # --- McSimulation hooks ----------------------------------------------

    def time_grid(self) -> TimeGrid:
        """Build the engine's TimeGrid from the option's last exercise date.

        # C++ parity: ``MCVanillaEngine::timeGrid`` (mcvanillaengine.hpp:153-164).
        """
        qassert.require(self._arguments.exercise is not None, "no exercise given")
        assert self._arguments.exercise is not None
        last_exercise_date = self._arguments.exercise.last_date()
        t = self._process.time(last_exercise_date)
        if self._time_steps is not None:
            return TimeGrid.regular(t, self._time_steps)
        assert self._time_steps_per_year is not None
        steps = int(self._time_steps_per_year * t)
        return TimeGrid.regular(t, max(steps, 1))

    def path_generator(self) -> PathGeneratorTypeProtocol[Path]:
        """Build a fresh ``PathGenerator`` per ``calculate()``.

        # C++ parity: ``MCVanillaEngine::pathGenerator`` (mcvanillaengine.hpp:72-81).
        """
        # 1-D process -> dimensions = 1; total Gaussian-sequence size is
        # ``factors * (grid.size() - 1)``.  For 1-D it's just (grid.size() - 1).
        dim_factors = self._process.factors()
        grid = self.time_grid()
        total_dim = dim_factors * (len(grid) - 1)
        # If seed=0 was passed, fall back to MT's nonzero-seed contract by
        # using a deterministic offset (the C++ default is also 0 but it
        # routes through SeedGenerator; we don't have that — use 1 instead).
        # Per L1 carve-out: SeedGenerator is deferred; pquantlib uses an
        # explicit nonzero default rather than a clock-based fallback.
        seed = self._seed if self._seed != 0 else 1
        gsg = make_pseudo_random_rsg(total_dim, seed)
        return PathGenerator.with_time_grid(
            self._process, grid, gsg, brownian_bridge=self._brownian_bridge
        )

    @abstractmethod
    def path_pricer(self) -> PathPricer[Path]:
        """Build the path pricer (concrete engine supplies this)."""

    # --- control variate (default no-op) ---------------------------------

    def control_variate_value(self) -> float | None:
        """Default: no CV value.  Subclasses override when they enable CV.

        # C++ parity: ``MCVanillaEngine::controlVariateValue``.
        """
        if not self._control_variate:
            return None
        # CV expects the subclass to override.
        raise LibraryException("control variate value not provided")


__all__ = ["MCVanillaEngine"]
