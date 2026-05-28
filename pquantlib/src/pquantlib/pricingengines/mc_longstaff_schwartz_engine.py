"""MCLongstaffSchwartzEngine — Monte Carlo engine for early-exercise options.

# C++ parity: ql/pricingengines/mclongstaffschwartzengine.hpp (v1.42.1).

C++ template chain:

    template <class GenericEngine, template <class> class MC,
              class RNG, class S = Statistics, class RNG_Calibration = RNG>
    class MCLongstaffSchwartzEngine
        : public GenericEngine, public McSimulation<MC, RNG, S>

— combines the engine interface (``calculate()`` filling
``results_.value``) with the MC orchestrator (``McSimulation``) plus a
*second* MC pass for the regression-coefficient calibration of the
``LongstaffSchwartzPathPricer``.

The Python port uses multiple inheritance ``MCVanillaEngine + ...``
through the concrete subclass — we don't redo the full ``GenericEngine
+ McSimulation`` glue, just specialise ``MCVanillaEngine`` enough to
plug in the LSM machinery.

Algorithm in ``calculate()``:

1. Build (and cache) the ``LongstaffSchwartzPathPricer`` via the
   subclass hook ``lsm_path_pricer()``.
2. Build a calibration ``PathGenerator`` with the calibration seed
   (``seed`` plus a deterministic offset to avoid path collisions).
3. Wrap the LSM pricer in a calibration ``MonteCarloModel`` and run
   ``calibration_samples`` draws — these get recorded into the pricer
   for the backward-induction regression.
4. Call ``self._cached_lsm_pricer.calibrate()`` — the pricer now
   transitions from "record" mode to "evaluate" mode.
5. Run the regular ``McSimulation.run_mc`` driver (which builds a
   fresh pricing ``PathGenerator``; reuses the *same* pricer
   instance via ``path_pricer()`` returning the cached one).
6. Fill ``results_.value`` and ``error_estimate``.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.longstaff_schwartz_path_pricer import (
    LongstaffSchwartzPathPricer,
)
from pquantlib.methods.montecarlo.monte_carlo_model import (
    MonteCarloModel,
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

# C++ ``MCLongstaffSchwartzEngine`` uses this constant as the offset for
# the calibration seed when the user passes a nonzero pricing seed (see
# mclongstaffschwartzengine.hpp:148-149):
#     seedCalibration = seed + 1768237423L
_CALIBRATION_SEED_OFFSET = 1768237423


class MCLongstaffSchwartzEngine(
    GenericEngine[OptionArguments, OneAssetOptionResults],
    McSimulation[Path],
):
    """Abstract MC engine for early-exercise options via Longstaff-Schwartz.

    # C++ parity: ``MCLongstaffSchwartzEngine<GenericEngine, MC, RNG, S, RNG_Calibration>``
    # (mclongstaffschwartzengine.hpp).

    Concrete subclasses (e.g. ``MCAmericanEngine``) supply
    :meth:`lsm_path_pricer` returning a ready-to-calibrate
    :class:`LongstaffSchwartzPathPricer`.
    """

    # Suppress pyright's unknown-base warning on the explicit Generic
    # base init: PEP-695 generic class hierarchies aren't yet fully
    # tracked.

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
        calibration_samples: int = 2048,
        brownian_bridge_calibration: bool | None = None,
        antithetic_variate_calibration: bool | None = None,
        seed_calibration: int | None = None,
    ) -> None:
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
        self._required_tolerance: float | None = required_tolerance
        self._max_samples: int | None = max_samples
        self._brownian_bridge: bool = brownian_bridge
        self._seed: int = seed
        self._calibration_samples: int = calibration_samples
        # C++ defaults: brownianBridgeCalibration = brownianBridge,
        #               antitheticVariateCalibration = antitheticVariate
        # if not given. seedCalibration = seed + offset (if seed != 0
        # else 0). Mirror those defaults.
        self._brownian_bridge_calibration: bool = (
            brownian_bridge_calibration
            if brownian_bridge_calibration is not None
            else brownian_bridge
        )
        self._antithetic_variate_calibration: bool = (
            antithetic_variate_calibration
            if antithetic_variate_calibration is not None
            else antithetic_variate
        )
        self._seed_calibration: int = (
            seed_calibration
            if seed_calibration is not None
            else (seed + _CALIBRATION_SEED_OFFSET if seed != 0 else 0)
        )

        # LSM pricer cache: built in ``calculate()``, consumed by ``path_pricer``.
        self._cached_lsm_pricer: LongstaffSchwartzPathPricer[Path, float] | None = None

        process.register_with(self)

    # --- engine entry-point -----------------------------------------------

    def calculate(self) -> None:
        """Run calibration + pricing MC; fill ``self._results``.

        # C++ parity: ``MCLongstaffSchwartzEngine::calculate``
        # (mclongstaffschwartzengine.hpp:178-210).
        """
        # 1) Build pricer (LSM); cache for the pricing-phase path_pricer() hook.
        self._cached_lsm_pricer = self.lsm_path_pricer()

        # 2) Drive the calibration MC.
        grid = self.time_grid()
        cal_pg = self._build_path_generator(self._seed_calibration, grid)
        cal_model = MonteCarloModel[Path](
            path_generator=cal_pg,
            path_pricer=self._cached_lsm_pricer,
            sample_accumulator=GeneralStatistics(),
            antithetic_variate=self._antithetic_variate_calibration,
        )
        cal_model.add_samples(self._calibration_samples)

        # 3) Train the regression.
        self._cached_lsm_pricer.calibrate()

        # 4) Drive the pricing MC via McSimulation.run_mc — which calls
        #    self.path_pricer() (returns the cached pricer) and
        #    self.path_generator() (builds a fresh GSG with the pricing seed).
        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )

        # 5) Fill results.
        assert self._mc_model is not None
        self._results.value = self._mc_model.sample_accumulator().mean()
        if self._mc_model.sample_accumulator().samples() > 1:
            self._results.error_estimate = (
                self._mc_model.sample_accumulator().error_estimate()
            )

    # --- McSimulation hooks ------------------------------------------------

    def time_grid(self) -> TimeGrid:
        """Build the engine's TimeGrid covering all exercise dates.

        # C++ parity: ``MCLongstaffSchwartzEngine::timeGrid``
        # (mclongstaffschwartzengine.hpp:214-240).
        """
        qassert.require(self._arguments.exercise is not None, "no exercise given")
        assert self._arguments.exercise is not None
        exercise = self._arguments.exercise

        required_times: list[float] = []
        if exercise.type().name == "American":
            last_t = self._process.time(exercise.last_date())
            required_times.append(last_t)
        else:
            for i in range(len(exercise.dates())):
                t = self._process.time(exercise.date(i))
                if t > 0.0:
                    required_times.append(t)

        if self._time_steps is not None:
            return TimeGrid.with_mandatory_and_steps(required_times, self._time_steps)
        assert self._time_steps_per_year is not None
        steps = int(self._time_steps_per_year * required_times[-1])
        return TimeGrid.with_mandatory_and_steps(required_times, max(steps, 1))

    def path_generator(self) -> PathGeneratorTypeProtocol[Path]:
        """Build a fresh pricing-phase path generator per ``calculate()``.

        # C++ parity: ``MCLongstaffSchwartzEngine::pathGenerator``
        # (mclongstaffschwartzengine.hpp:247-256).
        """
        return self._build_path_generator(self._seed, self.time_grid())

    def path_pricer(self) -> PathPricer[Path]:
        """Return the cached LSM pricer (already calibrated by ``calculate()``).

        # C++ parity: ``MCLongstaffSchwartzEngine::pathPricer``
        # (mclongstaffschwartzengine.hpp:170-174).
        """
        qassert.require(
            self._cached_lsm_pricer is not None,
            "path pricer unknown — call calculate() first",
        )
        assert self._cached_lsm_pricer is not None
        return self._cached_lsm_pricer

    # --- introspection -------------------------------------------------------

    def exercise_probability(self) -> float:
        """Mean exercised-indicator across priced paths.

        # C++ parity: surfaced by C++ via
        # ``results_.additionalResults["exerciseProbability"]``
        # (mclongstaffschwartzengine.hpp:204-205). We expose it as a
        # typed engine method so consumers don't have to fish in
        # ``additional_results``.

        Returns the post-pricing exercise probability tracked by the
        LSM pricer. Raises if ``calculate()`` hasn't run yet.
        """
        qassert.require(
            self._cached_lsm_pricer is not None,
            "exercise probability not available — call calculate() first",
        )
        assert self._cached_lsm_pricer is not None
        return self._cached_lsm_pricer.exercise_probability()

    # --- subclass contract --------------------------------------------------

    @abstractmethod
    def lsm_path_pricer(self) -> LongstaffSchwartzPathPricer[Path, float]:
        """Build a fresh, uncalibrated LSM path pricer per ``calculate()``.

        # C++ parity: ``MCLongstaffSchwartzEngine::lsmPathPricer``
        # (mclongstaffschwartzengine.hpp:91-92, pure-virtual).
        """

    # --- control variate (default no-op) ------------------------------------

    def control_variate_value(self) -> float | None:
        if not self._control_variate:
            return None
        raise LibraryException("control variate value not provided")

    # --- helpers ------------------------------------------------------------

    def _build_path_generator(self, seed: int, grid: TimeGrid) -> PathGenerator:
        """Build a 1-D PathGenerator over ``grid`` with the given seed.

        # C++ parity: same wiring as ``MCVanillaEngine::pathGenerator``:
        #   factors * (grid.size() - 1) dim Gaussian sequence over PseudoRandom.
        """
        dim_factors = self._process.factors()
        total_dim = dim_factors * (len(grid) - 1)
        # Mirror Phase 5 L5-C divergence: MT rejects seed=0, so use 1 as
        # a deterministic fallback.
        effective_seed = seed if seed != 0 else 1
        gsg = make_pseudo_random_rsg(total_dim, effective_seed)
        return PathGenerator.with_time_grid(
            self._process, grid, gsg, brownian_bridge=self._brownian_bridge
        )


__all__ = ["MCLongstaffSchwartzEngine"]
