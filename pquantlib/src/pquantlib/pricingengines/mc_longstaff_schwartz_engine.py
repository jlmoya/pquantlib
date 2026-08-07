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
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.math.statistics.general_statistics import GeneralStatistics
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
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.time.time_grid import TimeGrid

# C++ ``MCLongstaffSchwartzEngine`` *documents* this offset for the calibration
# seed when the user passes a nonzero pricing seed
# (mclongstaffschwartzengine.hpp:148-149)::
#
#     seedCalibration_(seedCalibration != Null<Real>()
#                          ? seedCalibration
#                          : (seed == 0 ? 0 : seed + 1768237423L))
#
# but that fallback is DEAD CODE, and reproducing the documented intent instead
# of the actual behaviour gets every American MC price wrong. ``seedCalibration``
# is a ``BigNatural`` whose default is ``Null<Size>()``, and the guard compares
# it against ``Null<Real>()`` -- a different specialisation. In v1.43
# ``Null<T>`` yields ``numeric_limits<int>::max()`` for integral ``T`` and
# ``numeric_limits<float>::max()`` for floating-point ``T``, so the default
# ``2147483647`` never equals ``3.4028235e38`` and the ternary always takes its
# first branch. The effective default calibration seed is therefore the literal
# ``Null<Size>()`` value, *independent of the pricing seed*.
#
# Cross-validated: with the offset rule, MCAmericanEngine's ATM put on the probe
# setup prices 6.140821805548784 against C++ 6.2236096554773 (1.3% out, 21 of
# 2047 paths taking a different exercise decision); with ``_NULL_SIZE`` it
# prices 6.223609655477297, i.e. 4.8e-16 relative.

#: ``Null<Size>()`` as an integer -- ``std::numeric_limits<int>::max()``.
_NULL_SIZE = 2147483647


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
        rng_traits: RngTraits = PseudoRandom,
        rng_traits_calibration: RngTraits | None = None,
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
        # C++ parity: mclongstaffschwartzengine.hpp:148-149 -- see the
        # ``_NULL_SIZE`` note above. ``None`` here means "the caller left
        # ``seedCalibration`` at its ``Null<Size>()`` default", which C++ then
        # uses verbatim as the seed.
        self._seed_calibration: int = (
            seed_calibration if seed_calibration is not None else _NULL_SIZE
        )
        # C++ ``RNG`` and ``RNG_Calibration = RNG`` template parameters.
        self._rng_traits: RngTraits = rng_traits
        self._rng_traits_calibration: RngTraits = (
            rng_traits_calibration if rng_traits_calibration is not None else rng_traits
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
        cal_pg = self._build_path_generator(
            self._seed_calibration,
            grid,
            traits=self._rng_traits_calibration,
            brownian_bridge=self._brownian_bridge_calibration,
        )
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
        accumulator = self._mc_model.sample_accumulator()
        self._results.value = accumulator.mean()
        # C++ parity: ``if constexpr (RNG::allowsErrorEstimate)``
        # (mclongstaffschwartzengine.hpp:206-209).
        if self._rng_traits.allows_error_estimate:
            self._results.error_estimate = accumulator.error_estimate()

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

    def _build_path_generator(
        self,
        seed: int,
        grid: TimeGrid,
        *,
        traits: RngTraits | None = None,
        brownian_bridge: bool | None = None,
    ) -> PathGenerator:
        """Build a 1-D PathGenerator over ``grid`` with the given seed.

        # C++ parity: ``MCLongstaffSchwartzEngine::pathGenerator``
        # (mclongstaffschwartzengine.hpp:247-256) for the pricing pass and
        # the inline generator construction in ``calculate()``
        # (mclongstaffschwartzengine.hpp:182-190) for the calibration pass.
        # Dimension is ``factors * (grid.size() - 1)`` in both.

        The seed is passed through untouched: seed 0 reaches ``SeedGenerator``
        via the Mersenne Twister and is clock-derived, exactly as in C++.
        """
        rng = traits if traits is not None else self._rng_traits
        bb = brownian_bridge if brownian_bridge is not None else self._brownian_bridge
        total_dim = self._process.factors() * (len(grid) - 1)
        gsg = rng.make_sequence_generator(total_dim, seed)
        return PathGenerator.with_time_grid(
            self._process, grid, gsg, brownian_bridge=bb
        )


__all__ = ["MCLongstaffSchwartzEngine"]
