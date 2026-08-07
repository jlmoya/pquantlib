"""MCLongstaffSchwartzPathEngine — MC engine for early-exercise basket options.

# C++ parity: ql/experimental/mcbasket/mclongstaffschwartzpathengine.hpp (v1.43).

References:
    Francis Longstaff, Eduardo Schwartz, 2001. *Valuing American Options by
    Simulation: A Simple Least-Squares Approach*, The Review of Financial
    Studies, Volume 14, No. 1, 113-147.

Abstract base: a subclass supplies :meth:`lsm_path_pricer`, returning an
uncalibrated
:class:`~pquantlib.experimental.mcbasket.longstaff_schwartz_multi_path_pricer.LongstaffSchwartzMultiPathPricer`.
:meth:`calculate` then runs the calibration pass, trains the regression and
prices.

Do not confuse this with
:class:`~pquantlib.pricingengines.mc_longstaff_schwartz_engine.MCLongstaffSchwartzEngine`,
the single-asset engine. That one draws its calibration paths from a *second*
generator seeded at ``seed + 1768237423``; this one does not.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import cast

from pquantlib import qassert
from pquantlib.experimental.mcbasket.longstaff_schwartz_multi_path_pricer import (
    LongstaffSchwartzMultiPathPricer,
)
from pquantlib.experimental.mcbasket.path_multi_asset_option import (
    PathMultiAssetOptionArguments,
)
from pquantlib.instruments.instrument import InstrumentResults
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.methods.montecarlo.monte_carlo_model import (
    MonteCarloModel,
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
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.time.time_grid import TimeGrid

_DEFAULT_CALIBRATION_SAMPLES = 2048
"""# C++ parity: ``nCalibrationSamples_((nCalibrationSamples == Null<Size>())
? 2048 : nCalibrationSamples)`` (mclongstaffschwartzpathengine.hpp:110)."""


class MCLongstaffSchwartzPathEngine(
    GenericEngine[PathMultiAssetOptionArguments, InstrumentResults],
    McSimulation[MultiPath],
):
    """Abstract Longstaff-Schwartz MC engine over multi-asset paths.

    # C++ parity: ``MCLongstaffSchwartzPathEngine<GenericEngine, MC, RNG, S>``.

    Args:
        process: the multi-asset process driving the basket.
        time_steps: total time steps (xor ``time_steps_per_year``).
        time_steps_per_year: time steps per year (xor ``time_steps``).
        brownian_bridge: Brownian-bridge path construction (unsupported for
            multi-variate paths in C++ too — raises at generation time).
        antithetic_variate: enable antithetic sampling.
        control_variate: enable control variate.
        required_samples: target sample count (xor ``required_tolerance``).
        required_tolerance: target absolute tolerance.
        max_samples: cap on samples.
        seed: RNG seed.
        n_calibration_samples: paths used to train the regression; ``None``
            means the C++ default of 2048.
        rng_traits: the random-number policy (C++ ``RNG`` template argument).
    """

    def __init__(
        self,
        process: StochasticProcess,
        time_steps: int | None,
        time_steps_per_year: int | None,
        brownian_bridge: bool,
        antithetic_variate: bool,
        control_variate: bool,
        required_samples: int | None,
        required_tolerance: float | None,
        max_samples: int | None,
        seed: int,
        n_calibration_samples: int | None = None,
        rng_traits: RngTraits = PseudoRandom,
    ) -> None:
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, PathMultiAssetOptionArguments(), InstrumentResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, antithetic_variate, control_variate
        )
        # C++ parity: mclongstaffschwartzpathengine.hpp:111-124.
        qassert.require(
            time_steps is not None or time_steps_per_year is not None,
            "no time steps provided",
        )
        qassert.require(
            time_steps is None or time_steps_per_year is None,
            "both time steps and time steps per year were provided",
        )
        qassert.require(time_steps != 0, f"timeSteps must be positive, {time_steps} not allowed")
        qassert.require(
            time_steps_per_year != 0,
            f"timeStepsPerYear must be positive, {time_steps_per_year} not allowed",
        )
        self._process: StochasticProcess = process
        self._time_steps: int | None = time_steps
        self._time_steps_per_year: int | None = time_steps_per_year
        self._brownian_bridge: bool = brownian_bridge
        self._required_samples: int | None = required_samples
        self._required_tolerance: float | None = required_tolerance
        self._max_samples: int | None = max_samples
        self._seed: int = seed
        self._n_calibration_samples: int = (
            _DEFAULT_CALIBRATION_SAMPLES if n_calibration_samples is None else n_calibration_samples
        )
        self._rng_traits: RngTraits = rng_traits
        self._lsm_path_pricer: LongstaffSchwartzMultiPathPricer | None = None
        process.register_with(self)

    # --- subclass contract --------------------------------------------------

    @abstractmethod
    def lsm_path_pricer(self) -> LongstaffSchwartzMultiPathPricer:
        """Build a fresh, uncalibrated LSM pricer.

        # C++ parity: ``lsmPathPricer()`` (pure virtual,
        # mclongstaffschwartzpathengine.hpp:68-69).
        """

    # --- engine entry-point -------------------------------------------------

    def calculate(self) -> None:
        """Calibrate the regression, then price.

        # C++ parity: ``calculate()`` (mclongstaffschwartzpathengine.hpp:136-157).

        # C++ parity note: the calibration model built here is *thrown away*
        # by the pricing pass. ``McSimulation::calculate`` unconditionally
        # rebuilds ``mcModel_`` from a fresh ``pathGenerator()`` — and that
        # generator is seeded with the same ``seed_``. So (a) the calibration
        # samples never reach ``results_.value``, and (b) the pricing paths
        # are the very paths the regression was fitted on: this engine prices
        # in-sample. Reproduced verbatim; it is C++ v1.43 behaviour.
        """
        pricer = self.lsm_path_pricer()
        self._lsm_path_pricer = pricer

        self._mc_model = MonteCarloModel[MultiPath](
            path_generator=self.path_generator(),
            path_pricer=pricer,
            sample_accumulator=GeneralStatistics(),
            antithetic_variate=self._antithetic_variate,
        )
        self._mc_model.add_samples(self._n_calibration_samples)
        pricer.calibrate()

        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )
        # ``run_mc`` has just rebuilt ``_mc_model`` (that is the whole point of
        # the C++ note above), so no None check is possible here.
        self._results.value = self._mc_model.sample_accumulator().mean()
        # C++ parity: ``if constexpr (RNG::allowsErrorEstimate)`` — a
        # low-discrepancy point set is not i.i.d., so no error is published.
        if self._rng_traits.allows_error_estimate:
            self._results.error_estimate = self._mc_model.sample_accumulator().error_estimate()

    # --- McSimulation hooks -------------------------------------------------

    def time_grid(self) -> TimeGrid:
        """Grid with one mandatory node per fixing date.

        # C++ parity: ``timeGrid()`` (mclongstaffschwartzpathengine.hpp:161-176).
        """
        fixings = self._arguments.fixing_dates
        fixing_times = [self._process.time(d) for d in fixings]
        if self._time_steps is not None:
            number_of_time_steps = self._time_steps
        else:
            assert self._time_steps_per_year is not None
            number_of_time_steps = int(self._time_steps_per_year * fixing_times[-1])
        return TimeGrid.with_mandatory_and_steps(fixing_times, number_of_time_steps)

    def path_generator(self) -> PathGeneratorTypeProtocol[MultiPath]:
        """Fresh multi-path generator.

        # C++ parity: ``pathGenerator()`` (mclongstaffschwartzpathengine.hpp:180-193).
        # Note the dimension is ``process_->factors()``, not ``size()``.
        """
        dimensions = self._process.factors()
        grid = self.time_grid()
        # ``RngTraits.make_sequence_generator`` is annotated with the
        # ``SequenceSample`` of ``math.randomnumbers.random_number_generator``;
        # ``MultiPathGenerator`` asks for the structurally identical one from
        # ``methods.montecarlo.gaussian_sequence_generator`` (same two fields,
        # ``value: NDArray[float64]`` and ``weight: float``). The cast is
        # nominal only — nothing about the object changes.
        generator = cast(
            "GaussianSequenceGeneratorProtocol",
            self._rng_traits.make_sequence_generator(dimensions * (len(grid) - 1), self._seed),
        )
        return MultiPathGenerator(
            self._process, grid, generator, brownian_bridge=self._brownian_bridge
        )

    def path_pricer(self) -> PathPricer[MultiPath]:
        """The cached (already calibrated) LSM pricer.

        # C++ parity: ``pathPricer()`` (mclongstaffschwartzpathengine.hpp:128-132).
        """
        qassert.require(self._lsm_path_pricer is not None, "path pricer unknown")
        assert self._lsm_path_pricer is not None
        return self._lsm_path_pricer


__all__ = ["MCLongstaffSchwartzPathEngine"]
