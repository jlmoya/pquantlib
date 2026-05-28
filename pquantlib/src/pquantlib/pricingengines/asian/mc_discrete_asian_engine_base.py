"""MCDiscreteAveragingAsianEngineBase — MC engine base for discrete-average Asians.

# C++ parity: ql/pricingengines/asian/mcdiscreteasianenginebase.hpp (v1.42.1) —
# ``template <template <class> class MC, class RNG, class S>
#  class MCDiscreteAveragingAsianEngineBase : public
#      DiscreteAveragingAsianOption::engine,
#      public McSimulation<MC, RNG, S>``.

Equivalent role to MCVanillaEngine but for Asians.  Builds the time
grid from the option's fixing-date list (per the C++
``initializeTimeGrid`` helper).  Subclasses supply ``path_pricer()``
(arithmetic vs geometric average) and may override
``control_path_pricer`` / ``control_pricing_engine`` to enable
geometric-average control variate.

Python divergences:

* The C++ template parameter ``MC`` is the variate trait (single vs
  multi). Asian engines are always single-variate in v1.42.1
  (multi-variate Asians are out of scope here); the Python port fixes
  this dimension and skips the trait altogether.
* The C++ class supports a ``timeSteps`` / ``timeStepsPerYear``
  override so callers can resample finer than the natural fixing grid
  for ``includeExerciseDate=true``; the Python port keeps the
  conservative-default branch (no extra steps) since L5-C's reference
  scenarios don't need it.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.instruments.asian_option import DiscreteAveragingAsianOptionArguments
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
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.time.time_grid import TimeGrid


class MCDiscreteAveragingAsianEngineBase(
    GenericEngine[DiscreteAveragingAsianOptionArguments, OneAssetOptionResults],
    McSimulation[Path],
):
    """Abstract MC engine base for discrete-average Asian options.

    # C++ parity: ``MCDiscreteAveragingAsianEngineBase<MC,RNG,S>``
    # (mcdiscreteasianenginebase.hpp:56-130).
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        *,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        control_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
    ) -> None:
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            DiscreteAveragingAsianOptionArguments(),
            OneAssetOptionResults(),
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self,
            antithetic_variate=antithetic_variate,
            control_variate=control_variate,
        )
        self._process: GeneralizedBlackScholesProcess = process
        self._brownian_bridge: bool = brownian_bridge
        self._required_samples: int | None = required_samples
        self._required_tolerance: float | None = required_tolerance
        self._max_samples: int | None = max_samples
        self._seed: int = seed
        process.register_with(self)

    # --- engine entry-point ----------------------------------------------

    def calculate(self) -> None:
        """Run the MC and fill ``self._results``.

        # C++ parity: ``MCDiscreteAveragingAsianEngineBase::calculate``
        # (mcdiscreteasianenginebase.hpp).
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
        """Build a time grid from the option's fixing dates.

        # C++ parity: ``MCDiscreteAveragingAsianEngineBase::timeGrid``.
        The default builds a grid whose mandatory times are exactly the
        future fixing times (translated to year-fractions via the
        process clock).
        """
        args = self._arguments
        process = self._process
        reference_date = process.risk_free_rate().reference_date()
        voldc = process.black_volatility().day_counter()
        fixing_times: list[float] = []
        for fd in args.fixing_dates:
            if fd >= reference_date:
                fixing_times.append(voldc.year_fraction(reference_date, fd))
        qassert.require(
            len(fixing_times) > 0,
            "no future fixing dates — all fixings are in the past",
        )
        return TimeGrid.with_mandatory(fixing_times)

    def path_generator(self) -> PathGeneratorTypeProtocol[Path]:
        """Build a fresh ``PathGenerator`` per ``calculate()``."""
        grid = self.time_grid()
        dim = self._process.factors() * (len(grid) - 1)
        seed = self._seed if self._seed != 0 else 1
        gsg = make_pseudo_random_rsg(dim, seed)
        return PathGenerator.with_time_grid(
            self._process, grid, gsg, brownian_bridge=self._brownian_bridge
        )

    # --- abstract subclass hooks -----------------------------------------

    @abstractmethod
    def path_pricer(self) -> PathPricer[Path]:
        """Subclass: build the arithmetic / geometric average pricer."""

    def control_path_pricer(self) -> PathPricer[Path] | None:
        return None

    def control_pricing_engine(self) -> PricingEngine | None:
        """Subclass overrides to provide the analytic CV engine.

        # C++ parity: ``MCDiscreteArithmeticAPEngine::controlPricingEngine``.
        """
        return None

    def control_variate_value(self) -> float | None:
        """Run the control pricing engine and return its NPV.

        # C++ parity: ``MCVanillaEngine::controlVariateValue`` — runs
        # the analytic CV engine with the *same* arguments as ``self``
        # and returns its ``value`` field.
        """
        if not self._control_variate:
            return None
        cpe = self.control_pricing_engine()
        qassert.require(
            cpe is not None,
            "engine does not provide control variation pricing engine",
        )
        assert cpe is not None
        # Copy our arguments into the CV engine's arguments slot.
        ctrl_args = cpe.get_arguments()
        qassert.require(
            isinstance(ctrl_args, DiscreteAveragingAsianOptionArguments),
            "engine is using inconsistent arguments",
        )
        assert isinstance(ctrl_args, DiscreteAveragingAsianOptionArguments)
        # Mirror Option::setupArguments() — copy payoff/exercise + the
        # extra Asian fields.
        ctrl_args.payoff = self._arguments.payoff
        ctrl_args.exercise = self._arguments.exercise
        ctrl_args.average_type = self._arguments.average_type
        ctrl_args.running_accumulator = self._arguments.running_accumulator
        ctrl_args.past_fixings = self._arguments.past_fixings
        ctrl_args.fixing_dates = list(self._arguments.fixing_dates)
        cpe.calculate()
        ctrl_results = cpe.get_results()
        qassert.require(
            isinstance(ctrl_results, OneAssetOptionResults),
            "engine returns an inconsistent result type",
        )
        assert isinstance(ctrl_results, OneAssetOptionResults)
        v = ctrl_results.value
        if v is None:
            raise LibraryException("control variate value not computed")
        return float(v)


__all__ = ["MCDiscreteAveragingAsianEngineBase"]
