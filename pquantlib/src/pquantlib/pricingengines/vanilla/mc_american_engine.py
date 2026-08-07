"""MCAmericanEngine — Monte Carlo pricing for American vanilla options.

# C++ parity: ql/pricingengines/vanilla/mcamericanengine.{hpp,cpp} (v1.42.1).

Implements the Longstaff-Schwartz American option algorithm:
calibrate a per-step exercise rule on a calibration MC pass, then
price on a fresh pricing MC pass.

The engine builds:

* An ``AmericanPathPricer`` — concrete
  :class:`EarlyExercisePathPricer[Path, float]` capturing the option
  payoff plus the polynomial basis system.
* A :class:`LongstaffSchwartzPathPricer[Path, float]` wrapping the
  above; this is what drives the regression and pricing.

The optional control-variate uses the analytic European value as the
zero-mean reference (C++ ``controlPathPricer`` returns the European
path pricer; ``controlPricingEngine`` returns ``AnalyticEuropeanEngine``).
"""

from __future__ import annotations

from collections.abc import Callable

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import EarlyExercise, EuropeanExercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.randomnumbers.rng_traits import PseudoRandom
from pquantlib.methods.montecarlo.early_exercise_path_pricer import (
    EarlyExercisePathPricer,
)
from pquantlib.methods.montecarlo.longstaff_schwartz_path_pricer import (
    LongstaffSchwartzPathPricer,
)
from pquantlib.methods.montecarlo.lsm_basis_system import (
    LsmBasisSystem,
    PolynomialType,
)
from pquantlib.methods.montecarlo.path import Path
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.option import OptionArguments
from pquantlib.payoffs import Payoff, PlainVanillaPayoff, StrikedTypePayoff
from pquantlib.pricingengines.mc_longstaff_schwartz_engine import (
    MCLongstaffSchwartzEngine,
)
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.pricingengines.vanilla.mc_european_engine import EuropeanPathPricer
from pquantlib.pricingengines.vanilla.mc_vanilla_engine import RngTraits
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


class AmericanPathPricer(EarlyExercisePathPricer[Path, float]):
    """Early-exercise pricer for American vanilla options.

    # C++ parity: ``AmericanPathPricer`` (mcamericanengine.hpp:85-102 +
    # mcamericanengine.cpp:31-72).

    Captures the option payoff plus a polynomial basis system; the
    exercise value at any time is ``payoff(path[t] * scaling)`` (the
    rescaling factor improves numerical stability of the regression).

    The C++ constructor restricts ``polynomialType`` to
    ``Monomial / Laguerre / Hermite / Hyperbolic / Chebyshev2nd`` —
    we mirror this check and additionally guard against the four C++
    types not yet ported (Hyperbolic / Legendre / Chebyshev).
    """

    __slots__ = ("_basis", "_payoff", "_scaling")

    _ALLOWED_TYPES: frozenset[PolynomialType] = frozenset(
        {
            PolynomialType.Monomial,
            PolynomialType.Laguerre,
            PolynomialType.Hermite,
            PolynomialType.Hyperbolic,
            PolynomialType.Chebyshev2nd,
        }
    )

    def __init__(
        self,
        payoff: Payoff,
        polynomial_order: int,
        polynomial_type: PolynomialType,
    ) -> None:
        qassert.require(
            polynomial_type in AmericanPathPricer._ALLOWED_TYPES,
            "insufficient polynomial type",
        )
        self._payoff: Payoff = payoff

        # C++ parity: rescale by 1 / strike (StrikedTypePayoff) to keep
        # the basis-system regression numerically stable. For non-strike
        # payoffs leave scaling = 1.
        # mcamericanengine.cpp:50-52.
        if isinstance(payoff, StrikedTypePayoff):
            self._scaling: float = 1.0 / payoff.strike()
        else:
            self._scaling = 1.0

        # Build the basis system. C++ appends the payoff itself as an
        # extra basis function (mcamericanengine.cpp:45-46).
        # NOTE: this gives basis size = polynomial_order + 2 (not + 1).
        self._basis: list[Callable[[float], float]] = list(
            LsmBasisSystem.path_basis_system(polynomial_order, polynomial_type)
        )
        # Payoff-as-basis: ``state -> payoff(state / scaling)`` which
        # is the same as ``state -> AmericanPathPricer.payoff(state)``.
        # We close over self.
        self._basis.append(self._payoff_basis)

    # --- EarlyExercisePathPricer contract ----------------------------------

    def __call__(self, path: Path, t: int) -> float:
        # C++ parity: operator()(path, t) = payoff(state(path, t)).
        return self._payoff_at_state(self.state(path, t))

    def state(self, path: Path, t: int) -> float:
        # C++ parity: state(path, t) = path[t] * scaling.
        return float(path[t]) * self._scaling

    def basis_system(self) -> list[Callable[[float], float]]:
        return self._basis

    # --- helpers -----------------------------------------------------------

    def _payoff_at_state(self, state: float) -> float:
        # C++ parity: ``payoff(state)`` private method
        # (mcamericanengine.cpp:55-57): ``return (*payoff_)(state / scalingValue_)``.
        return self._payoff(state / self._scaling)

    def _payoff_basis(self, state: float) -> float:
        # Captures self for the extra basis-function lambda the C++ adds
        # at construction. Mirrors mcamericanengine.cpp:45-46.
        return self._payoff_at_state(state)


class MCAmericanEngine(MCLongstaffSchwartzEngine):
    """Monte Carlo pricing engine for American vanilla options.

    # C++ parity: ``MCAmericanEngine<RNG, S, RNG_Calibration>``
    # (mcamericanengine.hpp:50-83).

    Specializes :class:`MCLongstaffSchwartzEngine` by providing:

    * :meth:`lsm_path_pricer` returning a
      ``LongstaffSchwartzPathPricer[Path, float]`` built around an
      :class:`AmericanPathPricer`.
    * Optional control variate via the analytic European engine
      (``controlPathPricer`` + ``controlPricingEngine`` + ``controlVariateValue``
      in C++).
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
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
        polynom_order: int = 2,
        polynom_type: PolynomialType = PolynomialType.Monomial,
        calibration_samples: int = 2048,
        antithetic_variate_calibration: bool | None = None,
        seed_calibration: int | None = None,
        rng_traits: RngTraits = PseudoRandom,
        rng_traits_calibration: RngTraits | None = None,
    ) -> None:
        super().__init__(
            process,
            time_steps=time_steps,
            time_steps_per_year=time_steps_per_year,
            brownian_bridge=brownian_bridge,
            antithetic_variate=antithetic_variate,
            control_variate=control_variate,
            required_samples=required_samples,
            required_tolerance=required_tolerance,
            max_samples=max_samples,
            seed=seed,
            calibration_samples=calibration_samples,
            antithetic_variate_calibration=antithetic_variate_calibration,
            seed_calibration=seed_calibration,
            rng_traits=rng_traits,
            rng_traits_calibration=rng_traits_calibration,
        )
        # C++ ``MCAmericanEngine`` passes ``brownianBridge = false``
        # unconditionally (mcamericanengine.hpp:161), so no bridge knob here.
        self._polynom_order: int = polynom_order
        self._polynom_type: PolynomialType = polynom_type

    def calculate(self) -> None:
        """Run MC + LSM regression, then clamp negative CV-driven NPVs to zero.

        # C++ parity: ``MCAmericanEngine::calculate``
        # (mcamericanengine.hpp:175-183) — when the control variate is
        # enabled, the CV-adjusted estimator can dip slightly negative
        # for deep-OTM options; C++ clamps via ``std::max(0.0, value)``.
        """
        super().calculate()
        if self._control_variate:
            self._results.value = max(0.0, self._results.value or 0.0)

    # --- MCLongstaffSchwartzEngine contract --------------------------------

    def lsm_path_pricer(self) -> LongstaffSchwartzPathPricer[Path, float]:
        """Build the AmericanPathPricer + LSM wrapper.

        # C++ parity: ``MCAmericanEngine::lsmPathPricer``
        # (mcamericanengine.hpp:186-209).
        """
        qassert.require(
            isinstance(self._process, GeneralizedBlackScholesProcess),
            "generalized Black-Scholes process required",
        )
        process = self._process
        assert isinstance(process, GeneralizedBlackScholesProcess)

        exercise = self._arguments.exercise
        qassert.require(exercise is not None, "no exercise given")
        assert exercise is not None
        qassert.require(
            isinstance(exercise, EarlyExercise),
            "wrong exercise given",
        )
        assert isinstance(exercise, EarlyExercise)
        qassert.require(
            not exercise.payoff_at_expiry(),
            "payoff at expiry not handled",
        )

        payoff = self._arguments.payoff
        qassert.require(payoff is not None, "no payoff given")
        assert payoff is not None

        early_pricer = AmericanPathPricer(
            payoff=payoff,
            polynomial_order=self._polynom_order,
            polynomial_type=self._polynom_type,
        )
        return LongstaffSchwartzPathPricer[Path, float](
            self.time_grid(),
            early_pricer,
            process.risk_free_rate(),
        )

    # --- control variate ---------------------------------------------------

    def control_path_pricer(self) -> PathPricer[Path] | None:
        """European-payoff path pricer used as the CV path pricer.

        # C++ parity: ``MCAmericanEngine::controlPathPricer``
        # (mcamericanengine.hpp:213-230).
        """
        if not self._control_variate:
            return None
        payoff = self._arguments.payoff
        qassert.require(
            isinstance(payoff, StrikedTypePayoff),
            "StrikedTypePayoff needed for control variate",
        )
        assert isinstance(payoff, StrikedTypePayoff)
        qassert.require(
            isinstance(self._process, GeneralizedBlackScholesProcess),
            "generalized Black-Scholes process required",
        )
        process = self._process
        assert isinstance(process, GeneralizedBlackScholesProcess)

        last_t = float(self.time_grid().back())
        discount = process.risk_free_rate().discount(last_t)
        return EuropeanPathPricer(
            option_type=payoff.option_type(),
            strike=payoff.strike(),
            discount=discount,
        )

    def control_pricing_engine(self) -> PricingEngine | None:
        """Analytic European engine supplying the CV reference value.

        # C++ parity: ``MCAmericanEngine::controlPricingEngine``
        # (mcamericanengine.hpp:234-242).
        """
        qassert.require(
            isinstance(self._process, GeneralizedBlackScholesProcess),
            "generalized Black-Scholes process required",
        )
        process = self._process
        assert isinstance(process, GeneralizedBlackScholesProcess)
        return AnalyticEuropeanEngine(process)

    def control_variate_value(self) -> float | None:
        """Analytic European NPV used as the CV reference value.

        # C++ parity: ``MCAmericanEngine::controlVariateValue``
        # (mcamericanengine.hpp:244-265). Note the C++ override does NOT
        # simply delegate to ``MCVanillaEngine::controlVariateValue``: it
        # copies the arguments and then *replaces* the exercise with a fresh
        # ``EuropeanExercise(arguments_.exercise->lastDate())``, because the
        # analytic engine cannot price the American exercise the MC engine
        # was handed.
        """
        control_engine = self.control_pricing_engine()
        qassert.require(
            control_engine is not None,
            "engine does not provide control variation pricing engine",
        )
        assert control_engine is not None

        payoff = self._arguments.payoff
        qassert.require(payoff is not None, "no payoff given")
        assert payoff is not None
        exercise = self._arguments.exercise
        qassert.require(exercise is not None, "no exercise given")
        assert exercise is not None
        if not isinstance(payoff, PlainVanillaPayoff):
            raise LibraryException(
                "control variate requires a PlainVanillaPayoff (AnalyticEuropeanEngine "
                "doesn't accept binary / asset-or-nothing payoffs)"
            )

        european_arguments = OptionArguments()
        european_arguments.payoff = payoff
        european_arguments.exercise = EuropeanExercise(exercise.last_date())
        control_arguments = control_engine.get_arguments()
        qassert.require(
            isinstance(control_arguments, OptionArguments),
            "engine is using inconsistent arguments",
        )
        assert isinstance(control_arguments, OptionArguments)
        control_arguments.payoff = european_arguments.payoff
        control_arguments.exercise = european_arguments.exercise
        control_engine.reset()
        control_arguments.validate()
        control_engine.calculate()
        results = control_engine.get_results()
        assert isinstance(results, OneAssetOptionResults)
        value = results.value
        if value is None:
            raise LibraryException("control engine did not produce a value")
        return value


class MakeMCAmericanEngine:
    """Fluent builder for :class:`MCAmericanEngine`.

    # C++ parity: ``MakeMCAmericanEngine<RNG, S, RNG_Calibration>``
    # (mcamericanengine.hpp:108-139, 268-399).

    Defaults reproduce the C++ member initialisers exactly::

        antithetic_ = false, controlVariate_ = false,
        calibrationSamples_ = 2048, seed_ = 0,
        polynomialOrder_ = 2, polynomialType_ = LsmBasisSystem::Monomial,
        antitheticCalibration_ = ext::nullopt, seedCalibration_ = Null<Size>()

    and ``steps_ / stepsPerYear_ / samples_ / maxSamples_ / tolerance_`` all
    start ``Null``. As in C++, the terminal conversion (here :meth:`engine`)
    is what enforces steps XOR stepsPerYear.
    """

    __slots__ = (
        "_antithetic",
        "_antithetic_calibration",
        "_calibration_samples",
        "_control_variate",
        "_max_samples",
        "_polynomial_order",
        "_polynomial_type",
        "_process",
        "_rng_traits",
        "_rng_traits_calibration",
        "_samples",
        "_seed",
        "_seed_calibration",
        "_steps",
        "_steps_per_year",
        "_tolerance",
    )

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        rng_traits: RngTraits = PseudoRandom,
        rng_traits_calibration: RngTraits | None = None,
    ) -> None:
        self._process: GeneralizedBlackScholesProcess = process
        self._rng_traits: RngTraits = rng_traits
        self._rng_traits_calibration: RngTraits | None = rng_traits_calibration
        self._antithetic: bool = False
        self._control_variate: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._calibration_samples: int = 2048
        self._tolerance: float | None = None
        self._seed: int = 0
        self._polynomial_order: int = 2
        self._polynomial_type: PolynomialType = PolynomialType.Monomial
        self._antithetic_calibration: bool | None = None
        self._seed_calibration: int | None = None

    def with_steps(self, steps: int) -> MakeMCAmericanEngine:
        """# C++ parity: ``withSteps`` (mcamericanengine.hpp:288-293)."""
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> MakeMCAmericanEngine:
        """# C++ parity: ``withStepsPerYear`` (mcamericanengine.hpp:295-301)."""
        self._steps_per_year = steps
        return self

    def with_samples(self, samples: int) -> MakeMCAmericanEngine:
        """# C++ parity: ``withSamples`` (mcamericanengine.hpp:303-310)."""
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> MakeMCAmericanEngine:
        """# C++ parity: ``withAbsoluteTolerance`` (mcamericanengine.hpp:312-323)."""
        qassert.require(self._samples is None, "number of samples already set")
        qassert.require(
            bool(self._rng_traits.allows_error_estimate),
            "chosen random generator policy does not allow an error estimate",
        )
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> MakeMCAmericanEngine:
        """# C++ parity: ``withMaxSamples`` (mcamericanengine.hpp:325-331)."""
        self._max_samples = samples
        return self

    def with_calibration_samples(self, samples: int) -> MakeMCAmericanEngine:
        """# C++ parity: ``withCalibrationSamples`` (mcamericanengine.hpp:333-339)."""
        self._calibration_samples = samples
        return self

    def with_seed(self, seed: int) -> MakeMCAmericanEngine:
        """# C++ parity: ``withSeed`` (mcamericanengine.hpp:341-346)."""
        self._seed = seed
        return self

    def with_antithetic_variate(self, b: bool = True) -> MakeMCAmericanEngine:
        """# C++ parity: ``withAntitheticVariate`` (mcamericanengine.hpp:348-354)."""
        self._antithetic = b
        return self

    def with_control_variate(self, b: bool = True) -> MakeMCAmericanEngine:
        """# C++ parity: ``withControlVariate`` (mcamericanengine.hpp:356-361)."""
        self._control_variate = b
        return self

    def with_polynomial_order(self, polynomial_order: int) -> MakeMCAmericanEngine:
        """# C++ parity: ``withPolynomialOrder`` (mcamericanengine.hpp:274-279)."""
        self._polynomial_order = polynomial_order
        return self

    def with_basis_system(self, polynomial_type: PolynomialType) -> MakeMCAmericanEngine:
        """# C++ parity: ``withBasisSystem`` (mcamericanengine.hpp:281-286)."""
        self._polynomial_type = polynomial_type
        return self

    def with_antithetic_variate_calibration(self, b: bool = True) -> MakeMCAmericanEngine:
        """# C++ parity: ``withAntitheticVariateCalibration``
        # (mcamericanengine.hpp:363-368)."""
        self._antithetic_calibration = b
        return self

    def with_seed_calibration(self, seed: int) -> MakeMCAmericanEngine:
        """# C++ parity: ``withSeedCalibration`` (mcamericanengine.hpp:370-376)."""
        self._seed_calibration = seed
        return self

    def engine(self) -> MCAmericanEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>() const``
        # (mcamericanengine.hpp:378-399)."""
        qassert.require(
            (self._steps is not None) or (self._steps_per_year is not None),
            "number of steps not given",
        )
        qassert.require(
            (self._steps is None) or (self._steps_per_year is None),
            "number of steps overspecified",
        )
        return MCAmericanEngine(
            self._process,
            time_steps=self._steps,
            time_steps_per_year=self._steps_per_year,
            antithetic_variate=self._antithetic,
            control_variate=self._control_variate,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            polynom_order=self._polynomial_order,
            polynom_type=self._polynomial_type,
            calibration_samples=self._calibration_samples,
            antithetic_variate_calibration=self._antithetic_calibration,
            seed_calibration=self._seed_calibration,
            rng_traits=self._rng_traits,
            rng_traits_calibration=self._rng_traits_calibration,
        )


__all__ = ["AmericanPathPricer", "MCAmericanEngine", "MakeMCAmericanEngine"]
