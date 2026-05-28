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
from pquantlib.exercise import EarlyExercise, EuropeanExercise, Exercise
from pquantlib.instruments.european_option import EuropeanOption
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
from pquantlib.payoffs import Payoff, PlainVanillaPayoff, StrikedTypePayoff
from pquantlib.pricingengines.mc_longstaff_schwartz_engine import (
    MCLongstaffSchwartzEngine,
)
from pquantlib.pricingengines.vanilla.analytic_european_engine import (
    AnalyticEuropeanEngine,
)
from pquantlib.pricingengines.vanilla.mc_european_engine import EuropeanPathPricer
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
        )
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

    def control_variate_value(self) -> float | None:
        """Analytic European NPV used as the CV reference value.

        # C++ parity: ``MCAmericanEngine::controlVariateValue``
        # (mcamericanengine.hpp:244-265).
        """
        if not self._control_variate:
            return None
        qassert.require(
            isinstance(self._process, GeneralizedBlackScholesProcess),
            "generalized Black-Scholes process required",
        )
        process = self._process
        assert isinstance(process, GeneralizedBlackScholesProcess)

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
        # Wrap as a European with the same last_date — C++ creates a
        # fresh EuropeanExercise from the exercise's last date.
        european_exercise: Exercise = EuropeanExercise(exercise.last_date())
        opt = EuropeanOption(payoff, european_exercise)
        opt.set_pricing_engine(AnalyticEuropeanEngine(process))
        return opt.npv()


__all__ = ["AmericanPathPricer", "MCAmericanEngine"]
