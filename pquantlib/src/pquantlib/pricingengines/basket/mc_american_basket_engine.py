"""MCAmericanBasketEngine — Longstaff-Schwartz MC for American baskets.

# C++ parity:
# ql/pricingengines/basket/mcamericanbasketengine.{hpp,cpp} (v1.43),
# ``QuantLib::MCAmericanBasketEngine``, ``QuantLib::AmericanBasketPathPricer``
# and ``QuantLib::MakeMCAmericanBasketEngine``.

Least-squares Monte Carlo. A calibration pass records paths and regresses the
continuation value on a polynomial basis; a pricing pass then applies the
learned exercise rule.

``AmericanBasketPathPricer`` is where a port most easily goes wrong:

* the regression **state** is the whole scaled asset vector
  ``[path[0][t] * scaling, ..., path[n-1][t] * scaling]``, and the basis is
  ``LsmBasisSystem.multi_path_basis_system(n_assets, order, type)`` — a
  multivariate polynomial basis over that vector. It is *not* a univariate
  basis over the max (or the average) of the basket;
* ``scaling = 1 / strike`` whenever the basket payoff wraps a
  ``StrikedTypePayoff``, and the exercise value re-divides by it:
  ``payoff(basketPayoff.accumulate(state) / scaling)``;
* the payoff itself is appended as one extra basis function, so the basis size
  is ``multi_path_basis_system(...).size() + 1``. The probe pins that size and
  the basis values at a fixed state for orders 1, 2 and 3;
* only ``Monomial / Laguerre / Hermite / Hyperbolic / Chebyshev2nd`` are
  accepted.

The C++ engine derives from
``MCLongstaffSchwartzEngine<BasketOption::engine, MultiVariate, RNG>``.
PQuantLib's :class:`MCLongstaffSchwartzEngine` is bound to ``Path`` /
``StochasticProcess1D`` / ``OneAssetOptionResults``, so it cannot be reused
here; the same calibration-then-pricing sequence is reproduced below over
``MultiPath``, including the 2048-sample calibration default and the
calibration seed C++ *actually* uses (see
``_NULL_SIZE_AS_CALIBRATION_SEED`` — it is not ``seed + 1768237423``).

Unlike PQuantLib's one-asset LSM engine, this one writes
``additional_results["exerciseProbability"]``, as C++ does
(mclongstaffschwartzengine.hpp:204-205).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

import numpy as np

from pquantlib import qassert
from pquantlib.exercise import EarlyExercise
from pquantlib.instruments.basket_option import BasketOptionResults, BasketPayoff
from pquantlib.math.array import Array
from pquantlib.math.statistics.general_statistics import GeneralStatistics
from pquantlib.methods.montecarlo.early_exercise_path_pricer import (
    EarlyExercisePathPricer,
)
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    make_pseudo_random_rsg,
)
from pquantlib.methods.montecarlo.longstaff_schwartz_path_pricer import (
    LongstaffSchwartzPathPricer,
)
from pquantlib.methods.montecarlo.lsm_basis_system import LsmBasisSystem, PolynomialType
from pquantlib.methods.montecarlo.monte_carlo_model import (
    MonteCarloModel,
    PathGeneratorTypeProtocol,
)
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.multi_path_generator import MultiPathGenerator
from pquantlib.methods.montecarlo.path_pricer import PathPricer
from pquantlib.option import OptionArguments
from pquantlib.payoffs import Payoff, StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.mc_simulation import McSimulation
from pquantlib.pricingengines.pricing_engine import PricingEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.processes.stochastic_process_array import StochasticProcessArray
from pquantlib.time.time_grid import TimeGrid

# C++ parity: nCalibrationSamples == Null<Size>() ? 2048 : nCalibrationSamples
# (mclongstaffschwartzengine.hpp:141).
_DEFAULT_CALIBRATION_SAMPLES = 2048

# The calibration seed actually used by C++.
#
# ``MCLongstaffSchwartzEngine`` writes
#     seedCalibration_ = (seedCalibration != Null<Real>())
#                            ? seedCalibration
#                            : (seed == 0 ? 0 : seed + 1768237423L);
# (mclongstaffschwartzengine.hpp:147-148). The condition is *always* true:
# ``seedCalibration`` is a ``BigNatural`` and ``Null<Real>()`` is ``FLT_MAX``
# (3.40282e38), so the two can never compare equal. The ``seed + 1768237423``
# branch is therefore dead code, and what reaches the calibration path
# generator is the constructor parameter's own default, ``Null<Size>()`` —
# which for an integral ``T`` is ``numeric_limits<int>::max()``, i.e.
# 2147483647, independent of the pricing seed.
#
# This is reproduced rather than "fixed": diverging would make the two
# libraries price differently. Pinned by ``mcab_lsm_internals`` in
# ``migration-harness/references/v143/pe/basket.json``, whose ``hand_npv``
# (built with 2147483647) equals the engine's NPV to 16 digits, while the
# ``seed + 1768237423`` reading gives 3.5591 against C++'s 3.7504.
_NULL_SIZE_AS_CALIBRATION_SEED = 2147483647


class AmericanBasketPathPricer(EarlyExercisePathPricer[MultiPath, Array]):
    """Early-exercise pricer over the scaled basket state vector.

    # C++ parity: ``AmericanBasketPathPricer``
    # (mcamericanbasketengine.hpp:103-125, mcamericanbasketengine.cpp:27-84).

    Args:
        asset_number: number of assets in the basket.
        payoff: the basket payoff. Must be a ``BasketPayoff``.
        polynomial_order: order of the multivariate polynomial basis.
        polynomial_type: polynomial family.
    """

    __slots__ = ("_asset_number", "_basis", "_payoff", "_scaling_value")

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
        asset_number: int,
        payoff: Payoff,
        polynomial_order: int = 2,
        polynomial_type: PolynomialType = PolynomialType.Monomial,
    ) -> None:
        # C++ parity: mcamericanbasketengine.cpp:27-53.
        qassert.require(
            polynomial_type in AmericanBasketPathPricer._ALLOWED_TYPES,
            "insufficient polynomial type",
        )
        self._asset_number: int = asset_number
        self._payoff: Payoff = payoff

        qassert.require(isinstance(payoff, BasketPayoff), "payoff not a basket payoff")
        assert isinstance(payoff, BasketPayoff)

        self._scaling_value: float = 1.0
        strike_payoff = payoff.base_payoff()
        if isinstance(strike_payoff, StrikedTypePayoff):
            self._scaling_value /= strike_payoff.strike()

        basis: list[Callable[[Array], float]] = list(
            LsmBasisSystem.multi_path_basis_system(
                asset_number, polynomial_order, polynomial_type
            )
        )
        basis.append(self._payoff_at_state)
        self._basis: list[Callable[[Array], float]] = basis

    # --- EarlyExercisePathPricer contract ------------------------------------

    def state(self, path: MultiPath, t: int) -> Array:
        """``[path[i][t] * scaling for i]``.

        # C++ parity: ``state`` (mcamericanbasketengine.cpp:55-65).
        """
        qassert.require(path.asset_number() == self._asset_number, "invalid multipath")
        return np.array(
            [float(path[i].values[t]) * self._scaling_value
             for i in range(self._asset_number)],
            dtype=np.float64,
        )

    def __call__(self, path: MultiPath, t: int) -> float:
        """# C++ parity: ``operator()`` (mcamericanbasketengine.cpp:76-79)."""
        return self._payoff_at_state(self.state(path, t))

    def basis_system(self) -> list[Callable[[Array], float]]:
        """# C++ parity: ``basisSystem`` (mcamericanbasketengine.cpp:81-84)."""
        return self._basis

    # --- helpers -------------------------------------------------------------

    def _payoff_at_state(self, state: Array) -> float:
        """``payoff(basketPayoff.accumulate(state) / scaling)``.

        # C++ parity: ``payoff(const Array&)`` (mcamericanbasketengine.cpp:67-74).
        """
        basket_payoff = self._payoff
        assert isinstance(basket_payoff, BasketPayoff)
        value = basket_payoff.accumulate([float(x) for x in state])
        return self._payoff(value / self._scaling_value)


class MCAmericanBasketEngine(
    GenericEngine[OptionArguments, BasketOptionResults],
    McSimulation[MultiPath],
):
    """Longstaff-Schwartz Monte Carlo engine for American basket options.

    # C++ parity: ``MCAmericanBasketEngine<RNG>``
    # (mcamericanbasketengine.hpp:46-69, 127-188) composed with
    # ``MCLongstaffSchwartzEngine`` (mclongstaffschwartzengine.hpp:178-256).

    .. warning:: The method is intrinsically weak for out-of-the-money options
       (C++ carries the same warning).
    """

    def __init__(
        self,
        processes: StochasticProcessArray,
        *,
        time_steps: int | None = None,
        time_steps_per_year: int | None = None,
        brownian_bridge: bool = False,
        antithetic_variate: bool = False,
        required_samples: int | None = None,
        required_tolerance: float | None = None,
        max_samples: int | None = None,
        seed: int = 0,
        calibration_samples: int | None = None,
        polynomial_order: int = 2,
        polynomial_type: PolynomialType = PolynomialType.Monomial,
        brownian_bridge_calibration: bool | None = None,
        antithetic_variate_calibration: bool | None = None,
        seed_calibration: int | None = None,
    ) -> None:
        GenericEngine.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, OptionArguments(), BasketOptionResults()
        )
        McSimulation.__init__(  # pyright: ignore[reportUnknownMemberType]
            self, antithetic_variate=antithetic_variate, control_variate=False
        )
        # C++ parity: mclongstaffschwartzengine.hpp:149-162.
        qassert.require(
            (time_steps is not None) or (time_steps_per_year is not None),
            "no time steps provided",
        )
        qassert.require(
            (time_steps is None) or (time_steps_per_year is None),
            "both time steps and time steps per year were provided",
        )
        if time_steps is not None:
            qassert.require(
                time_steps > 0, f"timeSteps must be positive, {time_steps} not allowed"
            )
        if time_steps_per_year is not None:
            qassert.require(
                time_steps_per_year > 0,
                f"timeStepsPerYear must be positive, {time_steps_per_year} not allowed",
            )

        self._processes: StochasticProcessArray = processes
        self._time_steps: int | None = time_steps
        self._time_steps_per_year: int | None = time_steps_per_year
        self._brownian_bridge: bool = brownian_bridge
        self._required_samples: int | None = required_samples
        self._required_tolerance: float | None = required_tolerance
        self._max_samples: int | None = max_samples
        self._seed: int = seed
        self._calibration_samples: int = (
            _DEFAULT_CALIBRATION_SAMPLES
            if calibration_samples is None
            else calibration_samples
        )
        self._polynomial_order: int = polynomial_order
        self._polynomial_type: PolynomialType = polynomial_type
        self._brownian_bridge_calibration: bool = (
            brownian_bridge
            if brownian_bridge_calibration is None
            else brownian_bridge_calibration
        )
        self._antithetic_variate_calibration: bool = (
            antithetic_variate
            if antithetic_variate_calibration is None
            else antithetic_variate_calibration
        )
        self._seed_calibration: int = (
            _NULL_SIZE_AS_CALIBRATION_SEED
            if seed_calibration is None
            else seed_calibration
        )
        self._cached_lsm_pricer: (
            LongstaffSchwartzPathPricer[MultiPath, Array] | None
        ) = None
        processes.register_with(self)

    # --- engine entry-point -------------------------------------------------

    def calculate(self) -> None:
        """Calibrate, then price.

        # C++ parity: ``MCLongstaffSchwartzEngine::calculate``
        # (mclongstaffschwartzengine.hpp:178-210).
        """
        self._results.reset()

        self._cached_lsm_pricer = self.lsm_path_pricer()

        grid = self.time_grid()
        cal_generator = self._build_path_generator(
            self._seed_calibration, grid, self._brownian_bridge_calibration
        )
        cal_model = MonteCarloModel[MultiPath](
            path_generator=cal_generator,
            path_pricer=self._cached_lsm_pricer,
            sample_accumulator=GeneralStatistics(),
            antithetic_variate=self._antithetic_variate_calibration,
        )
        cal_model.add_samples(self._calibration_samples)
        self._cached_lsm_pricer.calibrate()

        self.run_mc(
            required_tolerance=self._required_tolerance,
            required_samples=self._required_samples,
            max_samples=self._max_samples,
        )

        assert self._mc_model is not None
        self._results.value = self._mc_model.sample_accumulator().mean()
        self._results.additional_results["exerciseProbability"] = (
            self._cached_lsm_pricer.exercise_probability()
        )
        if self._mc_model.sample_accumulator().samples() > 1:
            self._results.error_estimate = (
                self._mc_model.sample_accumulator().error_estimate()
            )

    # --- McSimulation hooks -------------------------------------------------

    def time_grid(self) -> TimeGrid:
        """# C++ parity: ``timeGrid`` (mclongstaffschwartzengine.hpp:214-240)."""
        exercise = self._arguments.exercise
        qassert.require(exercise is not None, "no exercise given")
        assert exercise is not None

        required_times: list[float] = []
        if exercise.type().name == "American":
            required_times.append(self._processes.time(exercise.last_date()))
        else:
            for i in range(len(exercise.dates())):
                t = self._processes.time(exercise.date(i))
                if t > 0.0:
                    required_times.append(t)

        if self._time_steps is not None:
            return TimeGrid.with_mandatory_and_steps(required_times, self._time_steps)
        assert self._time_steps_per_year is not None
        steps = int(self._time_steps_per_year * required_times[-1])
        return TimeGrid.with_mandatory_and_steps(required_times, max(steps, 1))

    def path_generator(self) -> PathGeneratorTypeProtocol[MultiPath]:
        """# C++ parity: ``pathGenerator`` (mclongstaffschwartzengine.hpp:247-256)."""
        return self._build_path_generator(
            self._seed, self.time_grid(), self._brownian_bridge
        )

    def path_pricer(self) -> PathPricer[MultiPath]:
        """# C++ parity: ``pathPricer`` (mclongstaffschwartzengine.hpp:170-174)."""
        qassert.require(self._cached_lsm_pricer is not None, "path pricer unknown")
        assert self._cached_lsm_pricer is not None
        return self._cached_lsm_pricer

    # --- LSM contract -------------------------------------------------------

    def lsm_path_pricer(self) -> LongstaffSchwartzPathPricer[MultiPath, Array]:
        """# C++ parity: ``lsmPathPricer`` (mcamericanbasketengine.hpp:155-188)."""
        qassert.require(
            self._processes.size() > 0, "Stochastic process array required"
        )
        process = self._processes.process(0)
        qassert.require(
            isinstance(process, GeneralizedBlackScholesProcess),
            "generalized Black-Scholes process required",
        )
        assert isinstance(process, GeneralizedBlackScholesProcess)

        exercise = self._arguments.exercise
        qassert.require(exercise is not None, "wrong exercise given")
        assert exercise is not None
        qassert.require(isinstance(exercise, EarlyExercise), "wrong exercise given")
        assert isinstance(exercise, EarlyExercise)
        qassert.require(
            not exercise.payoff_at_expiry(), "payoff at expiry not handled"
        )

        payoff = self._arguments.payoff
        qassert.require(payoff is not None, "no payoff given")
        assert payoff is not None

        early_pricer = AmericanBasketPathPricer(
            self._processes.size(),
            payoff,
            self._polynomial_order,
            self._polynomial_type,
        )
        return LongstaffSchwartzPathPricer[MultiPath, Array](
            self.time_grid(), early_pricer, process.risk_free_rate()
        )

    def exercise_probability(self) -> float:
        """Post-pricing exercise probability tracked by the LSM pricer."""
        qassert.require(
            self._cached_lsm_pricer is not None,
            "exercise probability not available — call calculate() first",
        )
        assert self._cached_lsm_pricer is not None
        return self._cached_lsm_pricer.exercise_probability()

    # --- helpers -------------------------------------------------------------

    def _build_path_generator(
        self, seed: int, grid: TimeGrid, brownian_bridge: bool
    ) -> MultiPathGenerator:
        dimensions = self._processes.factors()
        gsg = make_pseudo_random_rsg(dimensions * (len(grid) - 1), seed)
        return MultiPathGenerator(
            self._processes, grid, gsg, brownian_bridge=brownian_bridge
        )

    def update(self) -> None:
        self.notify_observers()


class MakeMCAmericanBasketEngine:
    """Named-parameter factory for :class:`MCAmericanBasketEngine`.

    # C++ parity: ``MakeMCAmericanBasketEngine<RNG>``
    # (mcamericanbasketengine.hpp:73-100, 191-303).
    """

    __slots__ = (
        "_antithetic",
        "_brownian_bridge",
        "_calibration_samples",
        "_max_samples",
        "_polynomial_order",
        "_polynomial_type",
        "_process",
        "_samples",
        "_seed",
        "_steps",
        "_steps_per_year",
        "_tolerance",
    )

    def __init__(self, process: StochasticProcessArray) -> None:
        self._process: StochasticProcessArray = process
        self._brownian_bridge: bool = False
        self._antithetic: bool = False
        self._steps: int | None = None
        self._steps_per_year: int | None = None
        self._samples: int | None = None
        self._max_samples: int | None = None
        self._calibration_samples: int | None = None
        self._polynomial_order: int = 2
        self._polynomial_type: PolynomialType = PolynomialType.Monomial
        self._tolerance: float | None = None
        self._seed: int = 0

    def with_steps(self, steps: int) -> Self:
        self._steps = steps
        return self

    def with_steps_per_year(self, steps: int) -> Self:
        self._steps_per_year = steps
        return self

    def with_brownian_bridge(self, b: bool = True) -> Self:
        self._brownian_bridge = b
        return self

    def with_antithetic_variate(self, b: bool = True) -> Self:
        self._antithetic = b
        return self

    def with_samples(self, samples: int) -> Self:
        qassert.require(self._tolerance is None, "tolerance already set")
        self._samples = samples
        return self

    def with_absolute_tolerance(self, tolerance: float) -> Self:
        qassert.require(self._samples is None, "number of samples already set")
        self._tolerance = tolerance
        return self

    def with_max_samples(self, samples: int) -> Self:
        self._max_samples = samples
        return self

    def with_seed(self, seed: int) -> Self:
        self._seed = seed
        return self

    def with_calibration_samples(self, samples: int) -> Self:
        self._calibration_samples = samples
        return self

    def with_polynomial_order(self, polynomial_order: int) -> Self:
        self._polynomial_order = polynomial_order
        return self

    def with_basis_system(self, polynomial_type: PolynomialType) -> Self:
        self._polynomial_type = polynomial_type
        return self

    def engine(self) -> PricingEngine:
        """# C++ parity: ``operator ext::shared_ptr<PricingEngine>()`` (…hpp:282-303)."""
        qassert.require(
            self._steps is not None or self._steps_per_year is not None,
            "number of steps not given",
        )
        qassert.require(
            self._steps is None or self._steps_per_year is None,
            "number of steps overspecified",
        )
        return MCAmericanBasketEngine(
            self._process,
            time_steps=self._steps,
            time_steps_per_year=self._steps_per_year,
            brownian_bridge=self._brownian_bridge,
            antithetic_variate=self._antithetic,
            required_samples=self._samples,
            required_tolerance=self._tolerance,
            max_samples=self._max_samples,
            seed=self._seed,
            calibration_samples=self._calibration_samples,
            polynomial_order=self._polynomial_order,
            polynomial_type=self._polynomial_type,
        )


__all__ = [
    "AmericanBasketPathPricer",
    "MCAmericanBasketEngine",
    "MakeMCAmericanBasketEngine",
]
