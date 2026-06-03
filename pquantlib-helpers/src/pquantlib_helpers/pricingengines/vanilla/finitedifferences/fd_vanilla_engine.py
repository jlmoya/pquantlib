"""FDVanillaEngine — base finite-difference engine for BSM one-asset options.

# Retired-API compat layer — NOT a 1:1 port of C++ QuantLib v1.42.1 (the old
# ``ql/pricingengines/vanilla/fdvanillaengine.hpp`` family was retired; modern
# QuantLib uses ``FdBlackScholesVanillaEngine``). JQuantLib still carries it.

Java parity: ``org.jquantlib.pricingengines.vanilla.finitedifferences.FDVanillaEngine``
+ ``PayoffFunction``.

Despite the name this is a *base* class: its job is grid layout (log-grid limits,
strike-in-grid enforcement), intrinsic-value sampling, BSM operator construction
(via the FD-alpha2
:class:`~pquantlib_helpers.methods.finitedifferences.pde_operator.OperatorFactory`),
and Neumann boundary-condition setup. Concrete schemes
(:class:`~pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_multi_period_engine.FDMultiPeriodEngine`,
:class:`~pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_step_condition_engine.FDStepConditionEngine`)
subclass it and override :meth:`calculate`.

# Java arg-order note: the JQuantLib ``FDVanillaEngine`` constructor is
# ``(process, timeSteps, gridPoints, timeDependent)``, but several JQuantLib
# subclasses call ``super(process, gridPoints, timeSteps, timeDependent)`` with
# the middle two SWAPPED. Because the FD dividend chain always passes
# ``timeSteps == gridPoints`` from the helper (``new Engine(process, timeSteps)``
# defaults ``gridPoints`` to the same value via the dividend-engine ctor chain),
# the swap is benign on the cross-validated path. We keep the C++/declared order
# ``(process, time_steps, grid_points, time_dependent)`` here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib_helpers.math.sampled_curve import SampledCurve
from pquantlib_helpers.methods.finitedifferences.boundary_condition import (
    NeumannBC,
    Side,
)
from pquantlib_helpers.methods.finitedifferences.pde_operator import OperatorFactory

if TYPE_CHECKING:
    from pquantlib.instruments.one_asset_option import OneAssetOptionResults
    from pquantlib.payoffs import Payoff
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )
    from pquantlib.time.date import Date
    from pquantlib_helpers.instruments.dividend_vanilla_option import (
        DividendVanillaOptionArguments,
    )
    from pquantlib_helpers.methods.finitedifferences.mixed_scheme import (
        BoundaryCondition,
    )
    from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
        TridiagonalOperator,
    )

import math

# C++/Java ``safetyZoneFactor`` — keeps the strike strictly inside the grid.
_SAFETY_ZONE_FACTOR = 1.1


class PayoffFunction:
    """Callable adapter wrapping a payoff as ``op(x) = payoff(x)``.

    Java parity: ``PayoffFunction`` (implements ``Ops.DoubleOp``).
    """

    def __init__(self, payoff: Payoff) -> None:
        """Bind the payoff."""
        self._payoff = payoff

    def __call__(self, a: float) -> float:
        """Return ``payoff(a)``."""
        return self._payoff(a)


class FDVanillaEngine:
    """Grid-layout base for the legacy finite-difference BSM engines.

    Java parity: ``FDVanillaEngine``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        time_steps: int,
        grid_points: int,
        time_dependent: bool,
    ) -> None:
        """Bind the process + grid sizing; allocate the intrinsic-value curve."""
        self.process: GeneralizedBlackScholesProcess = process
        self.time_steps: int = time_steps
        self.grid_points: int = grid_points
        self.time_dependent: bool = time_dependent
        self.required_grid_value: float = 0.0
        self.exercise_date: Date | None = None
        self.payoff: Payoff | None = None
        self.finite_difference_operator: TridiagonalOperator | None = None
        self.intrinsic_values: SampledCurve = SampledCurve(grid_points)
        self.bc_s: list[BoundaryCondition] = []
        # temporaries
        self.s_min: float = 0.0
        self.center: float = 0.0
        self.s_max: float = 0.0

    # ------------------------------------------------------------------
    def grid(self) -> Array:
        """The current underlying grid. Java parity: ``grid()``."""
        return self.intrinsic_values.grid()

    def setup_arguments(self, a: DividendVanillaOptionArguments) -> None:
        """Read payoff + exercise date + strike off the engine arguments.

        Java parity: ``FDVanillaEngine.setupArguments``.
        """
        assert a.exercise is not None
        self.exercise_date = a.exercise.last_date()
        assert a.payoff is not None
        self.payoff = a.payoff
        qassert.require(
            isinstance(self.payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(self.payoff, StrikedTypePayoff)
        self.required_grid_value = self.payoff.strike()

    def get_residual_time(self) -> float:
        """Time to the exercise date. Java parity: ``getResidualTime``."""
        assert self.exercise_date is not None
        return self.process.time(self.exercise_date)

    def safe_grid_points(self, grid_points: int, residual_time: float) -> int:
        """Floor the grid size by a per-year minimum. Java parity: ``safeGridPoints``."""
        min_grid_points = 10
        min_grid_points_per_year = 2
        if residual_time > 1.0:
            floor = int(min_grid_points + (residual_time - 1.0) * min_grid_points_per_year)
        else:
            floor = min_grid_points
        return max(grid_points, floor)

    def set_grid_limits(self) -> None:
        """Lay out the [sMin, sMax] log-grid + enforce strike-in-grid.

        Java parity: ``setGridLimits()`` (the no-arg form).
        """
        self.set_grid_limits_at(
            self.process.state_variable().value(), self.get_residual_time()
        )
        self.ensure_strike_in_grid()

    def set_grid_limits_at(self, center: float, t: float) -> None:
        """Compute [sMin, sMax] around ``center`` for residual time ``t``.

        Java parity: ``setGridLimits(center, t)``.
        """
        qassert.require(center > 0.0, "negative or null underlying given")
        self.center = center
        new_grid_points = self.safe_grid_points(self.grid_points, t)
        if new_grid_points > self.intrinsic_values.size():
            self.intrinsic_values = SampledCurve(new_grid_points)

        vol_sqrt_time = math.sqrt(
            self.process.black_volatility().black_variance_at_time(t, center)
        )
        # the prefactor fine-tunes performance at small volatilities
        prefactor = 1.0 + 0.02 / vol_sqrt_time
        min_max_factor = math.exp(4.0 * prefactor * vol_sqrt_time)
        self.s_min = center / min_max_factor
        self.s_max = center * min_max_factor

    def ensure_strike_in_grid(self) -> None:
        """Widen [sMin, sMax] so the strike sits strictly inside.

        Java parity: ``ensureStrikeInGrid``.
        """
        if not isinstance(self.payoff, StrikedTypePayoff):
            return
        required_grid_value = self.payoff.strike()
        if self.s_min > required_grid_value / _SAFETY_ZONE_FACTOR:
            self.s_min = required_grid_value / _SAFETY_ZONE_FACTOR
            self.s_max = self.center / (self.s_min / self.center)
        if self.s_max < required_grid_value * _SAFETY_ZONE_FACTOR:
            self.s_max = required_grid_value * _SAFETY_ZONE_FACTOR
            self.s_min = self.center / (self.s_max / self.center)

    def initialize_initial_condition(self) -> None:
        """Sample the intrinsic payoff on the log-grid.

        Java parity: ``initializeInitialCondition``.
        """
        assert self.payoff is not None
        self.intrinsic_values.set_log_grid(self.s_min, self.s_max)
        self.intrinsic_values.sample(PayoffFunction(self.payoff))

    def initialize_operator(self) -> None:
        """Build the BSM tridiagonal operator on the current grid.

        Java parity: ``initializeOperator``.
        """
        self.finite_difference_operator = OperatorFactory.get_operator(
            self.process,
            self.intrinsic_values.grid(),
            self.get_residual_time(),
            self.time_dependent,
        )

    def initialize_boundary_conditions(self) -> None:
        """Install lower/upper Neumann boundary conditions.

        Java parity: ``initializeBoundaryConditions``.
        """
        iv = self.intrinsic_values
        n = iv.size()
        self.bc_s = []
        self.bc_s.append(NeumannBC(iv.value(1) - iv.value(0), Side.Lower))
        self.bc_s.append(NeumannBC(iv.value(n - 1) - iv.value(n - 2), Side.Upper))

    def calculate(self, results: OneAssetOptionResults) -> None:
        """Engine-specific scheme hook. Subclasses MUST override.

        Java parity: ``FDVanillaEngine.calculate`` (throws in the base).
        """
        raise NotImplementedError(
            "FDVanillaEngine.calculate: subclass must override this hook"
        )


__all__ = ["FDVanillaEngine", "PayoffFunction"]
