"""FDStepConditionEngine — control-variate FD engine via the system FD model.

# Retired-API compat layer — see fd_vanilla_engine docstring.

Java parity: ``org.jquantlib.pricingengines.vanilla.finitedifferences.FDStepConditionEngine``.

Extends :class:`FDVanillaEngine` to roll the option PDE *and* a European-reference
PDE forward as a 2-component system (via
:class:`~pquantlib_helpers.methods.finitedifferences.standard_system_finite_difference_model.StandardSystemFiniteDifferenceModel`),
then forms the answer as ``FD_option - FD_european + analytic_european`` — the
analytic European leg (:class:`~pquantlib.pricingengines.black_calculator.BlackCalculator`)
cancels the FD discretisation error of the European leg, a standard control
variate. The early-exercise step condition is supplied by the subclass
(:class:`~pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_american_condition.FDAmericanCondition`).
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING, cast

from pquantlib import qassert
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.black_calculator import BlackCalculator
from pquantlib_helpers.math.sampled_curve import SampledCurve
from pquantlib_helpers.methods.finitedifferences.boundary_condition import (
    BoundaryConditionSet,
)
from pquantlib_helpers.methods.finitedifferences.standard_system_finite_difference_model import (
    StandardSystemFiniteDifferenceModel,
)
from pquantlib_helpers.methods.finitedifferences.step_condition import (
    NullCondition,
    StepConditionSet,
)
from pquantlib_helpers.methods.finitedifferences.tridiagonal_operator import (
    TridiagonalOperator,
)
from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_vanilla_engine import (
    FDVanillaEngine,
)

if TYPE_CHECKING:
    from pquantlib.instruments.one_asset_option import OneAssetOptionResults
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )
    from pquantlib_helpers.methods.finitedifferences.boundary_condition import (
        BoundaryConditionLike,
    )
    from pquantlib_helpers.methods.finitedifferences.finite_difference_model import (
        StepCondition,
    )
    from pquantlib_helpers.methods.finitedifferences.mixed_scheme import (
        BoundaryCondition,
    )


class FDStepConditionEngine(FDVanillaEngine):
    """Control-variate FD engine using the 2-component system model.

    Java parity: ``FDStepConditionEngine``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        time_steps: int,
        grid_points: int,
        time_dependent: bool,
    ) -> None:
        """Bind the process + grid sizing."""
        super().__init__(process, time_steps, grid_points, time_dependent)
        self.step_condition: StepCondition = NullCondition()
        self.prices: SampledCurve = SampledCurve(0)
        self.control_prices: SampledCurve = SampledCurve(grid_points)
        self.control_operator: TridiagonalOperator | None = None
        self.control_bcs: list[BoundaryCondition] = []

    @abstractmethod
    def initialize_step_condition(self) -> None:
        """Install the early-exercise step condition."""

    def calculate(self, results: OneAssetOptionResults) -> None:
        """Run the control-variate 2-system rollback and form the answer.

        Java parity: ``FDStepConditionEngine.calculate``.
        """
        self.set_grid_limits()
        self.initialize_initial_condition()
        self.initialize_operator()
        self.initialize_boundary_conditions()
        self.initialize_step_condition()

        assert self.finite_difference_operator is not None

        self.prices = SampledCurve(self.intrinsic_values)
        self.control_prices = SampledCurve(self.intrinsic_values)
        # A deep copy of the FD operator: same coefficients, independent arrays.
        # Java parity: ``FDStepConditionEngine.calculate`` (jquantlib-final) takes
        # the same snapshot — ``new TridiagonalOperator(finiteDifferenceOperator)``
        # — at this point in calculate().  The control operator is therefore a
        # constant-coefficient snapshot of the BSM operator evaluated at t=T.
        # When ``time_dependent=True`` the *main* operator's coefficients are
        # refreshed each time step (via ``operator.set_time(t)``), but the control
        # operator has no ``time_setter`` and its coefficients NEVER update.  This
        # silently produces an inaccurate control-variate result for time-varying
        # vol/rate/dividend curves.  ``time_dependent=True`` is therefore NOT a
        # validated or accurate execution path for ``FDStepConditionEngine`` (it is
        # not covered by any reference probe).  The only validated path is
        # ``time_dependent=False`` (the constructor default), where both operators
        # hold the same constant-coefficient snapshot throughout the rollback.
        self.control_operator = TridiagonalOperator(
            low=self.finite_difference_operator.lower_diagonal(),
            mid=self.finite_difference_operator.diagonal(),
            high=self.finite_difference_operator.upper_diagonal(),
        )
        self.control_bcs = [self.bc_s[0], self.bc_s[1]]

        operator_set = [self.finite_difference_operator, self.control_operator]
        array_set = [self.prices.values(), self.control_prices.values()]

        # BoundaryConditionSet is typed over the structurally-identical
        # ``BoundaryConditionLike`` protocol; our lists hold the
        # ``BoundaryCondition`` protocol from ``mixed_scheme`` (same hook surface).
        bc_set = BoundaryConditionSet()
        bc_set.push_back(cast("list[BoundaryConditionLike]", self.bc_s))
        bc_set.push_back(cast("list[BoundaryConditionLike]", self.control_bcs))

        condition_set = StepConditionSet()
        condition_set.push_back(self.step_condition)
        condition_set.push_back(NullCondition())

        model = StandardSystemFiniteDifferenceModel(operator_set, bc_set)
        array_set = model.rollback(
            array_set, self.get_residual_time(), 0.0, self.time_steps, condition_set
        )

        self.prices.set_values(array_set[0])
        self.control_prices.set_values(array_set[1])

        qassert.require(
            isinstance(self.payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(self.payoff, StrikedTypePayoff)
        striked_payoff = self.payoff

        assert self.exercise_date is not None
        variance = self.process.black_volatility().black_variance(
            self.exercise_date, striked_payoff.strike()
        )
        dividend_discount = self.process.dividend_yield().discount(self.exercise_date)
        risk_free_discount = self.process.risk_free_rate().discount(self.exercise_date)
        spot = self.process.state_variable().value()
        forward_price = spot * dividend_discount / risk_free_discount

        black = BlackCalculator(
            striked_payoff, forward_price, math.sqrt(variance), risk_free_discount
        )

        results.value = (
            self.prices.value_at_center()
            - self.control_prices.value_at_center()
            + black.value()
        )
        results.delta = (
            self.prices.first_derivative_at_center()
            - self.control_prices.first_derivative_at_center()
            + black.delta(spot)
        )
        results.gamma = (
            self.prices.second_derivative_at_center()
            - self.control_prices.second_derivative_at_center()
            + black.gamma(spot)
        )
        results.additional_results["priceCurve"] = self.prices


__all__ = ["FDStepConditionEngine"]
