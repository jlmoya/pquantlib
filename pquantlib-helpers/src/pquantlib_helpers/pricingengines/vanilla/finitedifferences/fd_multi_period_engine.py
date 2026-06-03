"""FDMultiPeriodEngine — FD engine that steps between event (dividend) dates.

# Retired-API compat layer — see fd_vanilla_engine docstring.

Java parity: ``org.jquantlib.pricingengines.vanilla.finitedifferences.FDMultiPeriodEngine``.

Extends :class:`FDVanillaEngine` to roll back across a set of "stopping" times
(the dividend dates), re-applying an engine-specific intermediate step at each.
The rollback uses the scalar
:class:`~pquantlib_helpers.methods.finitedifferences.finite_difference_model.StandardFiniteDifferenceModel`.
The value + delta + gamma are read off the grid centre via
:class:`~pquantlib_helpers.math.sampled_curve.SampledCurve`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pquantlib import qassert
from pquantlib_helpers.math.sampled_curve import SampledCurve
from pquantlib_helpers.methods.finitedifferences.finite_difference_model import (
    StandardFiniteDifferenceModel,
)
from pquantlib_helpers.methods.finitedifferences.step_condition import NullCondition
from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_vanilla_engine import (
    FDVanillaEngine,
)

if TYPE_CHECKING:
    from pquantlib.cashflows.dividend import Dividend
    from pquantlib.instruments.one_asset_option import OneAssetOptionResults
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )
    from pquantlib_helpers.instruments.dividend_vanilla_option import (
        DividendVanillaOptionArguments,
    )
    from pquantlib_helpers.methods.finitedifferences.finite_difference_model import (
        StepCondition,
    )

_DATE_TOLERANCE = 1e-6


class FDMultiPeriodEngine(FDVanillaEngine):
    """FD engine rolling back across event (dividend) dates.

    Java parity: ``FDMultiPeriodEngine``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        time_steps: int,
        grid_points: int,
        time_dependent: bool,
    ) -> None:
        """Bind the process + grid sizing; record steps-per-period.

        Java parity — the parameter-name SWAP. ``FDMultiPeriodEngine``'s Java
        constructor declares its middle two parameters as ``(gridPoints,
        timeSteps)`` — the reverse of its ``FDVanillaEngine`` superclass
        ``(timeSteps, gridPoints)`` — and is reached from
        ``FDDividendEngineBase`` via ``super(process, timeSteps, gridPoints,
        timeDependent)`` (FDDividendEngineBase.java:85). Positionally, that maps
        the caller's ``gridPoints`` into the Java ``timeSteps`` parameter, and
        Java then sets ``timeStepPerPeriod = timeSteps`` (FDMultiPeriodEngine.java:92).
        Net effect: ``timeStepPerPeriod`` is bound to the *gridPoints* value, and
        the ``timeSteps`` ctor argument is a dead parameter on the multi-period
        dividend path (only ``gridPoints`` drives both the spatial grid and the
        steps-per-period). This is WHY the Java European result is bit-invariant
        to ``timeSteps``. We keep the declared ``(time_steps, grid_points)``
        order on our own signature for readability, then reproduce the swap here
        so ``_time_step_per_period`` tracks ``grid_points`` exactly as Java does.
        """
        super().__init__(process, time_steps, grid_points, time_dependent)
        self.events: list[Dividend] = []
        self.stopping_times: list[float] = []
        self.prices: SampledCurve = SampledCurve(0)
        self.step_condition: StepCondition = NullCondition()
        self.model: StandardFiniteDifferenceModel | None = None
        # Java parity: timeStepPerPeriod := gridPoints (see the ctor swap above).
        self._time_step_per_period: int = grid_points

    # ------------------------------------------------------------------
    def setup_arguments_with_schedule(
        self,
        args: DividendVanillaOptionArguments,
        schedule: list[Dividend],
    ) -> None:
        """Read base arguments + map each event date to a stopping time.

        Java parity: ``setupArguments(Arguments, List<Event>)``.
        """
        super().setup_arguments(args)
        self.events = schedule
        self.stopping_times = []
        for event in self.events:
            self.stopping_times.append(self.process.time(event.date()))

    def get_dividend_time(self, i: int) -> float:
        """Stopping time of event ``i``. Java parity: ``getDividendTime``."""
        return self.stopping_times[i]

    def initialize_model(self) -> None:
        """Wire the Crank-Nicolson system model. Java parity: ``initializeModel``."""
        assert self.finite_difference_operator is not None
        self.model = StandardFiniteDifferenceModel(
            self.finite_difference_operator, self.bc_s
        )

    def initialize_step_condition(self) -> None:
        """Default to a null step condition. Java parity: ``initializeStepCondition``."""
        self.step_condition = NullCondition()

    @abstractmethod
    def execute_intermediate_step(self, step: int) -> None:
        """Engine-specific action at event ``step``."""

    def calculate(self, results: OneAssetOptionResults) -> None:  # noqa: PLR0915
        """Roll back across the event dates and read centre value + greeks.

        Java parity: ``FDMultiPeriodEngine.calculate``.
        """
        date_number = len(self.stopping_times)
        last_date_is_res_time = False
        first_index = -1
        last_index = date_number - 1
        first_date_is_zero = False
        first_non_zero_date = self.get_residual_time()

        if date_number > 0:
            qassert.require(
                self.get_dividend_time(0) >= 0, "first date cannot be negative"
            )
            if (
                self.get_dividend_time(0)
                < self.get_residual_time() * _DATE_TOLERANCE
            ):
                first_date_is_zero = True
                first_index = 0
                if date_number >= 2:
                    first_non_zero_date = self.get_dividend_time(1)
            if (
                abs(self.get_dividend_time(last_index) - self.get_residual_time())
                < _DATE_TOLERANCE
            ):
                last_date_is_res_time = True
                last_index = date_number - 2

            if not first_date_is_zero:
                first_non_zero_date = self.get_dividend_time(0)

            if date_number >= 2:
                for j in range(1, date_number):
                    qassert.require(
                        self.get_dividend_time(j - 1) < self.get_dividend_time(j),
                        "dates must be in strictly increasing order",
                    )

        dt = self.get_residual_time() / (self._time_step_per_period * (date_number + 1))
        # Ensure dt is always smaller than the first non-zero date.
        if first_non_zero_date <= dt:
            dt = first_non_zero_date / 2.0

        self.set_grid_limits()
        self.initialize_initial_condition()
        self.initialize_operator()
        self.initialize_boundary_conditions()
        self.initialize_model()
        self.initialize_step_condition()

        self.prices = self.intrinsic_values.clone()
        assert self.model is not None

        if last_date_is_res_time:
            self.execute_intermediate_step(date_number - 1)

        j = last_index
        while True:
            begin_date = (
                self.get_residual_time()
                if j == date_number - 1
                else self.get_dividend_time(j + 1)
            )
            end_date = self.get_dividend_time(j) if j >= 0 else dt

            self.prices.set_values(
                self.model.rollback(
                    self.prices.values(),
                    begin_date,
                    end_date,
                    self._time_step_per_period,
                    self.step_condition,
                )
            )

            if j >= 0:
                self.execute_intermediate_step(j)

            j -= 1
            if j < first_index:
                break

        self.prices.set_values(
            self.model.rollback(self.prices.values(), dt, 0.0, 1, self.step_condition)
        )

        if first_date_is_zero:
            self.execute_intermediate_step(0)

        results.value = self.prices.value_at_center()
        results.delta = self.prices.first_derivative_at_center()
        results.gamma = self.prices.second_derivative_at_center()
        results.additional_results["priceCurve"] = self.prices


__all__ = ["FDMultiPeriodEngine"]
