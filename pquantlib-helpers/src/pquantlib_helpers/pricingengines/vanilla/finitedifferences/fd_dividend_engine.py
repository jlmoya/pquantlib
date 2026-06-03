"""FD escrowed-dividend engines (base + Merton-73 spot model).

# Retired-API compat layer — see fd_vanilla_engine docstring.

Java parity: ``FDDividendEngineBase`` (abstract), ``FDDividendEngineMerton73``,
and the ``FDDividendEngine`` alias (``typedef FDDividendEngineMerton73
FDDividendEngine``).

The Merton-73 variant is the classic escrowed-dividend scheme: it shifts the
grid centre down by the present value of all future dividends, then at each
dividend date *scales* (not shifts) the grid by ``1 + PV(div)/center`` so the
PDE stays consistent. This is the engine the European/American FD dividend
helpers drive (the C++/Java ``FDDividendEuropeanEngine`` = Merton-73).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from pquantlib_helpers.pricingengines.vanilla.finitedifferences.fd_multi_period_engine import (
    FDMultiPeriodEngine,
)

if TYPE_CHECKING:
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )
    from pquantlib_helpers.instruments.dividend_vanilla_option import (
        DividendVanillaOptionArguments,
    )


class FDDividendEngineBase(FDMultiPeriodEngine):
    """Abstract base for the FD escrowed-dividend engines.

    Java parity: ``FDDividendEngineBase``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        time_steps: int = 100,
        grid_points: int = 100,
        time_dependent: bool = False,
    ) -> None:
        """Bind the process + grid sizing."""
        super().__init__(process, time_steps, grid_points, time_dependent)

    def setup_arguments(self, a: DividendVanillaOptionArguments) -> None:
        """Map the dividend cash-flows to events + stopping times.

        Java parity: ``FDDividendEngineBase.setupArguments``.
        """
        events = list(a.cash_flow)
        self.setup_arguments_with_schedule(a, events)

    def get_dividend_amount(self, i: int) -> float:
        """Cash amount of dividend ``i``. Java parity: ``getDividendAmount``."""
        return self.events[i].amount()

    def get_discounted_dividend(self, i: int) -> float:
        """PV-of-dividend ``i`` discounted by the r/q discount ratio.

        Java parity: ``getDiscountedDividend`` —
        ``amount * riskFreeDiscount(date) / dividendDiscount(date)``.
        """
        dividend = self.get_dividend_amount(i)
        date = self.events[i].date()
        discount = self.process.risk_free_rate().discount(
            date
        ) / self.process.dividend_yield().discount(date)
        return dividend * discount

    @abstractmethod
    def set_grid_limits(self) -> None:
        """Engine-specific grid layout."""

    @abstractmethod
    def execute_intermediate_step(self, step: int) -> None:
        """Engine-specific action at dividend ``step``."""


class FDDividendEngineMerton73(FDDividendEngineBase):
    """Merton-73 escrowed-dividend FD engine.

    Java parity: ``FDDividendEngineMerton73``. The x-axis is the underlying NPV
    minus the value of the paid dividends; dividends are applied by *scaling*
    the grid (so they scale with the underlying), not shifting.
    """

    def set_grid_limits(self) -> None:
        """Centre the grid on spot minus the PV of all future dividends.

        Java parity: ``FDDividendEngineMerton73.setGridLimits``.
        """
        paid_dividends = 0.0
        for i in range(len(self.events)):
            if self.get_dividend_time(i) >= 0.0:
                paid_dividends += self.get_discounted_dividend(i)
        self.set_grid_limits_at(
            self.process.state_variable().value() - paid_dividends,
            self.get_residual_time(),
        )
        self.ensure_strike_in_grid()

    def execute_intermediate_step(self, step: int) -> None:
        """Scale the grid by ``1 + PV(div)/center`` at dividend ``step``.

        Java parity: ``FDDividendEngineMerton73.executeIntermediateStep``.
        """
        scale_factor = self.get_discounted_dividend(step) / self.center + 1.0
        self.s_min *= scale_factor
        self.s_max *= scale_factor
        self.center *= scale_factor

        self.intrinsic_values.scale_grid(scale_factor)
        self.initialize_initial_condition()
        self.prices.scale_grid(scale_factor)
        self.initialize_operator()
        self.initialize_model()

        self.initialize_step_condition()
        self.step_condition.apply_to(
            self.prices.values(), self.get_dividend_time(step)
        )


class FDDividendEngine(FDDividendEngineMerton73):
    """Default FD dividend engine (``typedef FDDividendEngineMerton73``).

    Java parity: ``FDDividendEngine`` (a thin alias subclass).
    """


__all__ = [
    "FDDividendEngine",
    "FDDividendEngineBase",
    "FDDividendEngineMerton73",
]
