"""FdBlackScholesShoutEngine — finite-difference Black-Scholes shout option engine.

# C++ parity: ql/pricingengines/vanilla/fdblackscholesshoutengine.{hpp,cpp}
# (v1.43) — ``class FdBlackScholesShoutEngine : public VanillaOption::engine``.

A *shout* option lets the holder lock in the intrinsic value once, at a time
of their choosing, while keeping the upside. The FD formulation is the vanilla
one with the early-exercise value replaced by
:class:`~pquantlib.methods.finitedifferences.utilities.fdm_shout_log_inner_value_calculator.FdmShoutLogInnerValueCalculator`,
which prices the residual option analytically at each node.

Cash dividends always go through the **escrowed** model here: the spot is
shifted by ``EscrowedDividendAdjustment`` and the schedule handed to the step
conditions carries zero amounts (so the dividend dates become stopping times
without dropping the grid twice).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend, FixedDividend
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_black_scholes_solver import (
    FdmBlackScholesSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.escrowed_dividend_adjustment import (
    EscrowedDividendAdjustment,
)
from pquantlib.methods.finitedifferences.utilities.fdm_shout_log_inner_value_calculator import (
    FdmShoutLogInnerValueCalculator,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class FdBlackScholesShoutEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Finite-difference Black-Scholes shout option engine.

    # C++ parity: ``class FdBlackScholesShoutEngine``.

    C++ has two constructors, the second inserting a ``DividendSchedule``
    after the process; Python folds that into the keyword-only ``dividends``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        t_grid: int = 100,
        x_grid: int = 100,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
        *,
        dividends: Sequence[Dividend] = (),
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._dividends: list[Dividend] = list(dividends)
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._damping_steps: int = damping_steps
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Douglas()``.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.douglas()
        )
        # C++ parity: ``registerWith(process_)``.
        process.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``FdBlackScholesShoutEngine::calculate`` (cpp:62-131)."""
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        exercise = args.exercise

        exercise_date = exercise.last_date()
        maturity = self._process.time(exercise_date)
        settlement_date = self._process.risk_free_rate().reference_date()

        escrowed_dividend_adj = EscrowedDividendAdjustment(
            self._dividends,
            self._process.risk_free_rate(),
            self._process.dividend_yield(),
            self._process.time,
            maturity,
        )

        div_adj = escrowed_dividend_adj.dividend_adjustment(
            self._process.time(settlement_date)
        )

        qassert.require(
            self._process.x0() + div_adj > 0.0, "spot minus dividends becomes negative"
        )

        qassert.require(
            isinstance(args.payoff, PlainVanillaPayoff), "non plain vanilla payoff given"
        )
        assert isinstance(args.payoff, PlainVanillaPayoff)
        payoff: PlainVanillaPayoff = args.payoff

        # C++ parity: the mesher gets an *empty* dividend schedule; the whole
        # dividend effect is carried by ``divAdj`` as the spot adjustment.
        mesher = FdmMesherComposite(
            FdmBlackScholesMesher(
                self._x_grid,
                self._process,
                maturity,
                payoff.strike(),
                None,
                None,
                0.0001,
                1.5,
                (payoff.strike(), 0.1),
                (),
                None,
                div_adj,
            )
        )

        inner_value_calculator = FdmShoutLogInnerValueCalculator(
            self._process.black_volatility(),
            escrowed_dividend_adj,
            maturity,
            payoff,
            mesher,
            0,
        )

        # C++ parity: the step conditions see the dividend *dates* with zero
        # amounts, so they become stopping times but move nothing.
        zero_dividend_schedule: list[Dividend] = [
            FixedDividend(0.0, cf.date()) for cf in self._dividends
        ]

        conditions = FdmStepConditionComposite.vanilla_composite(
            zero_dividend_schedule,
            exercise,
            mesher,
            inner_value_calculator,
            self._process.risk_free_rate().reference_date(),
            self._process.risk_free_rate().day_counter(),
        )

        solver_desc = FdmSolverDesc(
            mesher=mesher,
            condition=conditions,
            calculator=inner_value_calculator.avg_inner_value,
            maturity=maturity,
            time_steps=self._t_grid,
            damping_steps=self._damping_steps,
        )

        solver = FdmBlackScholesSolver(
            self._process, payoff.strike(), solver_desc, self._scheme_desc
        )

        spot = self._process.x0() + div_adj

        results.value = solver.value_at(spot)
        results.delta = solver.delta_at(spot)
        results.gamma = solver.gamma_at(spot)
        results.theta = solver.theta_at(spot)


__all__ = ["FdBlackScholesShoutEngine"]
