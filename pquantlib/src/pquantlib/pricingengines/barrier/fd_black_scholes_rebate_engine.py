"""FdBlackScholesRebateEngine — FD Black-Scholes barrier-rebate helper engine.

# C++ parity: ql/pricingengines/barrier/fdblackscholesrebateengine.{hpp,cpp}
# (v1.43) — ``class FdBlackScholesRebateEngine : public BarrierOption::engine``.

Prices *only* the rebate leg of a barrier option: the terminal payoff is a
``CashOrNothingPayoff(Call, 0.0, rebate)`` — i.e. the rebate is paid at expiry
on every path that never touched the barrier — and the barrier itself enters
as an ``FdmDirichletBoundary`` pinned at ``rebate`` on the knocked-out face of
the mesh.

Used standalone, and by ``FdBlackScholesBarrierEngine`` to assemble knock-*in*
prices via in-out parity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend
from pquantlib.exercise import Exercise
from pquantlib.instruments.barrier_option import BarrierOptionArguments, BarrierType
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    BoundaryConditionSide,
    FdmBoundaryCondition,
)
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver import NULL_REAL
from pquantlib.methods.finitedifferences.solvers.fdm_black_scholes_solver import (
    FdmBlackScholesSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_dirichlet_boundary import (
    FdmDirichletBoundary,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmLogInnerValue,
)
from pquantlib.payoffs import CashOrNothingPayoff, OptionType, StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class FdBlackScholesRebateEngine(
    GenericEngine[BarrierOptionArguments, OneAssetOptionResults]
):
    """Finite-difference Black-Scholes barrier-rebate engine.

    # C++ parity: ``class FdBlackScholesRebateEngine``.

    C++ has two constructors differing only by a ``DividendSchedule`` in
    second position; Python folds that into the keyword-only ``dividends``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        t_grid: int = 100,
        x_grid: int = 100,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
        local_vol: bool = False,
        illegal_local_vol_overwrite: float = -NULL_REAL,
        *,
        dividends: Sequence[Dividend] = (),
    ) -> None:
        super().__init__(BarrierOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._dividends: list[Dividend] = list(dividends)
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._damping_steps: int = damping_steps
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Douglas()``.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.douglas()
        )
        self._local_vol: bool = local_vol
        self._illegal_local_vol_overwrite: float = illegal_local_vol_overwrite
        # C++ parity: ``registerWith(process_)``.
        process.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``FdBlackScholesRebateEngine::calculate`` (cpp:68-145)."""
        args = self._arguments
        results = self._results

        # 1. Mesher
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        assert args.barrier is not None
        assert args.rebate is not None
        assert args.barrier_type is not None
        barrier_type: BarrierType = args.barrier_type

        maturity = self._process.time(args.exercise.last_date())

        x_min: float | None = None
        x_max: float | None = None
        if barrier_type in (BarrierType.DownIn, BarrierType.DownOut):
            x_min = math.log(args.barrier)
        if barrier_type in (BarrierType.UpIn, BarrierType.UpOut):
            x_max = math.log(args.barrier)

        equity_mesher = FdmBlackScholesMesher(
            self._x_grid,
            self._process,
            maturity,
            payoff.strike(),
            x_min,
            x_max,
            0.0001,
            1.5,
            None,
            self._dividends,
            None,
            0.0,
        )
        mesher = FdmMesherComposite(equity_mesher)

        # 2. Calculator
        rebate_payoff = CashOrNothingPayoff(OptionType.Call, 0.0, args.rebate)
        calculator = FdmLogInnerValue(rebate_payoff, mesher, 0)

        # 3. Step conditions
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "only european style option are supported",
        )

        conditions = FdmStepConditionComposite.vanilla_composite(
            self._dividends,
            args.exercise,
            mesher,
            calculator,
            self._process.risk_free_rate().reference_date(),
            self._process.risk_free_rate().day_counter(),
        )

        # 4. Boundary conditions
        boundaries: list[FdmBoundaryCondition] = []
        if barrier_type in (BarrierType.DownIn, BarrierType.DownOut):
            boundaries.append(
                FdmDirichletBoundary(mesher, args.rebate, 0, BoundaryConditionSide.LOWER)
            )
        if barrier_type in (BarrierType.UpIn, BarrierType.UpOut):
            boundaries.append(
                FdmDirichletBoundary(mesher, args.rebate, 0, BoundaryConditionSide.UPPER)
            )

        # 5. Solver
        solver_desc = FdmSolverDesc(
            mesher=mesher,
            condition=conditions,
            calculator=calculator.avg_inner_value,
            maturity=maturity,
            time_steps=self._t_grid,
            damping_steps=self._damping_steps,
            bc_set=tuple(boundaries),
        )

        solver = FdmBlackScholesSolver(
            self._process,
            payoff.strike(),
            solver_desc,
            self._scheme_desc,
            self._local_vol,
            self._illegal_local_vol_overwrite,
        )

        spot = self._process.x0()
        results.value = solver.value_at(spot)
        results.delta = solver.delta_at(spot)
        results.gamma = solver.gamma_at(spot)
        results.theta = solver.theta_at(spot)


__all__ = ["FdBlackScholesRebateEngine"]
