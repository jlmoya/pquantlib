"""FdBlackScholesBarrierEngine — FD Black/Scholes barrier-option engine.

# C++ parity: ql/pricingengines/barrier/fdblackscholesbarrierengine.{hpp,cpp}
# (v1.43) — ``class FdBlackScholesBarrierEngine : public BarrierOption::engine``.

Knock-**out** options are priced directly: the mesh is clipped at
``log(barrier)`` on the knocked side and an ``FdmDirichletBoundary`` pins that
face at the rebate.

Knock-**in** options are then assembled by in-out parity, exactly as C++ does:

    in = vanilla + rebate_leg - out

with the vanilla leg priced by ``FdBlackScholesVanillaEngine`` (damping steps
forced to 0) and the rebate leg by ``FdBlackScholesRebateEngine`` on a coarser
mesh (``max(50, xGrid/5)`` points, ``min(1, dampingSteps/2)`` damping steps).
Every one of those knobs is load-bearing for the number, so none of them is
"tidied up" here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend
from pquantlib.exercise import Exercise
from pquantlib.instruments.barrier_option import (
    BarrierOption,
    BarrierOptionArguments,
    BarrierType,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.instruments.vanilla_option import VanillaOption
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
from pquantlib.methods.finitedifferences.step_conditions.step_condition import StepCondition
from pquantlib.methods.finitedifferences.utilities.fdm_dirichlet_boundary import (
    FdmDirichletBoundary,
)
from pquantlib.methods.finitedifferences.utilities.fdm_dividend_handler import (
    FdmDividendHandler,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmLogInnerValue,
)
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.barrier.fd_black_scholes_rebate_engine import (
    FdBlackScholesRebateEngine,
)
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.fd_black_scholes_vanilla_engine import (
    FdBlackScholesVanillaEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class FdBlackScholesBarrierEngine(
    GenericEngine[BarrierOptionArguments, OneAssetOptionResults]
):
    """Finite-difference Black/Scholes barrier-option engine.

    # C++ parity: ``class FdBlackScholesBarrierEngine``.

    C++ has two constructors differing only by a ``DividendSchedule`` in
    second position; Python folds that into the keyword-only ``dividends``.
    """

    #: # C++ parity: ``const Size min_grid_size = 50`` inside ``calculate()``.
    MIN_GRID_SIZE: int = 50

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

    def _triggered(self, underlying: float) -> bool:
        """# C++ parity: ``BarrierOption::engine::triggered`` (barrieroption.cpp:126-137)."""
        barrier_type = self._arguments.barrier_type
        barrier = self._arguments.barrier
        assert barrier is not None
        if barrier_type in (BarrierType.DownIn, BarrierType.DownOut):
            return underlying < barrier
        if barrier_type in (BarrierType.UpIn, BarrierType.UpOut):
            return underlying > barrier
        qassert.fail("unknown type")

    def calculate(self) -> None:  # noqa: PLR0915
        """# C++ parity: ``FdBlackScholesBarrierEngine::calculate`` (cpp:73-207).

        Kept as one function because C++ is one function: splitting the
        knock-in assembly out would hide that the vanilla leg's zero damping
        steps and the rebate leg's coarse mesh are chosen *inside* this body.
        """
        args = self._arguments
        results = self._results

        # 1. Mesher
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "non-striked type payoff given"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff
        qassert.require(payoff.strike() > 0.0, "strike must be positive")

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "only european style option are supported",
        )
        assert args.barrier is not None
        assert args.rebate is not None
        assert args.barrier_type is not None
        barrier_type: BarrierType = args.barrier_type

        spot = self._process.x0()
        qassert.require(spot > 0.0, "negative or null underlying given")
        qassert.require(not self._triggered(spot), "barrier touched")

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
        calculator = FdmLogInnerValue(payoff, mesher, 0)

        # 3. Step conditions
        step_conditions: list[StepCondition] = []
        stopping_times: list[list[float]] = []

        # 3.1 Step condition if discrete dividends
        # C++ builds the handler unconditionally, then only uses it when the
        # schedule is non-empty.
        dividend_condition = FdmDividendHandler(
            self._dividends,
            mesher,
            self._process.risk_free_rate().reference_date(),
            self._process.risk_free_rate().day_counter(),
            0,
        )
        if self._dividends:
            step_conditions.append(dividend_condition)
            # this effectively excludes times after maturity
            stopping_times.append(
                [min(maturity, t) for t in dividend_condition.dividend_times()]
            )

        conditions = FdmStepConditionComposite(stopping_times, step_conditions)

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

        results.value = solver.value_at(spot)
        results.delta = solver.delta_at(spot)
        results.gamma = solver.gamma_at(spot)
        results.theta = solver.theta_at(spot)

        # 6. Calculate vanilla option and rebate for in-barriers
        if barrier_type in (BarrierType.DownIn, BarrierType.UpIn):
            # Calculate the vanilla option
            vanilla_option = VanillaOption(payoff, args.exercise)
            vanilla_option.set_pricing_engine(
                FdBlackScholesVanillaEngine(
                    self._process,
                    self._t_grid,
                    self._x_grid,
                    0,  # dampingSteps
                    self._scheme_desc,
                    self._local_vol,
                    self._illegal_local_vol_overwrite,
                    dividends=self._dividends,
                )
            )

            # Calculate the rebate value
            rebate_option = BarrierOption(
                barrier_type, args.barrier, args.rebate, payoff, args.exercise
            )

            rebate_damping_steps = (
                min(1, self._damping_steps // 2) if self._damping_steps > 0 else 0
            )

            rebate_option.set_pricing_engine(
                FdBlackScholesRebateEngine(
                    self._process,
                    self._t_grid,
                    max(self.MIN_GRID_SIZE, self._x_grid // 5),
                    rebate_damping_steps,
                    self._scheme_desc,
                    self._local_vol,
                    self._illegal_local_vol_overwrite,
                    dividends=self._dividends,
                )
            )

            assert results.value is not None
            assert results.delta is not None
            assert results.gamma is not None
            assert results.theta is not None
            results.value = vanilla_option.npv() + rebate_option.npv() - results.value
            results.delta = (
                vanilla_option.delta() + rebate_option.delta() - results.delta
            )
            results.gamma = (
                vanilla_option.gamma() + rebate_option.gamma() - results.gamma
            )
            results.theta = (
                vanilla_option.theta() + rebate_option.theta() - results.theta
            )


__all__ = ["FdBlackScholesBarrierEngine"]
