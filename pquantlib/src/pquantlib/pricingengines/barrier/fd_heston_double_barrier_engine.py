"""FdHestonDoubleBarrierEngine — FD Heston double-barrier option engine.

# C++ parity: ql/pricingengines/barrier/fdhestondoublebarrierengine.{hpp,cpp}
# (v1.43) — ``class FdHestonDoubleBarrierEngine : public
#  GenericModelEngine<HestonModel, DoubleBarrierOption::arguments,
#  DoubleBarrierOption::results>``.

Only ``DoubleBarrier::KnockOut`` and European exercise are supported. The
equity mesh is clamped at ``log(barrier_lo)`` / ``log(barrier_hi)`` and *both*
Dirichlet faces are pinned to ``arguments_.rebate``.

Note the equity mesher call is the **short** ``FdmBlackScholesMesher``
overload — ``(xGrid, process, maturity, strike, xMin, xMax)`` — so it takes the
header defaults ``eps = 0.0001``, ``scaleFactor = 1.5`` and a null critical
point, rather than the explicit arguments the single-barrier engine passes.
The numbers coincide, but a port that copies the single-barrier call has to
copy the right defaults too.

There is no *In* branch and no dividend handling: C++ builds an empty
``FdmStepConditionComposite`` unconditionally, and this constructor has no
``DividendSchedule`` parameter at all.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.double_barrier_option import (
    DoubleBarrierOptionArguments,
    DoubleBarrierType,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    BoundaryConditionSide,
    FdmBoundaryCondition,
)
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_heston_variance_mesher import (
    FdmHestonLocalVolatilityVarianceMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_heston_solver import FdmHestonSolver
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
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_model_engine import GenericModelEngine
from pquantlib.pricingengines.vanilla.fd_heston_vanilla_engine import process_helper
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)


class FdHestonDoubleBarrierEngine(
    GenericModelEngine[HestonModel, DoubleBarrierOptionArguments, OneAssetOptionResults]
):
    """Finite-differences Heston double-barrier option engine.

    # C++ parity: ``class FdHestonDoubleBarrierEngine``.
    """

    def __init__(
        self,
        model: HestonModel,
        t_grid: int = 100,
        x_grid: int = 100,
        v_grid: int = 50,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
        leverage_fct: LocalVolTermStructure | None = None,
        mixing_factor: float = 1.0,
    ) -> None:
        super().__init__(DoubleBarrierOptionArguments(), OneAssetOptionResults(), model)
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._v_grid: int = v_grid
        self._damping_steps: int = damping_steps
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._leverage_fct: LocalVolTermStructure | None = leverage_fct
        self._mixing_factor: float = mixing_factor

    def calculate(self) -> None:
        """# C++ parity: ``FdHestonDoubleBarrierEngine::calculate``
        # (fdhestondoublebarrierengine.cpp:49-122).
        """
        args = self._arguments
        results = self._results
        assert args.exercise is not None
        assert args.barrier_lo is not None
        assert args.barrier_hi is not None
        assert args.rebate is not None

        qassert.require(
            args.barrier_type == DoubleBarrierType.KnockOut,
            "only Knock-Out double barrier options are supported",
        )

        model = self.model()
        qassert.require(model is not None, "no model specified")
        assert model is not None

        # 1. Mesher
        process = model.process()
        maturity = process.time(args.exercise.last_date())

        # 1.1 The variance mesher
        t_grid_min = 5
        t_avg_steps = max(t_grid_min, self._t_grid // 50)
        v_mesher = FdmHestonLocalVolatilityVarianceMesher(
            self._v_grid,
            process,
            self._leverage_fct,
            maturity,
            t_avg_steps,
            0.0001,
            self._mixing_factor,
        )

        # 1.2 The equity mesher
        payoff = args.payoff
        qassert.require(isinstance(payoff, StrikedTypePayoff), "wrong payoff type given")
        assert isinstance(payoff, StrikedTypePayoff)

        x_min = math.log(args.barrier_lo)
        x_max = math.log(args.barrier_hi)

        helper = process_helper(
            process.s0(),
            process.dividend_yield(),
            process.risk_free_rate(),
            v_mesher.vola_estimate(),
        )
        # C++ parity: the SHORT overload, i.e. eps = 0.0001, scaleFactor = 1.5,
        # cPoint = (Null, Null) come from the header defaults.
        equity_mesher = FdmBlackScholesMesher(
            self._x_grid, helper, maturity, payoff.strike(), x_min, x_max
        )

        mesher = FdmMesherComposite(equity_mesher, v_mesher)

        # 2. Calculator
        calculator = FdmLogInnerValue(payoff, mesher, 0)

        # 3. Step conditions — always empty.
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "only european style option are supported",
        )
        conditions = FdmStepConditionComposite([], [])

        # 4. Boundary conditions — both faces held at the rebate.
        boundaries: list[FdmBoundaryCondition] = [
            FdmDirichletBoundary(mesher, args.rebate, 0, BoundaryConditionSide.LOWER),
            FdmDirichletBoundary(mesher, args.rebate, 0, BoundaryConditionSide.UPPER),
        ]

        # 5. Solver
        solver_desc = FdmSolverDesc(
            mesher,
            conditions,
            calculator.avg_inner_value,
            maturity,
            self._t_grid,
            self._damping_steps,
            tuple(boundaries),
        )
        solver = FdmHestonSolver(
            process,
            solver_desc,
            self._scheme_desc,
            None,
            self._leverage_fct,
            self._mixing_factor,
        )

        spot = process.s0().value()
        results.value = solver.value_at(spot, process.v0)
        results.delta = solver.delta_at(spot, process.v0)
        results.gamma = solver.gamma_at(spot, process.v0)
        results.theta = solver.theta_at(spot, process.v0)


__all__ = ["FdHestonDoubleBarrierEngine"]
