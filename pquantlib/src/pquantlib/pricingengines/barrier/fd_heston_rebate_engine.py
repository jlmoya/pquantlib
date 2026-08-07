"""FdHestonRebateEngine — FD Heston barrier-option rebate helper engine.

# C++ parity: ql/pricingengines/barrier/fdhestonrebateengine.{hpp,cpp} (v1.43)
# — ``class FdHestonRebateEngine : public GenericModelEngine<HestonModel,
#    BarrierOption::arguments, BarrierOption::results>``.

Prices only the rebate leg of a barrier option: the terminal payoff is a
``CashOrNothingPayoff(Call, 0.0, rebate)`` — strike zero, so it pays the rebate
at every node that has not been knocked out — and the barrier face is pinned to
the rebate by an :class:`FdmDirichletBoundary`. The engine is used standalone
and as the rebate leg inside :class:`FdHestonBarrierEngine` for *In* barriers.

Note the mesher is the same as the barrier engine's: scale factor ``1.5``,
``eps = 0.0001``, **no** critical point (an explicitly null ``cPoint``), and
``xMin``/``xMax`` clamped at ``log(barrier)`` on whichever side the barrier
lives.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

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
from pquantlib.payoffs import CashOrNothingPayoff, OptionType, StrikedTypePayoff
from pquantlib.pricingengines.generic_model_engine import GenericModelEngine
from pquantlib.pricingengines.vanilla.fd_heston_vanilla_engine import process_helper
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)

_DOWN_TYPES = (BarrierType.DownIn, BarrierType.DownOut)
_UP_TYPES = (BarrierType.UpIn, BarrierType.UpOut)


class FdHestonRebateEngine(
    GenericModelEngine[HestonModel, BarrierOptionArguments, OneAssetOptionResults]
):
    """Finite-differences Heston barrier-option rebate helper engine.

    # C++ parity: ``class FdHestonRebateEngine``.
    """

    def __init__(
        self,
        model: HestonModel,
        dividends: Sequence[Dividend] | None = None,
        t_grid: int = 100,
        x_grid: int = 100,
        v_grid: int = 50,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
        leverage_fct: LocalVolTermStructure | None = None,
        mixing_factor: float = 1.0,
    ) -> None:
        super().__init__(BarrierOptionArguments(), OneAssetOptionResults(), model)
        self._dividends: list[Dividend] = list(dividends) if dividends else []
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
        """# C++ parity: ``FdHestonRebateEngine::calculate``
        # (fdhestonrebateengine.cpp:69-159).
        """
        args = self._arguments
        results = self._results
        assert args.exercise is not None
        assert args.barrier_type is not None
        assert args.barrier is not None
        assert args.rebate is not None

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

        x_min = math.log(args.barrier) if args.barrier_type in _DOWN_TYPES else None
        x_max = math.log(args.barrier) if args.barrier_type in _UP_TYPES else None

        helper = process_helper(
            process.s0(),
            process.dividend_yield(),
            process.risk_free_rate(),
            v_mesher.vola_estimate(),
        )
        equity_mesher = FdmBlackScholesMesher(
            self._x_grid,
            helper,
            maturity,
            payoff.strike(),
            x_min,
            x_max,
            0.0001,
            1.5,
            (None, None),
            self._dividends,
        )

        mesher = FdmMesherComposite(equity_mesher, v_mesher)

        # 2. Calculator — the rebate is a cash-or-nothing call struck at zero.
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
            process.risk_free_rate().reference_date(),
            process.risk_free_rate().day_counter(),
        )

        # 4. Boundary conditions
        boundaries: list[FdmBoundaryCondition] = []
        if args.barrier_type in _DOWN_TYPES:
            boundaries.append(
                FdmDirichletBoundary(mesher, args.rebate, 0, BoundaryConditionSide.LOWER)
            )
        if args.barrier_type in _UP_TYPES:
            boundaries.append(
                FdmDirichletBoundary(mesher, args.rebate, 0, BoundaryConditionSide.UPPER)
            )

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


__all__ = ["FdHestonRebateEngine"]
