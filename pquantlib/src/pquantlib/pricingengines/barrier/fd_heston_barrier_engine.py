"""FdHestonBarrierEngine — finite-differences Heston barrier-option engine.

# C++ parity: ql/pricingengines/barrier/fdhestonbarrierengine.{hpp,cpp} (v1.43)
# — ``class FdHestonBarrierEngine : public GenericModelEngine<HestonModel,
#    BarrierOption::arguments, BarrierOption::results>``.

Solves the knock-out problem directly: the equity mesh is clamped at
``log(barrier)`` on the barrier side and an :class:`FdmDirichletBoundary`
holding ``arguments_.rebate`` pins that face.

For *In* barriers the engine then applies in-out parity, and this is the part a
port silently drops: it prices a plain vanilla with
:class:`FdHestonVanillaEngine` **and** a rebate leg with
:class:`FdHestonRebateEngine`, on a deliberately coarser grid

    xGrid = max(20, xGrid // 4),  vGrid = max(10, vGrid // 4),
    dampingSteps = min(1, dampingSteps // 2) if dampingSteps > 0 else 0

— note ``min``, not ``max``, so four damping steps become one — and reports
``vanilla + rebate - knockout`` for the value *and* for delta, gamma and theta.

Unlike the plain vanilla engine this one builds its step-condition composite by
hand (dividend handler only, no exercise dispatch) and requires European
exercise.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

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
from pquantlib.models.equity.heston_model import HestonModel
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.barrier.fd_heston_rebate_engine import FdHestonRebateEngine
from pquantlib.pricingengines.generic_model_engine import GenericModelEngine
from pquantlib.pricingengines.vanilla.fd_heston_vanilla_engine import (
    FdHestonVanillaEngine,
    process_helper,
)
from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
    LocalVolTermStructure,
)

_DOWN_TYPES = (BarrierType.DownIn, BarrierType.DownOut)
_UP_TYPES = (BarrierType.UpIn, BarrierType.UpOut)
_IN_TYPES = (BarrierType.DownIn, BarrierType.UpIn)


class FdHestonBarrierEngine(
    GenericModelEngine[HestonModel, BarrierOptionArguments, OneAssetOptionResults]
):
    """Finite-differences Heston barrier-option engine.

    # C++ parity: ``class FdHestonBarrierEngine``.
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

    def calculate(self) -> None:  # noqa: PLR0915 — one C++ function, kept whole
        """# C++ parity: ``FdHestonBarrierEngine::calculate``
        # (fdhestonbarrierengine.cpp:68-207).
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

        # 2. Calculator
        calculator = FdmLogInnerValue(payoff, mesher, 0)

        # 3. Step conditions — hand-rolled, not vanillaComposite.
        step_conditions: list[StepCondition] = []
        stopping_times: list[list[float]] = []

        # 3.1 Step condition if discrete dividends
        dividend_condition = FdmDividendHandler(
            self._dividends,
            mesher,
            process.risk_free_rate().reference_date(),
            process.risk_free_rate().day_counter(),
            0,
        )
        if self._dividends:
            step_conditions.append(dividend_condition)
            # C++ parity: "this effectively excludes times after maturity".
            stopping_times.append(
                [min(maturity, t) for t in dividend_condition.dividend_times()]
            )

        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "only european style option are supported",
        )
        conditions = FdmStepConditionComposite(stopping_times, step_conditions)

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

        # 6. Calculate vanilla option and rebate for in-barriers
        if args.barrier_type in _IN_TYPES:
            # Calculate the vanilla option
            vanilla_option = VanillaOption(payoff, args.exercise)
            vanilla_option.set_pricing_engine(
                FdHestonVanillaEngine(
                    model,
                    self._dividends,
                    None,
                    self._t_grid,
                    self._x_grid,
                    self._v_grid,
                    self._damping_steps,
                    self._scheme_desc,
                )
            )
            # Calculate the rebate value
            rebate_option = BarrierOption(
                args.barrier_type, args.barrier, args.rebate, payoff, args.exercise
            )
            x_grid_min = 20
            v_grid_min = 10
            # C++ parity: ``std::min(Size(1), dampingSteps_/2)`` — min, not max.
            rebate_damping_steps = (
                min(1, self._damping_steps // 2) if self._damping_steps > 0 else 0
            )
            rebate_option.set_pricing_engine(
                FdHestonRebateEngine(
                    model,
                    self._dividends,
                    self._t_grid,
                    max(x_grid_min, self._x_grid // 4),
                    max(v_grid_min, self._v_grid // 4),
                    rebate_damping_steps,
                    self._scheme_desc,
                )
            )

            assert results.value is not None
            assert results.delta is not None
            assert results.gamma is not None
            assert results.theta is not None
            results.value = vanilla_option.npv() + rebate_option.npv() - results.value
            results.delta = vanilla_option.delta() + rebate_option.delta() - results.delta
            results.gamma = vanilla_option.gamma() + rebate_option.gamma() - results.gamma
            results.theta = vanilla_option.theta() + rebate_option.theta() - results.theta


__all__ = ["FdHestonBarrierEngine"]
