"""Fd2dBlackScholesVanillaEngine — two-dimensional FD Black-Scholes basket engine.

# C++ parity: ql/pricingengines/basket/fd2dblackscholesvanillaengine.{hpp,cpp}
# (v1.43) — ``class Fd2dBlackScholesVanillaEngine : public BasketOption::engine``.

Both underlyings get their own ``FdmBlackScholesMesher`` (anchored on their own
spot, not on a strike — a basket has no single strike), and the correlation
enters through the mixed second-derivative term of ``Fdm2dBlackScholesOp``.
The default scheme is ``Hundsdorfer``, the ADI splitting that handles that
mixed term, **not** ``Douglas``.

The reported Greeks are the *basket* ones: delta is the sum of the two
single-asset deltas and gamma is ``gammaX + gammaY + 2 gammaXY``.
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.instruments.basket_option import BasketOptionResults, BasketPayoff
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver import NULL_REAL
from pquantlib.methods.finitedifferences.solvers.fdm_2d_black_scholes_solver import (
    Fdm2dBlackScholesSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmLogBasketInnerValue,
)
from pquantlib.option import OptionArguments
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class Fd2dBlackScholesVanillaEngine(GenericEngine[OptionArguments, BasketOptionResults]):
    """Two-dimensional finite-differences Black-Scholes basket engine.

    # C++ parity: ``class Fd2dBlackScholesVanillaEngine``.
    """

    def __init__(
        self,
        p1: GeneralizedBlackScholesProcess,
        p2: GeneralizedBlackScholesProcess,
        correlation: float,
        x_grid: int = 100,
        y_grid: int = 100,
        t_grid: int = 50,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
        local_vol: bool = False,
        illegal_local_vol_overwrite: float = -NULL_REAL,
    ) -> None:
        super().__init__(OptionArguments(), BasketOptionResults())
        self._p1: GeneralizedBlackScholesProcess = p1
        self._p2: GeneralizedBlackScholesProcess = p2
        self._correlation: float = correlation
        self._x_grid: int = x_grid
        self._y_grid: int = y_grid
        self._t_grid: int = t_grid
        self._damping_steps: int = damping_steps
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()``.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )
        self._local_vol: bool = local_vol
        self._illegal_local_vol_overwrite: float = illegal_local_vol_overwrite
        # C++ parity: ``registerWith(p1)`` / ``registerWith(p2)``.
        p1.register_with(self)
        p2.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``Fd2dBlackScholesVanillaEngine::calculate`` (cpp:52-109)."""
        args = self._arguments
        results = self._results

        # 1. Payoff
        qassert.require(isinstance(args.payoff, BasketPayoff), "basket payoff expected")
        assert isinstance(args.payoff, BasketPayoff)
        payoff: BasketPayoff = args.payoff
        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None

        # 2. Mesher
        maturity = self._p1.time(args.exercise.last_date())
        em1 = FdmBlackScholesMesher(
            self._x_grid,
            self._p1,
            maturity,
            self._p1.x0(),
            None,
            None,
            0.0001,
            1.5,
            (self._p1.x0(), 0.1),
        )
        em2 = FdmBlackScholesMesher(
            self._y_grid,
            self._p2,
            maturity,
            self._p2.x0(),
            None,
            None,
            0.0001,
            1.5,
            (self._p2.x0(), 0.1),
        )

        mesher = FdmMesherComposite(em1, em2)

        # 3. Calculator
        calculator = FdmLogBasketInnerValue(payoff, mesher)

        # 4. Step conditions
        conditions = FdmStepConditionComposite.vanilla_composite(
            (),
            args.exercise,
            mesher,
            calculator,
            self._p1.risk_free_rate().reference_date(),
            self._p1.risk_free_rate().day_counter(),
        )

        # 5. Boundary conditions: C++ passes a default-constructed set.
        # 6. Solver
        solver_desc = FdmSolverDesc(
            mesher=mesher,
            condition=conditions,
            calculator=calculator.avg_inner_value,
            maturity=maturity,
            time_steps=self._t_grid,
            damping_steps=self._damping_steps,
        )

        solver = Fdm2dBlackScholesSolver(
            self._p1,
            self._p2,
            self._correlation,
            solver_desc,
            self._scheme_desc,
            self._local_vol,
            self._illegal_local_vol_overwrite,
        )

        x = self._p1.x0()
        y = self._p2.x0()

        results.value = solver.value_at(x, y)
        results.delta = solver.delta_x_at(x, y) + solver.delta_y_at(x, y)
        results.gamma = (
            solver.gamma_x_at(x, y)
            + solver.gamma_y_at(x, y)
            + 2 * solver.gamma_xy_at(x, y)
        )
        results.theta = solver.theta_at(x, y)


__all__ = ["Fd2dBlackScholesVanillaEngine"]
