"""FdCEVVanillaEngine — finite-difference engine for the CEV model.

# C++ parity: ql/pricingengines/vanilla/fdcevvanillaengine.{hpp,cpp}
# (v1.43) — ``class FdCEVVanillaEngine : public VanillaOption::engine``.

The CEV forward ``dF = alpha F^beta dW`` is discretised on an
:class:`~pquantlib.methods.finitedifferences.meshers.fdm_cev_1d_mesher.FdmCEV1dMesher`
grid in *forward* (not log-forward) space, so the engine drives an
:class:`~pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver.Fdm1DimSolver`
directly rather than going through ``FdmBlackScholesSolver``: value, delta and
gamma are the raw spline and its two derivatives at ``f0``.

Two boundary conditions carry the model's behaviour at the mesh edges:

* the **upper** one is time-dependent and prices the option analytically at
  the top forward with :class:`~pquantlib.pricingengines.vanilla.analytic_cev_engine.CEVCalculator`;
* the **lower** one only exists when ``delta = (1-2 beta)/(1-beta) < 2``,
  i.e. when the origin is attainable and the terminal cash flow must be
  discounted back rather than left to diffuse.
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    BoundaryConditionSide,
    FdmBoundaryCondition,
)
from pquantlib.methods.finitedifferences.meshers.fdm_cev_1d_mesher import FdmCEV1dMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.operators.fdm_cev_op import FdmCEVOp
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_1dim_solver import Fdm1DimSolver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_discount_dirichlet_boundary import (
    FdmDiscountDirichletBoundary,
)
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmCellAveragingInnerValue,
)
from pquantlib.methods.finitedifferences.utilities.fdm_time_dep_dirichlet_boundary import (
    FdmTimeDepDirichletBoundary,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.pricingengines.vanilla.analytic_cev_engine import CEVCalculator
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


@final
class _PriceAtBoundary:
    """Analytic CEV price at the upper mesh boundary, as a function of time.

    # C++ parity: the anonymous-namespace ``class PriceAtBoundary``
    # (fdcevvanillaengine.cpp:42-68).
    """

    __slots__ = ("_calculator", "_maturity_time", "_payoff", "_r_ts")

    def __init__(
        self,
        maturity_time: float,
        payoff: StrikedTypePayoff,
        r_ts: YieldTermStructure,
        calculator: CEVCalculator,
    ) -> None:
        self._maturity_time: float = maturity_time
        self._payoff: StrikedTypePayoff = payoff
        self._r_ts: YieldTermStructure = r_ts
        self._calculator: CEVCalculator = calculator

    def __call__(self, t: float) -> float:
        """# C++ parity: ``Real PriceAtBoundary::operator()(Real t) const``."""
        time_to_expiry = max(1.0 / 365.0, self._maturity_time - t)
        df = self._r_ts.discount(self._maturity_time) / self._r_ts.discount(t)
        return df * self._calculator.value(
            self._payoff.option_type(), self._payoff.strike(), time_to_expiry
        )


@final
class FdCEVVanillaEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Finite-difference CEV vanilla option engine.

    # C++ parity: ``class FdCEVVanillaEngine``.
    """

    def __init__(
        self,
        f0: float,
        alpha: float,
        beta: float,
        discount_curve: YieldTermStructure,
        t_grid: int = 50,
        x_grid: int = 400,
        damping_steps: int = 0,
        scaling_factor: float = 1.0,
        eps: float = 1e-4,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._f0: float = f0
        self._alpha: float = alpha
        self._beta: float = beta
        self._discount_curve: YieldTermStructure = discount_curve
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._damping_steps: int = damping_steps
        self._scaling_factor: float = scaling_factor
        self._eps: float = eps
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Douglas()``.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.douglas()
        )
        # C++ parity: ``registerWith(discountCurve_)``.
        discount_curve.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``FdCEVVanillaEngine::calculate`` (cpp:84-163)."""
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

        r_ts = self._discount_curve
        dc = r_ts.day_counter()
        reference_date = r_ts.reference_date()
        maturity_date = args.exercise.last_date()
        maturity_time = dc.year_fraction(reference_date, maturity_date)

        cev_mesher = FdmCEV1dMesher(
            self._x_grid,
            self._f0,
            self._alpha,
            self._beta,
            maturity_time,
            self._eps,
            self._scaling_factor,
            (payoff.strike(), 0.1),
        )

        locations = cev_mesher.locations()
        lower_bound = float(locations[0])
        upper_bound = float(locations[-1])

        mesher = FdmMesherComposite(cev_mesher)

        # 2. Calculator
        calculator = FdmCellAveragingInnerValue(payoff, mesher, 0)

        # 3. Step conditions
        conditions = FdmStepConditionComposite.vanilla_composite(
            (), args.exercise, mesher, calculator, reference_date, dc
        )

        # 4. Boundary conditions
        boundaries: list[FdmBoundaryCondition] = []

        upper_bound_price = _PriceAtBoundary(
            maturity_time,
            payoff,
            r_ts,
            CEVCalculator(upper_bound, self._alpha, self._beta),
        )
        boundaries.append(
            FdmTimeDepDirichletBoundary(
                mesher,
                0,
                BoundaryConditionSide.UPPER,
                value_on_boundary=upper_bound_price,
            )
        )

        delta = (1 - 2 * self._beta) / (1 - self._beta)
        if delta < 2.0:
            terminal_cash_flow = payoff(lower_bound)
            boundaries.append(
                FdmDiscountDirichletBoundary(
                    mesher,
                    r_ts,
                    maturity_time,
                    terminal_cash_flow,
                    0,
                    BoundaryConditionSide.LOWER,
                )
            )

        # 5. Solver
        solver_desc = FdmSolverDesc(
            mesher=mesher,
            condition=conditions,
            calculator=calculator.avg_inner_value,
            maturity=maturity_time,
            time_steps=self._t_grid,
            damping_steps=self._damping_steps,
            bc_set=tuple(boundaries),
        )

        op = FdmCEVOp(mesher, self._discount_curve, self._f0, self._alpha, self._beta, 0)

        solver = Fdm1DimSolver(solver_desc, self._scheme_desc, op)

        results.value = solver.interpolate_at(self._f0)
        results.delta = solver.derivative_x(self._f0)
        results.gamma = solver.derivative_xx(self._f0)
        results.theta = solver.theta_at(self._f0)


__all__ = ["FdCEVVanillaEngine"]
