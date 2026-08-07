"""FdSabrVanillaEngine — finite-difference pricing engine for the SABR model.

# C++ parity: ql/pricingengines/vanilla/fdsabrvanillaengine.{hpp,cpp} (v1.43)
# — ``class FdSabrVanillaEngine : public VanillaOption::engine``.

Solves the two-dimensional SABR PDE on a (forward, log-vol) mesh: an
:class:`FdmCEV1dMesher` on the forward and a :class:`Concentrating1dMesher` on
``log alpha``, driven by :class:`FdmSabrOp`.

Details a port loses easily and that are reproduced verbatim:

* the constructor calls ``validateSabrParameters(alpha, 0.5, nu, rho)`` — the
  second argument is the **literal 0.5**, not ``beta``; ``beta`` is separately
  required to be ``< 1.0``. So a negative ``beta`` passes validation;
* the forward mesh's upper vol level is
  ``alpha * exp(nu * sqrt(T) * N^-1(0.75))`` — the 75th percentile, not the
  ``1 - eps`` percentile used for the vol axis;
* the inner value is an :class:`FdmCellAveragingInnerValue` with the *identity*
  grid mapping (the forward axis is in price space, not log space);
* both Dirichlet faces carry the **undiscounted** payoff evaluated at the CEV
  mesh's own end points, discounted back by
  :class:`FdmDiscountDirichletBoundary`;
* only ``results_.value`` is filled — the greeks stay unset.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.inverse_cumulative_normal import InverseCumulativeNormal
from pquantlib.math.interpolations.sabr_formula import validate_sabr_parameters
from pquantlib.methods.finitedifferences.fdm_boundary_condition import (
    BoundaryConditionSide,
    FdmBoundaryCondition,
)
from pquantlib.methods.finitedifferences.meshers.concentrating_1d_mesher import (
    Concentrating1dMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_cev_1d_mesher import FdmCEV1dMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.operators.fdm_sabr_op import FdmSabrOp
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_2dim_solver import Fdm2DimSolver
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
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.termstructures.yield_term_structure import YieldTermStructure

_INV_CUM_NORMAL = InverseCumulativeNormal()


class FdSabrVanillaEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Finite-differences pricing engine for the SABR model.

    # C++ parity: ``class FdSabrVanillaEngine``.
    """

    def __init__(
        self,
        f0: float,
        alpha: float,
        beta: float,
        nu: float,
        rho: float,
        r_ts: YieldTermStructure,
        t_grid: int = 50,
        f_grid: int = 400,
        x_grid: int = 50,
        damping_steps: int = 0,
        scaling_factor: float = 1.0,
        eps: float = 1e-4,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._f0: float = f0
        self._alpha: float = alpha
        self._beta: float = beta
        self._nu: float = nu
        self._rho: float = rho
        self._r_ts: YieldTermStructure = r_ts
        self._t_grid: int = t_grid
        self._f_grid: int = f_grid
        self._x_grid: int = x_grid
        self._damping_steps: int = damping_steps
        self._scaling_factor: float = scaling_factor
        self._eps: float = eps
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )

        # C++ parity: ``validateSabrParameters(alpha, 0.5, nu, rho)`` — the
        # second argument really is the constant 0.5, so ``beta`` is NOT
        # range-checked here, only by the QL_REQUIRE below.
        validate_sabr_parameters(alpha, 0.5, nu, rho)
        # ``{beta:g}`` reproduces C++'s default ``std::ostream`` formatting of a
        # ``Real`` (6 significant digits), so 1.0 prints as "1" exactly as
        # QuantLib does. Plain ``{beta}`` would print "1.0".
        qassert.require(beta < 1.0, f"beta must be smaller than 1.0: {beta:g} not allowed")

        # C++ parity: ``registerWith(rTS_)``.
        r_ts.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``FdSabrVanillaEngine::calculate``
        # (fdsabrvanillaengine.cpp:65-144).
        """
        args = self._arguments
        results = self._results

        payoff = args.payoff
        qassert.require(isinstance(payoff, StrikedTypePayoff), "non-striked payoff given")
        assert isinstance(payoff, StrikedTypePayoff)
        assert args.exercise is not None
        exercise: Exercise = args.exercise

        # 1. Mesher
        dc = self._r_ts.day_counter()
        reference_date = self._r_ts.reference_date()
        maturity_date = exercise.last_date()
        maturity_time = dc.year_fraction(reference_date, maturity_date)

        upper_alpha = self._alpha * math.exp(
            self._nu * math.sqrt(maturity_time) * _INV_CUM_NORMAL(0.75)
        )

        cev_mesher = FdmCEV1dMesher(
            self._f_grid,
            self._f0,
            upper_alpha,
            self._beta,
            maturity_time,
            self._eps,
            self._scaling_factor,
            (payoff.strike(), 0.025),
        )

        norm_inv_eps = _INV_CUM_NORMAL(1.0 - self._eps)
        log_drift = -0.5 * self._nu * self._nu * maturity_time
        vol_range = self._nu * math.sqrt(maturity_time) * norm_inv_eps * self._scaling_factor

        x_min = math.log(self._alpha) + log_drift - vol_range
        x_max = math.log(self._alpha) + log_drift + vol_range

        x_mesher = Concentrating1dMesher(
            x_min, x_max, self._x_grid, (math.log(self._alpha), 0.1)
        )

        mesher = FdmMesherComposite(cev_mesher, x_mesher)

        # 2. Calculator — identity grid mapping (the forward axis is in price
        # space), so this is FdmCellAveragingInnerValue, not FdmLogInnerValue.
        calculator = FdmCellAveragingInnerValue(payoff, mesher, 0)

        # 3. Step conditions
        conditions = FdmStepConditionComposite.vanilla_composite(
            [], exercise, mesher, calculator, reference_date, dc
        )

        # 4. Boundary conditions
        lower_bound = float(cev_mesher.locations()[0])
        upper_bound = float(cev_mesher.locations()[-1])

        boundaries: list[FdmBoundaryCondition] = [
            FdmDiscountDirichletBoundary(
                mesher,
                self._r_ts,
                maturity_time,
                payoff(upper_bound),
                0,
                BoundaryConditionSide.UPPER,
            ),
            FdmDiscountDirichletBoundary(
                mesher,
                self._r_ts,
                maturity_time,
                payoff(lower_bound),
                0,
                BoundaryConditionSide.LOWER,
            ),
        ]

        # 5. Solver
        solver_desc = FdmSolverDesc(
            mesher,
            conditions,
            calculator.avg_inner_value,
            maturity_time,
            self._t_grid,
            self._damping_steps,
            tuple(boundaries),
        )

        op = FdmSabrOp(
            mesher, self._r_ts, self._f0, self._alpha, self._beta, self._nu, self._rho
        )
        solver = Fdm2DimSolver(solver_desc, self._scheme_desc, op)

        results.value = solver.interpolate_at(self._f0, math.log(self._alpha))


__all__ = ["FdSabrVanillaEngine"]
