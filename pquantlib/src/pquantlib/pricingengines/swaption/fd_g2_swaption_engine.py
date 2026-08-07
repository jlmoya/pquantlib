"""FdG2SwaptionEngine — 2-D finite-difference G2++ swaption engine.

# C++ parity: ql/pricingengines/swaption/fdg2swaptionengine.{hpp,cpp} (v1.43)
# — ``class FdG2SwaptionEngine : public GenericModelEngine<G2,
#    Swaption::arguments, Swaption::results>``.

Rolls the swaption back on the two-factor ``(x, y)`` state mesh — one
Ornstein-Uhlenbeck mesher per factor, built with ``tAvgSteps = 1`` and
``epsilon = invEps`` — and reads ``valueAt(0.0, 0.0)``.

Like :class:`FdHullWhiteSwaptionEngine` it builds a **second** ``G2`` on the
swap's forwarding curve (same a / sigma / b / eta / rho, different term
structure) so discounting and forwarding curves stay separate, and requires
that the two curves share a day counter and a reference date.

The default scheme here is ``FdmSchemeDesc::Hundsdorfer()`` (the Hull-White
swaption engine defaults to ``Douglas()``).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.instruments.swaption import SwaptionArguments, SwaptionResults
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.fdm_simple_process_1d_mesher import (
    FdmSimpleProcess1dMesher,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_g2_solver import FdmG2Solver
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_affine_model_swap_inner_value import (
    FdmAffineModelSwapInnerValue,
)
from pquantlib.models.shortrate.twofactor.g2 import G2
from pquantlib.pricingengines.generic_model_engine import GenericModelEngine
from pquantlib.pricingengines.swaption.fd_hull_white_swaption_engine import (
    forwarding_curve,
)
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.time.date import Date


class FdG2SwaptionEngine(GenericModelEngine[G2, SwaptionArguments, SwaptionResults]):
    """Finite-differences G2++ swaption engine.

    # C++ parity: ``class FdG2SwaptionEngine``.
    """

    def __init__(
        self,
        model: G2,
        t_grid: int = 100,
        x_grid: int = 50,
        y_grid: int = 50,
        damping_steps: int = 0,
        inv_eps: float = 1e-5,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(SwaptionArguments(), SwaptionResults(), model)
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._y_grid: int = y_grid
        self._damping_steps: int = damping_steps
        self._inv_eps: float = inv_eps
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Hundsdorfer()`` default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.hundsdorfer()
        )

    def calculate(self) -> None:
        """# C++ parity: ``FdG2SwaptionEngine::calculate``
        # (fdg2swaptionengine.cpp:48-121).
        """
        args = self._arguments
        results = self._results
        model = self.model()
        qassert.require(model is not None, "no model specified")
        assert model is not None
        assert args.exercise is not None
        assert args.swap is not None

        # 1. Term structure
        ts = model.term_structure

        # 2. Mesher
        dc = ts.day_counter()
        reference_date = ts.reference_date()
        maturity = dc.year_fraction(reference_date, args.exercise.last_date())

        process1 = OrnsteinUhlenbeckProcess(model.a(), model.sigma())
        process2 = OrnsteinUhlenbeckProcess(model.b(), model.eta())

        x_mesher = FdmSimpleProcess1dMesher(
            self._x_grid, process1, maturity, 1, self._inv_eps
        )
        y_mesher = FdmSimpleProcess1dMesher(
            self._y_grid, process2, maturity, 1, self._inv_eps
        )
        mesher = FdmMesherComposite(x_mesher, y_mesher)

        # 3. Inner Value Calculator
        t2d: dict[float, Date] = {}
        for exercise_date in args.exercise.dates():
            t = dc.year_fraction(reference_date, exercise_date)
            qassert.require(t >= 0, "exercise dates must not contain past date")
            t2d[t] = exercise_date

        dis_ts = model.term_structure
        fwd_ts = forwarding_curve(args.swap)

        qassert.require(
            fwd_ts.day_counter() == dis_ts.day_counter(),
            "day counter of forward and discount curve must match",
        )
        qassert.require(
            fwd_ts.reference_date() == dis_ts.reference_date(),
            "reference date of forward and discount curve must match",
        )

        fwd_model = G2(
            fwd_ts, model.a(), model.sigma(), model.b(), model.eta(), model.rho()
        )
        calculator = FdmAffineModelSwapInnerValue(
            model, fwd_model, args.swap, t2d, mesher, 0
        )

        # 4. Step conditions
        # C++ parity: ``FdmStepConditionComposite::vanillaComposite(
        # DividendSchedule(), exercise, mesher, calculator, refDate, dc)``.
        conditions = FdmStepConditionComposite.vanilla_composite(
            [], args.exercise, mesher, calculator, reference_date, dc
        )

        # 5. Boundary conditions — none. 6. Solver.
        solver_desc = FdmSolverDesc(
            mesher,
            conditions,
            calculator.avg_inner_value,
            maturity,
            self._t_grid,
            self._damping_steps,
        )
        solver = FdmG2Solver(model, solver_desc, self._scheme_desc)

        results.value = solver.value_at(0.0, 0.0)


__all__ = ["FdG2SwaptionEngine"]
