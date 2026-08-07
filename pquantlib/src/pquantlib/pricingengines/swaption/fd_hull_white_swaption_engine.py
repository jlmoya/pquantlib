"""FdHullWhiteSwaptionEngine — 1-D finite-difference Hull-White swaption engine.

# C++ parity: ql/pricingengines/swaption/fdhullwhiteswaptionengine.{hpp,cpp}
# (v1.43) — ``class FdHullWhiteSwaptionEngine : public
#  GenericModelEngine<HullWhite, Swaption::arguments, Swaption::results>``.

Rolls the swaption back on a single Ornstein-Uhlenbeck short-rate mesh and
reads ``valueAt(0.0)``.

Two details a port loses easily:

* the engine builds a **second** ``HullWhite`` model on the *swap's*
  forwarding curve and hands both models to
  :class:`FdmAffineModelSwapInnerValue`, so discounting and forwarding are
  genuinely separate curves. It then requires that the two curves share a day
  counter and a reference date;
* the default scheme is ``FdmSchemeDesc::Douglas()``, not ``Hundsdorfer()``
  as in every Heston engine.

``exercise->dates()`` is mapped to a ``{time: Date}`` dictionary, with
``QL_REQUIRE(t >= 0, "exercise dates must not contain past date")``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.indexes.ibor_index import IborIndex
from pquantlib.instruments.fixed_vs_floating_swap import FixedVsFloatingSwap
from pquantlib.instruments.swaption import SwaptionArguments, SwaptionResults
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.fdm_simple_process_1d_mesher import (
    FdmSimpleProcess1dMesher,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_hull_white_solver import (
    FdmHullWhiteSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.utilities.fdm_affine_model_swap_inner_value import (
    FdmAffineModelSwapInnerValue,
)
from pquantlib.models.shortrate.onefactor.hull_white import HullWhite
from pquantlib.pricingengines.generic_model_engine import GenericModelEngine
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.termstructures.yield_term_structure import YieldTermStructure
from pquantlib.time.date import Date


def forwarding_curve(swap: FixedVsFloatingSwap) -> YieldTermStructure:
    """The swap's floating-leg forecasting curve.

    # C++ parity: ``arguments_.swap->iborIndex()->forwardingTermStructure()``
    # (fdhullwhiteswaptionengine.cpp:76-77, fdg2swaptionengine.cpp:87-88).

    C++ types ``iborIndex()`` as ``ext::shared_ptr<IborIndex>`` and
    ``forwardingTermStructure()`` as ``Handle<YieldTermStructure>``. pquantlib
    types the former as the narrower ``IborIndexProtocol`` (no curve accessor)
    and the latter as ``YieldTermStructureProtocol | None``, so the two C++
    static types are restored here. The accessor is spelled
    ``forecast_term_structure``.
    """
    index = swap.ibor_index()
    qassert.require(isinstance(index, IborIndex), "swap index is not an IborIndex")
    assert isinstance(index, IborIndex)
    fwd_ts = index.forecast_term_structure()
    qassert.require(
        isinstance(fwd_ts, YieldTermStructure), "no forwarding term structure given"
    )
    assert isinstance(fwd_ts, YieldTermStructure)
    return fwd_ts


class FdHullWhiteSwaptionEngine(
    GenericModelEngine[HullWhite, SwaptionArguments, SwaptionResults]
):
    """Finite-differences Hull-White swaption engine.

    # C++ parity: ``class FdHullWhiteSwaptionEngine``.
    """

    def __init__(
        self,
        model: HullWhite,
        t_grid: int = 100,
        x_grid: int = 100,
        damping_steps: int = 0,
        inv_eps: float = 1e-5,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(SwaptionArguments(), SwaptionResults(), model)
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._damping_steps: int = damping_steps
        self._inv_eps: float = inv_eps
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Douglas()`` default — note
        # this differs from every Heston engine's Hundsdorfer default.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.douglas()
        )

    def calculate(self) -> None:
        """# C++ parity: ``FdHullWhiteSwaptionEngine::calculate``
        # (fdhullwhiteswaptionengine.cpp:48-106).
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

        process = OrnsteinUhlenbeckProcess(model.a(), model.sigma())
        short_rate_mesher = FdmSimpleProcess1dMesher(
            self._x_grid, process, maturity, 1, self._inv_eps
        )
        mesher = FdmMesherComposite(short_rate_mesher)

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

        fwd_model = HullWhite(fwd_ts, model.a(), model.sigma())
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
        solver = FdmHullWhiteSolver(model, solver_desc, self._scheme_desc)

        results.value = solver.value_at(0.0)


__all__ = ["FdHullWhiteSwaptionEngine", "forwarding_curve"]
