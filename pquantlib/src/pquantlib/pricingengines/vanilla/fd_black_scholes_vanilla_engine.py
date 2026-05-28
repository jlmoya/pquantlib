"""FdBlackScholesVanillaEngine — finite-difference Black-Scholes vanilla engine.

# C++ parity: ql/pricingengines/vanilla/fdblackscholesvanillaengine.{hpp,cpp}
# (v1.42.1) — ``class FdBlackScholesVanillaEngine : public VanillaOption::engine``.

The engine builds the 1-D log-spot mesh, the BSM linear operator, the
step conditions (American only — Bermudan/dividend support deferred),
and the backward solver. It then sets the initial payoff at the
maturity time and rolls back to t=0, interpolating the result at
``log(spot)``.

**Carve-outs (vs C++):**

* Multi-D operator splittings (Craig-Sneyd, Hundsdorfer-Verwer,
  TR-BDF2, Method-of-Lines) — deferred.
* Cash dividend models (Spot / Escrowed) — deferred.
* Local-vol / quanto branches — deferred.
* ``MakeFdBlackScholesVanillaEngine`` builder — Python's kwargs make
  this redundant; the engine accepts the same parameters directly.
* Greek extraction (delta/gamma/theta from the grid) — the L5-D
  scope reports NPV only. Greeks via finite-difference on the
  solved grid are a Phase 6 enhancement.
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.operators.fdm_black_scholes_op import (
    FdmBlackScholesOp,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_backward_solver import (
    FdmBackwardSolver,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_american_step_condition import (
    FdmAmericanStepCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class FdBlackScholesVanillaEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Finite-difference Black-Scholes vanilla option engine.

    # C++ parity: ``class FdBlackScholesVanillaEngine``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        t_grid: int = 100,
        x_grid: int = 100,
        damping_steps: int = 0,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._damping_steps: int = damping_steps
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.crank_nicolson()
        )
        process.register_with(self)

    def calculate(self) -> None:
        """Compute NPV via backward FD rollback.

        # C++ parity: ``FdBlackScholesVanillaEngine::calculate``.
        """
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        qassert.require(args.payoff is not None, "no payoff given")
        assert args.exercise is not None
        assert args.payoff is not None
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff),
            "non-striked payoff given",
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff
        exercise: Exercise = args.exercise

        # 0. Time to maturity (year fraction via the process).
        exercise_date = exercise.last_date()
        maturity = self._process.time(exercise_date)
        qassert.require(maturity > 0.0, "maturity must be strictly positive")

        # 1. Mesher: 1-D log-spot mesh anchored at the strike.
        equity_mesher = FdmBlackScholesMesher(self._x_grid, self._process, maturity, payoff.strike())
        mesher = FdmMesherComposite(equity_mesher)

        # 2. Operator.
        op = FdmBlackScholesOp(mesher, self._process, payoff.strike())

        # 3. Step conditions — American or empty (European).
        conditions: list[StepCondition] = []
        stopping_times: list[list[float]] = []
        if exercise.type() == Exercise.Type.American:
            # Exercise starts at the earliest exercise date relative to ref.
            ref_date = self._process.risk_free_rate().reference_date()
            dc = self._process.risk_free_rate().day_counter()
            exercise_start = dc.year_fraction(ref_date, exercise.date(0))
            american = FdmAmericanStepCondition(mesher, payoff, exercise_start)
            conditions.append(american)
        elif exercise.type() == Exercise.Type.European:
            pass
        else:
            raise LibraryException(
                f"FdBlackScholesVanillaEngine: only European and American exercises supported, "
                f"got {exercise.type().name}"
            )
        composite = FdmStepConditionComposite(stopping_times, conditions)

        # 4. Initial values: payoff(exp(x)) at each grid node.
        spots = np.exp(mesher.locations(0))
        rhs = np.array([payoff(float(s)) for s in spots], dtype=np.float64)

        # 5. Roll back from maturity to t=0.
        solver = FdmBackwardSolver(op, composite, self._scheme_desc)
        rhs = solver.rollback(rhs, maturity, 0.0, self._t_grid, self._damping_steps)

        # 6. Interpolate at log(spot).
        # The locations on the 1-D log-spot grid are sorted ascending.
        log_spots = equity_mesher.locations()
        # rhs has the same length and the same iteration order (1-D layout
        # uses spacing=1 so flat-index == coord). No reordering needed.
        target = float(np.log(self._process.x0()))
        interp = LinearInterpolation(log_spots, rhs)
        npv = interp(target)

        results.value = npv
        # Greeks via grid finite-differencing are deferred — leave None.
        results.additional_results = {
            "spot": self._process.x0(),
            "strike": payoff.strike(),
            "timeToExpiry": maturity,
            "tGrid": self._t_grid,
            "xGrid": self._x_grid,
            "scheme": self._scheme_desc.type.name,
        }


__all__ = ["FdBlackScholesVanillaEngine"]
