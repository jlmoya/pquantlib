"""FdOrnsteinUhlenbeckVanillaEngine — FD vanilla engine under OU dynamics.

# C++ parity: ql/experimental/finitedifferences/fdornsteinuhlenbeckvanillaengine.{hpp,cpp}
# (v1.42.1).

Prices a European vanilla option (call or put) where the underlying
state ``x`` follows the Ornstein-Uhlenbeck SDE

.. math::

    dx_t = a (b - x_t) dt + \\sigma dW_t

(arithmetic — ``x`` is the spot itself, not ``log(spot)``).

The engine builds:

1. A process-driven 1-D mesher centred on ``x0``.
2. The Black-Scholes-equivalent OU operator
   ``L = a(b - x) D_x + 0.5 sigma^2 D_xx - r I``.
3. The American/European step conditions (American currently
   deferred — only European exercise is supported).
4. The backward solver with the requested ``FdmSchemeDesc``.

NPV is the linear-interpolated value at the operator's grid at
``x = x0``.

**Carve-outs vs C++:**

* Dividend schedules — deferred (the OU PDE is on the spot, not
  log-spot; the C++ engine treats dividends via mesher offsets +
  step conditions).
* Greeks (delta / gamma / theta) — not extracted (mirrors the
  L5-D BSM engine).
* American exercise — deferred (the existing
  ``FdmAmericanStepCondition`` is wired against
  ``FdmMesher`` so could be ported here but is out of scope for
  the W5-C cluster).
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.fdm_simple_process_1d_mesher import (
    FdmSimpleProcess1dMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_ornstein_uhlenbeck_op import (
    FdmOrnsteinUhlenbeckOp,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_backward_solver import (
    FdmBackwardSolver,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.option import OptionArguments
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.ornstein_uhlenbeck_process import OrnsteinUhlenbeckProcess
from pquantlib.termstructures.yield_term_structure import YieldTermStructure


@final
class FdOrnsteinUhlenbeckVanillaEngine(GenericEngine[OptionArguments, OneAssetOptionResults]):
    """Finite-difference engine for vanilla options under OU dynamics.

    # C++ parity: ``class FdOrnsteinUhlenbeckVanillaEngine : public
    # VanillaOption::engine``.
    """

    def __init__(
        self,
        process: OrnsteinUhlenbeckProcess,
        rTS: YieldTermStructure,  # noqa: N803 (C++ field name preserved)
        t_grid: int = 100,
        x_grid: int = 100,
        damping_steps: int = 0,
        epsilon: float = 1e-4,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(OptionArguments(), OneAssetOptionResults())
        self._process: OrnsteinUhlenbeckProcess = process
        self._rTS: YieldTermStructure = rTS
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._damping_steps: int = damping_steps
        self._epsilon: float = epsilon
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.crank_nicolson()
        )

    def calculate(self) -> None:
        """Compute NPV via backward FD rollback.

        # C++ parity: ``FdOrnsteinUhlenbeckVanillaEngine::calculate``.
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

        # 0. Time to maturity (year fraction via the rTS day counter).
        ref_date = self._rTS.reference_date()
        dc = self._rTS.day_counter()
        exercise_date = exercise.last_date()
        maturity = dc.year_fraction(ref_date, exercise_date)
        qassert.require(maturity > 0.0, "maturity must be strictly positive")

        # 1. Mesher: 1-D process-driven mesh.
        equity_mesher = FdmSimpleProcess1dMesher(
            self._x_grid, self._process, maturity, 1, self._epsilon
        )
        mesher = FdmMesherComposite(equity_mesher)

        # 2. OU operator.
        op = FdmOrnsteinUhlenbeckOp(mesher, self._process, self._rTS, direction=0)

        # 3. Step conditions — European only (American deferred).
        if exercise.type() == Exercise.Type.European:
            pass
        else:
            raise LibraryException(
                f"FdOrnsteinUhlenbeckVanillaEngine: only European exercise supported, "
                f"got {exercise.type().name}"
            )
        composite = FdmStepConditionComposite([], [])

        # 4. Initial values: payoff(x) at each grid node (arithmetic spot).
        x_locations = mesher.locations(0)
        rhs = np.array([payoff(float(x)) for x in x_locations], dtype=np.float64)

        # 5. Roll back from maturity to t=0.
        solver = FdmBackwardSolver(op, composite, self._scheme_desc)
        rhs = solver.rollback(rhs, maturity, 0.0, self._t_grid, self._damping_steps)

        # 6. Interpolate at x = x0.
        target = float(self._process.x0())
        interp = LinearInterpolation(equity_mesher.locations(), rhs)
        npv = interp(target)

        results.value = npv
        results.additional_results = {
            "x0": self._process.x0(),
            "strike": payoff.strike(),
            "timeToExpiry": maturity,
            "tGrid": self._t_grid,
            "xGrid": self._x_grid,
            "scheme": self._scheme_desc.type.name,
        }


__all__ = ["FdOrnsteinUhlenbeckVanillaEngine"]
