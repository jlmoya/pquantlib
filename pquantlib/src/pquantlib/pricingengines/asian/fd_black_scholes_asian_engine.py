"""FdBlackScholesAsianEngine — FD Black-Scholes arithmetic-average Asian engine.

# C++ parity: ql/pricingengines/asian/fdblackscholesasianengine.{hpp,cpp}
# (v1.43) — ``class FdBlackScholesAsianEngine : public
# GenericEngine<DiscreteAveragingAsianOption::arguments,
#               DiscreteAveragingAsianOption::results>``.

Two axes: log-spot on an ``FdmBlackScholesMesher`` and the running arithmetic
*average* on a second ``FdmBlackScholesMesher`` whose bounds are overridden to
straddle both ``log(avg)`` and ``log(spot)``. The payoff is read off the
*average* axis (``FdmLogInnerValue(payoff, mesher, 1)``), and
``FdmArithmeticAverageCondition`` re-weights the average axis at each fixing
date.

C++ fills value, delta and gamma but **not** theta for this engine.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.instruments.asian_option import DiscreteAveragingAsianOptionArguments
from pquantlib.instruments.average_type import AverageType
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_simple_2d_bs_solver import (
    FdmSimple2dBSSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_arithmetic_average_condition import (
    FdmArithmeticAverageCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import StepCondition
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmLogInnerValue,
)
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class FdBlackScholesAsianEngine(
    GenericEngine[DiscreteAveragingAsianOptionArguments, OneAssetOptionResults]
):
    """Finite-difference Black-Scholes arithmetic Asian option engine.

    # C++ parity: ``class FdBlackScholesAsianEngine``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        t_grid: int = 100,
        x_grid: int = 100,
        a_grid: int = 50,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(DiscreteAveragingAsianOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        self._a_grid: int = a_grid
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Douglas()``.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.douglas()
        )
        # C++ parity: the constructor does not ``registerWith`` anything.

    def calculate(self) -> None:
        """# C++ parity: ``FdBlackScholesAsianEngine::calculate`` (cpp:47-124)."""
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        qassert.require(
            args.exercise.type() == Exercise.Type.European,
            "European exercise supported only",
        )
        qassert.require(
            args.average_type == AverageType.Arithmetic,
            "Arithmetic averaging supported only",
        )
        assert args.running_accumulator is not None
        assert args.past_fixings is not None
        qassert.require(
            args.running_accumulator == 0 or args.past_fixings > 0,
            "Running average requires at least one past fixing",
        )

        # 1. Mesher
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "non-striked payoff given"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff
        maturity = self._process.time(args.exercise.last_date())
        equity_mesher = FdmBlackScholesMesher(
            self._x_grid, self._process, maturity, payoff.strike()
        )

        spot = self._process.x0()
        qassert.require(spot > 0.0, "negative or null underlying given")

        avg = (
            spot
            if args.running_accumulator == 0
            else args.running_accumulator / args.past_fixings
        )

        norm_inv_eps = InverseCumulativeNormal()(1 - 0.0001)
        # C++ parity: ``blackVol(maturity, strike)`` — the third parameter
        # defaults to ``extrapolate = false``, so the range check is live.
        sigma_sqrt_t = self._process.black_volatility().black_vol_at_time(
            maturity, payoff.strike()
        ) * math.sqrt(maturity)
        r = sigma_sqrt_t * norm_inv_eps

        x_min = min(math.log(avg) - 0.25 * r, math.log(spot) - 1.5 * r)
        x_max = max(math.log(avg) + 0.25 * r, math.log(spot) + 1.5 * r)

        average_mesher = FdmBlackScholesMesher(
            self._a_grid, self._process, maturity, payoff.strike(), x_min, x_max
        )

        mesher = FdmMesherComposite(equity_mesher, average_mesher)

        # 2. Calculator — the payoff lives on the *average* axis.
        calculator = FdmLogInnerValue(payoff, mesher, 1)

        # 3. Step conditions
        step_conditions: list[StepCondition] = []
        stopping_times: list[list[float]] = []

        # 3.1 Arithmetic average step conditions
        average_times: list[float] = []
        for fixing_date in args.fixing_dates:
            t = self._process.time(fixing_date)
            qassert.require(t >= 0, "Fixing dates must not contain past date")
            average_times.append(t)
        stopping_times.append(average_times)
        step_conditions.append(
            FdmArithmeticAverageCondition(
                average_times, args.running_accumulator, args.past_fixings, mesher, 0
            )
        )

        conditions = FdmStepConditionComposite(stopping_times, step_conditions)

        # 4. Boundary conditions: C++ passes a default-constructed set.
        # 5. Solver
        solver_desc = FdmSolverDesc(
            mesher=mesher,
            condition=conditions,
            calculator=calculator.avg_inner_value,
            maturity=maturity,
            time_steps=self._t_grid,
            damping_steps=0,
        )
        solver = FdmSimple2dBSSolver(
            self._process, payoff.strike(), solver_desc, self._scheme_desc
        )

        results.value = solver.value_at(spot, avg)
        results.delta = solver.delta_at(spot, avg, spot * 0.01)
        results.gamma = solver.gamma_at(spot, avg, spot * 0.01)
        # C++ parity: theta is deliberately NOT filled by this engine.


__all__ = ["FdBlackScholesAsianEngine"]
