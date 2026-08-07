"""FdSimpleBSSwingEngine — finite-difference Black-Scholes engine for simple swing options.

# C++ parity: ql/pricingengines/vanilla/fdsimplebsswingengine.{hpp,cpp}
# (v1.43) — ``class FdSimpleBSSwingEngine : public
# GenericEngine<VanillaSwingOption::arguments, VanillaSwingOption::results>``.

Two axes: log-spot on an ``FdmBlackScholesMesher``, and the *number of
exercise rights already used* on a ``Uniform1dMesher`` with
``maxExerciseRights + 1`` integer nodes. The terminal condition is identically
zero (``FdmZeroInnerValue``) — all the value comes from
``FdmSimpleSwingCondition``, which at each exercise time moves a node up the
rights axis whenever exercising beats waiting.

The reported value is at ``(spot, 1)``: one right already used, i.e. the value
of the whole strip seen from a state where the holder still owns
``maxExerciseRights`` rights (C++ counts *down* the axis).

**Location divergence (reported, not fixed).** C++ v1.43 declares
``VanillaSwingOption``/``SwingExercise`` in ``ql/instruments/vanillaswingoption.hpp``;
this port keeps them under ``pquantlib.experimental.finitedifferences``.

**Results carrier.** C++'s ``VanillaSwingOption`` derives from
``OneAssetOption``, so ``VanillaSwingOption::results`` carries the Greeks that
``calculate()`` fills. The Python ``VanillaSwingOption`` derives from ``Option``
and its ``fetch_results`` pulls only the value, so the Greeks are read off the
engine's own results object.
"""

from __future__ import annotations

from typing import final

from pquantlib import qassert
from pquantlib.exercise import Exercise
from pquantlib.experimental.finitedifferences.vanilla_swing_option import (
    VanillaSwingOptionArguments,
)
from pquantlib.instruments.one_asset_option import OneAssetOptionResults
from pquantlib.methods.finitedifferences.meshers.fdm_black_scholes_mesher import (
    FdmBlackScholesMesher,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import Uniform1dMesher
from pquantlib.methods.finitedifferences.schemes.fdm_scheme_desc import FdmSchemeDesc
from pquantlib.methods.finitedifferences.solvers.fdm_simple_2d_bs_solver import (
    FdmSimple2dBSSolver,
)
from pquantlib.methods.finitedifferences.solvers.fdm_solver_desc import FdmSolverDesc
from pquantlib.methods.finitedifferences.step_conditions.fdm_simple_swing_condition import (
    FdmSimpleSwingCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_step_condition_composite import (
    FdmStepConditionComposite,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import StepCondition
from pquantlib.methods.finitedifferences.utilities.fdm_inner_value_calculator import (
    FdmLogInnerValue,
    FdmZeroInnerValue,
)
from pquantlib.payoffs import StrikedTypePayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)


@final
class FdSimpleBSSwingEngine(
    GenericEngine[VanillaSwingOptionArguments, OneAssetOptionResults]
):
    """Finite-difference Black-Scholes engine for simple swing options.

    # C++ parity: ``class FdSimpleBSSwingEngine``.
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        t_grid: int = 50,
        x_grid: int = 100,
        scheme_desc: FdmSchemeDesc | None = None,
    ) -> None:
        super().__init__(VanillaSwingOptionArguments(), OneAssetOptionResults())
        self._process: GeneralizedBlackScholesProcess = process
        self._t_grid: int = t_grid
        self._x_grid: int = x_grid
        # C++ parity: ``schemeDesc = FdmSchemeDesc::Douglas()``.
        self._scheme_desc: FdmSchemeDesc = (
            scheme_desc if scheme_desc is not None else FdmSchemeDesc.douglas()
        )
        # C++ parity: the constructor does not ``registerWith`` anything.

    def calculate(self) -> None:
        """# C++ parity: ``FdSimpleBSSwingEngine::calculate`` (cpp:46-112)."""
        args = self._arguments
        results = self._results

        qassert.require(args.exercise is not None, "no exercise given")
        assert args.exercise is not None
        exercise = args.exercise
        qassert.require(
            exercise.type() == Exercise.Type.Bermudan, "Bermudan exercise supported only"
        )

        # 1. Mesher
        qassert.require(
            isinstance(args.payoff, StrikedTypePayoff), "Strike type payoff expected"
        )
        assert isinstance(args.payoff, StrikedTypePayoff)
        payoff: StrikedTypePayoff = args.payoff
        assert args.max_exercise_rights is not None
        assert args.min_exercise_rights is not None
        max_exercise_rights: int = args.max_exercise_rights

        maturity = self._process.time(exercise.last_date())
        equity_mesher = FdmBlackScholesMesher(
            self._x_grid, self._process, maturity, payoff.strike()
        )
        exercise_mesher = Uniform1dMesher(
            0.0, float(max_exercise_rights), max_exercise_rights + 1
        )

        mesher = FdmMesherComposite(equity_mesher, exercise_mesher)

        # 2. Calculator
        calculator = FdmZeroInnerValue()

        # 3. Step conditions
        step_conditions: list[StepCondition] = []
        stopping_times: list[list[float]] = []

        # 3.1 Bermudan step conditions
        exercise_times: list[float] = []
        for d in exercise.dates():
            t = self._process.time(d)
            qassert.require(t >= 0, "exercise dates must not contain past date")
            exercise_times.append(t)
        stopping_times.append(exercise_times)

        exercise_calculator = FdmLogInnerValue(payoff, mesher, 0)

        step_conditions.append(
            FdmSimpleSwingCondition(
                exercise_times, mesher, exercise_calculator, 1, args.min_exercise_rights
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

        spot = self._process.x0()

        results.value = solver.value_at(spot, 1)
        results.delta = solver.delta_at(spot, 1, spot * 0.01)
        results.gamma = solver.gamma_at(spot, 1, spot * 0.01)
        results.theta = solver.theta_at(spot, 1)


__all__ = ["FdSimpleBSSwingEngine"]
