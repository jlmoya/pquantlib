"""FdmStepConditionComposite — list of step conditions applied in sequence.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.{hpp,cpp}
# (v1.43).

Carries a list of step conditions and a list of *stopping times* (the
times at which any of the conditions wants to be invoked). The
backward solver consults ``stopping_times()`` to refine the step
schedule, then calls ``apply_to`` at each step.

Both C++ static builders are ported: ``join_conditions``
(C++ ``joinConditions``) and ``vanilla_composite`` (C++
``vanillaComposite``), the latter being the exercise/dividend dispatch
that every vanilla-flavoured FD engine in QuantLib routes through.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from pquantlib import qassert
from pquantlib.cashflows.dividend import Dividend
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.exercise import Exercise
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.step_conditions.fdm_american_step_condition import (
    FdmAmericanStepCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_bermudan_step_condition import (
    FdmBermudanStepCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.fdm_snapshot_condition import (
    FdmSnapshotCondition,
)
from pquantlib.methods.finitedifferences.step_conditions.inner_value_calculator_protocol import (
    FdmInnerValueCalculatorLike,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)
from pquantlib.methods.finitedifferences.utilities.fdm_dividend_handler import (
    FdmDividendHandler,
)
from pquantlib.time.date import Date


@final
class FdmStepConditionComposite(StepCondition):
    """Composite of step conditions applied in sequence.

    # C++ parity: ``class FdmStepConditionComposite : public StepCondition<Array>``.
    """

    def __init__(
        self,
        stopping_times: list[list[float]],
        conditions: list[StepCondition],
    ) -> None:
        self._conditions: list[StepCondition] = list(conditions)
        # Flatten + dedupe + sort stopping times.
        all_times: set[float] = set()
        for ts in stopping_times:
            all_times.update(ts)
        self._stopping_times: list[float] = sorted(all_times)

    def conditions(self) -> list[StepCondition]:
        return list(self._conditions)

    def stopping_times(self) -> list[float]:
        return list(self._stopping_times)

    def apply_to(self, a: Array, t: float) -> None:
        for c in self._conditions:
            c.apply_to(a, t)

    @staticmethod
    def join_conditions(
        c1: FdmSnapshotCondition,
        c2: FdmStepConditionComposite,
    ) -> FdmStepConditionComposite:
        """Glue a snapshot condition onto an existing composite.

        # C++ parity: ``FdmStepConditionComposite::joinConditions``.

        The snapshot's own time joins the stopping times, so the solver is
        forced to land exactly on it — which is what makes
        ``FdmSnapshotCondition``'s exact ``t == t_`` test fire. Order matters:
        ``c2`` runs first, then the snapshot, so the recorded values are the
        *post-condition* ones.
        """
        return FdmStepConditionComposite(
            [c2.stopping_times(), [c1.get_time()]],
            [c2, c1],
        )

    @staticmethod
    def vanilla_composite(
        cash_flow: Sequence[Dividend],
        exercise: Exercise,
        mesher: FdmMesher,
        calculator: FdmInnerValueCalculatorLike,
        ref_date: Date,
        day_counter: DayCounter,
    ) -> FdmStepConditionComposite:
        """Build the dividend + exercise composite used by the vanilla FD engines.

        # C++ parity: ``FdmStepConditionComposite::vanillaComposite``
        # (fdmstepconditioncomposite.cpp:80-145).
        """
        stopping_times: list[list[float]] = []
        step_conditions: list[StepCondition] = []

        if cash_flow:
            maturity_date = exercise.last_date()
            # C++ parity: the ``std::copy_if`` keeping only dividends inside
            # ``[refDate, maturityDate]``.
            dividends = [
                div for div in cash_flow if div.date() >= ref_date and div.date() <= maturity_date
            ]

            dividend_condition = FdmDividendHandler(
                dividends, mesher, ref_date, day_counter, 0
            )
            step_conditions.append(dividend_condition)

            dividend_times = list(dividend_condition.dividend_times())
            maturity_time = day_counter.year_fraction(ref_date, exercise.last_date())

            # this effectively excludes times after maturity
            stopping_times.append([min(maturity_time, t) for t in dividend_times])

            # smoother convergence behavior with number of time steps
            stopping_times.append([min(maturity_time, t + 1e-5) for t in dividend_times])

        qassert.require(
            exercise.type()
            in (Exercise.Type.American, Exercise.Type.European, Exercise.Type.Bermudan),
            "exercise type is not supported",
        )
        if exercise.type() == Exercise.Type.American:
            exercise_start = day_counter.year_fraction(ref_date, exercise.date(0))
            step_conditions.append(
                FdmAmericanStepCondition(mesher, calculator, exercise_start)
            )
        elif exercise.type() == Exercise.Type.Bermudan:
            bermudan_condition = FdmBermudanStepCondition(
                exercise.dates(), ref_date, day_counter, mesher, calculator
            )
            step_conditions.append(bermudan_condition)
            stopping_times.append(bermudan_condition.exercise_times())

        return FdmStepConditionComposite(stopping_times, step_conditions)


__all__ = ["FdmStepConditionComposite"]
