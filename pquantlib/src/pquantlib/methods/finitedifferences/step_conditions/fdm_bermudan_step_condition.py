"""FdmBermudanStepCondition — early exercise on a discrete date schedule.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmbermudanstepcondition.{hpp,cpp}
# (v1.43).

The American condition floors the solution at the exercise value on
*every* step; the Bermudan condition does it only on the steps that fall
exactly on one of the exercise times. Exercise *dates* are converted to
times once, in the constructor, via ``dayCounter.yearFraction(refDate, d)``
— the same conversion the solver uses to build its time grid, which is
what makes the exact ``==`` comparison in ``applyTo`` land.

``exercise_times()`` exists so that ``FdmStepConditionComposite`` can pull
the schedule into its stopping times (see
``FdmStepConditionComposite::vanillaComposite`` in C++).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from pquantlib import qassert
from pquantlib.daycounters.day_counter import DayCounter
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.step_conditions.inner_value_calculator_protocol import (
    FdmInnerValueCalculatorLike,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)
from pquantlib.time.date import Date


@final
class FdmBermudanStepCondition(StepCondition):
    """Bermudan early-exercise floor, applied on the exercise times only.

    # C++ parity: ``class FdmBermudanStepCondition : public StepCondition<Array>``.
    """

    __slots__ = ("_calculator", "_exercise_times", "_mesher")

    def __init__(
        self,
        exercise_dates: Sequence[Date],
        reference_date: Date,
        day_counter: DayCounter,
        mesher: FdmMesher,
        calculator: FdmInnerValueCalculatorLike,
    ) -> None:
        # C++ parity: the constructor body is exactly this loop.
        self._exercise_times: list[float] = [
            day_counter.year_fraction(reference_date, d) for d in exercise_dates
        ]
        self._mesher: FdmMesher = mesher
        self._calculator: FdmInnerValueCalculatorLike = calculator

    def exercise_times(self) -> list[float]:
        """The exercise schedule as year fractions off the reference date.

        # C++ parity: ``FdmBermudanStepCondition::exerciseTimes`` returns
        # ``const std::vector<Time>&``; Python returns a copy so the
        # schedule cannot be mutated from outside.
        """
        return list(self._exercise_times)

    def apply_to(self, a: Array, t: float) -> None:
        """Floor ``a`` at the inner value, but only exactly on an exercise time.

        # C++ parity: ``FdmBermudanStepCondition::applyTo``.
        """
        # C++ uses std::find + exact equality; `in` on a list of floats is
        # the same test.
        if t not in self._exercise_times:
            return

        layout = self._mesher.layout()
        qassert.require(
            layout.size() == a.size,
            f"inconsistent array dimensions: a has {a.size}, layout has {layout.size()}",
        )

        # C++ also fills a local `Array locations(dims)` inside the node loop
        # from mesher_->location(iter, i); nothing ever reads it. The port
        # omits that dead store.
        for iterator in layout.iter():
            inner_value = self._calculator.inner_value(iterator, t)
            # C++ writes only when strictly greater; max() returns its left
            # operand on a tie, so the two forms store the same double.
            a[iterator.index] = max(float(a[iterator.index]), inner_value)


__all__ = ["FdmBermudanStepCondition"]
