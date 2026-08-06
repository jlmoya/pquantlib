"""FdmSimpleSwingCondition — exercise decision for a swing option.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmsimpleswingcondition.{hpp,cpp}
# (v1.43).

A swing option grants a fixed number of exercise rights over a set of
exercise dates. One mesh direction (``swing_direction``) is discrete and
counts the rights *already used*: coordinate ``k`` along that direction
means "``k`` exercises consumed so far". The last coordinate,
``dim[swing_direction] - 1``, is the absorbing "all rights used" state.

At an exercise time the holder compares

  * doing nothing — value ``a[iter]``, versus
  * exercising once — the cashflow ``calculator.inner_value(iter, t)`` plus
    the value of the *same* node with one more right consumed, i.e. the
    neighbour one step along ``swing_direction``.

and takes the better. On top of that the ``min_exercises`` constraint
forces exercise when there are no longer enough remaining exercise dates
to satisfy the minimum: ``d`` is the number of exercise times from the
current one to the end (inclusive), so ``exercisesUsed + d <= minExercises``
means "even using every remaining date the minimum cannot be met unless we
exercise now".

All updates are written to a **copy** and swapped in at the end, so within
one sweep every node sees the pre-update neighbour value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.step_conditions.inner_value_calculator_protocol import (
    FdmInnerValueCalculatorLike,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)


@final
class FdmSimpleSwingCondition(StepCondition):
    """Swing-option exercise condition on a discrete exercise-count axis.

    # C++ parity: ``class FdmSimpleSwingCondition : public StepCondition<Array>``.
    """

    __slots__ = (
        "_calculator",
        "_exercise_times",
        "_mesher",
        "_min_exercises",
        "_swing_direction",
    )

    def __init__(
        self,
        exercise_times: Sequence[float],
        mesher: FdmMesher,
        calculator: FdmInnerValueCalculatorLike,
        swing_direction: int,
        min_exercises: int = 0,
    ) -> None:
        self._exercise_times: list[float] = list(exercise_times)
        self._mesher: FdmMesher = mesher
        self._calculator: FdmInnerValueCalculatorLike = calculator
        self._min_exercises: int = min_exercises
        self._swing_direction: int = swing_direction

    def apply_to(self, a: Array, t: float) -> None:
        """Apply the swing exercise decision at ``t``.

        # C++ parity: ``FdmSimpleSwingCondition::applyTo``.
        """
        layout = self._mesher.layout()
        # C++ computes maxExerciseValue before the time test; it is a pure
        # function of the layout, so the placement is immaterial.
        max_exercise_value = layout.dim()[self._swing_direction] - 1

        if t not in self._exercise_times:
            return

        ret_val = a.copy()

        # C++ `std::distance(iter, exerciseTimes_.end())`: the number of
        # exercise times from the matched one to the end, inclusive.
        d = len(self._exercise_times) - self._exercise_times.index(t)

        qassert.require(
            layout.size() == a.size,
            f"inconsistent array dimensions: a has {a.size}, layout has {layout.size()}",
        )

        for iterator in layout.iter():
            exercises_used = iterator.coordinates[self._swing_direction]

            if exercises_used < max_exercise_value:
                cashflow = self._calculator.inner_value(iterator, t)
                current_value = float(a[iterator.index])
                value_plus_one_exercise = float(a[layout.neighbourhood(iterator, self._swing_direction, 1)])

                if (
                    current_value < value_plus_one_exercise + cashflow
                    or exercises_used + d <= self._min_exercises
                ):
                    ret_val[iterator.index] = value_plus_one_exercise + cashflow

        # C++ `a = retVal;` — element-wise copy back into the caller's array.
        a[:] = ret_val


__all__ = ["FdmSimpleSwingCondition"]
