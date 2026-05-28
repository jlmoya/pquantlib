"""DiscretizedOption — option on a discretized underlying.

# C++ parity: ql/discretizedasset.{hpp,cpp} (v1.42.1) — ``class
#             DiscretizedOption``.

Concrete subclass of ``DiscretizedAsset`` that carries a reference
to an underlying discretized asset plus an exercise specification
(type + exercise times). The ``post_adjust_values_impl`` template
method does the exercise check at every node where ``t`` lies on an
exercise date.

In the lattice rollback C++ goes backwards in time; this is why the
``postAdjustValuesImpl`` rolls the *underlying* back to ``time()``
first (via ``partialRollback``), so the option's exercise condition
``max(underlying, continuation)`` can read the just-rolled underlying
values.

The ``reset(size)`` method allocates the option's value array to
zeros and then triggers the initial pre+post adjustment.

The ``mandatory_times()`` method unions the underlying's mandatory
times with the non-negative exercise times.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from pquantlib.exceptions import LibraryException
from pquantlib.exercise import Exercise
from pquantlib.methods.lattices.discretized_asset import DiscretizedAsset


class DiscretizedOption(DiscretizedAsset):
    """Discretized option on a ``DiscretizedAsset`` underlying.

    # C++ parity: ``DiscretizedOption`` (discretizedasset.hpp:160-176 +
    # .cpp:25-51).
    """

    def __init__(
        self,
        underlying: DiscretizedAsset,
        exercise_type: Exercise.Type,
        exercise_times: Sequence[float],
    ) -> None:
        super().__init__()
        # C++ parity: ``DiscretizedOption`` ctor — moves underlying +
        # exerciseTimes into the held members. Python copies the list
        # so the caller's mutations cannot bleed in.
        self._underlying: DiscretizedAsset = underlying
        self._exercise_type: Exercise.Type = exercise_type
        self._exercise_times: list[float] = list(exercise_times)

    # -- low-level interface ----------------------------------------------

    def reset(self, size: int) -> None:
        """Allocate ``values`` zero-filled then run initial adjustment.

        # C++ parity: ``DiscretizedOption::reset`` (discretizedasset.hpp:221-227).

        Validates that the option and the underlying share the same
        ``Lattice`` instance (C++ uses pointer equality on
        ``shared_ptr<Lattice>``).
        """
        m_opt = self.method()
        m_und = self._underlying.method()
        if m_opt is not m_und:
            raise LibraryException(
                "option and underlying were initialized on different methods",
            )
        self._values = np.zeros(size, dtype=np.float64)
        self.adjust_values()

    def mandatory_times(self) -> list[float]:
        """Union of underlying's times with non-negative exercise times.

        # C++ parity: ``DiscretizedOption::mandatoryTimes``
        # (discretizedasset.hpp:229-237).
        """
        # Copy underlying's list (it may not be deduped).
        times = list(self._underlying.mandatory_times())
        # Append the positive (>= 0) exercise times in declaration order
        # — C++ ``find_if(t >= 0.0)`` then ``insert(end, i, end)``.
        for t in self._exercise_times:
            if t >= 0.0:
                times.append(t)
        return times

    # -- exercise machinery ----------------------------------------------

    def _apply_exercise_condition(self) -> None:
        """Pointwise ``max(underlying, current)`` over ``values``.

        # C++ parity: ``DiscretizedOption::applyExerciseCondition``
        # (discretizedasset.hpp:239-242).
        """
        # numpy element-wise max — C++ does a plain loop.
        self._values = np.maximum(self._underlying.values, self._values)

    def _post_adjust_values_impl(self) -> None:
        """Roll underlying to ``time``, then apply exercise condition.

        # C++ parity: ``DiscretizedOption::postAdjustValuesImpl``
        # (discretizedasset.cpp:25-51).
        """
        # Step 1 — roll the underlying back to our current time, then
        # pre-adjust it. Comment in C++ source: "with time flowing
        # backward, options must be exercised before performing the
        # [final] adjustment."
        self._underlying.partial_rollback(self._time)
        self._underlying.pre_adjust_values()

        # Step 2 — apply the exercise condition at the right times.
        et = self._exercise_type
        if et == Exercise.Type.American:
            # American: continuously exercisable between ``[earliest, latest]``.
            # The C++ code reads ``exerciseTimes_[0]`` (earliest) and
            # ``exerciseTimes_[1]`` (latest) — strictly two-element list.
            if (
                self._time >= self._exercise_times[0]
                and self._time <= self._exercise_times[1]
            ):
                self._apply_exercise_condition()
        elif et in (Exercise.Type.Bermudan, Exercise.Type.European):
            # Bermudan/European: exercise at any (possibly the only)
            # exercise date that the lattice is currently on.
            for t in self._exercise_times:
                if t >= 0.0 and self.is_on_time(t):
                    self._apply_exercise_condition()
        else:
            # C++ parity: ``QL_FAIL("invalid exercise type")``.
            raise LibraryException(f"invalid exercise type: {et!r}")

        # Step 3 — post-adjust the underlying after the option has
        # had its say.
        self._underlying.post_adjust_values()

    # -- accessors --------------------------------------------------------

    @property
    def underlying(self) -> DiscretizedAsset:
        return self._underlying

    @property
    def exercise_type(self) -> Exercise.Type:
        return self._exercise_type

    @property
    def exercise_times(self) -> list[float]:
        # Defensive copy — callers should not mutate our state.
        return list(self._exercise_times)
