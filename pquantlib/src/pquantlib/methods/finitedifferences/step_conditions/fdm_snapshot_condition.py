"""FdmSnapshotCondition — capture the solution vector at one instant.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmsnapshotcondition.{hpp,cpp}
# (v1.43).

A step condition that never modifies the solution: it only records a
copy of it when the backward sweep passes *exactly* through ``t``.
``FdmStepConditionComposite::joinConditions`` glues one of these onto an
existing composite so a solver can read the grid out at an intermediate
time (used by ``FdmBackwardSolver``-driven engines that need the value
surface at, say, a barrier-monitoring date).

Two details of the C++ implementation are load-bearing and preserved:

* the time test is **exact** ``t == t_`` — no tolerance. The solver is
  responsible for putting a grid point exactly on ``t_`` (which is why
  ``joinConditions`` pushes ``c1->getTime()`` into the stopping times).
* ``values_ = a`` is a **copy** (C++ ``Array::operator=``), not an alias,
  so later mutation of the solution vector cannot corrupt the snapshot.
  ``values_`` is ``mutable`` in C++ precisely because ``applyTo`` is
  ``const``; Python has no such constraint.
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)


@final
class FdmSnapshotCondition(StepCondition):
    """Record the solution vector when the sweep reaches time ``t``.

    # C++ parity: ``class FdmSnapshotCondition : public StepCondition<Array>``.
    """

    __slots__ = ("_t", "_values")

    def __init__(self, t: float) -> None:
        self._t: float = t
        # C++ default-constructs `Array values_` — size 0 until the first
        # snapshot is taken.
        self._values: Array = np.empty(0, dtype=np.float64)

    def apply_to(self, a: Array, t: float) -> None:
        """Store a copy of ``a`` iff ``t`` equals the snapshot time exactly.

        # C++ parity: ``FdmSnapshotCondition::applyTo`` — note the array is
        # never modified.
        """
        if t == self._t:
            self._values = a.copy()

    def get_time(self) -> float:
        """The snapshot time.

        # C++ parity: ``FdmSnapshotCondition::getTime``.
        """
        return self._t

    def get_values(self) -> Array:
        """The snapshotted values; length 0 until a snapshot has been taken.

        # C++ parity: ``FdmSnapshotCondition::getValues`` returns
        # ``const Array&``. Python has no const, so this returns the stored
        # array itself (same convention as ``Fdm1dMesher.locations``):
        # callers must not mutate it.
        """
        return self._values


__all__ = ["FdmSnapshotCondition"]
