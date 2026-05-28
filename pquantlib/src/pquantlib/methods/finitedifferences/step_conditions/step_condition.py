"""StepCondition — abstract per-step in-place transform.

# C++ parity: ql/methods/finitedifferences/stepcondition.hpp
# (v1.42.1) — ``template <class array_type> class StepCondition``.

The C++ template is parameterised on the array type; concrete step
conditions (FdmAmericanStepCondition, FdmBermudanStepCondition, ...)
specialise on ``Array``. The Python port is a plain ABC because
the only array type used in L5-D is ``numpy.ndarray``.

A step condition's ``apply_to(a, t)`` mutates ``a`` in place at
time ``t``. Composite step conditions (``FdmStepConditionComposite``)
chain multiple conditions; callers apply the composite at each
backward step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pquantlib.math.array import Array


class StepCondition(ABC):
    """Abstract array-valued step condition.

    # C++ parity: ``class StepCondition<Array>``.
    """

    @abstractmethod
    def apply_to(self, a: Array, t: float) -> None:
        """Apply the step condition in place to ``a`` at time ``t``."""


__all__ = ["StepCondition"]
