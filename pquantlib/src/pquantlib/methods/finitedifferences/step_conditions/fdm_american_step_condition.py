"""FdmAmericanStepCondition — early-exercise floor for American options.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmamericanstepcondition.{hpp,cpp}
# (v1.43) — ``FdmAmericanStepCondition(ext::shared_ptr<FdmMesher>,
# ext::shared_ptr<FdmInnerValueCalculator>, Time exerciseStart)``.

At every time step from ``exercise_start`` onwards the FD value is
replaced by the **max** of itself and the immediate-exercise value at
each grid node — implementing the American early-exercise barrier on the
backward-induction sweep.

The immediate-exercise value comes from an
:class:`~pquantlib.methods.finitedifferences.step_conditions.inner_value_calculator_protocol.FdmInnerValueCalculatorLike`,
exactly as in C++. An earlier revision of this port took a bare
``Payoff`` and hard-coded ``payoff(exp(location(iter, 0)))``; that is only
correct for a plain single-asset log-spot grid and silently mispriced
every engine whose exercise value is *not* that expression — the escrowed
cash-dividend model (``FdmEscrowedLogInnerValueCalculator``), the shout
engine (``FdmShoutLogInnerValueCalculator``), and every multi-dimensional
grid (CIR, 2-D basket, n-dim basket), where direction 0 is not the only
axis.
"""

from __future__ import annotations

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
class FdmAmericanStepCondition(StepCondition):
    """American early-exercise step condition.

    # C++ parity: ``class FdmAmericanStepCondition : public StepCondition<Array>``.
    """

    __slots__ = ("_calculator", "_exercise_start", "_mesher")

    def __init__(
        self,
        mesher: FdmMesher,
        calculator: FdmInnerValueCalculatorLike,
        exercise_start: float = 0.0,
    ) -> None:
        self._mesher: FdmMesher = mesher
        self._calculator: FdmInnerValueCalculatorLike = calculator
        self._exercise_start: float = exercise_start

    def apply_to(self, a: Array, t: float) -> None:
        """Replace ``a[i]`` with ``max(a[i], innerValue(i, t))``.

        # C++ parity: ``FdmAmericanStepCondition::applyTo``.
        """
        if t < self._exercise_start:
            return
        layout = self._mesher.layout()
        qassert.require(
            a.size == layout.size(),
            f"inconsistent array dimensions: a has {a.size}, layout has {layout.size()}",
        )
        for iterator in layout.iter():
            inner_value = self._calculator.inner_value(iterator, t)
            # C++ writes only when strictly greater; max() returns its left
            # operand on a tie, so the two forms store the same double.
            a[iterator.index] = max(float(a[iterator.index]), inner_value)


__all__ = ["FdmAmericanStepCondition"]
