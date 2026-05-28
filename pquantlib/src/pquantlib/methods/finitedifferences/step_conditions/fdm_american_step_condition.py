"""FdmAmericanStepCondition — early-exercise floor for American options.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmamericanstepcondition.{hpp,cpp}
# (v1.42.1).

At every time step the FD value is replaced by the **max** of itself
and the immediate-exercise payoff at each grid node — implementing
the American early-exercise barrier on the backward-induction sweep.

The C++ class takes an ``FdmInnerValueCalculator`` (computes the
payoff value at a given iter/time). The Python port simplifies: the
calculator interface is collapsed to a callable returning ``payoff(S)``
at each grid node. That keeps the L5-D scope tight — for the vanilla
single-asset path the inner value is always ``payoff(exp(x))``.

C++ also accepts an ``exerciseStart`` time. The Python port keeps
the same parameter for forward compatibility (American options can
also be configured with a *non-zero* exercise start, mirroring C++).
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)
from pquantlib.payoffs import Payoff


@final
class FdmAmericanStepCondition(StepCondition):
    """American early-exercise step condition.

    # C++ parity: ``class FdmAmericanStepCondition``.
    """

    def __init__(
        self,
        mesher: FdmMesher,
        payoff: Payoff,
        exercise_start: float = 0.0,
    ) -> None:
        self._mesher: FdmMesher = mesher
        self._payoff: Payoff = payoff
        self._exercise_start: float = exercise_start
        # Pre-compute spot values per node: spot = exp(log-spot).
        self._spots: Array = np.exp(mesher.locations(0))

    def apply_to(self, a: Array, t: float) -> None:
        """Replace ``a[i]`` with ``max(a[i], payoff(spot[i]))``.

        # C++ parity: ``FdmAmericanStepCondition::applyTo``.
        """
        if t < self._exercise_start:
            return
        qassert.require(
            a.size == self._mesher.layout().size(),
            f"inconsistent array dimensions: a has {a.size}, layout has {self._mesher.layout().size()}",
        )
        # Vectorised: a[i] <- max(a[i], payoff(spot[i])).
        # The payoff isn't vectorised — call element-wise.
        for i in range(a.size):
            inner = self._payoff(float(self._spots[i]))
            a[i] = max(a[i], inner)


__all__ = ["FdmAmericanStepCondition"]
