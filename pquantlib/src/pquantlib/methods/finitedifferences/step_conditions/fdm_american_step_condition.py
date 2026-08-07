"""FdmAmericanStepCondition — early-exercise floor for American options.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmamericanstepcondition.{hpp,cpp}
# (v1.43).

At every time step the FD value is replaced by the **max** of itself
and the immediate-exercise value at each grid node — implementing the
American early-exercise barrier on the backward-induction sweep.

C++ takes an ``FdmInnerValueCalculator`` and calls
``calculator->innerValue(iter, t)`` per node (fdmamericanstepcondition.cpp:41).
This port accepts that, and additionally keeps the earlier
``Payoff``-shaped convenience form, which is exactly
``payoff(exp(mesher.location(iter, 0)))``.

The distinction is not cosmetic. ``payoff(exp(x))`` is only the inner
value when direction 0 of the mesh is a log-spot — true for the
Black-Scholes engines that were the original callers, false for every
model whose inner value comes from a shape curve or a basket (the
extended-OU / Kluge power engines, for one, where direction 0 is an OU
factor and the price is ``shape(t) * exp(x)``). Those engines must pass
a calculator.
"""

from __future__ import annotations

from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.step_conditions.inner_value_calculator_protocol import (
    FdmInnerValueCalculatorLike,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)
from pquantlib.payoffs import Payoff


@final
class FdmAmericanStepCondition(StepCondition):
    """American early-exercise step condition.

    # C++ parity: ``class FdmAmericanStepCondition``.

    Args:
        mesher: the FD mesh.
        payoff: either an ``FdmInnerValueCalculator``-shaped object (C++
            parity) or a ``Payoff``, in which case the inner value is
            ``payoff(exp(location(iter, 0)))``.
        exercise_start: no early exercise before this time
            (``if (t < exerciseStart_) return;``).
    """

    def __init__(
        self,
        mesher: FdmMesher,
        payoff: Payoff | FdmInnerValueCalculatorLike,
        exercise_start: float = 0.0,
    ) -> None:
        self._mesher: FdmMesher = mesher
        self._exercise_start: float = exercise_start
        self._calculator: FdmInnerValueCalculatorLike | None = (
            payoff if isinstance(payoff, FdmInnerValueCalculatorLike) else None
        )
        self._payoff: Payoff | None = None
        self._spots: Array | None = None
        if self._calculator is None:
            assert isinstance(payoff, Payoff)
            self._payoff = payoff
            # Pre-compute spot values per node: spot = exp(log-spot).
            self._spots = np.exp(mesher.locations(0))

    def apply_to(self, a: Array, t: float) -> None:
        """Replace ``a[i]`` with ``max(a[i], inner_value(i, t))``.

        # C++ parity: ``FdmAmericanStepCondition::applyTo``.
        """
        if t < self._exercise_start:
            return
        qassert.require(
            a.size == self._mesher.layout().size(),
            f"inconsistent array dimensions: a has {a.size}, layout has {self._mesher.layout().size()}",
        )
        if self._calculator is not None:
            # C++ writes `if (innerValue > a[i]) a[i] = innerValue;`; max()
            # is the same for finite values and NaN-free grids.
            for it in self._mesher.layout().iter():
                inner = self._calculator.inner_value(it, t)
                a[it.index] = max(a[it.index], inner)
            return
        assert self._payoff is not None
        assert self._spots is not None
        # The payoff isn't vectorised — call element-wise.
        for i in range(a.size):
            inner = self._payoff(float(self._spots[i]))
            a[i] = max(a[i], inner)


__all__ = ["FdmAmericanStepCondition"]
