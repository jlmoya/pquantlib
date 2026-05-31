"""FdmExpExtOUInnerValueCalculator — payoff on ``exp(f(t) + x)``.

# C++ parity: ql/experimental/finitedifferences/fdmexpextouinnervaluecalculator.hpp
# (v1.42.1).

At a grid node with location ``u`` (along the active ``direction``)
and optional time-dependent seasonality shape ``f(t)``, returns

    payoff(exp(f(t) + u))

If ``shape`` is None, ``f(t) = 0``. The shape is a list of
``(time, value)`` pairs; the C++ binary search uses ``lower_bound``
to find the seasonal value at the queried time (within
``sqrt(QL_EPSILON)`` tolerance).

The Python port mirrors this exactly.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib.math.array import Array
from pquantlib.math.constants import QL_EPSILON
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)
from pquantlib.payoffs import Payoff

Shape = list[tuple[float, float]]


@final
class FdmExpExtOUInnerValueCalculator:
    """Inner-value calculator for an exp-OU grid.

    # C++ parity: ``class FdmExpExtOUInnerValueCalculator : public
    # FdmInnerValueCalculator``.
    """

    __slots__ = ("_direction", "_mesher", "_payoff", "_shape")

    def __init__(
        self,
        payoff: Payoff,
        mesher: FdmMesher,
        shape: Shape | None = None,
        direction: int = 0,
    ) -> None:
        self._payoff: Payoff = payoff
        self._mesher: FdmMesher = mesher
        self._shape: Shape | None = shape
        self._direction: int = direction

    def inner_value(self, iter_: FdmLinearOpIterator, t: float) -> float:
        """Return ``payoff(exp(f(t) + u))`` at the grid node.

        # C++ parity: ``innerValue(iter, t)``.
        """
        u = self._mesher.location(iter_, self._direction)
        f = self._seasonal_shape(t)
        return self._payoff(math.exp(f + u))

    def avg_inner_value(self, iter_: FdmLinearOpIterator, t: float) -> float:
        """Same as ``inner_value`` — the average is degenerate at a single node.

        # C++ parity: ``avgInnerValue`` forwards to ``innerValue``.
        """
        return self.inner_value(iter_, t)

    def __call__(self, iter_: FdmLinearOpIterator, t: float) -> float:
        """Call-syntax for the ``InnerValueCalculator`` Callable Protocol."""
        return self.inner_value(iter_, t)

    def _seasonal_shape(self, t: float) -> float:
        """Look up the seasonal shape value at time ``t`` via lower_bound.

        # C++ parity: shape lookup via std::lower_bound on
        # ``(time - sqrt(QL_EPSILON), 0.0)``.
        """
        if self._shape is None:
            return 0.0
        target_t = t - math.sqrt(QL_EPSILON)
        # Find the first entry whose time >= target_t (lower_bound).
        # Match C++: lower_bound(begin, end, (t - eps, 0.0)) — compares
        # by `time` since pair-of-(time, value) is lex.
        for time_val, val in self._shape:
            if time_val >= target_t:
                return val
        # Falls past end — match C++ undefined behaviour by returning
        # the last value (typical seasonal interpretation).
        return self._shape[-1][1]


# Re-export array for convenience (typing of innerValue across array-typed
# operations).
__all__ = ["Array", "FdmExpExtOUInnerValueCalculator", "Shape"]
