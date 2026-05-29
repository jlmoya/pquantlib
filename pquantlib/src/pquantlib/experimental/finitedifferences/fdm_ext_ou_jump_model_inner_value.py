"""FdmExtOUJumpModelInnerValue — inner-value calculator for ExtOU+Jump payoffs.

# C++ parity: ql/experimental/finitedifferences/fdmextoujumpmodelinnervalue.hpp
# (v1.42.1).

The Kluge / Extended-Ornstein-Uhlenbeck-with-Jumps model represents
the log-spot as a sum of three components

.. math::

    \\log S(t) = f(t) + x(t) + y(t)

where ``x(t)`` is an OU process, ``y(t)`` is a compound Poisson jump
process, and ``f(t)`` is a deterministic shape function. The inner
value at grid coordinate ``(x, y)`` and time ``t`` is

.. math::

    \\text{inner}(x, y, t) = \\text{payoff}\\bigl(\\exp(f(t) + x + y)\\bigr)

with ``f(t)`` resolved via lower-bound lookup on the piecewise-
constant shape curve ``[(t_0, f_0), (t_1, f_1), ...]``.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from pquantlib.math.constants import QL_EPSILON
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)


@runtime_checkable
class _CallablePayoff(Protocol):
    """Minimal Payoff Protocol — just ``__call__(price) -> float``."""

    def __call__(self, price: float, /) -> float: ...


# Shape type: list of (time, log-offset) tuples sorted ascending by time.
Shape = list[tuple[float, float]]


class FdmExtOUJumpModelInnerValue:
    """Inner-value calculator for ExtOU+Jump payoffs.

    # C++ parity: ``class FdmExtOUJumpModelInnerValue : public
    # FdmInnerValueCalculator``.

    The shape curve is optional; when omitted ``f(t) = 0`` and the
    inner value reduces to ``payoff(exp(x + y))``.
    """

    def __init__(
        self,
        payoff: _CallablePayoff,
        mesher: FdmMesher,
        shape: Shape | None = None,
    ) -> None:
        self._payoff: _CallablePayoff = payoff
        self._mesher: FdmMesher = mesher
        self._shape: Shape | None = shape

    def inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Payoff at ``(x, y)`` and time ``t``.

        # C++ parity: ``FdmExtOUJumpModelInnerValue::innerValue`` —
        # ``payoff(exp(f(t) + x + y))`` where ``f(t)`` is the
        # piecewise-constant shape with a lower-bound lookup using
        # the ``sqrt(QL_EPSILON)`` tolerance.
        """
        x = self._mesher.location(iterator, 0)
        y = self._mesher.location(iterator, 1)
        f = self._resolve_shape(t)
        return self._payoff(math.exp(f + x + y))

    def avg_inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Same as ``inner_value`` (no averaging in C++).

        # C++ parity: ``FdmExtOUJumpModelInnerValue::avgInnerValue``
        # delegates to ``innerValue`` directly.
        """
        return self.inner_value(iterator, t)

    # --- helpers -------------------------------------------------------

    def _resolve_shape(self, t: float) -> float:
        """Piecewise-constant shape lookup at time ``t``.

        # C++ parity: ``std::lower_bound(shape->begin(), shape->end(),
        # std::pair<Time, Real>(t - sqrt(QL_EPSILON), 0.0))->second``.

        With the curve ``[(t_0, f_0), (t_1, f_1), ...]`` (sorted by
        time), ``lower_bound(t_eps)`` returns the first entry whose
        time is >= ``t_eps``. The shape value at ``t`` is ``f`` of
        that entry. ``t_eps = t - sqrt(QL_EPSILON)`` provides a small
        left-bias so a query at an exact knot returns *that* knot
        rather than the next one.
        """
        if self._shape is None:
            return 0.0
        # sqrt(QL_EPSILON) ≈ 1.49e-8 — matches C++ exactly.
        t_eps = t - math.sqrt(QL_EPSILON)
        # std::lower_bound semantics: return the first iter where
        # !(*iter < t_eps), i.e. the first entry with time >= t_eps.
        for time_pt, value in self._shape:
            if time_pt >= t_eps:
                return value
        # Past the end — return the last value (C++ would dereference
        # ``end()``, which is UB; in practice this branch is not hit
        # in valid usage).
        return self._shape[-1][1]


__all__ = ["FdmExtOUJumpModelInnerValue", "Shape"]
