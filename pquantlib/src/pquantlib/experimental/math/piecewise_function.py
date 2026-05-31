"""PiecewiseFunction — right-continuous piecewise-constant step function.

# C++ parity: ql/experimental/math/piecewisefunction.hpp @ v1.42.1 (099987f0).

The C++ source exposes this as the macro ``QL_PIECEWISE_FUNCTION(X, Y, x)``:

    Y[min(upper_bound(X, x) - X.begin(), len(Y) - 1)]

defining a right-continuous-with-left-limits (RCLL) step function that takes the
values ``Y[0], Y[1], ..., Y[n]`` on the intervals
``(-inf, X[0]), [X[0], X[1]), ..., [X[n-1], +inf)``.

The Python port wraps the same lookup in a small callable class. ``Y`` should
normally have length ``len(X) + 1``; extra ``Y`` values are ignored, and if
fewer are given the last value is held for the remaining intervals. With an
empty ``X`` the function is the constant ``Y[0]``.

Warning: if ``Y`` is empty the lookup raises (the C++ macro performs an invalid
access in that case, which we surface as an explicit error).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from pquantlib import qassert


class PiecewiseFunction:
    """Right-continuous piecewise-constant step function.

    :param x: the (ascending) breakpoints.
    :param y: the per-interval values; ``len(y)`` normally ``len(x) + 1``.
    """

    __slots__ = ("_x", "_y")

    def __init__(self, x: Sequence[float], y: Sequence[float]) -> None:
        qassert.require(len(y) > 0, "PiecewiseFunction needs at least one y value")
        self._x = list(x)
        self._y = list(y)

    def __call__(self, x: float) -> float:
        """Value of the step function at ``x`` (RCLL semantics)."""
        # upper_bound: first index with X[idx] > x  ==  bisect_right.
        idx = bisect_right(self._x, x)
        return self._y[min(idx, len(self._y) - 1)]
