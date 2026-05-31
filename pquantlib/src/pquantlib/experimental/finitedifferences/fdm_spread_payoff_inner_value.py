"""FdmSpreadPayoffInnerValue — spread payoff inner-value calculator.

# C++ parity: ql/experimental/finitedifferences/fdmspreadpayoffinnervalue.hpp
# (v1.42.1).

Wraps a ``BasketPayoff`` and two per-axis inner-value calculators
(``calc1`` for asset 0 along one direction, ``calc2`` for asset 1
along another). At each grid node, computes the underlying values
from the two child calculators, packs them into a 2-element array
and passes them to the basket payoff's ``evaluate(prices)`` method.

Used by spread-option FD engines (Kluge electricity-vs-gas spreads).
"""

from __future__ import annotations

from typing import Protocol, final

from pquantlib.instruments.basket_option import BasketPayoff
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)


class _InnerValueCalculator(Protocol):
    """Inner-value calc protocol — same as ``FdmExpExtOUInnerValueCalculator``."""

    def inner_value(self, iter_: FdmLinearOpIterator, t: float) -> float: ...

    def avg_inner_value(self, iter_: FdmLinearOpIterator, t: float) -> float: ...


@final
class FdmSpreadPayoffInnerValue:
    """Inner-value calculator for a 2-asset spread payoff.

    # C++ parity: ``class FdmSpreadPayoffInnerValue : public
    # FdmInnerValueCalculator``.
    """

    __slots__ = ("_calc1", "_calc2", "_payoff")

    def __init__(
        self,
        payoff: BasketPayoff,
        calc1: _InnerValueCalculator,
        calc2: _InnerValueCalculator,
    ) -> None:
        self._payoff: BasketPayoff = payoff
        self._calc1: _InnerValueCalculator = calc1
        self._calc2: _InnerValueCalculator = calc2

    def inner_value(self, iter_: FdmLinearOpIterator, t: float) -> float:
        """Return ``payoff([calc1(iter, t), calc2(iter, t)])``.

        # C++ parity: ``innerValue(iter, t)``.
        """
        v1 = self._calc1.inner_value(iter_, t)
        v2 = self._calc2.inner_value(iter_, t)
        return self._payoff.evaluate([v1, v2])

    def avg_inner_value(self, iter_: FdmLinearOpIterator, t: float) -> float:
        """Same as ``inner_value`` (no spatial averaging).

        # C++ parity: ``avgInnerValue`` forwards to ``innerValue``.
        """
        return self.inner_value(iter_, t)

    def __call__(self, iter_: FdmLinearOpIterator, t: float) -> float:
        """Call-syntax for the ``InnerValueCalculator`` Callable Protocol."""
        return self.inner_value(iter_, t)


__all__ = ["FdmSpreadPayoffInnerValue"]
