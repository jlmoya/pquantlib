"""Inner-value calculators — the payoff seen by an FD grid node.

# C++ parity: ql/methods/finitedifferences/utilities/fdminnervaluecalculator.{hpp,cpp}
# @ v1.43 (6b57206e0).

Five classes live in this C++ header:

* :class:`FdmInnerValueCalculator` — the ABC (``innerValue`` / ``avgInnerValue``).
* :class:`FdmCellAveragingInnerValue` — payoff at the node, plus a cell-averaged
  variant that Simpson-integrates the payoff over ``[loc - dminus/2, loc + dplus/2]``.
* :class:`FdmLogInnerValue` — ``FdmCellAveragingInnerValue`` with ``exp`` as the
  grid mapping (log-spot grids).
* :class:`FdmLogBasketInnerValue` — multi-asset log grid fed to a ``BasketPayoff``.
* :class:`FdmZeroInnerValue` — constant 0.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import final

from pquantlib.exceptions import LibraryException
from pquantlib.instruments.basket_option import BasketPayoff
from pquantlib.math.integrals.simpson import SimpsonIntegral
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpIterator,
)
from pquantlib.payoffs import Payoff


class FdmInnerValueCalculator(ABC):
    """Abstract inner-value calculator.

    # C++ parity: ``class FdmInnerValueCalculator``.
    """

    @abstractmethod
    def inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Payoff at the node addressed by ``iterator``.

        # C++ parity: ``innerValue(const FdmLinearOpIterator&, Time)``.
        """

    @abstractmethod
    def avg_inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Cell-averaged payoff at the node addressed by ``iterator``.

        # C++ parity: ``avgInnerValue(const FdmLinearOpIterator&, Time)``.
        """


def _identity(x: float) -> float:
    """# C++ parity: the default ``gridMapping`` ``[](Real x){ return x; }``."""
    return x


class FdmCellAveragingInnerValue(FdmInnerValueCalculator):
    """Payoff evaluated at a node, with an optional cell-average variant.

    # C++ parity: ``class FdmCellAveragingInnerValue : public
    # FdmInnerValueCalculator``.
    """

    def __init__(
        self,
        payoff: Payoff,
        mesher: FdmMesher,
        direction: int,
        grid_mapping: Callable[[float], float] = _identity,
    ) -> None:
        self._payoff: Payoff = payoff
        self._mesher: FdmMesher = mesher
        self._direction: int = direction
        self._grid_mapping: Callable[[float], float] = grid_mapping
        self._avg_inner_values: list[float] = []

    def inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """``payoff(gridMapping(location))``.

        # C++ parity: ``FdmCellAveragingInnerValue::innerValue`` — ``t`` is
        # unused (the C++ parameter is unnamed).
        """
        del t
        loc = self._mesher.location(iterator, self._direction)
        return self._payoff(self._grid_mapping(loc))

    def avg_inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Cell-averaged payoff, cached per coordinate along ``direction``.

        # C++ parity: ``FdmCellAveragingInnerValue::avgInnerValue`` — the
        # cache is filled on the first call by walking the layout once and
        # computing one value per distinct coordinate.
        """
        layout = self._mesher.layout()
        if not self._avg_inner_values:
            size = layout.dim()[self._direction]
            values: list[float] = [0.0] * size
            initialized: list[bool] = [False] * size
            for i in layout.iter():
                xn = i.coordinates[self._direction]
                if not initialized[xn]:
                    initialized[xn] = True
                    values[xn] = self._avg_inner_value_calc(i, t)
            self._avg_inner_values = values

        return self._avg_inner_values[iterator.coordinates[self._direction]]

    def _avg_inner_value_calc(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Simpson average of the payoff over the node's cell.

        # C++ parity: ``FdmCellAveragingInnerValue::avgInnerValueCalc``.

        Boundary nodes fall back to the point value. Any ``Error`` thrown
        by the integrator (including the ``Integrator`` constructor's
        accuracy check when ``f(a)+f(b)`` underflows) makes C++ fall back
        to the point value too; ``LibraryException`` is the Python analogue.
        """
        dim = self._mesher.layout().dim()[self._direction]
        coord = iterator.coordinates[self._direction]

        if coord == 0 or coord == dim - 1:
            return self.inner_value(iterator, t)

        loc = self._mesher.location(iterator, self._direction)
        a = loc - self._mesher.dminus(iterator, self._direction) / 2.0
        b = loc + self._mesher.dplus(iterator, self._direction) / 2.0

        # C++ parity: the anonymous-namespace ``mapped_payoff`` functor.
        def f(x: float) -> float:
            return self._payoff(self._grid_mapping(x))

        try:
            fa = f(a)
            fb = f(b)
            acc = (fa + fb) * 5e-5 if (fa != 0.0 or fb != 0.0) else 1e-4
            return SimpsonIntegral(acc, 8)(f, a, b) / (b - a)
        except LibraryException:
            # use default value
            return self.inner_value(iterator, t)


@final
class FdmLogInnerValue(FdmCellAveragingInnerValue):
    """Cell-averaging inner value on a log-spot grid.

    # C++ parity: ``class FdmLogInnerValue : public
    # FdmCellAveragingInnerValue`` — the grid mapping is ``std::exp``.
    """

    def __init__(self, payoff: Payoff, mesher: FdmMesher, direction: int) -> None:
        super().__init__(payoff, mesher, direction, math.exp)


@final
class FdmLogBasketInnerValue(FdmInnerValueCalculator):
    """Basket payoff over a multi-asset log grid.

    # C++ parity: ``class FdmLogBasketInnerValue : public
    # FdmInnerValueCalculator``.
    """

    def __init__(self, payoff: BasketPayoff, mesher: FdmMesher) -> None:
        self._payoff: BasketPayoff = payoff
        self._mesher: FdmMesher = mesher

    def inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """``payoff(exp(location) for each direction)``.

        # C++ parity: ``FdmLogBasketInnerValue::innerValue`` — ``t`` unused.
        """
        del t
        n = len(self._mesher.layout().dim())
        x = [math.exp(self._mesher.location(iterator, i)) for i in range(n)]
        return self._payoff.evaluate(x)

    def avg_inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Same as :meth:`inner_value`.

        # C++ parity: ``FdmLogBasketInnerValue::avgInnerValue``.
        """
        return self.inner_value(iterator, t)


@final
class FdmZeroInnerValue(FdmInnerValueCalculator):
    """Inner value that is identically zero.

    # C++ parity: ``class FdmZeroInnerValue : public FdmInnerValueCalculator``.
    """

    def inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Always ``0.0``. # C++ parity: ``return 0.0;``."""
        del iterator, t
        return 0.0

    def avg_inner_value(self, iterator: FdmLinearOpIterator, t: float) -> float:
        """Always ``0.0``. # C++ parity: ``return 0.0;``."""
        del iterator, t
        return 0.0


__all__ = [
    "FdmCellAveragingInnerValue",
    "FdmInnerValueCalculator",
    "FdmLogBasketInnerValue",
    "FdmLogInnerValue",
    "FdmZeroInnerValue",
]
