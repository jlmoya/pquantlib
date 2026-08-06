"""FdmSimpleStorageCondition — inject/withdraw decision for gas storage.

# C++ parity: ql/methods/finitedifferences/stepconditions/fdmsimplestoragecondition.{hpp,cpp}
# (v1.43).

The mesh is 2-D: direction 0 carries the (log) commodity price, direction 1
the amount currently in storage. At each exercise time the operator may
change the stored volume by at most ``change_rate`` in either direction,
paying/receiving the spot price — ``calculator.inner_value(iter, t)`` — per
unit moved.

The value after the decision is the best of

  * doing nothing:        ``a[iter]``
  * injecting the max:    ``V(x, y + maxInject) - price * maxInject``
  * withdrawing the max:  ``V(x, y - maxWithDraw) + price * maxWithDraw``

(the "bang-bang-wait" strategy), and then, because the value function is not
necessarily concave in the stored volume, C++ also scans every *grid* volume
strictly inside ``(y - maxWithDraw, y + maxInject)`` and takes the best of
those too.

``V`` is the current solution re-read through a ``BilinearInterpolation``
over the (x, y) grid, which is how off-grid volumes ``y ± change`` get a
value at all.

Interpolation parity note
-------------------------
C++ builds ``Matrix m(y_.size(), x_.size())`` and ``std::copy``s the flat
solution vector into it, i.e. ``m[j][i] == a[j * x_.size() + i]`` — which is
exactly the layout's flat index ``i * spacing[0] + j * spacing[1]`` for a
2-D mesher. A C-order ``reshape((len(y), len(x)))`` reproduces that byte for
byte. The interpolant is then queried with C++'s default
``allowExtrapolation = false``; ``PQuantLib``'s ``Interpolation2D`` applies
the same ``close()``-tolerant range check, so the boundary queries
``y - maxWithDraw == y_[0]`` and ``y + maxInject == y_[-1]`` behave
identically instead of tripping a spurious range error.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.bilinear import BilinearInterpolation
from pquantlib.methods.finitedifferences.meshers.fdm_mesher import FdmMesher
from pquantlib.methods.finitedifferences.step_conditions.inner_value_calculator_protocol import (
    FdmInnerValueCalculatorLike,
)
from pquantlib.methods.finitedifferences.step_conditions.step_condition import (
    StepCondition,
)


@final
class FdmSimpleStorageCondition(StepCondition):
    """Gas-storage inject/withdraw step condition on a 2-D (price, volume) mesh.

    # C++ parity: ``class FdmSimpleStorageCondition : public StepCondition<Array>``.
    """

    __slots__ = (
        "_calculator",
        "_change_rate",
        "_exercise_times",
        "_mesher",
        "_x",
        "_y",
    )

    def __init__(
        self,
        exercise_times: Sequence[float],
        mesher: FdmMesher,
        calculator: FdmInnerValueCalculatorLike,
        change_rate: float,
    ) -> None:
        self._exercise_times: list[float] = list(exercise_times)
        self._mesher: FdmMesher = mesher
        self._calculator: FdmInnerValueCalculatorLike = calculator
        self._change_rate: float = change_rate

        # C++ parity: harvest the per-direction axes by walking the layout and
        # picking the nodes on each axis' zero-line.
        layout = mesher.layout()
        xs: list[float] = []
        ys: list[float] = []
        for iterator in layout.iter():
            if iterator.coordinates[1] == 0:
                xs.append(mesher.location(iterator, 0))
            if iterator.coordinates[0] == 0:
                ys.append(mesher.location(iterator, 1))
        self._x: Array = np.array(xs, dtype=np.float64)
        self._y: Array = np.array(ys, dtype=np.float64)

    def apply_to(self, a: Array, t: float) -> None:
        """Apply the storage inject/withdraw decision at ``t``.

        # C++ parity: ``FdmSimpleStorageCondition::applyTo``.
        """
        if t not in self._exercise_times:
            return

        # C++ places this QL_REQUIRE *after* building the Matrix and the
        # interpolant. Python has to check first: reshaping a wrong-length
        # array raises numpy's ValueError, which would mask the library error
        # C++ reports. Still inside the time test, so a wrong-length array at a
        # non-exercise time stays a no-op exactly as in C++.
        layout = self._mesher.layout()
        qassert.require(
            layout.size() == a.size,
            f"inconsistent array dimensions: a has {a.size}, layout has {layout.size()}",
        )

        ret_val: Array = np.empty(a.size, dtype=np.float64)

        n_x = int(self._x.shape[0])
        n_y = int(self._y.shape[0])
        # C++: Matrix m(y_.size(), x_.size()); std::copy(a.begin(), a.end(), m.begin());
        # np.array(...) copies; a reshaped view would alias `a`, and
        # PQuantLib's Interpolation2D keeps whatever array it is handed
        # (np.ascontiguousarray is a no-op on contiguous float64 input) rather
        # than copying, despite what its docstring says.
        m = np.array(a, dtype=np.float64).reshape(n_y, n_x)
        interpl = BilinearInterpolation(self._x, self._y, m)

        y_front = float(self._y[0])
        y_back = float(self._y[-1])

        for iterator in layout.iter():
            coor = iterator.coordinates
            x = float(self._x[coor[0]])
            y = float(self._y[coor[1]])

            price = self._calculator.inner_value(iterator, t)

            max_with_draw = min(y - y_front, self._change_rate)
            sell_price = interpl(x, y - max_with_draw)

            max_inject = min(y_back - y, self._change_rate)
            buy_price = interpl(x, y + max_inject)

            # bang-bang-wait strategy. C++ std::max on an initializer_list and
            # Python's max() both return the leftmost maximum, so ties agree.
            current_value = max(
                float(a[iterator.index]),
                buy_price - price * max_inject,
                sell_price + price * max_with_draw,
            )

            # check if intermediate grid points give a better value.
            # C++ std::upper_bound == numpy.searchsorted(side="right").
            y_index = int(np.searchsorted(self._y, y - max_with_draw, side="right"))
            while y_index < n_y:
                y_node = float(self._y[y_index])
                if not y_node < y + max_inject:
                    break
                if y_node != y:
                    change = y_node - y
                    storage_price = interpl(x, y_node)
                    current_value = max(current_value, storage_price - change * price)
                y_index += 1

            ret_val[iterator.index] = current_value

        # C++ `a = retVal;` — element-wise copy back into the caller's array.
        a[:] = ret_val


__all__ = ["FdmSimpleStorageCondition"]
