"""Backward-flat in x, linear in y, over a 2-D grid.

# C++ parity: ql/math/interpolations/backwardflatlinearinterpolation.hpp (v1.43).

A hybrid 2-D interpolation: piecewise constant in the first component,
taking the value at the *right* end of each x-interval (backward-flat), and
piecewise linear in the second.

The x selection has three arms and they are not interchangeable::

    x <= xs[0]      -> column 0
    x == xs[i]      -> column i        (exact equality, i = locate_x(x))
    otherwise       -> column i + 1

The middle arm matters: at a node the value is that node's own column, not
the column to its right, so the function is right-continuous *except*
exactly on the nodes, where it takes the left limit. The equality is C++'s
literal ``x == this->xBegin_[i]``, not a tolerance comparison, and is
reproduced as such — a ``close_enough`` here would change the answer for
arguments one ULP off a node.

For ``x`` past the last abscissa ``locate_x`` clamps to ``n-2``, the equality
fails and the ``i+1`` arm picks the last column, so the surface is flat to
the right. For ``x`` below the first abscissa the first arm picks column 0,
so it is flat to the left too. The y direction, by contrast, extrapolates
linearly off both ends (``u`` simply leaves ``[0, 1]``).

**Indexing convention** (matches C++ ``zData_[j][i]``): ``z[y_index,
x_index]`` — rows are y, columns are x.
"""

from __future__ import annotations

from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation_2d import Interpolation2D
from pquantlib.math.matrix import Matrix


class BackwardflatLinearInterpolation(Interpolation2D):
    """Backward-flat in x, linear in y, on a grid indexed as ``z[y, x]``.

    # C++ parity: ``class BackwardflatLinearInterpolation : public Interpolation2D``
    # (backwardflatlinearinterpolation.hpp:77-89), Impl at lines 33-70.
    """

    __slots__ = ()

    def __init__(self, xs: Array, ys: Array, z: Matrix) -> None:
        super().__init__(xs, ys, z, required_points=2)

    def _value(self, x: float, y: float) -> float:
        # C++ parity: backwardflatlinearinterpolation.hpp:46-69.
        j = self.locate_y(y)
        z = self._z
        if x <= float(self._xs[0]):
            z1 = float(z[j, 0])
            z2 = float(z[j + 1, 0])
        else:
            i = self.locate_x(x)
            if x == float(self._xs[i]):
                z1 = float(z[j, i])
                z2 = float(z[j + 1, i])
            else:
                z1 = float(z[j, i + 1])
                z2 = float(z[j + 1, i + 1])

        u = (y - float(self._ys[j])) / float(self._ys[j + 1] - self._ys[j])
        return (1.0 - u) * z1 + u * z2


class BackwardflatLinear:
    """Backward-flat-linear-interpolation factory.

    # C++ parity: ``class BackwardflatLinear``
    # (backwardflatlinearinterpolation.hpp:91-99).

    See :class:`~pquantlib.math.interpolations.bilinear.Bilinear` for why the
    C++ traits struct becomes an instantiable class here.
    """

    __slots__ = ()

    def interpolate(
        self, xs: Array, ys: Array, z: Matrix
    ) -> BackwardflatLinearInterpolation:
        """Build a :class:`BackwardflatLinearInterpolation` over ``(xs, ys, z)``."""
        return BackwardflatLinearInterpolation(xs, ys, z)


__all__ = ["BackwardflatLinear", "BackwardflatLinearInterpolation"]
