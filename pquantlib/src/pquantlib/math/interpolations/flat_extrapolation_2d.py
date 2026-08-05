"""Flat-extrapolation decorator for 2-D interpolations.

# C++ parity: ql/math/interpolations/flatextrapolation2d.hpp (v1.43).

Turns any :class:`~pquantlib.math.interpolations.interpolation_2d.Interpolation2D`
into one that extrapolates flat: outside the grid, both coordinates are
clamped to the boundary before the decorated surface is asked::

    value(x, y) = decorated(clamp(x, x_min, x_max), clamp(y, y_min, y_max))

That is the whole decorator. Everything else — ``x_min``/``x_max``,
``y_min``/``y_max``, ``x_values``/``y_values``, ``z_data``, ``locate_x``/
``locate_y``, ``is_in_range`` — forwards to the decorated interpolation
rather than being recomputed, exactly as the C++ Impl does.

Two details are easy to lose and are reproduced deliberately:

- The decorator carries **its own** extrapolation flag. ``is_in_range``
  delegates, so an out-of-range call still raises until
  ``enable_extrapolation()`` is called *on the decorator* — enabling it on
  the decorated surface does not help, because the decorator's own
  ``_check_range`` runs first against the decorator's own flag.
- The delegated call is made **without** ``allow_extrapolation``
  (C++ ``(*decoratedInterp_)(x, y)`` leaves the default ``false``). It is
  safe only because the arguments have already been clamped into range; the
  1-D ``FlatExtrapolator`` differs here and passes ``true``, matching *its*
  C++ counterpart. Both are faithful to their own header.

C++ also runs ``decoratedInterp_->update()`` from the Impl constructor and
again from ``calculate()``; :meth:`update` mirrors that.
"""

from __future__ import annotations

from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation_2d import Interpolation2D
from pquantlib.math.matrix import Matrix


class FlatExtrapolator2D(Interpolation2D):
    """Decorator making any 2-D interpolation extrapolate flat.

    # C++ parity: ``class FlatExtrapolator2D : public Interpolation2D``
    # (flatextrapolation2d.hpp:36-89), Impl at lines 43-88.
    """

    __slots__ = ("_decorated",)

    def __init__(self, decorated_interpolation: Interpolation2D) -> None:
        # The base keeps a copy of the grid, but every accessor that would
        # read it is overridden to delegate — matching C++, where the Impl
        # forwards xMin/xMax/xValues/locateX/yMin/yMax/yValues/locateY/zData/
        # isInRange to the decorated interpolation rather than caching them.
        super().__init__(
            decorated_interpolation.x_values,
            decorated_interpolation.y_values,
            decorated_interpolation.z_data,
        )
        self._decorated: Interpolation2D = decorated_interpolation
        # C++ parity: FlatExtrapolator2DImpl's ctor calls calculate(), which is
        # decoratedInterp_->update() (flatextrapolation2d.hpp:47, 61-63).
        self._decorated.update()

    @property
    def decorated_interpolation(self) -> Interpolation2D:
        """The interpolation this decorator wraps."""
        return self._decorated

    # ----- delegated accessors -------------------------------------------

    @property
    def x_min(self) -> float:
        return self._decorated.x_min

    @property
    def x_max(self) -> float:
        return self._decorated.x_max

    @property
    def y_min(self) -> float:
        return self._decorated.y_min

    @property
    def y_max(self) -> float:
        return self._decorated.y_max

    @property
    def x_values(self) -> Array:
        return self._decorated.x_values

    @property
    def y_values(self) -> Array:
        return self._decorated.y_values

    @property
    def z_data(self) -> Matrix:
        return self._decorated.z_data

    def locate_x(self, x: float) -> int:
        return self._decorated.locate_x(x)

    def locate_y(self, y: float) -> int:
        return self._decorated.locate_y(y)

    def is_in_range(self, x: float, y: float) -> bool:
        return self._decorated.is_in_range(x, y)

    def update(self) -> None:
        # C++ parity: FlatExtrapolator2DImpl::calculate (flatextrapolation2d.hpp:61-63).
        self._decorated.update()

    # ----- the decorated value -------------------------------------------

    def _value(self, x: float, y: float) -> float:
        # C++ parity: flatextrapolation2d.hpp:64-68.
        return self._decorated(self._bind_x(x), self._bind_y(y))

    def _bind_x(self, x: float) -> float:
        # C++ parity: flatextrapolation2d.hpp:73-79.
        if x < self.x_min:
            return self.x_min
        if x > self.x_max:
            return self.x_max
        return x

    def _bind_y(self, y: float) -> float:
        # C++ parity: flatextrapolation2d.hpp:80-86.
        if y < self.y_min:
            return self.y_min
        if y > self.y_max:
            return self.y_max
        return y


__all__ = ["FlatExtrapolator2D"]
