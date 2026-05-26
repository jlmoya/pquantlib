"""Abstract base for 1-D interpolations.

# C++ parity: ql/math/interpolation.hpp (v1.42.1) — ``class Interpolation``.

C++ ``Interpolation`` is a thin handle around a PIMPL ``Impl`` exposing
``value(x)``, ``primitive(x)``, ``derivative(x)``, ``secondDerivative(x)``,
``xMin``/``xMax``, ``isInRange``, and ``update``. The Impl is a templated
``templateImpl<I1, I2>`` carrying iterator pairs into externally-owned
x and y sequences (C++ stores iterators — the *user* must keep the
underlying data alive).

The Python port collapses this to a concrete abstract class:

- We **copy** ``xs`` and ``ys`` into owned numpy arrays at construction.
  C++ used iterators to avoid copies; Python does not have the same
  lifetime concerns and the copy cost is negligible for the typical
  curve sizes (~50 nodes). Documented divergence.
- ``Extrapolator`` (C++ base mixin) collapses into the ``allow_extrapolation``
  kwarg on ``__call__``/``primitive``/``derivative``/``second_derivative``;
  no separate ``enableExtrapolation()`` flag class.
- ``Impl`` PIMPL hierarchy disappears — each concrete interpolation is
  one class inheriting from ``Interpolation`` directly.

Some methods may not apply to specific concretes (e.g. flat
interpolations have zero ``derivative`` and ``second_derivative``);
the abstract default ``second_derivative`` returns 0.0 to match the
C++ ``templateImpl::secondDerivative`` default. Concretes override.

The C++ ``locate(x)`` helper uses ``upper_bound`` on the half-open
sub-range ``[xBegin, xEnd-1)``. Replicated here in ``_locate`` via
``numpy.searchsorted(side='right')`` minus one, with the boundary
clamps ``x < xs[0] -> 0`` and ``x > xs[-1] -> n-2``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.closeness import close


class Interpolation(ABC):
    """Abstract base for 1-D interpolations.

    Subclasses must implement ``_value``. They may override ``_primitive``,
    ``_derivative``, ``_second_derivative`` if those have a closed form;
    default implementations raise ``NotImplementedError`` (callers wrap
    in ``LibraryException`` via ``__call__``-style entry points).
    """

    def __init__(self, x_seq: Array, y_seq: Array, *, required_points: int = 2) -> None:
        # Defensive copy + dtype normalization. C++ stores iterators
        # into externally-owned vectors; the Python port owns its data.
        xs = np.ascontiguousarray(x_seq, dtype=np.float64)
        ys = np.ascontiguousarray(y_seq, dtype=np.float64)
        qassert.require(
            xs.ndim == 1 and ys.ndim == 1,
            "Interpolation requires 1-D x and y sequences",
        )
        qassert.require(
            xs.shape[0] == ys.shape[0],
            f"x and y sequences must have the same length (got {xs.shape[0]} and {ys.shape[0]})",
        )
        qassert.require(
            xs.shape[0] >= required_points,
            f"not enough points to interpolate: at least {required_points} required, {xs.shape[0]} provided",
        )
        self._xs: Array = xs
        self._ys: Array = ys
        self._allow_extrapolation: bool = False

    # ----- public API -----------------------------------------------------

    def __call__(self, x: float, *, allow_extrapolation: bool = False) -> float:
        self._check_range(x, allow_extrapolation)
        return self._value(x)

    def primitive(self, x: float, *, allow_extrapolation: bool = False) -> float:
        self._check_range(x, allow_extrapolation)
        return self._primitive(x)

    def derivative(self, x: float, *, allow_extrapolation: bool = False) -> float:
        self._check_range(x, allow_extrapolation)
        return self._derivative(x)

    def second_derivative(self, x: float, *, allow_extrapolation: bool = False) -> float:
        self._check_range(x, allow_extrapolation)
        return self._second_derivative(x)

    def update(self) -> None:  # noqa: B027 — intentional default no-op; subclasses override
        """Recompute cached state if the underlying x/y data changed.

        Default no-op — subclasses that cache coefficients override (e.g.,
        ``LinearInterpolation`` recomputes slopes and the primitive array).
        Concretes that need no cached state (``BilinearInterpolation``
        when carved off the 1-D base) skip overriding.
        """

    @property
    def x_min(self) -> float:
        return float(self._xs[0])

    @property
    def x_max(self) -> float:
        return float(self._xs[-1])

    def is_in_range(self, x: float) -> bool:
        x1 = self.x_min
        x2 = self.x_max
        return (x1 <= x <= x2) or close(x, x1) or close(x, x2)

    @property
    def allows_extrapolation(self) -> bool:
        return self._allow_extrapolation

    def enable_extrapolation(self, b: bool = True) -> None:
        self._allow_extrapolation = b

    # ----- hooks for subclasses ------------------------------------------

    @abstractmethod
    def _value(self, x: float) -> float: ...

    def _primitive(self, x: float) -> float:
        del x
        raise NotImplementedError("primitive not implemented for this interpolation")

    def _derivative(self, x: float) -> float:
        del x
        raise NotImplementedError("derivative not implemented for this interpolation")

    def _second_derivative(self, x: float) -> float:
        # C++ ``templateImpl::secondDerivative`` defaults to 0.0 for the
        # linear-impl base; concretes override. We follow the C++ pattern.
        del x
        return 0.0

    # ----- helpers --------------------------------------------------------

    def _locate(self, x: float) -> int:
        """Index ``i`` such that ``xs[i] <= x < xs[i+1]`` (C++ ``locate``).

        Clamped at the boundaries:
        - ``x < xs[0]`` returns 0
        - ``x > xs[-1]`` returns n-2
        """
        xs = self._xs
        n = xs.shape[0]
        if x < float(xs[0]):
            return 0
        if x > float(xs[-1]):
            return n - 2
        # C++ ``upper_bound(xBegin, xEnd-1, x) - xBegin - 1``: search excludes
        # the last element, so for x == xs[-1] we still get n-2 (clamped above).
        # numpy.searchsorted(side='right') returns the upper_bound index into
        # the full array; subtract 1 for the C++ convention.
        idx = int(np.searchsorted(xs[:-1], x, side="right")) - 1
        return idx

    def _check_range(self, x: float, allow_extrapolation: bool) -> None:
        if allow_extrapolation or self._allow_extrapolation:
            return
        qassert.require(
            self.is_in_range(x),
            f"interpolation range is [{self.x_min}, {self.x_max}]: extrapolation at {x} not allowed",
        )
