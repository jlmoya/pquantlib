"""Flat-extrapolation decorator for 1-D interpolations.

# C++ parity: ql/math/interpolations/flatextrapolation.hpp (v1.43) —
# new in that release.

Turns any 1-D interpolation into one that extrapolates flat — clamping to
the boundary values — outside the original data range::

    value(x)             = decorated(clamp(x, x_min, x_max))
    derivative(x)        = 0                            if x < x_min or x > x_max
                         = decorated.derivative(x)      otherwise
    second_derivative(x) = 0                            if x < x_min or x > x_max
                         = decorated.second_derivative(x)  otherwise
    primitive(x)         = P(x_min) + f(x_min) * (x - x_min)   if x < x_min
                         = P(x_max) + f(x_max) * (x - x_max)   if x > x_max
                         = decorated.primitive(x)              otherwise

Three details are easy to get wrong and are reproduced deliberately:

- The decorator carries **its own** extrapolation flag. Enabling
  extrapolation on the decorated interpolation does not propagate: an
  out-of-range call still raises until ``enable_extrapolation()`` is called
  on the decorator. Internally every delegated call passes
  ``allow_extrapolation=True``, so the decorated object's own flag never
  matters.
- The out-of-range test in ``derivative`` and ``second_derivative`` is
  **strict** (``x < x_min or x > x_max``). Exactly at a boundary the value
  comes from the decorated interpolation, not from the flat branch. Writing
  ``<=`` / ``>=`` there silently zeroes the endpoint slopes — and a natural
  spline, whose second derivative is 0 at both ends by construction, hides
  the bug. A not-a-knot spline does not, which is why the probe uses one.
- ``primitive`` extends **linearly** outside the range — the integral of a
  constant is affine — rather than staying flat or clamping.

C++ implements this as an ``Interpolation::Impl`` and gets ``checkRange``
plus the ``Extrapolator`` flag from the ``Interpolation`` handle. The Python
port has no PIMPL layer, so ``FlatExtrapolator`` subclasses ``Interpolation``
directly and overrides the four ``_value`` / ``_primitive`` / ``_derivative``
/ ``_second_derivative`` hooks; ``Interpolation.__call__`` and friends then
supply the same range check against the decorator's own flag.
"""

from __future__ import annotations

from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation


class FlatExtrapolator(Interpolation):
    """Decorator making any 1-D interpolation extrapolate flat.

    # C++ parity: ``class FlatExtrapolator`` in
    # ql/math/interpolations/flatextrapolation.hpp:38-101 (v1.43).
    """

    def __init__(self, decorated_interpolation: Interpolation) -> None:
        # The base keeps its own copy of the knots, but every accessor that
        # would read them (``x_min``, ``x_max``, ``is_in_range``) is overridden
        # to delegate — matching C++, where the Impl forwards ``xMin``,
        # ``xMax``, ``xValues``, ``yValues`` and ``isInRange`` to the
        # decorated interpolation rather than caching them.
        super().__init__(decorated_interpolation.x_values, decorated_interpolation.y_values)
        self._decorated: Interpolation = decorated_interpolation
        # C++ parity: FlatExtrapolatorImpl's constructor runs ``calculate()``,
        # which is ``decoratedInterp_->update()``.
        self._decorated.update()

    @property
    def decorated_interpolation(self) -> Interpolation:
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
    def x_values(self) -> Array:
        return self._decorated.x_values

    @property
    def y_values(self) -> Array:
        return self._decorated.y_values

    def is_in_range(self, x: float) -> bool:
        return self._decorated.is_in_range(x)

    def update(self) -> None:
        self._decorated.update()

    # ----- the decorated calculus ----------------------------------------

    def _bind(self, x: float) -> float:
        """Clamp ``x`` into the decorated range — the "flat" in flat extrapolation.

        # C++ parity: flatextrapolation.hpp:96-98 — ``std::clamp``.
        """
        return min(max(x, self.x_min), self.x_max)

    def _value(self, x: float) -> float:
        # C++ parity: flatextrapolation.hpp:67-69.
        return self._decorated(self._bind(x), allow_extrapolation=True)

    def _primitive(self, x: float) -> float:
        # C++ parity: flatextrapolation.hpp:70-78 — a LINEAR extension outside
        # the range, with slope equal to the boundary value of the integrand.
        x_min = self.x_min
        x_max = self.x_max
        if x < x_min:
            return self._decorated.primitive(x_min, allow_extrapolation=True) + self._decorated(
                x_min, allow_extrapolation=True
            ) * (x - x_min)
        if x > x_max:
            return self._decorated.primitive(x_max, allow_extrapolation=True) + self._decorated(
                x_max, allow_extrapolation=True
            ) * (x - x_max)
        return self._decorated.primitive(x, allow_extrapolation=True)

    def _derivative(self, x: float) -> float:
        # C++ parity: flatextrapolation.hpp:79-83. STRICT comparison: exactly
        # at x_min / x_max the decorated slope is used, not the flat 0.
        if x < self.x_min or x > self.x_max:
            return 0.0
        return self._decorated.derivative(x, allow_extrapolation=True)

    def _second_derivative(self, x: float) -> float:
        # C++ parity: flatextrapolation.hpp:84-88 — strict, as above.
        if x < self.x_min or x > self.x_max:
            return 0.0
        return self._decorated.second_derivative(x, allow_extrapolation=True)
