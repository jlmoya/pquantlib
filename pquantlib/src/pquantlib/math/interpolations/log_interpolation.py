"""Log-linear, log-cubic and log-mixed-linear-cubic interpolation.

# C++ parity: ql/math/interpolations/loginterpolation.hpp (v1.43).

Every class here is the same idea: interpolate ``log(y)`` with some inner
scheme, then exponentiate. C++ expresses that once, in the templated
``detail::LogInterpolationImpl``, and then spells out one wrapper class and
one factory/traits class per inner scheme plus a pile of presets. This
module keeps that shape: :class:`LogInterpolation` is the shared wrapper,
and each C++ class below it is transcribed with its own preset arguments —
those arguments are the class's entire meaning.

All ``y`` must be strictly positive; C++ checks this inside ``update()``
with a message naming the offending index, and so does this port.

``primitive`` is not available: C++ ``QL_FAIL``s
("LogInterpolation primitive not implemented") because ``exp`` of a spline
has no closed-form integral. Reproduced.

Documented divergences from the C++ source:

* C++'s inner interpolation holds iterators into the member ``logY_``
  vector, so ``update()`` refills that vector in place and then calls
  ``interpolation_.update()``. This port copies, so ``update()`` rebuilds
  the inner interpolation. Same values, more allocation.
* ``math.log`` / ``math.exp`` are used rather than the numpy ufuncs so the
  port goes through the same platform libm as C++'s ``std::log`` /
  ``std::exp``. numpy's vectorised kernels are accurate but not
  bit-identical, and a 1-ULP difference here is amplified by the spline.
"""

from __future__ import annotations

import math
from typing import Protocol, final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition,
    Cubic,
    DerivativeApprox,
)
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import Linear
from pquantlib.math.interpolations.mixed_interpolation import (
    MixedInterpolation,
    MixedLinearCubic,
)


class Interpolator(Protocol):
    """The C++ ``Interpolator`` template-parameter concept.

    C++ passes these as template arguments; Python passes them as objects.
    Every factory/traits class in ``ql/math/interpolations`` satisfies it.
    """

    required_points: int

    def interpolate(
        self, x_seq: Array, y_seq: Array, update: bool = True
    ) -> Interpolation:
        """Build an interpolation over ``(x, y)``."""
        ...


class LogInterpolation(Interpolation):
    """``exp`` of an inner interpolation of ``log(y)``.

    # C++ parity: ``detail::LogInterpolationImpl``
    #             (loginterpolation.hpp:365-403).
    """

    def __init__(
        self,
        x_seq: Array,
        y_seq: Array,
        factory: Interpolator,
        update: bool = True,
    ) -> None:
        super().__init__(x_seq, y_seq, required_points=factory.required_points)
        self._factory: Interpolator = factory
        self._log_y: Array = np.zeros(int(self._xs.shape[0]), dtype=np.float64)
        self._inner: Interpolation = factory.interpolate(self._xs, self._log_y, False)
        if update:
            self.update()

    def update(self) -> None:
        """# C++ parity: ``LogInterpolationImpl::update`` (379-387)."""
        ys = self._ys
        log_y = np.empty(int(ys.shape[0]), dtype=np.float64)
        for i in range(int(ys.shape[0])):
            yi = float(ys[i])
            qassert.require(yi > 0.0, f"invalid value ({yi}) at index {i}")
            log_y[i] = math.log(yi)
        self._log_y = log_y
        self._inner = self._factory.interpolate(self._xs, log_y)

    def _value(self, x: float) -> float:
        return math.exp(self._inner(x, allow_extrapolation=True))

    def _primitive(self, x: float) -> float:
        del x
        qassert.fail("LogInterpolation primitive not implemented")

    def _derivative(self, x: float) -> float:
        return self._value(x) * self._inner.derivative(x, allow_extrapolation=True)

    def _second_derivative(self, x: float) -> float:
        return self._derivative(x) * self._inner.derivative(
            x, allow_extrapolation=True
        ) + self._value(x) * self._inner.second_derivative(x, allow_extrapolation=True)


@final
class LogLinearInterpolation(LogInterpolation):
    """Linear interpolation of ``log(y)``.

    # C++ parity: ``LogLinearInterpolation`` (loginterpolation.hpp:44-55).
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(x_seq, y_seq, Linear())


@final
class LogLinear:
    """Log-linear interpolation factory and traits.

    # C++ parity: ``class LogLinear`` (loginterpolation.hpp:59-68).
    """

    global_ = False  # C++ ``static const bool global = false``.
    required_points = 2  # C++ ``static const Size requiredPoints = 2``.

    def interpolate(
        self, x_seq: Array, y_seq: Array, update: bool = True
    ) -> LogLinearInterpolation:
        """Build a :class:`LogLinearInterpolation` over ``(x, y)``."""
        f = LogLinearInterpolation(x_seq, y_seq)
        if update:
            f.update()
        return f


class LogCubicInterpolation(LogInterpolation):
    """Cubic interpolation of ``log(y)``.

    # C++ parity: ``LogCubicInterpolation`` (loginterpolation.hpp:72-93).
    """

    def __init__(
        self,
        x_seq: Array,
        y_seq: Array,
        derivative_approx: DerivativeApprox,
        monotonic: bool,
        left_condition: BoundaryCondition,
        left_condition_value: float,
        right_condition: BoundaryCondition,
        right_condition_value: float,
        update: bool = True,
    ) -> None:
        super().__init__(
            x_seq,
            y_seq,
            Cubic(
                derivative_approx,
                monotonic,
                left_condition,
                left_condition_value,
                right_condition,
                right_condition_value,
            ),
            update,
        )


class LogCubic:
    """Log-cubic interpolation factory and traits.

    # C++ parity: ``class LogCubic`` (loginterpolation.hpp:97-125).

    Note the default ``monotonic = True`` — the opposite of
    :class:`~pquantlib.math.interpolations.cubic_interpolation.Cubic`, whose
    default is ``False``. That asymmetry is C++'s, and it is why
    ``LogCubic(Kruger)`` is a *filtered* Kruger while ``Cubic(Kruger)`` is
    not.
    """

    global_ = True  # C++ ``static const bool global = true``.
    required_points = 2  # C++ ``static const Size requiredPoints = 2``.

    def __init__(
        self,
        derivative_approx: DerivativeApprox,
        monotonic: bool = True,
        left_condition: BoundaryCondition = BoundaryCondition.SecondDerivative,
        left_condition_value: float = 0.0,
        right_condition: BoundaryCondition = BoundaryCondition.SecondDerivative,
        right_condition_value: float = 0.0,
    ) -> None:
        self._da: DerivativeApprox = derivative_approx
        self._monotonic: bool = monotonic
        self._left_type: BoundaryCondition = left_condition
        self._left_value: float = left_condition_value
        self._right_type: BoundaryCondition = right_condition
        self._right_value: float = right_condition_value

    def interpolate(
        self, x_seq: Array, y_seq: Array, update: bool = True
    ) -> LogCubicInterpolation:
        """Build a :class:`LogCubicInterpolation` over ``(x, y)``."""
        return LogCubicInterpolation(
            x_seq,
            y_seq,
            self._da,
            self._monotonic,
            self._left_type,
            self._left_value,
            self._right_type,
            self._right_value,
            update,
        )


@final
class MonotonicLogCubic(LogCubic):
    """``LogCubic(Spline, monotonic, natural, natural)``.

    # C++ parity: ``MonotonicLogCubic`` (loginterpolation.hpp:138-144).
    """

    def __init__(self) -> None:
        super().__init__(
            DerivativeApprox.Spline,
            True,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class KrugerLog(LogCubic):
    """``LogCubic(Kruger, non-monotonic, natural, natural)``.

    # C++ parity: ``KrugerLog`` (loginterpolation.hpp:146-152). Replaces the
    # deprecated ``DefaultLogCubic``, which differed: it left ``monotonic``
    # at the ``LogCubic`` default of ``true``.
    """

    def __init__(self) -> None:
        super().__init__(
            DerivativeApprox.Kruger,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class LogCubicNaturalSpline(LogCubicInterpolation):
    """# C++ parity: ``LogCubicNaturalSpline`` (loginterpolation.hpp:155-166)."""

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Spline,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class MonotonicLogCubicNaturalSpline(LogCubicInterpolation):
    """# C++ parity: ``MonotonicLogCubicNaturalSpline`` (loginterpolation.hpp:168-179)."""

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Spline,
            True,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class KrugerLogCubic(LogCubicInterpolation):
    """# C++ parity: ``KrugerLogCubic`` (loginterpolation.hpp:181-192)."""

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Kruger,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class HarmonicLogCubic(LogCubicInterpolation):
    """# C++ parity: ``HarmonicLogCubic`` (loginterpolation.hpp:194-205)."""

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Harmonic,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class FritschButlandLogCubic(LogCubicInterpolation):
    """# C++ parity: ``FritschButlandLogCubic`` (loginterpolation.hpp:207-218).

    ``monotonic = false`` here, unlike the standalone ``FritschButlandCubic``.
    """

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.FritschButland,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class LogParabolic(LogCubicInterpolation):
    """# C++ parity: ``LogParabolic`` (loginterpolation.hpp:220-231)."""

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Parabolic,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class MonotonicLogParabolic(LogCubicInterpolation):
    """# C++ parity: ``MonotonicLogParabolic`` (loginterpolation.hpp:233-244)."""

    def __init__(self, x_seq: Array, y_seq: Array) -> None:
        super().__init__(
            x_seq,
            y_seq,
            DerivativeApprox.Parabolic,
            True,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


class LogMixedLinearCubicInterpolation(LogInterpolation):
    """Mixed linear/cubic interpolation of ``log(y)``.

    # C++ parity: ``LogMixedLinearCubicInterpolation``
    #             (loginterpolation.hpp:248-270).
    """

    def __init__(
        self,
        x_seq: Array,
        y_seq: Array,
        n: int,
        behavior: MixedInterpolation.Behavior,
        derivative_approx: DerivativeApprox,
        monotonic: bool,
        left_condition: BoundaryCondition,
        left_condition_value: float | None,
        right_condition: BoundaryCondition,
        right_condition_value: float,
        update: bool = True,
    ) -> None:
        super().__init__(
            x_seq,
            y_seq,
            MixedLinearCubic(
                n,
                behavior,
                derivative_approx,
                monotonic,
                left_condition,
                left_condition_value,
                right_condition,
                right_condition_value,
            ),
            update,
        )


class LogMixedLinearCubic:
    """Log-mixed-linear-cubic factory and traits.

    # C++ parity: ``class LogMixedLinearCubic`` (loginterpolation.hpp:274-308).
    """

    global_ = True  # C++ ``static const bool global = true``.
    required_points = 3  # C++ ``static const Size requiredPoints = 3``.

    def __init__(
        self,
        n: int,
        behavior: MixedInterpolation.Behavior,
        derivative_approx: DerivativeApprox,
        monotonic: bool = True,
        left_condition: BoundaryCondition = BoundaryCondition.SecondDerivative,
        left_condition_value: float | None = 0.0,
        right_condition: BoundaryCondition = BoundaryCondition.SecondDerivative,
        right_condition_value: float = 0.0,
    ) -> None:
        self._n: int = n
        self._behavior: MixedInterpolation.Behavior = behavior
        self._da: DerivativeApprox = derivative_approx
        self._monotonic: bool = monotonic
        self._left_type: BoundaryCondition = left_condition
        self._left_value: float | None = left_condition_value
        self._right_type: BoundaryCondition = right_condition
        self._right_value: float = right_condition_value

    def interpolate(
        self, x_seq: Array, y_seq: Array, update: bool = True
    ) -> LogMixedLinearCubicInterpolation:
        """Build a :class:`LogMixedLinearCubicInterpolation` over ``(x, y)``."""
        return LogMixedLinearCubicInterpolation(
            x_seq,
            y_seq,
            self._n,
            self._behavior,
            self._da,
            self._monotonic,
            self._left_type,
            self._left_value,
            self._right_type,
            self._right_value,
            update,
        )


@final
class MonotonicLogMixedLinearCubic(LogMixedLinearCubic):
    """# C++ parity: ``MonotonicLogMixedLinearCubic`` (loginterpolation.hpp:325-334)."""

    def __init__(
        self,
        n: int,
        behavior: MixedInterpolation.Behavior = MixedInterpolation.Behavior.ShareRanges,
    ) -> None:
        super().__init__(
            n,
            behavior,
            DerivativeApprox.Spline,
            True,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class KrugerLogMixedLinearCubic(LogMixedLinearCubic):
    """# C++ parity: ``KrugerLogMixedLinearCubic`` (loginterpolation.hpp:336-345)."""

    def __init__(
        self,
        n: int,
        behavior: MixedInterpolation.Behavior = MixedInterpolation.Behavior.ShareRanges,
    ) -> None:
        super().__init__(
            n,
            behavior,
            DerivativeApprox.Kruger,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class LogMixedLinearCubicNaturalSpline(LogMixedLinearCubicInterpolation):
    """# C++ parity: ``LogMixedLinearCubicNaturalSpline`` (loginterpolation.hpp:348-360)."""

    def __init__(
        self,
        x_seq: Array,
        y_seq: Array,
        n: int,
        behavior: MixedInterpolation.Behavior = MixedInterpolation.Behavior.ShareRanges,
    ) -> None:
        super().__init__(
            x_seq,
            y_seq,
            n,
            behavior,
            DerivativeApprox.Spline,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


# ``Linear`` is re-exported so callers can build the C++ ``LogLinear``
# equivalent (``LogInterpolation(x, y, Linear())``) without a second import.
__all__ = [
    "FritschButlandLogCubic",
    "HarmonicLogCubic",
    "Interpolator",
    "KrugerLog",
    "KrugerLogCubic",
    "KrugerLogMixedLinearCubic",
    "Linear",
    "LogCubic",
    "LogCubicInterpolation",
    "LogCubicNaturalSpline",
    "LogInterpolation",
    "LogLinear",
    "LogLinearInterpolation",
    "LogMixedLinearCubic",
    "LogMixedLinearCubicInterpolation",
    "LogMixedLinearCubicNaturalSpline",
    "LogParabolic",
    "MonotonicLogCubic",
    "MonotonicLogCubicNaturalSpline",
    "MonotonicLogMixedLinearCubic",
    "MonotonicLogParabolic",
]
