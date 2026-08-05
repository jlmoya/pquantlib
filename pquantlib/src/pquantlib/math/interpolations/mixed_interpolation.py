"""Mixed linear/cubic interpolation between discrete points.

# C++ parity: ql/math/interpolations/mixedinterpolation.hpp (v1.43).

A curve that is linear over the first part of the range and cubic over the
rest. The switch point is the pillar at index ``n``.

``MixedInterpolation.Behavior`` picks how the two sub-interpolations see
the data:

* ``ShareRanges`` — both are defined over the *whole* range and the
  evaluator simply chooses between them at the switch point. The cubic
  therefore "sees" the linear region's data when it solves for its slopes.
* ``SplitRanges`` — the linear one covers ``x[0..n]`` and the cubic covers
  ``x[n..]``. With ``SplitRanges`` you can additionally ask for the
  derivative to match across the seam by passing ``left_condition =
  FirstDerivative`` and ``left_condition_value = None`` (C++ spells the
  latter ``Null<Real>()``); the switch function then overwrites the cubic's
  left condition with the linear segment's slope at the switch point,
  between the two sub-updates.

Documented divergences from the C++ source:

* C++'s ``Null<Real>()`` sentinel becomes ``None``. Same convention as the
  rest of pquantlib (see ``quotes/simple_quote.py``).
* C++'s sub-interpolations hold iterators into the caller's arrays, so
  ``update()`` re-reads them in place. This port copies, so ``update()``
  rebuilds both sub-interpolations from the current data. Observationally
  identical; it just does more work.
"""

from __future__ import annotations

from enum import IntEnum
from typing import final

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition,
    Cubic,
    CubicInterpolation,
    DerivativeApprox,
)
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import Linear, LinearInterpolation


class MixedInterpolation:
    """Namespace for the mixed-interpolation behaviour flag.

    # C++ parity: ``struct MixedInterpolation`` (mixedinterpolation.hpp:38-48).
    """

    class Behavior(IntEnum):
        """How the two sub-interpolations divide up the data range."""

        ShareRanges = 0
        """Both interpolations span the whole range (the C++ default)."""
        SplitRanges = 1
        """The first covers ``x[0..n]``, the second ``x[n..]``."""


class MixedLinearCubicInterpolation(Interpolation):
    """Linear below the switch pillar, cubic above it.

    # C++ parity: ``MixedLinearCubicInterpolation``
    #             (mixedinterpolation.hpp:59-94) + the templated
    #             ``detail::MixedInterpolationImpl`` (223-296).

    Args:
        x_seq: pillar abscissae, sorted ascending.
        y_seq: pillar ordinates.
        n: index of the switch pillar. Must satisfy ``n <= len(x) - 1``.
        behavior: :class:`MixedInterpolation.Behavior`.
        derivative_approx: cubic slope scheme for the second segment.
        monotonic: apply the Hyman filter to the cubic segment.
        left_condition: cubic's left end condition.
        left_condition_value: value for it, or ``None`` to request
            derivative matching at the switch point (C++ ``Null<Real>()``).
        right_condition: cubic's right end condition.
        right_condition_value: value for it.
        update: run :meth:`update` at construction.
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
        # C++ ``MixedInterpolationImpl`` passes requiredPoints = 1; the
        # per-segment counts are checked by the sub-interpolations.
        super().__init__(x_seq, y_seq, required_points=1)
        # C++ parity: mixedinterpolation.hpp:73-76.
        match_derivatives = (
            left_condition == BoundaryCondition.FirstDerivative
            and left_condition_value is None
        )
        qassert.require(
            not match_derivatives or behavior == MixedInterpolation.Behavior.SplitRanges,
            "matching derivatives is only supported with SplitRanges",
        )
        qassert.require(
            match_derivatives or left_condition_value is not None,
            "a null left condition value is only meaningful with "
            "FirstDerivative (it requests derivative matching)",
        )
        # C++ parity: mixedinterpolation.hpp:239-243.
        max_n = int(self._xs.shape[0]) - 1
        qassert.require(n <= max_n, f"n is too large ({n} > {max_n})")
        qassert.require(behavior in tuple(MixedInterpolation.Behavior),
                        f"unknown mixed-interpolation behavior: {behavior}")

        self._n: int = n
        self._behavior: MixedInterpolation.Behavior = behavior
        self._match_derivatives: bool = match_derivatives
        self._factory1: Linear = Linear()
        # When matching derivatives the left value is overwritten before the
        # cubic's own update() runs, so the placeholder never reaches the
        # arithmetic — exactly as C++'s Null<Real>() never does.
        self._factory2: Cubic = Cubic(
            derivative_approx,
            monotonic,
            left_condition,
            0.0 if left_condition_value is None else left_condition_value,
            right_condition,
            right_condition_value,
        )
        self._interpolation1: LinearInterpolation = self._factory1.interpolate(
            self._xs[: n + 1] if behavior == MixedInterpolation.Behavior.SplitRanges
            else self._xs,
            self._ys[: n + 1] if behavior == MixedInterpolation.Behavior.SplitRanges
            else self._ys,
        )
        self._interpolation2: CubicInterpolation = self._factory2.interpolate(
            self._xs[n:] if behavior == MixedInterpolation.Behavior.SplitRanges
            else self._xs,
            self._ys[n:] if behavior == MixedInterpolation.Behavior.SplitRanges
            else self._ys,
            False,
        )
        if update:
            self.update()

    @property
    def _x_switch(self) -> float:
        """The switch abscissa, C++ ``*xBegin2_``."""
        return float(self._xs[self._n])

    def update(self) -> None:
        """# C++ parity: ``MixedInterpolationImpl::update`` (265-269)."""
        n = self._n
        split = self._behavior == MixedInterpolation.Behavior.SplitRanges
        x1 = self._xs[: n + 1] if split else self._xs
        y1 = self._ys[: n + 1] if split else self._ys
        x2 = self._xs[n:] if split else self._xs
        y2 = self._ys[n:] if split else self._ys
        self._interpolation1 = self._factory1.interpolate(x1, y1)
        self._interpolation2 = self._factory2.interpolate(x2, y2, False)
        if self._match_derivatives:
            # C++ parity: mixedinterpolation.hpp:78-83 — the switch function
            # reaches into the cubic's CubicInterpolationBaseImpl and sets
            # leftValue_ to the linear segment's slope at the switch point.
            self._interpolation2.update_left_condition_value(
                self._interpolation1.derivative(self._x_switch, allow_extrapolation=True)
            )
        self._interpolation2.update()

    def _value(self, x: float) -> float:
        if x < self._x_switch:
            return self._interpolation1(x, allow_extrapolation=True)
        return self._interpolation2(x, allow_extrapolation=True)

    def _primitive(self, x: float) -> float:
        # C++ parity: mixedinterpolation.hpp:275-281 — the second segment's
        # primitive is re-based so the two agree at the switch point.
        if x < self._x_switch:
            return self._interpolation1.primitive(x, allow_extrapolation=True)
        return (
            self._interpolation2.primitive(x, allow_extrapolation=True)
            - self._interpolation2.primitive(self._x_switch, allow_extrapolation=True)
            + self._interpolation1.primitive(self._x_switch, allow_extrapolation=True)
        )

    def _derivative(self, x: float) -> float:
        if x < self._x_switch:
            return self._interpolation1.derivative(x, allow_extrapolation=True)
        return self._interpolation2.derivative(x, allow_extrapolation=True)

    def _second_derivative(self, x: float) -> float:
        if x < self._x_switch:
            return self._interpolation1.second_derivative(x, allow_extrapolation=True)
        return self._interpolation2.second_derivative(x, allow_extrapolation=True)


@final
class MixedLinearCubic:
    """Mixed linear/cubic factory and traits.

    # C++ parity: ``class MixedLinearCubic`` (mixedinterpolation.hpp:98-133).
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
    ) -> MixedLinearCubicInterpolation:
        """Build a :class:`MixedLinearCubicInterpolation` over ``(x, y)``."""
        return MixedLinearCubicInterpolation(
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


# ---------------------------------------------------------------------------
# convenience classes (mixedinterpolation.hpp:135-219)
# ---------------------------------------------------------------------------


@final
class MixedLinearCubicNaturalSpline(MixedLinearCubicInterpolation):
    """Linear then natural cubic spline, unfiltered.

    # C++ parity: ``MixedLinearCubicNaturalSpline`` (mixedinterpolation.hpp:137-149).
    """

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


@final
class MixedLinearMonotonicCubicNaturalSpline(MixedLinearCubicInterpolation):
    """Linear then Hyman-filtered natural cubic spline.

    # C++ parity: ``MixedLinearMonotonicCubicNaturalSpline``
    #             (mixedinterpolation.hpp:151-163).
    """

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
            True,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class MixedLinearKrugerCubic(MixedLinearCubicInterpolation):
    """Linear then Kruger cubic, unfiltered.

    # C++ parity: ``MixedLinearKrugerCubic`` (mixedinterpolation.hpp:165-177).
    """

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
            DerivativeApprox.Kruger,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class MixedLinearFritschButlandCubic(MixedLinearCubicInterpolation):
    """Linear then Fritsch-Butland cubic, unfiltered.

    # C++ parity: ``MixedLinearFritschButlandCubic``
    #             (mixedinterpolation.hpp:179-191). Note ``monotonic=false``
    #             here, unlike the standalone ``FritschButlandCubic``.
    """

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
            DerivativeApprox.FritschButland,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class MixedLinearParabolic(MixedLinearCubicInterpolation):
    """Linear then parabolic cubic, unfiltered.

    # C++ parity: ``MixedLinearParabolic`` (mixedinterpolation.hpp:193-205).
    """

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
            DerivativeApprox.Parabolic,
            False,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


@final
class MixedLinearMonotonicParabolic(MixedLinearCubicInterpolation):
    """Linear then Hyman-filtered parabolic cubic.

    # C++ parity: ``MixedLinearMonotonicParabolic``
    #             (mixedinterpolation.hpp:207-219).
    """

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
            DerivativeApprox.Parabolic,
            True,
            BoundaryCondition.SecondDerivative,
            0.0,
            BoundaryCondition.SecondDerivative,
            0.0,
        )


__all__ = [
    "MixedInterpolation",
    "MixedLinearCubic",
    "MixedLinearCubicInterpolation",
    "MixedLinearCubicNaturalSpline",
    "MixedLinearFritschButlandCubic",
    "MixedLinearKrugerCubic",
    "MixedLinearMonotonicCubicNaturalSpline",
    "MixedLinearMonotonicParabolic",
    "MixedLinearParabolic",
]
