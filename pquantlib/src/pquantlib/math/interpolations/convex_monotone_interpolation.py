"""Convex-monotone yield-curve interpolation (Hagan-West).

# C++ parity: ql/math/interpolations/convexmonotoneinterpolation.hpp (v1.43).

Reference: P. Hagan & G. West, *Interpolation Methods for Curve
Construction*, Applied Mathematical Finance 13(2), 2006, with QuantLib's
enhancements (``monotonicity < 1`` and ``quadraticity > 0`` smoothing, plus
the ``forcePositive`` non-negativity machinery).

The scheme is a small state machine, not a formula. For each segment it
derives the discrete forwards ``f``, forms the two gradients

    g_prev = f[i-1] - y[i]        g_next = f[i] - y[i]

and dispatches on their signs and relative magnitudes to one of a family of
:class:`SectionHelper` shapes:

===========================  ===============================================
:class:`ConstantGradHelper`  both gradients vanish (``< 1e-14``)
:class:`ConvexMonotone2Helper`  flat then rising parabola
:class:`ConvexMonotone3Helper`  falling parabola then flat
:class:`ConvexMonotone4Helper`  parabola-parabola, split at ``eta``
:class:`ConvexMonotone4MinHelper`  ditto, but flattened to 0 where it would
                             otherwise go negative (``force_positive``)
:class:`QuadraticHelper`     a single quadratic through the segment
:class:`QuadraticMinHelper`  ditto, with the same non-negativity split
:class:`ComboHelper`         a ``quadraticity``-weighted blend of a
                             quadratic and a convex-monotone helper
:class:`EverywhereConstantHelper`  the single-period case, the flat final
                             period, and the extrapolation beyond the last
                             pillar
===========================  ===============================================

``monotonicity`` sets the bounds ``b2 = (1+m)/2`` and ``b3 = (1-m)/2`` that
clamp the split point ``eta``; ``quadraticity`` blends toward the pure
quadratic. ``ConvexMonotone``'s own defaults are ``0.3 / 0.7 / True``.

Documented divergences from the C++ source:

* C++ keys its section helpers by the *upper* x of each segment in a
  ``std::map`` and resolves with ``upper_bound``. This port keeps a list
  indexed by the same pillar index and resolves with ``bisect_right``,
  which is the same lookup; :meth:`ConvexMonotoneInterpolation.get_existing_helpers`
  still hands back the x-keyed mapping that ``localInterpolate`` needs.
* C++ uses iterator pairs into caller-owned arrays; this port copies (see
  the :class:`~pquantlib.math.interpolations.interpolation.Interpolation`
  divergence note).
* ``derivative`` / ``second_derivative`` ``QL_FAIL`` in C++; here they
  raise ``NotImplementedError`` from the base class' defaults, which is
  what the existing call sites and tests expect.
"""

from __future__ import annotations

import bisect
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation

# C++ parity: ``ConvexMonotone::requiredPoints`` (convexmonotoneinterpolation.hpp:91).
_REQUIRED_POINTS: int = 2
# C++ parity: the zero-gradient threshold inside ``ConvexMonotoneImpl::update``
# (convexmonotoneinterpolation.hpp:640-641).
_ZERO_GRAD_EPS: float = 1.0e-14


class SectionHelper(ABC):
    """One segment of the interpolation: value, primitive, and ``f_next``.

    # C++ parity: ``detail::SectionHelper``
    #             (convexmonotoneinterpolation.hpp:164-170).
    """

    __slots__ = ()

    @abstractmethod
    def value(self, x: float) -> float:
        """The interpolated value at ``x``."""

    @abstractmethod
    def primitive(self, x: float) -> float:
        """The running integral up to ``x``."""

    @abstractmethod
    def f_next(self) -> float:
        """The forward at the segment's right edge (C++ ``fNext()``)."""


@final
class EverywhereConstantHelper(SectionHelper):
    """A constant segment.

    # C++ parity: ``EverywhereConstantHelper``
    #             (convexmonotoneinterpolation.hpp:268-282).
    """

    __slots__ = ("_prev_primitive", "_value", "_x_prev")

    def __init__(self, value: float, prev_primitive: float, x_prev: float) -> None:
        self._value: float = value
        self._prev_primitive: float = prev_primitive
        self._x_prev: float = x_prev

    def value(self, x: float) -> float:
        del x
        return self._value

    def primitive(self, x: float) -> float:
        return self._prev_primitive + (x - self._x_prev) * self._value

    def f_next(self) -> float:
        return self._value


@final
class ConstantGradHelper(SectionHelper):
    """A straight line through ``(x_prev, f_prev)`` and ``(x_next, f_next)``.

    # C++ parity: ``ConstantGradHelper``
    #             (convexmonotoneinterpolation.hpp:463-479).
    """

    __slots__ = ("_f_grad", "_f_next", "_f_prev", "_prev_primitive", "_x_prev")

    def __init__(
        self,
        f_prev: float,
        prev_primitive: float,
        x_prev: float,
        x_next: float,
        f_next: float,
    ) -> None:
        self._f_prev: float = f_prev
        self._prev_primitive: float = prev_primitive
        self._x_prev: float = x_prev
        self._f_grad: float = (f_next - f_prev) / (x_next - x_prev)
        self._f_next: float = f_next

    def value(self, x: float) -> float:
        return self._f_prev + (x - self._x_prev) * self._f_grad

    def primitive(self, x: float) -> float:
        return self._prev_primitive + (x - self._x_prev) * (
            self._f_prev + 0.5 * (x - self._x_prev) * self._f_grad
        )

    def f_next(self) -> float:
        return self._f_next


@final
class QuadraticHelper(SectionHelper):
    """The quadratic through ``f_prev``, ``f_next`` with mean ``f_average``.

    # C++ parity: ``QuadraticHelper``
    #             (convexmonotoneinterpolation.hpp:481-511).
    """

    __slots__ = (
        "_a",
        "_b",
        "_c",
        "_f_average",
        "_f_next",
        "_f_prev",
        "_prev_primitive",
        "_x_next",
        "_x_prev",
        "_x_scaling",
    )

    def __init__(
        self,
        x_prev: float,
        x_next: float,
        f_prev: float,
        f_next: float,
        f_average: float,
        prev_primitive: float,
    ) -> None:
        self._x_prev: float = x_prev
        self._x_next: float = x_next
        self._f_prev: float = f_prev
        self._f_next: float = f_next
        self._f_average: float = f_average
        self._prev_primitive: float = prev_primitive
        self._a: float = 3 * f_prev + 3 * f_next - 6 * f_average
        self._b: float = -(4 * f_prev + 2 * f_next - 6 * f_average)
        self._c: float = f_prev
        self._x_scaling: float = x_next - x_prev

    def value(self, x: float) -> float:
        xv = (x - self._x_prev) / self._x_scaling
        return self._a * xv * xv + self._b * xv + self._c

    def primitive(self, x: float) -> float:
        xv = (x - self._x_prev) / self._x_scaling
        return (
            self._prev_primitive
            + self._x_scaling
            * (self._a / 3 * xv * xv + self._b / 2 * xv + self._c)
            * xv
        )

    def f_next(self) -> float:
        return self._f_next


@final
class QuadraticMinHelper(SectionHelper):
    """The quadratic, flattened to zero over the region where it dips below.

    # C++ parity: ``QuadraticMinHelper``
    #             (convexmonotoneinterpolation.hpp:513-585).

    The split only happens when the quadratic actually has real roots
    (``d > 0``) *and* the auxiliary discriminant ``dAv`` is non-negative;
    otherwise every coefficient stays as :class:`QuadraticHelper` computes
    it and the two classes agree bit for bit. That is why substituting the
    plain quadratic passes on well-behaved forward curves and silently
    diverges on the ones the helper exists for.
    """

    __slots__ = (
        "_a",
        "_b",
        "_c",
        "_f_average",
        "_f_next",
        "_f_prev",
        "_primitive1",
        "_primitive2",
        "_split_region",
        "_x1",
        "_x2",
        "_x3",
        "_x4",
        "_x_ratio",
        "_x_scaling",
    )

    def __init__(
        self,
        x_prev: float,
        x_next: float,
        f_prev: float,
        f_next: float,
        f_average: float,
        prev_primitive: float,
    ) -> None:
        self._x1: float = x_prev
        self._x4: float = x_next
        self._primitive1: float = prev_primitive
        self._f_average: float = f_average
        self._f_prev: float = f_prev
        self._f_next: float = f_next
        self._split_region: bool = False
        self._x_ratio: float = 1.0
        self._x2: float = 0.0
        self._x3: float = 0.0
        self._primitive2: float = 0.0
        self._a: float = 3 * f_prev + 3 * f_next - 6 * f_average
        self._b: float = -(4 * f_prev + 2 * f_next - 6 * f_average)
        self._c: float = f_prev
        d = self._b * self._b - 4 * self._a * self._c
        self._x_scaling: float = self._x4 - self._x1
        if d > 0:
            a_av = 36.0
            b_av = -24 * (self._f_prev + self._f_next)
            c_av = 4 * (
                self._f_prev * self._f_prev
                + self._f_prev * self._f_next
                + self._f_next * self._f_next
            )
            d_av = b_av * b_av - 4.0 * a_av * c_av
            if d_av >= 0.0:
                self._split_region = True
                av_root = (-b_av - math.sqrt(d_av)) / (2 * a_av)
                self._x_ratio = self._f_average / av_root
                self._x_scaling *= self._x_ratio
                self._a = 3 * self._f_prev + 3 * self._f_next - 6 * av_root
                self._b = -(4 * self._f_prev + 2 * self._f_next - 6 * av_root)
                self._c = self._f_prev
                x_root = -self._b / (2 * self._a)
                self._x2 = self._x1 + self._x_ratio * (self._x4 - self._x1) * x_root
                self._x3 = self._x4 - self._x_ratio * (self._x4 - self._x1) * (1 - x_root)
                self._primitive2 = (
                    self._primitive1
                    + self._x_scaling
                    * (self._a / 3 * x_root * x_root + self._b / 2 * x_root + self._c)
                    * x_root
                )

    def value(self, x: float) -> float:
        xv = (x - self._x1) / (self._x4 - self._x1)
        if self._split_region:
            if x <= self._x2:
                xv /= self._x_ratio
            elif x < self._x3:
                return 0.0
            else:
                xv = 1.0 - (1.0 - xv) / self._x_ratio
        return self._c + self._b * xv + self._a * xv * xv

    def primitive(self, x: float) -> float:
        xv = (x - self._x1) / (self._x4 - self._x1)
        if self._split_region:
            # NOTE the strict `<` here against the `<=` in value(): the two
            # boundaries genuinely differ in C++ and are transcribed as-is.
            if x < self._x2:
                xv /= self._x_ratio
            elif x < self._x3:
                return self._primitive2
            else:
                xv = 1.0 - (1.0 - xv) / self._x_ratio
        return (
            self._primitive1
            + self._x_scaling
            * (self._a / 3 * xv * xv + self._b / 2 * xv + self._c)
            * xv
        )

    def f_next(self) -> float:
        return self._f_next


@final
class ConvexMonotone2Helper(SectionHelper):
    """Flat, then a rising parabola from ``eta2``.

    # C++ parity: ``ConvexMonotone2Helper``
    #             (convexmonotoneinterpolation.hpp:284-317).
    """

    __slots__ = (
        "_eta2",
        "_f_average",
        "_g_next",
        "_g_prev",
        "_prev_primitive",
        "_x_prev",
        "_x_scaling",
    )

    def __init__(
        self,
        x_prev: float,
        x_next: float,
        g_prev: float,
        g_next: float,
        f_average: float,
        eta2: float,
        prev_primitive: float,
    ) -> None:
        self._x_prev: float = x_prev
        self._x_scaling: float = x_next - x_prev
        self._g_prev: float = g_prev
        self._g_next: float = g_next
        self._f_average: float = f_average
        self._eta2: float = eta2
        self._prev_primitive: float = prev_primitive

    def value(self, x: float) -> float:
        xv = (x - self._x_prev) / self._x_scaling
        if xv <= self._eta2:
            return self._f_average + self._g_prev
        return (
            self._f_average
            + self._g_prev
            + (self._g_next - self._g_prev)
            / ((1 - self._eta2) * (1 - self._eta2))
            * (xv - self._eta2)
            * (xv - self._eta2)
        )

    def primitive(self, x: float) -> float:
        xv = (x - self._x_prev) / self._x_scaling
        if xv <= self._eta2:
            return self._prev_primitive + self._x_scaling * (
                self._f_average * xv + self._g_prev * xv
            )
        return self._prev_primitive + self._x_scaling * (
            self._f_average * xv
            + self._g_prev * xv
            + (self._g_next - self._g_prev)
            / ((1 - self._eta2) * (1 - self._eta2))
            * (
                1.0 / 3.0 * (xv * xv * xv - self._eta2 * self._eta2 * self._eta2)
                - self._eta2 * xv * xv
                + self._eta2 * self._eta2 * xv
            )
        )

    def f_next(self) -> float:
        return self._f_average + self._g_next


@final
class ConvexMonotone3Helper(SectionHelper):
    """A falling parabola up to ``eta3``, then flat.

    # C++ parity: ``ConvexMonotone3Helper``
    #             (convexmonotoneinterpolation.hpp:319-352).
    """

    __slots__ = (
        "_eta3",
        "_f_average",
        "_g_next",
        "_g_prev",
        "_prev_primitive",
        "_x_prev",
        "_x_scaling",
    )

    def __init__(
        self,
        x_prev: float,
        x_next: float,
        g_prev: float,
        g_next: float,
        f_average: float,
        eta3: float,
        prev_primitive: float,
    ) -> None:
        self._x_prev: float = x_prev
        self._x_scaling: float = x_next - x_prev
        self._g_prev: float = g_prev
        self._g_next: float = g_next
        self._f_average: float = f_average
        self._eta3: float = eta3
        self._prev_primitive: float = prev_primitive

    def value(self, x: float) -> float:
        xv = (x - self._x_prev) / self._x_scaling
        if xv <= self._eta3:
            return (
                self._f_average
                + self._g_next
                + (self._g_prev - self._g_next)
                / (self._eta3 * self._eta3)
                * (self._eta3 - xv)
                * (self._eta3 - xv)
            )
        return self._f_average + self._g_next

    def primitive(self, x: float) -> float:
        xv = (x - self._x_prev) / self._x_scaling
        if xv <= self._eta3:
            return self._prev_primitive + self._x_scaling * (
                self._f_average * xv
                + self._g_next * xv
                + (self._g_prev - self._g_next)
                / (self._eta3 * self._eta3)
                * (
                    1.0 / 3.0 * xv * xv * xv
                    - self._eta3 * xv * xv
                    + self._eta3 * self._eta3 * xv
                )
            )
        return self._prev_primitive + self._x_scaling * (
            self._f_average * xv
            + self._g_next * xv
            + (self._g_prev - self._g_next)
            / (self._eta3 * self._eta3)
            * (1.0 / 3.0 * self._eta3 * self._eta3 * self._eta3)
        )

    def f_next(self) -> float:
        return self._f_average + self._g_next


class ConvexMonotone4Helper(SectionHelper):
    """Parabola-parabola meeting at ``eta4``.

    # C++ parity: ``ConvexMonotone4Helper``
    #             (convexmonotoneinterpolation.hpp:354-392).
    """

    __slots__ = (
        "_a",
        "_eta4",
        "_f_average",
        "_g_next",
        "_g_prev",
        "_prev_primitive",
        "_x_prev",
        "_x_scaling",
    )

    def __init__(
        self,
        x_prev: float,
        x_next: float,
        g_prev: float,
        g_next: float,
        f_average: float,
        eta4: float,
        prev_primitive: float,
    ) -> None:
        self._x_prev: float = x_prev
        self._x_scaling: float = x_next - x_prev
        self._g_prev: float = g_prev
        self._g_next: float = g_next
        self._f_average: float = f_average
        self._eta4: float = eta4
        self._prev_primitive: float = prev_primitive
        # C++ ``A_``; renamed because a bare ``a`` reads as a polynomial
        # coefficient here, which it is not.
        self._a: float = -0.5 * (eta4 * g_prev + (1 - eta4) * g_next)

    def value(self, x: float) -> float:
        xv = (x - self._x_prev) / self._x_scaling
        if xv <= self._eta4:
            return (
                self._f_average
                + self._a
                + (self._g_prev - self._a)
                * (self._eta4 - xv)
                * (self._eta4 - xv)
                / (self._eta4 * self._eta4)
            )
        return (
            self._f_average
            + self._a
            + (self._g_next - self._a)
            * (xv - self._eta4)
            * (xv - self._eta4)
            / ((1 - self._eta4) * (1 - self._eta4))
        )

    def primitive(self, x: float) -> float:
        xv = (x - self._x_prev) / self._x_scaling
        if xv <= self._eta4:
            return (
                self._prev_primitive
                + self._x_scaling
                * (
                    self._f_average
                    + self._a
                    + (self._g_prev - self._a)
                    / (self._eta4 * self._eta4)
                    * (self._eta4 * self._eta4 - self._eta4 * xv + 1.0 / 3.0 * xv * xv)
                )
                * xv
            )
        return self._prev_primitive + self._x_scaling * (
            self._f_average * xv
            + self._a * xv
            + (self._g_prev - self._a) * (1.0 / 3.0 * self._eta4)
            + (self._g_next - self._a)
            / ((1 - self._eta4) * (1 - self._eta4))
            * (
                1.0 / 3.0 * xv * xv * xv
                - self._eta4 * xv * xv
                + self._eta4 * self._eta4 * xv
                - 1.0 / 3.0 * self._eta4 * self._eta4 * self._eta4
            )
        )

    def f_next(self) -> float:
        return self._f_average + self._g_next


@final
class ConvexMonotone4MinHelper(ConvexMonotone4Helper):
    """:class:`ConvexMonotone4Helper` with a zero plateau where it would go negative.

    # C++ parity: ``ConvexMonotone4MinHelper``
    #             (convexmonotoneinterpolation.hpp:394-461).

    When ``A + f_average <= 0`` the parabola pair would dip below zero. The
    helper then shifts the average up by the smallest amount that keeps the
    integral right, compresses the two parabolas into ``[x_prev, x2]`` and
    ``[x3, x_next]``, and returns exactly ``0`` in between. Otherwise it *is*
    the base helper — which is why ``force_positive`` looks like a no-op
    until it is not.
    """

    __slots__ = ("_split_region", "_x2", "_x3", "_x_ratio")

    def __init__(
        self,
        x_prev: float,
        x_next: float,
        g_prev: float,
        g_next: float,
        f_average: float,
        eta4: float,
        prev_primitive: float,
    ) -> None:
        super().__init__(x_prev, x_next, g_prev, g_next, f_average, eta4, prev_primitive)
        self._split_region: bool = False
        self._x_ratio: float = 1.0
        self._x2: float = 0.0
        self._x3: float = 0.0
        if self._a + self._f_average <= 0.0:
            self._split_region = True
            f_prev = self._g_prev + self._f_average
            f_next = self._g_next + self._f_average
            reqd_shift = (
                self._eta4 * f_prev + (1 - self._eta4) * f_next
            ) / 3.0 - self._f_average
            reqd_period = reqd_shift * self._x_scaling / (self._f_average + reqd_shift)
            x_adjust = self._x_scaling - reqd_period
            self._x_ratio = x_adjust / self._x_scaling
            self._f_average += reqd_shift
            self._g_next = f_next - self._f_average
            self._g_prev = f_prev - self._f_average
            self._a = -(self._eta4 * self._g_prev + (1.0 - self._eta4) * self._g_next) / 2.0
            self._x2 = self._x_prev + x_adjust * self._eta4
            self._x3 = self._x_prev + self._x_scaling - x_adjust * (1.0 - self._eta4)

    def value(self, x: float) -> float:
        if not self._split_region:
            return super().value(x)
        xv = (x - self._x_prev) / self._x_scaling
        if x <= self._x2:
            xv /= self._x_ratio
            return (
                self._f_average
                + self._a
                + (self._g_prev - self._a)
                * (self._eta4 - xv)
                * (self._eta4 - xv)
                / (self._eta4 * self._eta4)
            )
        if x < self._x3:
            return 0.0
        xv = 1.0 - (1.0 - xv) / self._x_ratio
        return (
            self._f_average
            + self._a
            + (self._g_next - self._a)
            * (xv - self._eta4)
            * (xv - self._eta4)
            / ((1 - self._eta4) * (1 - self._eta4))
        )

    def primitive(self, x: float) -> float:
        if not self._split_region:
            return super().primitive(x)
        xv = (x - self._x_prev) / self._x_scaling
        if x <= self._x2:
            xv /= self._x_ratio
            return (
                self._prev_primitive
                + self._x_scaling
                * self._x_ratio
                * (
                    self._f_average
                    + self._a
                    + (self._g_prev - self._a)
                    / (self._eta4 * self._eta4)
                    * (self._eta4 * self._eta4 - self._eta4 * xv + 1.0 / 3.0 * xv * xv)
                )
                * xv
            )
        if x <= self._x3:
            # NOTE `<=` here against the `<` in value(): C++'s own asymmetry.
            return self._prev_primitive + self._x_scaling * self._x_ratio * (
                self._f_average * self._eta4
                + self._a * self._eta4
                + (self._g_prev - self._a)
                / (self._eta4 * self._eta4)
                * (1.0 / 3.0 * self._eta4 * self._eta4 * self._eta4)
            )
        xv = 1.0 - (1.0 - xv) / self._x_ratio
        return self._prev_primitive + self._x_scaling * self._x_ratio * (
            self._f_average * xv
            + self._a * xv
            + (self._g_prev - self._a) * (1.0 / 3.0 * self._eta4)
            + (self._g_next - self._a)
            / ((1.0 - self._eta4) * (1.0 - self._eta4))
            * (
                1.0 / 3.0 * xv * xv * xv
                - self._eta4 * xv * xv
                + self._eta4 * self._eta4 * xv
                - 1.0 / 3.0 * self._eta4 * self._eta4 * self._eta4
            )
        )


@final
class ComboHelper(SectionHelper):
    """``quadraticity``-weighted blend of a quadratic and a convex-monotone helper.

    # C++ parity: ``ComboHelper``
    #             (convexmonotoneinterpolation.hpp:241-266).
    """

    __slots__ = ("_conv_mono", "_quadratic", "_quadraticity")

    def __init__(
        self,
        quadratic_helper: SectionHelper,
        conv_mono_helper: SectionHelper,
        quadraticity: float,
    ) -> None:
        qassert.require(
            quadraticity < 1.0 and quadraticity > 0.0,
            "Quadratic value must lie between 0 and 1",
        )
        self._quadraticity: float = quadraticity
        self._quadratic: SectionHelper = quadratic_helper
        self._conv_mono: SectionHelper = conv_mono_helper

    def value(self, x: float) -> float:
        return self._quadraticity * self._quadratic.value(x) + (
            1.0 - self._quadraticity
        ) * self._conv_mono.value(x)

    def primitive(self, x: float) -> float:
        return self._quadraticity * self._quadratic.primitive(x) + (
            1.0 - self._quadraticity
        ) * self._conv_mono.primitive(x)

    def f_next(self) -> float:
        return self._quadraticity * self._quadratic.f_next() + (
            1.0 - self._quadraticity
        ) * self._conv_mono.f_next()


class ConvexMonotoneInterpolation(Interpolation):
    """Convex-monotone yield-curve interpolation.

    # C++ parity: ``ConvexMonotoneInterpolation<I1, I2>``
    #             (convexmonotoneinterpolation.hpp:53-84) plus
    #             ``detail::ConvexMonotoneImpl`` (173-238, 587-848).

    Args:
        x_seq: pillar x values, ascending, at least two.
        y_seq: discretely-averaged values. **The first element is ignored**
            (C++: "the first value in the y-vector is ignored") — ``y[i]``
            is the average over ``[x[i-1], x[i]]``, and there is no such
            interval at the left edge.
        quadratic_constraint: ``True`` (default) selects the C++
            ``ConvexMonotone`` factory defaults ``quadraticity = 0.3``,
            ``monotonicity = 0.7``; ``False`` selects the pure Hagan-West
            ``0.0 / 1.0``. Explicit kwargs override either way.
        quadraticity: blend weight toward the quadratic helper, in [0, 1].
        monotonicity: shape parameter in [0, 1] bounding the split point.
        force_positive: clamp the boundary forwards at zero and select the
            ``*MinHelper`` variants that keep the curve non-negative.
        flat_final_period: C++ ``constantLastPeriod`` — make the final
            period an ``EverywhereConstantHelper``. Used by ``LocalBootstrap``
            while the curve is still growing.
        pre_existing_helpers: helpers for the leading segments, keyed by
            their upper x, as returned by :meth:`get_existing_helpers`.
        update: run :meth:`update` at construction.
    """

    def __init__(
        self,
        x_seq: Sequence[float] | Array,
        y_seq: Sequence[float] | Array,
        quadratic_constraint: bool = True,
        *,
        quadraticity: float | None = None,
        monotonicity: float | None = None,
        force_positive: bool = True,
        flat_final_period: bool = False,
        pre_existing_helpers: dict[float, SectionHelper] | None = None,
        update: bool = True,
    ) -> None:
        xs_arr: Array = np.asarray(x_seq, dtype=np.float64)
        ys_arr: Array = np.asarray(y_seq, dtype=np.float64)
        super().__init__(xs_arr, ys_arr, required_points=_REQUIRED_POINTS)

        # C++ factory defaults at convexmonotoneinterpolation.hpp:94-96.
        if quadraticity is None:
            quadraticity = 0.3 if quadratic_constraint else 0.0
        if monotonicity is None:
            monotonicity = 0.7 if quadratic_constraint else 1.0

        # C++ parity: convexmonotoneinterpolation.hpp:201-209.
        qassert.require(
            0.0 <= monotonicity <= 1.0, "Monotonicity must lie between 0 and 1"
        )
        qassert.require(
            0.0 <= quadraticity <= 1.0, "Quadraticity must lie between 0 and 1"
        )
        length = int(self._xs.shape[0])
        qassert.require(
            length >= 2,
            "Single point provided, not supported by convex monotone method as "
            "first point is ignored",
        )
        pre = dict(pre_existing_helpers) if pre_existing_helpers else {}
        qassert.require(
            (length - len(pre)) > 1, "Too many existing helpers have been supplied"
        )

        self._quadraticity: float = quadraticity
        self._monotonicity: float = monotonicity
        self._force_positive: bool = force_positive
        self._constant_last_period: bool = flat_final_period
        self._pre_section_helpers: dict[float, SectionHelper] = pre

        # Indexed by pillar: slot ``i`` covers ``(x[i-1], x[i]]``. Slot 0 is
        # unused, mirroring the ignored ``y[0]``.
        self._section_helpers: list[SectionHelper | None] = [None] * length
        self._extrapolation_helper: SectionHelper | None = None

        if update:
            self.update()

    # ------------------------------------------------------------------
    # update — port of ``ConvexMonotoneImpl::update``
    # (convexmonotoneinterpolation.hpp:587-830)
    # ------------------------------------------------------------------

    def update(self) -> None:  # noqa: PLR0915 — one-for-one with the C++ body
        """Rebuild the section helpers from the current pillar data."""
        xs = self._xs
        ys = self._ys
        length = int(xs.shape[0])
        self._section_helpers = [None] * length

        if length == 2:  # single period
            single = EverywhereConstantHelper(float(ys[1]), 0.0, float(xs[0]))
            self._section_helpers[1] = single
            self._extrapolation_helper = single
            return

        f: list[float] = [0.0] * length
        pre_keys = sorted(self._pre_section_helpers)
        for slot, key in enumerate(pre_keys, start=1):
            self._section_helpers[slot] = self._pre_section_helpers[key]
        start_point = len(pre_keys) + 1

        # first derive the boundary forwards
        for i in range(start_point, length - 1):
            dx_prev = float(xs[i]) - float(xs[i - 1])
            dx = float(xs[i + 1]) - float(xs[i])
            f[i] = dx / (dx + dx_prev) * float(ys[i]) + dx_prev / (dx + dx_prev) * float(
                ys[i + 1]
            )

        if start_point > 1:
            # C++ ``preSectionHelpers_.rbegin()->second->fNext()`` — the
            # largest-keyed pre-existing helper.
            last_pre = self._pre_section_helpers[pre_keys[-1]]
            f[start_point - 1] = last_pre.f_next()
        if start_point == 1:
            f[0] = 1.5 * float(ys[1]) - 0.5 * f[1]

        f[length - 1] = 1.5 * float(ys[length - 1]) - 0.5 * f[length - 2]

        if self._force_positive:
            if f[0] < 0:
                f[0] = 0.0
            if f[length - 1] < 0.0:  # noqa: PLR1730 — C++ writes the branch, not max()
                f[length - 1] = 0.0

        primitive = 0.0
        for i in range(start_point - 1):
            primitive += float(ys[i + 1]) * (float(xs[i + 1]) - float(xs[i]))

        end_point = length
        if self._constant_last_period:
            end_point -= 1

        for i in range(start_point, end_point):
            x_prev = float(xs[i - 1])
            x_next = float(xs[i])
            y_avg = float(ys[i])
            g_prev = f[i - 1] - y_avg
            g_next = f[i] - y_avg

            # first deal with the zero gradient case
            if abs(g_prev) < _ZERO_GRAD_EPS and abs(g_next) < _ZERO_GRAD_EPS:
                self._section_helpers[i] = ConstantGradHelper(
                    f[i - 1], primitive, x_prev, x_next, f[i]
                )
            else:
                quadraticity = self._quadraticity
                quadratic_helper: SectionHelper | None = None
                conv_monotone_helper: SectionHelper | None = None

                if self._quadraticity > 0.0:
                    if g_prev >= -2.0 * g_next and g_prev > -0.5 * g_next and self._force_positive:
                        quadratic_helper = QuadraticMinHelper(
                            x_prev, x_next, f[i - 1], f[i], y_avg, primitive
                        )
                    else:
                        quadratic_helper = QuadraticHelper(
                            x_prev, x_next, f[i - 1], f[i], y_avg, primitive
                        )

                if self._quadraticity < 1.0:
                    b2 = (1.0 + self._monotonicity) / 2.0
                    b3 = (1.0 - self._monotonicity) / 2.0
                    if (g_prev > 0.0 and -0.5 * g_prev >= g_next >= -2.0 * g_prev) or (
                        g_prev < 0.0 and -0.5 * g_prev <= g_next <= -2.0 * g_prev
                    ):
                        quadraticity = 1.0
                        if self._quadraticity == 0:
                            if self._force_positive:
                                quadratic_helper = QuadraticMinHelper(
                                    x_prev, x_next, f[i - 1], f[i], y_avg, primitive
                                )
                            else:
                                quadratic_helper = QuadraticHelper(
                                    x_prev, x_next, f[i - 1], f[i], y_avg, primitive
                                )
                    elif (g_prev < 0.0 and g_next > -2.0 * g_prev) or (
                        g_prev > 0.0 and g_next < -2.0 * g_prev
                    ):
                        eta = (g_next + 2.0 * g_prev) / (g_next - g_prev)
                        if eta < b2:
                            conv_monotone_helper = ConvexMonotone2Helper(
                                x_prev, x_next, g_prev, g_next, y_avg, eta, primitive
                            )
                        elif self._force_positive:
                            conv_monotone_helper = ConvexMonotone4MinHelper(
                                x_prev, x_next, g_prev, g_next, y_avg, b2, primitive
                            )
                        else:
                            conv_monotone_helper = ConvexMonotone4Helper(
                                x_prev, x_next, g_prev, g_next, y_avg, b2, primitive
                            )
                    elif (g_prev > 0.0 and g_next < 0.0 and g_next > -0.5 * g_prev) or (
                        g_prev < 0.0 and g_next > 0.0 and g_next < -0.5 * g_prev
                    ):
                        eta = g_next / (g_next - g_prev) * 3.0
                        if eta > b3:
                            conv_monotone_helper = ConvexMonotone3Helper(
                                x_prev, x_next, g_prev, g_next, y_avg, eta, primitive
                            )
                        elif self._force_positive:
                            conv_monotone_helper = ConvexMonotone4MinHelper(
                                x_prev, x_next, g_prev, g_next, y_avg, b3, primitive
                            )
                        else:
                            conv_monotone_helper = ConvexMonotone4Helper(
                                x_prev, x_next, g_prev, g_next, y_avg, b3, primitive
                            )
                    else:
                        # `g_prev + g_next` cannot be zero here: g_next ==
                        # -g_prev with either sign lands in the first branch
                        # above, and g_prev == 0 forces g_next != 0 (the
                        # zero-gradient case was handled earlier). So the
                        # division is safe without C++'s implicit IEEE
                        # inf/nan fallback.
                        eta = g_next / (g_prev + g_next)
                        b2 = (1.0 + self._monotonicity) / 2.0
                        b3 = (1.0 - self._monotonicity) / 2.0
                        if eta > b2:  # noqa: PLR1730 — C++ clamps with two ifs
                            eta = b2
                        if eta < b3:  # noqa: PLR1730
                            eta = b3
                        if self._force_positive:
                            conv_monotone_helper = ConvexMonotone4MinHelper(
                                x_prev, x_next, g_prev, g_next, y_avg, eta, primitive
                            )
                        else:
                            conv_monotone_helper = ConvexMonotone4Helper(
                                x_prev, x_next, g_prev, g_next, y_avg, eta, primitive
                            )

                if quadraticity == 1.0:
                    qassert.require(
                        quadratic_helper is not None,
                        "ConvexMonotone: quadratic branch must be active when quadraticity=1",
                    )
                    self._section_helpers[i] = quadratic_helper
                elif quadraticity == 0.0:
                    qassert.require(
                        conv_monotone_helper is not None,
                        "ConvexMonotone: convex-monotone branch must be active "
                        "when quadraticity=0",
                    )
                    self._section_helpers[i] = conv_monotone_helper
                else:
                    qassert.require(
                        quadratic_helper is not None and conv_monotone_helper is not None,
                        "ConvexMonotone: both branches must be active for combo",
                    )
                    assert quadratic_helper is not None
                    assert conv_monotone_helper is not None
                    self._section_helpers[i] = ComboHelper(
                        quadratic_helper, conv_monotone_helper, quadraticity
                    )

            primitive += y_avg * (float(xs[i]) - float(xs[i - 1]))

        if self._constant_last_period:
            last = EverywhereConstantHelper(
                float(ys[length - 1]), primitive, float(xs[length - 2])
            )
            self._section_helpers[length - 1] = last
            self._extrapolation_helper = last
        else:
            # C++ ``sectionHelpers_.rbegin()->second`` is the helper at the
            # largest key, i.e. x[length-1].
            last_in_range = self._section_helpers[length - 1]
            qassert.require(
                last_in_range is not None,
                "ConvexMonotone: final section helper was not built",
            )
            assert last_in_range is not None
            self._extrapolation_helper = EverywhereConstantHelper(
                last_in_range.value(float(xs[length - 1])),
                primitive,
                float(xs[length - 1]),
            )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get_existing_helpers(self) -> dict[float, SectionHelper]:
        """The section helpers keyed by their upper x.

        # C++ parity: ``ConvexMonotoneImpl::getExistingHelpers``
        #             (convexmonotoneinterpolation.hpp:224-229) — drops the
        #             final helper when ``constantLastPeriod`` is set,
        #             because that one is a placeholder for a pillar that has
        #             not been bootstrapped yet.
        """
        xs = self._xs
        out: dict[float, SectionHelper] = {}
        for i, helper in enumerate(self._section_helpers):
            if helper is not None:
                out[float(xs[i])] = helper
        if self._constant_last_period:
            out.pop(float(xs[-1]), None)
        return out

    @property
    def section_helpers(self) -> list[SectionHelper | None]:
        """Section helpers indexed by pillar; slot 0 is ``None``."""
        if self._constant_last_period:
            return [*self._section_helpers[:-1], None]
        return list(self._section_helpers)

    def _segment_for(self, x: float) -> SectionHelper:
        xs = self._xs
        if x >= float(xs[-1]):
            assert self._extrapolation_helper is not None
            return self._extrapolation_helper
        # C++ ``sectionHelpers_.upper_bound(x)`` — the first segment whose
        # upper x is strictly greater than ``x``. The helper keys are
        # ``xs[1..n-1]``, so that index is ``bisect_right(xs, x)`` clamped
        # into ``[1, n-1]``.
        i = bisect.bisect_right(xs.tolist(), x)
        i = max(i, 1)
        i = min(i, int(xs.shape[0]) - 1)
        helper = self._section_helpers[i]
        if helper is None:
            assert self._extrapolation_helper is not None
            return self._extrapolation_helper
        return helper

    def _value(self, x: float) -> float:
        return self._segment_for(x).value(x)

    def _primitive(self, x: float) -> float:
        return self._segment_for(x).primitive(x)

    def _derivative(self, x: float) -> float:
        del x
        # C++ ``QL_FAIL`` (convexmonotoneinterpolation.hpp:216-218).
        raise NotImplementedError("Convex-monotone spline derivative not implemented")

    def _second_derivative(self, x: float) -> float:
        del x
        # C++ ``QL_FAIL`` (convexmonotoneinterpolation.hpp:219-222).
        raise NotImplementedError(
            "Convex-monotone spline second derivative not implemented"
        )

    @property
    def quadraticity(self) -> float:
        return self._quadraticity

    @property
    def monotonicity(self) -> float:
        return self._monotonicity

    @property
    def force_positive(self) -> bool:
        return self._force_positive


@final
class ConvexMonotone:
    """Convex-monotone interpolation factory and traits.

    # C++ parity: ``class ConvexMonotone``
    #             (convexmonotoneinterpolation.hpp:88-159).
    """

    global_ = True  # C++ ``static const bool global = true``.
    required_points = 2  # C++ ``static const Size requiredPoints = 2``.
    data_size_adjustment = 1  # C++ ``static const Size dataSizeAdjustment = 1``.

    def __init__(
        self,
        quadraticity: float = 0.3,
        monotonicity: float = 0.7,
        force_positive: bool = True,
    ) -> None:
        self._quadraticity: float = quadraticity
        self._monotonicity: float = monotonicity
        self._force_positive: bool = force_positive

    def interpolate(
        self, x_seq: Array, y_seq: Array, update: bool = True
    ) -> ConvexMonotoneInterpolation:
        """Build a :class:`ConvexMonotoneInterpolation` over ``(x, y)``."""
        return ConvexMonotoneInterpolation(
            x_seq,
            y_seq,
            quadraticity=self._quadraticity,
            monotonicity=self._monotonicity,
            force_positive=self._force_positive,
            flat_final_period=False,
            update=update,
        )

    def local_interpolate(
        self,
        x_seq: Array,
        y_seq: Array,
        localisation: int,
        prev_interpolation: ConvexMonotoneInterpolation | None,
        final_size: int,
    ) -> ConvexMonotoneInterpolation:
        """Grow the interpolation by one pillar, reusing the settled helpers.

        # C++ parity: ``ConvexMonotone::localInterpolate``
        #             (convexmonotoneinterpolation.hpp:112-155).

        ``LocalBootstrap`` calls this once per pillar. Until the curve
        reaches ``final_size`` the last period is held flat, because its
        pillar is still being solved for; the helpers for everything to its
        left are carried over verbatim so the bootstrap does not re-solve
        settled sections.
        """
        length = int(np.asarray(x_seq, dtype=np.float64).shape[0])
        if length - localisation == 1:  # the first time this function is called
            return ConvexMonotoneInterpolation(
                x_seq,
                y_seq,
                quadraticity=self._quadraticity,
                monotonicity=self._monotonicity,
                force_positive=self._force_positive,
                flat_final_period=length != final_size,
            )
        qassert.require(
            prev_interpolation is not None,
            "localInterpolate needs the previous interpolation once the curve "
            "has grown past its first segment",
        )
        assert prev_interpolation is not None
        return ConvexMonotoneInterpolation(
            x_seq,
            y_seq,
            quadraticity=self._quadraticity,
            monotonicity=self._monotonicity,
            force_positive=self._force_positive,
            flat_final_period=length != final_size,
            pre_existing_helpers=prev_interpolation.get_existing_helpers(),
        )


__all__ = [
    "ComboHelper",
    "ConstantGradHelper",
    "ConvexMonotone",
    "ConvexMonotone2Helper",
    "ConvexMonotone3Helper",
    "ConvexMonotone4Helper",
    "ConvexMonotone4MinHelper",
    "ConvexMonotoneInterpolation",
    "EverywhereConstantHelper",
    "QuadraticHelper",
    "QuadraticMinHelper",
    "SectionHelper",
]
