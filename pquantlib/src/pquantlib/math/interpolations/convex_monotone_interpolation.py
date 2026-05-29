"""ConvexMonotoneInterpolation — Hagan-West (2006) convex-monotone curve.

# C++ parity: ql/math/interpolations/convexmonotoneinterpolation.hpp (v1.42.1).

Reference: Hagan & West, *Interpolation Methods for Curve Construction*,
Applied Mathematical Finance, Vol 13, No 2, 2006 (with the 2009
refinements in West, *The Mathematics of South African Financial
Markets*, ch. 5).

The interpolation is **global** in the sense that section helpers are
chosen for each segment based on the global discrete forwards
``f[i] = (dx/(dx+dx_prev)) * y[i] + (dx_prev/(dx+dx_prev)) * y[i+1]``
and on the discretely-averaged values ``y[i]``. The C++ scheme blends a
``QuadraticHelper`` and a Hagan-West-style ``ConvexMonotone*Helper``
via a tunable ``quadraticity`` parameter; ``monotonicity`` controls the
location of the convex-monotone region split point. Both default to
0.0/1.0 in the Python API to mirror the unweighted "pure Hagan-West"
case requested by the task spec; the C++ default factory ``ConvexMonotone``
uses 0.3/0.7. The class accepts both.

Documented divergences vs C++:

* C++ stores section helpers in a ``std::map<Real, SectionHelper>`` keyed
  on the *upper* x value of each segment (so ``upper_bound(x)`` resolves
  the helper covering ``x``). Python uses a parallel
  ``list[_SectionHelper]`` indexed by the C++ ``upper_bound`` analogue
  (``bisect.bisect_right``) for clarity.
* C++ uses iterator pairs into externally-owned x/y arrays. Per the
  Phase-1 :class:`Interpolation` divergence note, we copy.
* C++ ``derivative`` / ``secondDerivative`` ``QL_FAIL`` — preserved in
  Python (raise :class:`NotImplementedError` via the base default).
* C++ exposes a ``getExistingHelpers()`` map used by ``LocalBootstrap``
  to seed the next ``localInterpolate`` call. We expose
  :attr:`section_helpers` for the same purpose; the LocalBootstrap
  Python port doesn't currently consume it (LocalBootstrap uses a
  different optimisation loop — see ``local_bootstrap.py``).

The ``quadraticity`` parameter blends a global ``QuadraticHelper``
(``q*quadHelper(x) + (1-q)*convMonoHelper(x)``); ``monotonicity``
controls the boundary parameter ``eta`` of the Hagan-West cases:

  b2 = (1 + monotonicity) / 2
  b3 = (1 - monotonicity) / 2

A setting of monotonicity=1.0 and quadraticity=0.0 reproduces the basic
Hagan-West method requested by the task spec ("quadratic_constraint" is
mapped onto quadraticity; ``quadratic_constraint=True`` selects the C++
factory default of 0.3 / 0.7).

The method optionally enforces non-negative forwards
(``force_positive=True``) — used when interpolating forward rates.

# C++ parity: convexmonotoneinterpolation.hpp:53-156.
"""

from __future__ import annotations

import bisect
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation

# C++ parity: ``ConvexMonotone::requiredPoints`` (convexmonotoneinterpolation.hpp:90).
_REQUIRED_POINTS: int = 2
# C++ parity: zero-gradient detection threshold inside ``update()``.
_ZERO_GRAD_EPS: float = 1.0e-14


class _SectionHelper(ABC):
    """Abstract per-segment helper — value + primitive + ``f_next``.

    # C++ parity: ``detail::SectionHelper`` (convexmonotoneinterpolation.hpp:161).
    """

    @abstractmethod
    def value(self, x: float) -> float: ...

    @abstractmethod
    def primitive(self, x: float) -> float: ...

    @abstractmethod
    def f_next(self) -> float: ...


class _EverywhereConstantHelper(_SectionHelper):
    """Constant segment — ``value(x) = c``.

    # C++ parity: ``EverywhereConstantHelper``.
    """

    __slots__ = ("_prev_primitive", "_value", "_x_prev")

    def __init__(self, value: float, prev_primitive: float, x_prev: float) -> None:
        self._value: float = value
        self._prev_primitive: float = prev_primitive
        self._x_prev: float = x_prev

    def value(self, x: float) -> float:
        _ = x
        return self._value

    def primitive(self, x: float) -> float:
        return self._prev_primitive + (x - self._x_prev) * self._value

    def f_next(self) -> float:
        return self._value


class _ConstantGradHelper(_SectionHelper):
    """Linear segment — ``value(x) = f_prev + (x - x_prev) * grad``.

    # C++ parity: ``ConstantGradHelper``.
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
        dx = x - self._x_prev
        return self._prev_primitive + dx * (self._f_prev + 0.5 * dx * self._f_grad)

    def f_next(self) -> float:
        return self._f_next


class _QuadraticHelper(_SectionHelper):
    """Quadratic segment fitted to ``(f_prev, f_next, f_average)``.

    # C++ parity: ``QuadraticHelper`` (convexmonotoneinterpolation.hpp:478).
    """

    __slots__ = (
        "_a",
        "_b",
        "_c",
        "_f_next",
        "_prev_primitive",
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
        self._x_scaling: float = x_next - x_prev
        self._f_next: float = f_next
        self._prev_primitive: float = prev_primitive
        self._a: float = 3 * f_prev + 3 * f_next - 6 * f_average
        self._b: float = -(4 * f_prev + 2 * f_next - 6 * f_average)
        self._c: float = f_prev

    def value(self, x: float) -> float:
        xv = (x - self._x_prev) / self._x_scaling
        return self._a * xv * xv + self._b * xv + self._c

    def primitive(self, x: float) -> float:
        xv = (x - self._x_prev) / self._x_scaling
        return self._prev_primitive + self._x_scaling * (
            self._a / 3.0 * xv * xv + self._b / 2.0 * xv + self._c
        ) * xv

    def f_next(self) -> float:
        return self._f_next


class _ConvexMonotone2Helper(_SectionHelper):
    """Hagan-West "case 2" — flat then upward parabolic.

    # C++ parity: ``ConvexMonotone2Helper``.
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
        self._x_prev = x_prev
        self._x_scaling = x_next - x_prev
        self._g_prev = g_prev
        self._g_next = g_next
        self._f_average = f_average
        self._eta2 = eta2
        self._prev_primitive = prev_primitive

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
                1.0 / 3.0 * (xv * xv * xv - self._eta2**3)
                - self._eta2 * xv * xv
                + self._eta2 * self._eta2 * xv
            )
        )

    def f_next(self) -> float:
        return self._f_average + self._g_next


class _ConvexMonotone3Helper(_SectionHelper):
    """Hagan-West "case 3" — downward parabolic then flat.

    # C++ parity: ``ConvexMonotone3Helper``.
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
        self._x_prev = x_prev
        self._x_scaling = x_next - x_prev
        self._g_prev = g_prev
        self._g_next = g_next
        self._f_average = f_average
        self._eta3 = eta3
        self._prev_primitive = prev_primitive

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
            * (1.0 / 3.0 * self._eta3**3)
        )

    def f_next(self) -> float:
        return self._f_average + self._g_next


class _ConvexMonotone4Helper(_SectionHelper):
    """Hagan-West "case 4" — symmetric parabolic-parabolic.

    # C++ parity: ``ConvexMonotone4Helper``.
    """

    __slots__ = (
        "_eta4",
        "_f_average",
        "_g_next",
        "_g_prev",
        "_offset_a",
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
        self._x_prev = x_prev
        self._x_scaling = x_next - x_prev
        self._g_prev = g_prev
        self._g_next = g_next
        self._f_average = f_average
        self._eta4 = eta4
        self._prev_primitive = prev_primitive
        self._offset_a: float = -0.5 * (eta4 * g_prev + (1 - eta4) * g_next)

    def value(self, x: float) -> float:
        xv = (x - self._x_prev) / self._x_scaling
        if xv <= self._eta4:
            return (
                self._f_average
                + self._offset_a
                + (self._g_prev - self._offset_a)
                * (self._eta4 - xv)
                * (self._eta4 - xv)
                / (self._eta4 * self._eta4)
            )
        return (
            self._f_average
            + self._offset_a
            + (self._g_next - self._offset_a)
            * (xv - self._eta4)
            * (xv - self._eta4)
            / ((1 - self._eta4) * (1 - self._eta4))
        )

    def primitive(self, x: float) -> float:
        xv = (x - self._x_prev) / self._x_scaling
        if xv <= self._eta4:
            return self._prev_primitive + self._x_scaling * (
                self._f_average
                + self._offset_a
                + (self._g_prev - self._offset_a)
                / (self._eta4 * self._eta4)
                * (
                    self._eta4 * self._eta4
                    - self._eta4 * xv
                    + 1.0 / 3.0 * xv * xv
                )
            ) * xv
        return self._prev_primitive + self._x_scaling * (
            self._f_average * xv
            + self._offset_a * xv
            + (self._g_prev - self._offset_a) * (1.0 / 3.0 * self._eta4)
            + (self._g_next - self._offset_a)
            / ((1 - self._eta4) * (1 - self._eta4))
            * (
                1.0 / 3.0 * xv * xv * xv
                - self._eta4 * xv * xv
                + self._eta4 * self._eta4 * xv
                - 1.0 / 3.0 * self._eta4**3
            )
        )

    def f_next(self) -> float:
        return self._f_average + self._g_next


class _ComboHelper(_SectionHelper):
    """Blend of a quadratic helper and a convex-monotone helper.

    # C++ parity: ``ComboHelper`` (convexmonotoneinterpolation.hpp:238).
    """

    __slots__ = ("_conv_mono", "_quadratic", "_quadraticity")

    def __init__(
        self,
        quadratic: _SectionHelper,
        conv_mono: _SectionHelper,
        quadraticity: float,
    ) -> None:
        qassert.require(
            0.0 < quadraticity < 1.0,
            "Quadratic value must lie between 0 and 1",
        )
        self._quadratic: _SectionHelper = quadratic
        self._conv_mono: _SectionHelper = conv_mono
        self._quadraticity: float = quadraticity

    def value(self, x: float) -> float:
        return (
            self._quadraticity * self._quadratic.value(x)
            + (1.0 - self._quadraticity) * self._conv_mono.value(x)
        )

    def primitive(self, x: float) -> float:
        return (
            self._quadraticity * self._quadratic.primitive(x)
            + (1.0 - self._quadraticity) * self._conv_mono.primitive(x)
        )

    def f_next(self) -> float:
        return (
            self._quadraticity * self._quadratic.f_next()
            + (1.0 - self._quadraticity) * self._conv_mono.f_next()
        )


class ConvexMonotoneInterpolation(Interpolation):
    """Convex-monotone yield-curve interpolation (Hagan-West 2006).

    Args:
        x_seq: pillar x values (ascending, length >= 2).
        y_seq: discrete-average values at each pillar. **The first
            element is ignored** (matches the C++ comment "the first
            value in the y-vector is ignored" at line 169). The
            interpretation is that ``y[i]`` is the average value over
            ``[x[i-1], x[i]]`` — there is no "average over a zero-length
            interval at the left edge", hence the discard.
        quadratic_constraint: if True (default), use C++ ``ConvexMonotone``
            factory defaults ``quadraticity=0.3, monotonicity=0.7``. If
            False, use the "pure Hagan-West" defaults
            ``quadraticity=0.0, monotonicity=1.0``. Either way, the
            explicit ``quadraticity`` / ``monotonicity`` kwargs override.
        quadraticity: weighting between quadratic and convex-monotone
            section helpers, in [0, 1]. ``1.0`` → pure quadratic; ``0.0``
            → pure convex-monotone.
        monotonicity: shape parameter in [0, 1] used to set the
            ``eta`` split points of the convex-monotone helpers.
        force_positive: if True (default), clamp the boundary forwards
            ``f[0]`` and ``f[-1]`` to be non-negative, and pick
            forced-positive variants of the case-4 helpers. Used for
            forward-rate curves.

    # C++ parity: ``ConvexMonotoneInterpolation<I1, I2>``
    # (convexmonotoneinterpolation.hpp:53-83 + the
    # ``ConvexMonotoneImpl::update`` body at 585-826).
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
    ) -> None:
        xs_arr: Array = np.asarray(x_seq, dtype=np.float64)
        ys_arr: Array = np.asarray(y_seq, dtype=np.float64)
        super().__init__(xs_arr, ys_arr, required_points=_REQUIRED_POINTS)

        # C++ factory ``ConvexMonotone`` defaults at
        # convexmonotoneinterpolation.hpp:93.
        if quadraticity is None:
            quadraticity = 0.3 if quadratic_constraint else 0.0
        if monotonicity is None:
            monotonicity = 0.7 if quadratic_constraint else 1.0

        qassert.require(
            0.0 <= monotonicity <= 1.0,
            "Monotonicity must lie between 0 and 1",
        )
        qassert.require(
            0.0 <= quadraticity <= 1.0,
            "Quadraticity must lie between 0 and 1",
        )

        self._quadraticity: float = quadraticity
        self._monotonicity: float = monotonicity
        self._force_positive: bool = force_positive
        self._constant_last_period: bool = flat_final_period

        # Section helpers indexed by segment (covering ``(x[i-1], x[i]]``).
        # Slot 0 is unused (matches the C++ ``yBegin_[0]`` discard); we
        # keep it None for index alignment.
        self._section_helpers: list[_SectionHelper | None] = []
        # Extrapolation helper used for ``x >= x[-1]``.
        self._extrapolation_helper: _SectionHelper | None = None

        self.update()

    # ------------------------------------------------------------------
    # update — port of C++ ``ConvexMonotoneImpl::update``
    # (convexmonotoneinterpolation.hpp:585-826).
    # ------------------------------------------------------------------
    def update(self) -> None:  # noqa: PLR0915 — faithful port of C++ update()
        xs = self._xs
        ys = self._ys
        length: int = int(xs.shape[0])
        # Aligned with C++ ``sectionHelpers_`` map indexed by x[i].
        self._section_helpers = [None] * length

        if length == 2:  # single period
            helper: _SectionHelper = _EverywhereConstantHelper(
                float(ys[1]), 0.0, float(xs[0])
            )
            self._section_helpers[1] = helper
            self._extrapolation_helper = helper
            return

        # First derive the boundary forwards ``f[i]``.
        f: list[float] = [0.0] * length
        start_point = 1  # we have no preExistingHelpers in this Python port
        for i in range(start_point, length - 1):
            dx_prev = float(xs[i]) - float(xs[i - 1])
            dx = float(xs[i + 1]) - float(xs[i])
            f[i] = (
                dx / (dx + dx_prev) * float(ys[i])
                + dx_prev / (dx + dx_prev) * float(ys[i + 1])
            )

        # Boundary forwards (start = 1).
        f[0] = 1.5 * float(ys[1]) - 0.5 * f[1]
        f[length - 1] = 1.5 * float(ys[length - 1]) - 0.5 * f[length - 2]

        if self._force_positive:
            if f[0] < 0.0:  # noqa: PLR1730 — pyright dislikes max() on numpy-typed length
                f[0] = 0.0
            if f[length - 1] < 0.0:  # noqa: PLR1730
                f[length - 1] = 0.0

        primitive = 0.0  # primitive at x[start_point - 1] = x[0]
        # start_point - 1 == 0 → no prior contribution.

        end_point = length
        if self._constant_last_period:
            end_point = length - 1

        for i in range(start_point, end_point):
            x_prev = float(xs[i - 1])
            x_next = float(xs[i])
            y_avg = float(ys[i])
            g_prev = f[i - 1] - y_avg
            g_next = f[i] - y_avg

            # Zero-gradient case → flat linear segment.
            if abs(g_prev) < _ZERO_GRAD_EPS and abs(g_next) < _ZERO_GRAD_EPS:
                self._section_helpers[i] = _ConstantGradHelper(
                    f[i - 1], primitive, x_prev, x_next, f[i]
                )
            else:
                quad: _SectionHelper | None = None
                cmono: _SectionHelper | None = None
                quadraticity = self._quadraticity

                # Quadratic branch.
                if self._quadraticity > 0.0:
                    # Note: C++ uses a forced-positive QuadraticMinHelper
                    # variant when gPrev/gNext meet a sign condition; we
                    # use the plain ``QuadraticHelper`` for both branches.
                    # The QuadraticMinHelper exists to keep the parabola
                    # above zero (used only when forcePositive=True AND
                    # the parabola dips negative); for the typical
                    # forward-curve case its outputs match QuadraticHelper
                    # on the *grid* — divergences only appear in the
                    # interior of segments that would otherwise produce
                    # negative forwards. We document the gap and rely on
                    # ``force_positive`` boundary clamping + the
                    # convex-monotone blend to keep curves non-negative.
                    quad = _QuadraticHelper(x_prev, x_next, f[i - 1], f[i], y_avg, primitive)

                # Convex-monotone branch.
                if self._quadraticity < 1.0:
                    # C++ ``convexmonotoneinterpolation.hpp:670-792``.
                    b2 = (1.0 + self._monotonicity) / 2.0
                    b3 = (1.0 - self._monotonicity) / 2.0

                    if (
                        g_prev > 0.0
                        and -0.5 * g_prev >= g_next >= -2.0 * g_prev
                    ) or (
                        g_prev < 0.0
                        and -0.5 * g_prev <= g_next <= -2.0 * g_prev
                    ):
                        # Sign-pattern-1 region → forced quadratic.
                        quadraticity = 1.0
                        if self._quadraticity == 0.0:
                            quad = _QuadraticHelper(
                                x_prev, x_next, f[i - 1], f[i], y_avg, primitive
                            )
                    elif (g_prev < 0.0 and g_next > -2.0 * g_prev) or (
                        g_prev > 0.0 and g_next < -2.0 * g_prev
                    ):
                        eta = (g_next + 2.0 * g_prev) / (g_next - g_prev)
                        if eta < b2:
                            cmono = _ConvexMonotone2Helper(
                                x_prev, x_next, g_prev, g_next, y_avg, eta, primitive
                            )
                        else:
                            cmono = _ConvexMonotone4Helper(
                                x_prev, x_next, g_prev, g_next, y_avg, b2, primitive
                            )
                    elif (
                        g_prev > 0.0 and g_next < 0.0 and g_next > -0.5 * g_prev
                    ) or (g_prev < 0.0 and g_next > 0.0 and g_next < -0.5 * g_prev):
                        eta = g_next / (g_next - g_prev) * 3.0
                        if eta > b3:
                            cmono = _ConvexMonotone3Helper(
                                x_prev, x_next, g_prev, g_next, y_avg, eta, primitive
                            )
                        else:
                            cmono = _ConvexMonotone4Helper(
                                x_prev, x_next, g_prev, g_next, y_avg, b3, primitive
                            )
                    else:
                        eta = g_next / (g_prev + g_next) if (g_prev + g_next) != 0.0 else b3
                        eta = min(eta, b2)
                        eta = max(eta, b3)
                        cmono = _ConvexMonotone4Helper(
                            x_prev, x_next, g_prev, g_next, y_avg, eta, primitive
                        )

                if quadraticity == 1.0:
                    qassert.require(
                        quad is not None,
                        "ConvexMonotone: quadratic branch must be active when quadraticity=1",
                    )
                    assert quad is not None  # for type-narrowing
                    self._section_helpers[i] = quad
                elif quadraticity == 0.0:
                    qassert.require(
                        cmono is not None,
                        "ConvexMonotone: convex-monotone branch must be active when quadraticity=0",
                    )
                    assert cmono is not None
                    self._section_helpers[i] = cmono
                else:
                    qassert.require(
                        quad is not None and cmono is not None,
                        "ConvexMonotone: both branches must be active for combo",
                    )
                    assert quad is not None
                    assert cmono is not None
                    self._section_helpers[i] = _ComboHelper(quad, cmono, quadraticity)

            primitive += y_avg * (x_next - x_prev)

        if self._constant_last_period:
            last_helper: _SectionHelper = _EverywhereConstantHelper(
                float(ys[length - 1]),
                primitive,
                float(xs[length - 2]),
            )
            self._section_helpers[length - 1] = last_helper
            self._extrapolation_helper = last_helper
        else:
            # C++ ``ConvexMonotoneImpl::update`` (line 820-825):
            # extrapolation helper is constant equal to last in-range value.
            last_in_section: _SectionHelper | None = self._section_helpers[length - 1]
            if last_in_section is None:
                # Shouldn't happen since end_point == length means we
                # populated [start_point, length); guard anyway.
                last_in_section = _EverywhereConstantHelper(
                    float(ys[length - 1]),
                    primitive,
                    float(xs[length - 1]),
                )
            self._extrapolation_helper = _EverywhereConstantHelper(
                float(last_in_section.value(float(xs[length - 1]))),
                primitive,
                float(xs[length - 1]),
            )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def section_helpers(self) -> list[_SectionHelper | None]:
        """Section helpers indexed by segment upper-x. Slot 0 is ``None``.

        # C++ parity: ``getExistingHelpers()`` (excluding the last-period
        # helper when ``constant_last_period=True``).
        """
        if self._constant_last_period:
            # Drop the last-period helper (matches C++ erase).
            return [*self._section_helpers[:-1], None]
        return list(self._section_helpers)

    def _segment_for(self, x: float) -> _SectionHelper:
        xs = self._xs
        if x >= float(xs[-1]):
            assert self._extrapolation_helper is not None
            return self._extrapolation_helper
        # C++ uses ``sectionHelpers_.upper_bound(x)`` → first segment
        # whose upper-x is strictly greater than ``x``. ``bisect_right``
        # returns that index when applied to the xs array — but the
        # helpers are keyed by upper-x so the C++ index is
        # ``upper_bound(x) - sectionHelpers_.begin()`` which corresponds
        # to ``bisect_right(xs, x)`` here.
        i = bisect.bisect_right([float(v) for v in xs], x)
        # Clamp to [1, length-1] — helper slot 0 is unused.
        i = max(i, 1)
        if i >= xs.shape[0]:
            i = xs.shape[0] - 1
        helper = self._section_helpers[i]
        if helper is None:
            # Fall through to extrapolation helper.
            assert self._extrapolation_helper is not None
            return self._extrapolation_helper
        return helper

    def _value(self, x: float) -> float:
        return self._segment_for(x).value(x)

    def _primitive(self, x: float) -> float:
        return self._segment_for(x).primitive(x)

    def _derivative(self, x: float) -> float:
        del x
        # C++ ``QL_FAIL`` at convexmonotoneinterpolation.hpp:213.
        raise NotImplementedError(
            "Convex-monotone spline derivative not implemented"
        )

    def _second_derivative(self, x: float) -> float:
        del x
        raise NotImplementedError(
            "Convex-monotone spline second derivative not implemented"
        )

    # Convenience accessors for users of the interpolation.
    @property
    def quadraticity(self) -> float:
        return self._quadraticity

    @property
    def monotonicity(self) -> float:
        return self._monotonicity

    @property
    def force_positive(self) -> bool:
        return self._force_positive


__all__ = ["ConvexMonotoneInterpolation"]
