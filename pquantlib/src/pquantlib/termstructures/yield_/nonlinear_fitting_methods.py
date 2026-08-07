"""Nonlinear methods to fit a bond discount function.

# C++ parity: ql/termstructures/yield/nonlinearfittingmethods.{hpp,cpp} (v1.43)

Seven concrete :class:`~pquantlib.termstructures.yield_.fitted_bond_discount_curve.FittingMethod`
strategies for :class:`~pquantlib.termstructures.yield_.fitted_bond_discount_curve.FittedBondDiscountCurve`:

- :class:`ExponentialSplinesFitting` — Merrill Lynch exponential splines.
- :class:`NelsonSiegelFitting` / :class:`SvenssonFitting` — parametric zero curves.
- :class:`CubicBSplinesFitting` — McCulloch cubic B-splines over a knot vector.
- :class:`NaturalCubicFitting` — natural cubic spline through nodal discounts.
- :class:`SimplePolynomialFitting` — a Bernstein-basis polynomial.
- :class:`SpreadFittingMethod` — any of the above as a spread over a given curve.

Each C++ class declares two or three constructors that differ only in argument
ORDER (a positional shortcut for the l2-without-optimizer case, and for
exponential splines a numCoeffs/fixedKappa-first form).  Python's keyword
arguments make those redundant, so each class here has ONE signature — the
widest C++ one, in the same order.

``Null<Real>()`` is spelled ``None``: it appears only as
``ExponentialSplinesFitting``'s "kappa is a free parameter" sentinel, and
``fixedKappa_ != Null<Real>()`` (cpp:72, 80) becomes ``is not None``.

**FMA contraction.** Three of the discount functions accumulate with
``d += x[i] * <something>``.  Clang defaults to ``-ffp-contract=on`` and fuses
each of those into a single FMA, whose result differs from the separately
rounded one by up to half an ulp — and this is not academic: the fit is a
simplex, so one flipped comparison sends the whole iterate path elsewhere.
Measured against the v1.43 probe, the plain transcription misses 9 of 40
exponential-spline discount factors and lands its fit at 241 function
evaluations instead of 257; with :func:`math.fma` in the accumulation it
matches all 40 and reproduces the fit — solution, cost, iteration count —
bit for bit.  Same call, same reasoning as
``pquantlib.experimental.math.std_random`` and
``pquantlib.methods.finitedifferences.operators.numerical_differentiation``.

``NelsonSiegelFitting`` and ``SvenssonFitting`` are left as plain expressions:
they already reproduce every probed value exactly, so there is nothing to
infer about how (or whether) the compiler contracted them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.bernstein_polynomial import BernsteinPolynomial
from pquantlib.math.bspline import BSpline
from pquantlib.math.constants import QL_EPSILON, QL_MAX_REAL
from pquantlib.math.interpolations.cubic_interpolation import CubicNaturalSpline
from pquantlib.termstructures.yield_.fitted_bond_discount_curve import FittingMethod

if TYPE_CHECKING:
    from pquantlib.math.optimization.constraint import Constraint
    from pquantlib.math.optimization.optimization_method import OptimizationMethod
    from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class ExponentialSplinesFitting(FittingMethod):
    r"""Exponential-splines fitting method.

    # C++ parity: ``ExponentialSplinesFitting``
    # (nonlinearfittingmethods.hpp:50-83, cpp:28-100).

    Fits ``d(t) = sum_i c_i exp(-kappa_i t)``.  See Li, B., E. DeWetering,
    G. Lucas, R. Brenner and A. Shapiro (2001): "Merrill Lynch Exponential
    Spline Model."

    ``fixed_kappa`` pins kappa and drops it from the optimization, shrinking
    :meth:`size` by one; ``None`` (C++ ``Null<Real>()``) leaves it free, in
    which case it is the LAST element of the parameter vector.

    Convergence may be slow.
    """

    def __init__(
        self,
        constrain_at_zero: bool = True,
        weights: Sequence[float] | Array | None = None,
        optimization_method: OptimizationMethod | None = None,
        l2: Sequence[float] | Array | None = None,
        min_cutoff_time: float = 0.0,
        max_cutoff_time: float = QL_MAX_REAL,
        num_coeffs: int = 9,
        fixed_kappa: float | None = None,
        constraint: Constraint | None = None,
    ) -> None:
        # C++ parity: cpp:28-42.
        super().__init__(
            constrain_at_zero,
            weights,
            optimization_method,
            l2,
            min_cutoff_time,
            max_cutoff_time,
            constraint,
        )
        self._num_coeffs: int = num_coeffs
        self._fixed_kappa: float | None = fixed_kappa
        qassert.require(self.size() > 0, "At least 1 unconstrained coefficient required")

    def clone(self) -> ExponentialSplinesFitting:
        """# C++ parity: cpp:64-67."""
        return _as(ExponentialSplinesFitting, self._cloned())

    def size(self) -> int:
        # C++ parity: cpp:69-73. One fewer optimization parameter if kappa is fixed.
        n = self._num_coeffs if self._constrain_at_zero else self._num_coeffs + 1
        return n - 1 if self._fixed_kappa is not None else n

    def _discount_function(self, x: Array, t: float) -> float:
        # C++ parity: cpp:75-100.
        d = 0.0
        n = self.size()
        # Use the internal fixed kappa if given, otherwise take kappa from x[].
        kappa = self._fixed_kappa if self._fixed_kappa is not None else float(x[n - 1])
        coeff = 0.0

        # ``d += x[i] * std::exp(...)`` is contracted into an FMA by Clang; see
        # the module docstring for the measurement that pins that down.
        if not self._constrain_at_zero:
            for i in range(n - 1):
                d = math.fma(float(x[i]), math.exp(-kappa * (i + 1) * t), d)
        else:
            #  notation:
            #  d(t) = coeff * exp(-kappa*1*t) + x[0]*exp(-kappa*2*t) +
            #         x[1]*exp(-kappa*3*t) + .. + x[7]*exp(-kappa*9*t)
            for i in range(n - 1):
                d = math.fma(float(x[i]), math.exp(-kappa * (i + 2) * t), d)
                coeff += float(x[i])
            coeff = 1.0 - coeff
            d = math.fma(coeff, math.exp(-kappa * t), d)

        return d


class NelsonSiegelFitting(FittingMethod):
    r"""Nelson-Siegel fitting method.

    # C++ parity: ``NelsonSiegelFitting``
    # (nonlinearfittingmethods.hpp:96-114, cpp:103-140).

    Fits ``d(t) = exp(-r t)`` with
    ``r = c0 + (c1 + c2) (1 - e^{-kt}) / (kt) - c2 e^{-kt}``.  See Nelson, C.
    and A. Siegel (1985), NBER Working Paper 1594.

    ``constrainAtZero`` is hard-coded ``true`` (cpp:110), so there is no
    constructor argument for it.
    """

    def __init__(
        self,
        weights: Sequence[float] | Array | None = None,
        optimization_method: OptimizationMethod | None = None,
        l2: Sequence[float] | Array | None = None,
        min_cutoff_time: float = 0.0,
        max_cutoff_time: float = QL_MAX_REAL,
        constraint: Constraint | None = None,
    ) -> None:
        # C++ parity: cpp:103-111.
        super().__init__(
            True, weights, optimization_method, l2, min_cutoff_time, max_cutoff_time, constraint
        )

    def clone(self) -> NelsonSiegelFitting:
        """# C++ parity: cpp:122-125."""
        return _as(NelsonSiegelFitting, self._cloned())

    def size(self) -> int:
        """# C++ parity: cpp:127-129."""
        return 4

    def _discount_function(self, x: Array, t: float) -> float:
        # C++ parity: cpp:131-140. The QL_EPSILON offsets on BOTH kappa and t
        # are what keep t = 0 (and kappa = 0) finite; they also bias the rate
        # very slightly, and reproducing that bias is the point of copying the
        # expression verbatim.
        kappa = float(x[self.size() - 1])
        zero_rate = (
            float(x[0])
            + (float(x[1]) + float(x[2]))
            * (1.0 - math.exp(-kappa * t))
            / ((kappa + QL_EPSILON) * (t + QL_EPSILON))
            - float(x[2]) * math.exp(-kappa * t)
        )
        return math.exp(-zero_rate * t)


class SvenssonFitting(FittingMethod):
    r"""Svensson fitting method.

    # C++ parity: ``SvenssonFitting``
    # (nonlinearfittingmethods.hpp:129-147, cpp:143-181).

    Nelson-Siegel plus a second hump term with its own decay ``kappa_1``.  See
    Svensson, L. (1994), CEPR Discussion Paper 1051.
    """

    def __init__(
        self,
        weights: Sequence[float] | Array | None = None,
        optimization_method: OptimizationMethod | None = None,
        l2: Sequence[float] | Array | None = None,
        min_cutoff_time: float = 0.0,
        max_cutoff_time: float = QL_MAX_REAL,
        constraint: Constraint | None = None,
    ) -> None:
        # C++ parity: cpp:143-150.
        super().__init__(
            True, weights, optimization_method, l2, min_cutoff_time, max_cutoff_time, constraint
        )

    def clone(self) -> SvenssonFitting:
        """# C++ parity: cpp:160-163."""
        return _as(SvenssonFitting, self._cloned())

    def size(self) -> int:
        """# C++ parity: cpp:165-167."""
        return 6

    def _discount_function(self, x: Array, t: float) -> float:
        # C++ parity: cpp:169-181.
        kappa = float(x[self.size() - 2])
        kappa_1 = float(x[self.size() - 1])

        zero_rate = (
            float(x[0])
            + (float(x[1]) + float(x[2]))
            * (1.0 - math.exp(-kappa * t))
            / ((kappa + QL_EPSILON) * (t + QL_EPSILON))
            - float(x[2]) * math.exp(-kappa * t)
            + float(x[3])
            * (
                (1.0 - math.exp(-kappa_1 * t)) / ((kappa_1 + QL_EPSILON) * (t + QL_EPSILON))
                - math.exp(-kappa_1 * t)
            )
        )
        return math.exp(-zero_rate * t)


class CubicBSplinesFitting(FittingMethod):
    r"""Cubic B-splines fitting method.

    # C++ parity: ``CubicBSplinesFitting``
    # (nonlinearfittingmethods.hpp:169-197, cpp:184-266).

    Fits ``d(t) = sum_i c_i N_{i,3}(t)`` over the cubic B-spline basis induced
    by ``knots``.  See McCulloch, J. (1971, 1975).

    "The results are extremely sensitive to the number and location of the knot
    points, and there is no optimal way of selecting them." — James, J. and
    N. Webber, *Interest Rate Modelling*, Wiley 2000, p. 440.

    Note the C++ construction ORDER: ``splines_(3, knots.size() - 5, knots)``
    runs in the member-initialiser list, BEFORE the ``knots.size() >= 8``
    check in the body — so for a short knot vector it is ``BSpline``'s own
    "must have p <= n" that fires first, not the eight-knot message.  That
    order is reproduced here.
    """

    def __init__(
        self,
        knots: Sequence[float],
        constrain_at_zero: bool = True,
        weights: Sequence[float] | Array | None = None,
        optimization_method: OptimizationMethod | None = None,
        l2: Sequence[float] | Array | None = None,
        min_cutoff_time: float = 0.0,
        max_cutoff_time: float = QL_MAX_REAL,
        constraint: Constraint | None = None,
    ) -> None:
        # C++ parity: cpp:184-214.
        super().__init__(
            constrain_at_zero,
            weights,
            optimization_method,
            l2,
            min_cutoff_time,
            max_cutoff_time,
            constraint,
        )
        self._splines: BSpline = BSpline(3, len(knots) - 5, knots)

        qassert.require(len(knots) >= 8, "At least 8 knots are required")
        basis_functions = len(knots) - 4

        if constrain_at_zero:
            self._size: int = basis_functions - 1

            # Note: a small but nonzero N_th basis function at t=0 may lead to
            # an ill conditioned problem
            self._n: int = 1

            qassert.require(
                abs(self._splines(self._n, 0.0)) > QL_EPSILON,
                "N_th cubic B-spline must be nonzero at t=0",
            )
        else:
            self._size = basis_functions
            self._n = 0

    def basis_function(self, i: int, t: float) -> float:
        """Cubic B-spline basis function ``N_{i,3}(t)``.

        # C++ parity: cpp:227-229 — ``Real basisFunction(Integer i, Time t)``.
        """
        return self._splines(i, t)

    def clone(self) -> CubicBSplinesFitting:
        """# C++ parity: cpp:231-234."""
        return _as(CubicBSplinesFitting, self._cloned())

    def size(self) -> int:
        """# C++ parity: cpp:236-238."""
        return self._size

    def _discount_function(self, x: Array, t: float) -> float:
        # C++ parity: cpp:240-266. In the constrained branch basis function
        # N_ is SKIPPED in the parameter vector and its coefficient is solved
        # for so that d(0) == 1 — which is why the loop indexes splines_(i+1)
        # once past N_.
        d = 0.0

        # Every ``d += x[i] * splines_(...)`` here is FMA-contracted by Clang;
        # see the module docstring.
        if not self._constrain_at_zero:
            for i in range(self._size):
                d = math.fma(float(x[i]), self._splines(i, t), d)
        else:
            big_t = 0.0
            total = 0.0
            for i in range(self._size):
                if i < self._n:
                    d = math.fma(float(x[i]), self._splines(i, t), d)
                    total = math.fma(float(x[i]), self._splines(i, big_t), total)
                else:
                    d = math.fma(float(x[i]), self._splines(i + 1, t), d)
                    total = math.fma(float(x[i]), self._splines(i + 1, big_t), total)
            # Two separate C++ statements, so no contraction across them.
            coeff = 1.0 - total
            coeff /= self._splines(self._n, big_t)
            d = math.fma(coeff, self._splines(self._n, t), d)

        return d


class NaturalCubicFitting(FittingMethod):
    """Natural cubic spline fitting method.

    # C++ parity: ``NaturalCubicFitting``
    # (nonlinearfittingmethods.hpp:206-233, cpp:268-338).

    The parameters ARE the nodal discount values ``d(t_i)``; ``d(0)`` is fixed
    at 1.0, so the parameter vector holds the remaining nodes and
    :meth:`size` is ``len(knots) - 1``.

    The constructor appends ``0.0`` to the knot vector, sorts, and de-duplicates
    with a ``1e-14`` tolerance (cpp:279-284) — so a knot list that already
    contains 0 keeps its length and one that does not grows by one, and both
    end up with the same :meth:`size` when they describe the same knot set.
    """

    def __init__(
        self,
        knot_times: Sequence[float],
        weights: Sequence[float] | Array | None = None,
        optimization_method: OptimizationMethod | None = None,
        l2: Sequence[float] | Array | None = None,
        min_cutoff_time: float = 0.0,
        max_cutoff_time: float = QL_MAX_REAL,
        constraint: Constraint | None = None,
    ) -> None:
        # C++ parity: cpp:268-299.
        super().__init__(
            True, weights, optimization_method, l2, min_cutoff_time, max_cutoff_time, constraint
        )
        sorted_times = sorted([*(float(k) for k in knot_times), 0.0])

        # C++ ``std::unique`` with a binary predicate compares each element
        # against the LAST KEPT one, not against its immediate predecessor in
        # the input — which matters for a run of near-duplicates.
        deduped: list[float] = [sorted_times[0]]
        for value in sorted_times[1:]:
            if not abs(value - deduped[-1]) <= 1e-14:
                deduped.append(value)
        self._knot_times: list[float] = deduped

        qassert.require(
            len(self._knot_times) >= 2,
            "NaturalCubicFitting: at least two knot times required",
        )

        n = len(self._knot_times)
        self._size: int = n - 1

        for i in range(n - 1):
            h = self._knot_times[i + 1] - self._knot_times[i]
            qassert.require(
                h > 1e-14,
                "NaturalCubicFitting: knot times must be strictly increasing "
                "(non-zero spacing)",
            )
            qassert.require(math.isfinite(h), "NaturalCubicFitting: non-finite knot spacing")

    def clone(self) -> NaturalCubicFitting:
        """# C++ parity: cpp:311-314."""
        return _as(NaturalCubicFitting, self._cloned())

    def size(self) -> int:
        """# C++ parity: cpp:316-318."""
        return self._size

    def _discount_function(self, x: Array, t: float) -> float:
        # C++ parity: cpp:320-338.
        n = len(self._knot_times)
        expected = self.size()
        qassert.require(
            x.size == expected,
            "NaturalCubicFitting::discountFunction(): parameter size mismatch: "
            f"expected {expected} got {x.size}",
        )

        y = np.empty(n, dtype=np.float64)
        y[0] = 1.0
        for i in range(1, n):
            y[i] = x[i - 1]

        for i in range(n):
            qassert.require(
                math.isfinite(float(y[i])),
                "NaturalCubicFitting::discountFunction(): non-finite nodal value",
            )

        spline = CubicNaturalSpline(np.asarray(self._knot_times, dtype=np.float64), y)
        # C++ calls update() explicitly even though the constructor already ran
        # it; harmless there and here, and kept so the two read alike.
        spline.update()
        # std::clamp — so t outside the knot range evaluates at the end knot
        # rather than extrapolating.
        return spline(min(max(t, self._knot_times[0]), self._knot_times[-1]))


class SimplePolynomialFitting(FittingMethod):
    r"""Simple polynomial fitting method.

    # C++ parity: ``SimplePolynomialFitting``
    # (nonlinearfittingmethods.hpp:245-268, cpp:341-387).

    Fits ``d(t) = sum_i c_i t^i``.  Crude, but fast and robust.

    The basis is written as ``BernsteinPolynomial::get(i, i, t)``, and since
    ``B_n^n(t) = C(n,n) t^n (1-t)^0 = t^n`` that IS the monomial basis — the
    Bernstein call is how C++ spells it and is reproduced literally, because
    for ``t > 1`` the two agree only because the ``(1-t)^0`` factor is exactly
    1.
    """

    def __init__(
        self,
        degree: int,
        constrain_at_zero: bool = True,
        weights: Sequence[float] | Array | None = None,
        optimization_method: OptimizationMethod | None = None,
        l2: Sequence[float] | Array | None = None,
        min_cutoff_time: float = 0.0,
        max_cutoff_time: float = QL_MAX_REAL,
        constraint: Constraint | None = None,
    ) -> None:
        # C++ parity: cpp:341-352.
        super().__init__(
            constrain_at_zero,
            weights,
            optimization_method,
            l2,
            min_cutoff_time,
            max_cutoff_time,
            constraint,
        )
        self._size: int = degree if constrain_at_zero else degree + 1

    def clone(self) -> SimplePolynomialFitting:
        """# C++ parity: cpp:365-368."""
        return _as(SimplePolynomialFitting, self._cloned())

    def size(self) -> int:
        """# C++ parity: cpp:370-372."""
        return self._size

    def _discount_function(self, x: Array, t: float) -> float:
        # C++ parity: cpp:374-387.
        d = 0.0

        # FMA-contracted accumulation; see the module docstring.
        if not self._constrain_at_zero:
            for i in range(self._size):
                d = math.fma(float(x[i]), BernsteinPolynomial.get(i, i, t), d)
        else:
            d = 1.0
            for i in range(self._size):
                d = math.fma(float(x[i]), BernsteinPolynomial.get(i + 1, i + 1, t), d)
        return d


class SpreadFittingMethod(FittingMethod):
    """Fit a spread curve on top of an existing discount curve.

    # C++ parity: ``SpreadFittingMethod``
    # (nonlinearfittingmethods.hpp:275-295, cpp:389-429).

    Delegates the parametric shape to ``method`` and multiplies its discount
    factor by the discounting curve's — with EXTRAPOLATION ENABLED on that
    call (cpp:415), so the spread curve can run past the base curve's max date.

    ``init()`` rebases: when the two curves have different reference dates,
    every discount factor is divided by the base curve's discount to this
    curve's reference date (cpp:418-429).
    """

    def __init__(
        self,
        method: FittingMethod,
        discount_curve: YieldTermStructure,
        min_cutoff_time: float = 0.0,
        max_cutoff_time: float = QL_MAX_REAL,
    ) -> None:
        # C++ parity: cpp:389-403 — the base-class arguments are READ OFF the
        # delegate, so constrainAtZero / weights / optimizationMethod / l2 are
        # whatever the wrapped method carries. C++ reads each behind a
        # ``method != nullptr ? ... : <default>`` because the parameter is a
        # shared_ptr that may be empty, then rejects the empty case outright
        # (cpp:401-402) — so the ternaries are dead and only these two checks
        # matter. Both are kept: a caller ignoring the annotations (which C++
        # has no equivalent of) still gets the C++ message rather than an
        # AttributeError three frames down.
        qassert.require(
            method is not None,  # pyright: ignore[reportUnnecessaryComparison]
            "Fitting method is empty",
        )
        qassert.require(
            discount_curve is not None,  # pyright: ignore[reportUnnecessaryComparison]
            "Discounting curve cannot be empty",
        )
        super().__init__(
            method.constrain_at_zero(),
            method.weights(),
            method.optimization_method(),
            method.l2(),
            min_cutoff_time,
            max_cutoff_time,
        )
        self._method: FittingMethod = method
        self._discounting_curve: YieldTermStructure = discount_curve
        # C++ leaves rebase_ uninitialised until init() (hpp:292); 1.0 is the
        # value init() assigns when the reference dates agree.
        self._rebase: float = 1.0

    def clone(self) -> SpreadFittingMethod:
        """# C++ parity: cpp:405-408.

        The wrapped ``method_`` is a ``shared_ptr`` in C++, so the copy SHARES
        it; :meth:`FittingMethod._cloned` shares it too.
        """
        return _as(SpreadFittingMethod, self._cloned())

    def size(self) -> int:
        """# C++ parity: cpp:410-412."""
        return self._method.size()

    def _init(self) -> None:
        # C++ parity: cpp:418-429.
        curve = self._require_curve()
        # In case the discount curve has a different reference date, discount
        # to this curve's reference date
        if curve.reference_date() != self._discounting_curve.reference_date():
            self._rebase = self._discounting_curve.discount(curve.reference_date())
        else:
            self._rebase = 1.0
        # Call regular init
        super()._init()

    def _discount_function(self, x: Array, t: float) -> float:
        # C++ parity: cpp:414-416. Note ``discount(t, true)`` — extrapolation
        # on the BASE curve is forced, regardless of its own flag.
        return self._method.discount(x, t) * self._discounting_curve.discount(t, True) / self._rebase


def _as[T: FittingMethod](cls: type[T], obj: FittingMethod) -> T:
    """Narrow :meth:`FittingMethod._cloned`'s return type back to ``cls``.

    ``_cloned`` goes through ``copy.copy``, which preserves the runtime class
    but is typed as returning the base. Every ``clone()`` above is C++'s
    ``std::make_unique<Derived>(*this)``, i.e. same class by construction.
    """
    assert isinstance(obj, cls), f"clone() produced {type(obj).__name__}, expected {cls.__name__}"
    return obj


__all__ = [
    "CubicBSplinesFitting",
    "ExponentialSplinesFitting",
    "NaturalCubicFitting",
    "NelsonSiegelFitting",
    "SimplePolynomialFitting",
    "SpreadFittingMethod",
    "SvenssonFitting",
]
