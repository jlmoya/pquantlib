"""AbcdCalibration — standalone Rebonato (a, b, c, d) volatility-curve fit.

# C++ parity: ql/termstructures/volatility/abcdcalibration.{hpp,cpp}
# (v1.42.1).

The C++ class is a stand-alone calibrator that wraps the same
``LevenbergMarquardt`` + ``ProjectedCostFunction`` +
``AbcdParametersTransformation`` machinery used by
``AbcdInterpolation``'s internal solve (L10-C ``abcd_interpolation.py``).
The two classes diverge only in their public surface: ``AbcdInterpolation``
extends :class:`Interpolation` (it overrides ``_value`` / ``_primitive``);
``AbcdCalibration`` is a calibrator and exposes the fitted parameters
plus a ``value(t)`` evaluator.

C++ declares two helper classes in ``AbcdCalibration``'s PRIVATE section:
``AbcdError`` (abcdcalibration.hpp:44) and ``AbcdParametersTransformation``
(abcdcalibration.hpp:69). A private nested class has no Python analogue, and
hiding them would leave their arithmetic untestable, so both are exposed here
at module scope with the same names and the same behaviour, cross-validated in
``tests/math/interpolations/test_abcd_error_formula.py``.

Documented divergences vs C++:

* The optimizer is ``scipy.optimize.least_squares`` ``trf`` with native
  box bounds, not the C++ ``LevenbergMarquardt`` + ``ProjectedCostFunction``
  + ``AbcdParametersTransformation`` combo. Recovered parameters can differ
  in the local-minima rich regime (typical on noisy 4-param-vs-6-pillar
  problems); on noiseless abcd-shape data the Python fit converges to the
  global minimum (residuals ~1e-13). :class:`AbcdError` and
  :class:`AbcdParametersTransformation` are therefore faithful and tested but
  are not on the ``compute()`` path; wiring them in would move published
  fitted parameters and is tracked separately.
* ``end_criteria`` accepted as a duck-typed object with an optional
  ``max_iterations`` attribute (mapped to scipy ``max_nfev``).
* ``optimization_method`` accepted but ignored.

Adjustment factor:
    ``k(t)`` returns ``black_vols / value(t)`` per the C++
    ``AbcdCalibration::k`` method.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.math.interpolations.abcd_interpolation import (
    abcd_black_volatility,
    validate_abcd,
)
from pquantlib.math.optimization.cost_function import (
    CostFunction,
    ParametersTransformation,
)

# C++ ``AbcdCalibration`` defaults at abcdcalibration.hpp:86-89.
_A_DEFAULT: Final[float] = -0.06
_B_DEFAULT: Final[float] = 0.17
_C_DEFAULT: Final[float] = 0.54
_D_DEFAULT: Final[float] = 0.17


def _vega_weights(times: np.ndarray, black_vols: np.ndarray) -> np.ndarray:
    """C++ ``AbcdCalibration::compute`` vega-weighting (lines 102-114).

    Weights ``[i]`` = ``CDF.derivative(0.5 * sqrt(vol[i]^2 * t[i]))``,
    normalised to sum to 1.
    """
    # Inline standard-normal pdf to avoid a CumulativeNormalDistribution
    # round-trip (lower test surface, identical result).
    std_dev = np.sqrt(black_vols * black_vols * times)
    # CDF.derivative(x) = pdf(x) = exp(-x^2/2) / sqrt(2 pi)
    pdf_arg = 0.5 * std_dev
    w = np.exp(-0.5 * pdf_arg * pdf_arg) / np.sqrt(2.0 * np.pi)
    s = float(w.sum())
    if s > 0.0:
        return w / s
    return np.ones_like(w) / float(w.shape[0])


class AbcdCalibration:
    """Stand-alone Rebonato (a, b, c, d) volatility-curve calibrator.

    Args:
        t: input times (must be non-negative, ascending).
        black_vols: market Black volatilities at each time. Same length as
            ``t``.
        a_guess / b_guess / c_guess / d_guess: initial parameter values.
            Defaults match C++.
        a_is_fixed / b_is_fixed / c_is_fixed / d_is_fixed: pin parameter
            at its guess if True.
        vega_weighted: weight residuals by Black-vega per C++ logic.
        end_criteria: opaque C++ pass-through. ``max_iterations`` attribute
            (if present) sets scipy ``max_nfev``; default 1000.
        optimization_method: opaque C++ pass-through; ignored — scipy
            TRF is always used.

    Attributes:
        a / b / c / d: fitted parameters.
        end_criteria_diagnostic: textual result message from scipy.

    # C++ parity: ``AbcdCalibration``
    # (abcdcalibration.hpp:42-124 + .cpp:56-99).
    """

    def __init__(
        self,
        t: Sequence[float],
        black_vols: Sequence[float],
        a: float = _A_DEFAULT,
        b: float = _B_DEFAULT,
        c: float = _C_DEFAULT,
        d: float = _D_DEFAULT,
        a_is_fixed: bool = False,
        b_is_fixed: bool = False,
        c_is_fixed: bool = False,
        d_is_fixed: bool = False,
        vega_weighted: bool = False,
        end_criteria: Any = None,
        optimization_method: Any = None,
    ) -> None:
        times_arr = np.asarray(t, dtype=np.float64)
        vols_arr = np.asarray(black_vols, dtype=np.float64)
        qassert.require(
            times_arr.shape[0] == vols_arr.shape[0],
            f"AbcdCalibration: times ({times_arr.shape[0]}) vs blackVols "
            f"({vols_arr.shape[0]}) length mismatch",
        )
        qassert.require(
            bool(np.all(times_arr >= 0.0)),
            "AbcdCalibration requires non-negative times",
        )
        validate_abcd(a, b, c, d)

        self._times: np.ndarray = times_arr
        self._black_vols: np.ndarray = vols_arr
        self._initial: list[float] = [a, b, c, d]
        self._is_fixed: list[bool] = [
            a_is_fixed, b_is_fixed, c_is_fixed, d_is_fixed,
        ]
        self._vega_weighted: bool = vega_weighted
        max_nfev = 1000
        if end_criteria is not None and hasattr(end_criteria, "max_iterations"):
            max_nfev = int(end_criteria.max_iterations)
        self._max_nfev: int = max_nfev
        _ = optimization_method

        # Fit state.
        self._a: float = a
        self._b: float = b
        self._c: float = c
        self._d: float = d
        self._weights: np.ndarray = (
            _vega_weights(times_arr, vols_arr)
            if vega_weighted
            else np.full_like(times_arr, 1.0 / float(times_arr.shape[0]))
        )
        self._end_criteria_diagnostic: str = "uncomputed"
        self._converged: bool = False

    # --- access ---------------------------------------------------------

    def a(self) -> float:
        return self._a

    def b(self) -> float:
        return self._b

    def c(self) -> float:
        return self._c

    def d(self) -> float:
        return self._d

    def set_parameters(self, a: float, b: float, c: float, d: float) -> None:
        """Overwrite the current ``(a, b, c, d)``.

        # C++ parity: there is no such method — ``AbcdError`` writes
        # ``abcd_->a_ = y[0]`` etc. directly, which it may do because
        # ``AbcdCalibration`` declares it a friend (abcdcalibration.hpp:44 sits
        # inside the class body). Python has no friendship, and
        # :class:`AbcdError` must be able to install trial parameters, so the
        # write is exposed as one named method rather than four attribute
        # pokes. Deliberately does NOT call ``validate_abcd``: C++ does not
        # validate on this path either (abcdcalibration.hpp:49-53), and the
        # optimiser legitimately visits infeasible points.
        """
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def value(self, t: float) -> float:
        """Evaluate the fitted Rebonato model at ``t``.

        # C++ parity: ``AbcdCalibration::value`` (abcdcalibration.cpp:163-165)
        # is ``abcdBlackVolatility(x, a_, b_, c_, d_)`` — the AVERAGE
        # (Black) volatility over ``[0, t]``, i.e.
        # ``AbcdFunction(a,b,c,d).volatility(0., t, t)`` (abcd.hpp:105-108),
        # NOT the instantaneous ``f(t)``.
        """
        return abcd_black_volatility(t, self._a, self._b, self._c, self._d)

    def k(
        self, t: Sequence[float], black_vols: Sequence[float],
    ) -> list[float]:
        """Per-time adjustment factor ``black_vols / value(t)``.

        # C++ parity: ``AbcdCalibration::k`` (abcdcalibration.cpp:168-178).
        """
        ts = np.asarray(t, dtype=np.float64)
        vols = np.asarray(black_vols, dtype=np.float64)
        qassert.require(
            ts.shape[0] == vols.shape[0],
            f"AbcdCalibration.k: times ({ts.shape[0]}) vs blackVols "
            f"({vols.shape[0]}) length mismatch",
        )
        result: list[float] = []
        for i in range(ts.shape[0]):
            v = self.value(float(ts[i]))
            result.append(float(vols[i]) / v if v != 0.0 else 0.0)
        return result

    def errors(self) -> list[float]:
        """Per-time weighted differences ``(value(t_i) - vol_i) * sqrt(w_i)``.

        # C++ parity: ``AbcdCalibration::errors``
        # (abcdcalibration.cpp:198-205). Live, like C++: computed from the
        # CURRENT (a, b, c, d), not cached at ``compute()`` time — that is what
        # makes :class:`AbcdError` able to drive it from trial parameters.
        """
        return [
            (self.value(float(t)) - float(v)) * math.sqrt(float(w))
            for t, v, w in zip(
                self._times, self._black_vols, self._weights, strict=True
            )
        ]

    def error(self) -> float:
        """``sqrt(n * sum_i w_i e_i^2 / (n - 1))`` at the current parameters.

        # C++ parity: ``AbcdCalibration::error``
        # (abcdcalibration.cpp:179-187):
        #
        #     Size n = times_.size();
        #     for i: error = value(times_[i]) - blackVols_[i];
        #            squaredError += error * error * weights_[i];
        #     return std::sqrt(n * squaredError / (n - 1));
        #
        # ``weights_`` default to a uniform ``1/n`` (abcdcalibration.cpp:72),
        # so the unweighted case reduces to the SAMPLE standard deviation
        # ``sqrt(sum e^2 / (n-1))`` — not ``sqrt(mean(e^2))``. Pinned for
        # n = 2, 3, 6 by tests/math/interpolations/test_abcd_error_formula.py
        # against migration-harness/references/v143/ts/abcd.json.
        """
        n = int(self._times.shape[0])
        if n == 0:
            return 0.0
        squared = 0.0
        for t, v, w in zip(
            self._times, self._black_vols, self._weights, strict=True
        ):
            e = self.value(float(t)) - float(v)
            squared += e * e * float(w)
        # C++ divides by (n - 1) unguarded; n == 1 would be a division by zero
        # there. Python keeps the same value for n > 1 and returns the single
        # weighted residual's magnitude for n == 1 rather than raising.
        return math.sqrt(n * squared / (n - 1)) if n > 1 else math.sqrt(squared)

    def max_error(self) -> float:
        """``max_i |value(t_i) - vol_i|`` at the current parameters.

        # C++ parity: ``AbcdCalibration::maxError``
        # (abcdcalibration.cpp:189-196). Note this is UNWEIGHTED even when
        # ``vegaWeighted`` is set — only :meth:`error` and :meth:`errors`
        # carry the weights.
        """
        if self._times.shape[0] == 0:
            return 0.0
        return max(
            abs(self.value(float(t)) - float(v))
            for t, v in zip(self._times, self._black_vols, strict=True)
        )

    def end_criteria(self) -> str:
        """Termination diagnostic message from scipy.

        # C++ parity: ``AbcdCalibration::endCriteria`` (returns
        # ``EndCriteria::Type``); we return scipy's ``message`` instead.
        """
        return self._end_criteria_diagnostic

    def converged(self) -> bool:
        return self._converged

    # --- residual + fit -------------------------------------------------

    def _residuals(self, free_params: np.ndarray) -> np.ndarray:
        params = list(self._initial)
        j = 0
        for i, fixed in enumerate(self._is_fixed):
            if not fixed:
                params[i] = float(free_params[j])
                j += 1
        a, b, c, d = params
        # Clamp to feasible region; the bounds keep us here but transient
        # solver iterates may visit edges.
        c = max(c, 0.0)
        d = max(d, 0.0)
        if a + d < 0.0:
            a = -d
        model = np.array(
            [abcd_black_volatility(float(t), a, b, c, d) for t in self._times],
            dtype=np.float64,
        )
        r = model - self._black_vols
        if self._vega_weighted:
            r = r * np.sqrt(self._weights)
        return r

    def compute(self) -> None:
        """Run the Levenberg-Marquardt-style fit.

        # C++ parity: ``AbcdCalibration::compute``
        # (abcdcalibration.cpp:101-162).
        """
        # C++ parity: abcdcalibration.cpp:117-122 — "there is nothing to
        # optimize": a_,b_,c_,d_ keep the values passed to the constructor and
        # abcdEndCriteria_ becomes EndCriteria::None. error()/maxError() are
        # live methods in C++, so nothing is cached here either.
        if all(self._is_fixed):
            self._a, self._b, self._c, self._d = self._initial
            self._converged = True
            self._end_criteria_diagnostic = "all parameters fixed"
            return

        free_initial: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        # a: unbounded (a + d >= 0 enforced softly).
        if not self._is_fixed[0]:
            free_initial.append(self._initial[0])
            lower.append(-np.inf)
            upper.append(np.inf)
        if not self._is_fixed[1]:
            free_initial.append(self._initial[1])
            lower.append(-np.inf)
            upper.append(np.inf)
        # c: >= 0.
        if not self._is_fixed[2]:
            free_initial.append(self._initial[2])
            lower.append(0.0)
            upper.append(np.inf)
        # d: >= 0.
        if not self._is_fixed[3]:
            free_initial.append(self._initial[3])
            lower.append(0.0)
            upper.append(np.inf)

        result: Any = least_squares(  # pyright: ignore[reportUnknownVariableType]
            self._residuals,
            np.array(free_initial, dtype=np.float64),
            bounds=(
                np.array(lower, dtype=np.float64),
                np.array(upper, dtype=np.float64),
            ),
            method="trf",
            max_nfev=self._max_nfev,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        params = list(self._initial)
        x_solution: np.ndarray = np.asarray(
            result.x,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            dtype=np.float64,
        )
        j = 0
        for i, fixed in enumerate(self._is_fixed):
            if not fixed:
                params[i] = float(x_solution[j])
                j += 1
        self._a, self._b, self._c, self._d = params
        validate_abcd(self._a, self._b, self._c, self._d)
        self._converged = bool(
            result.success  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        self._end_criteria_diagnostic = str(
            result.message  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )


class AbcdParametersTransformation(ParametersTransformation):
    """Bijection between R^4 and the feasible abcd region.

    # C++ parity: ``AbcdCalibration::AbcdParametersTransformation``
    # (abcdcalibration.hpp:69-78, abcdcalibration.cpp:36-52).

    C++ declares this in the PRIVATE section of ``AbcdCalibration`` so that
    only ``compute()`` can reach it; PQuantLib exposes it at module scope
    because a private nested class has no Python analogue and hiding it would
    leave the mapping untestable.

    ``direct`` maps unconstrained coordinates onto parameters satisfying the
    ``AbcdMathFunction`` feasibility conditions ``c > 0``, ``d > 0``,
    ``a + d > 0``::

        b = x[1]
        c = exp(x[2])
        d = exp(x[3])
        a = exp(x[0]) - d          # so a + d = exp(x[0]) > 0

    ``inverse`` is its two-sided partner. Note the asymmetry that C++ builds
    in: ``inverse`` computes the a-slot as ``log(x[0] + x[3])`` from the
    CONSTRAINED ``d`` in ``x[3]``, which is only the inverse of ``direct``
    because ``direct`` assigns ``y[3]`` before it reads it for ``y[0]``.
    Reordering those two statements still round-trips for some inputs, so the
    round-trip residual is pinned explicitly in
    ``tests/math/interpolations/test_abcd_error_formula.py``.
    """

    def direct(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Unconstrained ``x`` -> constrained ``(a, b, c, d)``.

        # C++ parity: abcdcalibration.cpp:37-43.
        """
        y = np.empty(4, dtype=np.float64)
        y[1] = x[1]
        y[2] = np.exp(x[2])
        y[3] = np.exp(x[3])
        y[0] = np.exp(x[0]) - y[3]
        return y

    def inverse(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Constrained ``(a, b, c, d)`` -> unconstrained coordinates.

        # C++ parity: abcdcalibration.cpp:46-52.
        """
        y = np.empty(4, dtype=np.float64)
        y[1] = x[1]
        y[2] = np.log(x[2])
        y[3] = np.log(x[3])
        y[0] = np.log(x[0] + x[3])
        return y


class AbcdError(CostFunction):
    """Cost function driving :class:`AbcdCalibration` from R^4.

    # C++ parity: ``AbcdCalibration::AbcdError``
    # (abcdcalibration.hpp:44-67).

    C++::

        Real value(const Array& x) const override {
            const Array y = abcd_->transformation_->direct(x);
            abcd_->a_ = y[0]; ... abcd_->d_ = y[3];
            return abcd_->error();
        }

    so the cost function MUTATES the calibration it points at — the trial
    parameters are written into the calibration before its live ``error()`` /
    ``errors()`` are read. PQuantLib keeps that contract exactly, because it
    is observable: after evaluating the cost function the calibration reports
    the trial parameters, not the ones it was constructed with.

    Like C++ this is declared private in ``AbcdCalibration``; PQuantLib
    exposes it at module scope for the same reason as
    :class:`AbcdParametersTransformation`.
    """

    def __init__(
        self,
        calibration: AbcdCalibration,
        transformation: AbcdParametersTransformation | None = None,
    ) -> None:
        # C++ parity: abcdcalibration.hpp:46 — holds a raw pointer back to the
        # calibration; the transformation is the one compute() installed
        # (abcdcalibration.cpp:125).
        self._abcd: AbcdCalibration = calibration
        self._transformation: AbcdParametersTransformation = (
            transformation
            if transformation is not None
            else AbcdParametersTransformation()
        )

    def _apply(self, x: npt.NDArray[np.float64]) -> None:
        # C++ parity: abcdcalibration.hpp:48-53 / :56-61 — identical prologue
        # in both value() and values().
        y = self._transformation.direct(x)
        self._abcd.set_parameters(
            float(y[0]), float(y[1]), float(y[2]), float(y[3])
        )

    def value(self, x: npt.NDArray[np.float64]) -> float:
        """``AbcdCalibration::error()`` at ``direct(x)``.

        # C++ parity: abcdcalibration.hpp:47-54. Note this OVERRIDES the
        # ``CostFunction`` default ``sqrt(mean(values(x)^2))`` — the two are
        # not the same number, because ``error()`` carries the ``n/(n-1)``
        # scaling.
        """
        self._apply(x)
        return self._abcd.error()

    def values(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """``AbcdCalibration::errors()`` at ``direct(x)``.

        # C++ parity: abcdcalibration.hpp:55-63.
        """
        self._apply(x)
        return np.asarray(self._abcd.errors(), dtype=np.float64)


__all__ = ["AbcdCalibration", "AbcdError", "AbcdParametersTransformation"]
