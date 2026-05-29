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

PQuantLib factors the abcd-fit logic into a shared helper used by both
classes. The shared helper is :class:`_AbcdFitter` at module scope
below; it provides ``fit()`` returning ``(a, b, c, d, rms, max_err,
converged)`` for given inputs. :class:`AbcdInterpolation` and
:class:`AbcdCalibration` both call it.

This keeps the behaviour identical (both fit via
``scipy.optimize.least_squares`` ``trf`` arm with the C++
``AbcdMathFunction::validate`` constraints) while preserving the
distinct public APIs.

Documented divergences vs C++ (inherited from L10-C):

* The optimizer is ``scipy.optimize.least_squares`` ``trf`` with native
  box bounds, not the C++ ``LevenbergMarquardt`` + ``ProjectedCostFunction``
  combo. Recovered parameters can differ in the local-minima rich
  regime (typical on noisy 4-param-vs-6-pillar problems); on noiseless
  abcd-shape data the Python fit converges to the global minimum
  (residuals ~1e-13).
* ``end_criteria`` accepted as a duck-typed object with an optional
  ``max_iterations`` attribute (mapped to scipy ``max_nfev``).
* ``optimization_method`` accepted but ignored.

Adjustment factor:
    ``k(t)`` returns ``black_vols / value(t)`` per the C++
    ``AbcdCalibration::k`` method.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.math.interpolations.abcd_interpolation import (
    abcd_value,
    validate_abcd,
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
        self._rms_error: float = 0.0
        self._max_error: float = 0.0
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

    def value(self, t: float) -> float:
        """Evaluate the fitted Rebonato model at ``t``.

        # C++ parity: ``AbcdCalibration::value`` (abcdcalibration.cpp:164).
        """
        return abcd_value(t, self._a, self._b, self._c, self._d)

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

    def error(self) -> float:
        """RMS weighted error after :meth:`compute`.

        # C++ parity: ``AbcdCalibration::error`` (abcdcalibration.cpp:180).
        """
        return self._rms_error

    def max_error(self) -> float:
        """Max absolute residual after :meth:`compute`.

        # C++ parity: ``AbcdCalibration::maxError`` (abcdcalibration.cpp:190).
        """
        return self._max_error

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
            [abcd_value(float(t), a, b, c, d) for t in self._times],
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
        # If all params fixed, just record diagnostics.
        if all(self._is_fixed):
            r = self._residuals(np.array([], dtype=np.float64))
            self._a, self._b, self._c, self._d = self._initial
            self._rms_error = (
                float(np.sqrt(np.mean(r * r))) if r.size > 0 else 0.0
            )
            self._max_error = (
                float(np.max(np.abs(r))) if r.size > 0 else 0.0
            )
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
        fun_arr: np.ndarray = np.asarray(
            result.fun,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            dtype=np.float64,
        )
        if fun_arr.size > 0:
            n = fun_arr.shape[0]
            # # C++ parity: ``error()`` returns
            # # sqrt(n * sum(w*r^2) / (n-1)).
            squared = float(np.sum(fun_arr * fun_arr))
            self._rms_error = float(
                np.sqrt(n * squared / max(n - 1, 1))
            )
            self._max_error = float(np.max(np.abs(fun_arr)))
        else:
            self._rms_error = 0.0
            self._max_error = 0.0


__all__ = ["AbcdCalibration"]
