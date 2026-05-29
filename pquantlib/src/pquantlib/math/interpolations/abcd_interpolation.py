"""AbcdInterpolation — Rebonato (a, b, c, d) parametric volatility curve.

# C++ parity: ql/math/interpolations/abcdinterpolation.hpp
#             + ql/termstructures/volatility/abcd.{hpp,cpp}
#             + ql/termstructures/volatility/abcdcalibration.{hpp,cpp}
#             + ql/math/abcdmathfunction.{hpp,cpp} (v1.42.1).

The "abcd" functional form (Rebonato, 2003) is

.. math::

   f(t) = (a + b\\,t) e^{-c\\,t} + d

for non-negative ``t``. It is widely used as the instantaneous
volatility parameterization for forward LIBOR rates in the Joshi-Rebonato
swap-market-model literature. The four parameters control:

- ``a + d`` is the short-term volatility (at t=0).
- ``d`` is the long-term volatility (at t=infinity).
- ``c`` controls the rate of decay from short- to long-term.
- ``b`` controls the location of the volatility peak. The peak is at
  ``t* = 1/c - a/b`` (if ``b > 0`` and ``a < c*b``), with peak value
  ``f(t*)``.

The validity constraints (matching C++
``AbcdMathFunction::validate``):

- ``c >= 0``  (decay rate non-negative)
- ``d >= 0``  (long-term vol non-negative)
- ``a + d >= 0``  (short-term vol non-negative)
- ``b`` can be any sign

The fit minimises the residual ``vol_market - f(t_i)`` via
``scipy.optimize.least_squares`` with a trust-region-reflective method
(precedent: L9-C ``SabrInterpolation``).

Documented divergences from C++:

* **Optimizer.** C++ uses a bespoke ``AbcdCalibration::compute()`` that
  layers ``LevenbergMarquardt`` + ``BoundConstraint`` + per-iteration
  weighting; we use ``scipy.optimize.least_squares(method='trf')`` with
  natively-enforced box bounds. Cross-validation against the L10-C
  C++ probe is at LOOSE tier on the recovered (a, b, c, d) parameters.
* **Vega weighting.** The C++ ``vegaWeighted=True`` branch weights
  each residual by the local Black vega; ports of that hook are
  identical to the L9-C SABR implementation (we re-use the same
  approximation).
* **end_criteria / optimization_method overrides.** C++ accepts
  ``ext::shared_ptr<EndCriteria>`` and ``ext::shared_ptr<OptimizationMethod>``
  pass-throughs; we accept them as opaque kwargs but only use ``end_criteria``
  to set ``max_nfev``. The optimization method itself is locked to the
  scipy TRF arm.

Cross-validated against the C++ probe in
``migration-harness/cpp/probes/cluster_l10c/probe.cpp`` — recovered
(a, b, c, d) on a synthetic abcd-shaped vol curve match the C++ probe
at LOOSE tier.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.interpolation import Interpolation

# Defaults per C++ ``AbcdInterpolation`` ctor (abcdinterpolation.hpp:163-166).
_A_DEFAULT: Final[float] = -0.06
_B_DEFAULT: Final[float] = 0.17
_C_DEFAULT: Final[float] = 0.54
_D_DEFAULT: Final[float] = 0.17


def abcd_value(t: float, a: float, b: float, c: float, d: float) -> float:
    """Evaluate the abcd functional form at ``t``.

    # C++ parity: ``AbcdMathFunction::operator()`` (abcdmathfunction.hpp:104-107).

    Returns 0 for negative ``t`` to mirror the C++ guard.
    """
    if t < 0.0:
        return 0.0
    return (a + b * t) * float(np.exp(-c * t)) + d


def validate_abcd(a: float, b: float, c: float, d: float) -> None:
    """Validate (a, b, c, d) per C++ ``AbcdMathFunction::validate``.

    # C++ parity: abcdmathfunction.cpp ``AbcdMathFunction::validate``.

    - ``c >= 0``
    - ``d >= 0``
    - ``a + d >= 0``
    """
    qassert.require(c >= 0.0, f"c parameter must be non-negative: {c} not allowed")
    qassert.require(d >= 0.0, f"d parameter must be non-negative: {d} not allowed")
    qassert.require(
        a + d >= 0.0,
        f"a+d must be non-negative: a={a}, d={d}, a+d={a + d} not allowed",
    )


class AbcdInterpolation(Interpolation):
    """Rebonato-style abcd parametric volatility curve.

    # C++ parity: ``AbcdInterpolation`` (abcdinterpolation.hpp:156-207).

    Fits the four parameters (a, b, c, d) of

    .. math::

       \\sigma(t) = (a + b\\,t)\\,e^{-c\\,t} + d

    to a curve of times and volatilities via
    ``scipy.optimize.least_squares``.

    Args:
        x_seq: times (must be non-negative, monotonically ascending).
        y_seq: market volatilities at each time.
        a, b, c, d: initial parameter values. Defaults match C++.
        a_is_fixed, b_is_fixed, c_is_fixed, d_is_fixed: pin the
            parameter at its initial value if True.
        vega_weighted: if True, residuals are weighted by an
            approximate Black vega (matches C++ semantics; identical
            approximation to L9-C SabrInterpolation).
        end_criteria: opaque C++ pass-through. If non-None and its
            ``max_iterations`` attribute is set, that is used as
            ``max_nfev`` for the scipy solver. Otherwise default 1000.
        optimization_method: opaque C++ pass-through. Ignored — the
            scipy TRF arm is always used.
    """

    def __init__(
        self,
        x_seq: Array,
        y_seq: Array,
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
        super().__init__(x_seq, y_seq, required_points=2)
        qassert.require(
            bool(np.all(self._xs >= 0.0)),
            "AbcdInterpolation requires non-negative times",
        )
        # C++ defers validation to ``AbcdMathFunction::validate`` after
        # ``compute()``; we validate the initial guess too (the solver
        # will keep the iterate feasible via the box bounds).
        validate_abcd(a, b, c, d)
        self._initial: list[float] = [a, b, c, d]
        self._is_fixed: list[bool] = [
            a_is_fixed,
            b_is_fixed,
            c_is_fixed,
            d_is_fixed,
        ]
        self._vega_weighted: bool = vega_weighted
        # Read max iterations from end_criteria if present (C++ allows
        # an EndCriteria object). We accept ``max_iterations`` as a
        # simple attribute.
        max_nfev = 1000
        if end_criteria is not None and hasattr(end_criteria, "max_iterations"):
            max_nfev = int(end_criteria.max_iterations)
        self._max_nfev: int = max_nfev
        # ``optimization_method`` is accepted but ignored — see docstring.
        _ = optimization_method
        # Fitted state — populated by ``update``.
        self._a: float = a
        self._b: float = b
        self._c: float = c
        self._d: float = d
        self._rms_error: float = 0.0
        self._max_error: float = 0.0
        self._converged: bool = False
        self.update()

    # --- fit -------------------------------------------------------------

    def _vega_weights(self) -> np.ndarray:
        """Approximate vega weights per knot (same scheme as SabrInterpolation).

        We use a flat sqrt-time weighting normalised to 1 — matches the
        L9-C SABR fitter's documented divergence.
        """
        n = self._xs.shape[0]
        w = np.where(self._xs > 0.0, np.sqrt(self._xs), 1.0)
        total = float(np.sum(w))
        w = w / total if total > 0.0 else np.ones(n, dtype=np.float64) / n
        return np.sqrt(w)

    def _residuals(self, free_params: np.ndarray) -> np.ndarray:
        params = list(self._initial)
        j = 0
        for i, fixed in enumerate(self._is_fixed):
            if not fixed:
                params[i] = float(free_params[j])
                j += 1
        a, b, c, d = params
        # Clamp to feasible region; the bounds in least_squares keep us
        # here, but transient solver iterates may visit edges.
        c = max(c, 0.0)
        d = max(d, 0.0)
        if a + d < 0.0:
            a = -d
        model = np.array(
            [abcd_value(float(t), a, b, c, d) for t in self._xs],
            dtype=np.float64,
        )
        r = model - self._ys
        if self._vega_weighted:
            r = r * self._vega_weights()
        return r

    def update(self) -> None:
        """Fit (a, b, c, d) to the input curve."""
        free_initial: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        # a: unbounded (a + d >= 0 enforced softly in _residuals).
        if not self._is_fixed[0]:
            free_initial.append(self._initial[0])
            lower.append(-np.inf)
            upper.append(np.inf)
        # b: unbounded.
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
        if not free_initial:
            r = self._residuals(np.array([], dtype=np.float64))
            self._update_diagnostics(r)
            self._converged = True
            self._a, self._b, self._c, self._d = self._initial
            return
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
        self._update_diagnostics(
            np.asarray(
                result.fun,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                dtype=np.float64,
            )
        )

    def _update_diagnostics(self, residuals: np.ndarray) -> None:
        if residuals.size == 0:
            self._rms_error = 0.0
            self._max_error = 0.0
            return
        self._rms_error = float(np.sqrt(np.mean(residuals * residuals)))
        self._max_error = float(np.max(np.abs(residuals)))

    # --- public API ------------------------------------------------------

    def a(self) -> float:
        return self._a

    def b(self) -> float:
        return self._b

    def c(self) -> float:
        return self._c

    def d(self) -> float:
        return self._d

    def rms_error(self) -> float:
        return self._rms_error

    def max_error(self) -> float:
        return self._max_error

    def converged(self) -> bool:
        return self._converged

    def _value(self, x: float) -> float:
        return abcd_value(x, self._a, self._b, self._c, self._d)


