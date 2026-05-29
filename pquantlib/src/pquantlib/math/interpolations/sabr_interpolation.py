"""SabrInterpolation — fits SABR (alpha, beta, nu, rho) to a strike-vol slice.

# C++ parity: ql/math/interpolations/sabrinterpolation.hpp (v1.42.1).

The C++ class wires the generic ``XABRInterpolationImpl`` template
through ``SABRSpecs``: a 4-parameter constrained optimisation using
QuantLib's own ``LevenbergMarquardt`` plus a multi-start "max-guesses"
grid (50 randomized restarts by default). PQuantLib delegates the
optimisation to ``scipy.optimize.least_squares(method='trf', bounds=...)``
which is QuantLib's de-facto equivalent for this kind of bound-constrained
non-linear least squares — see the divergence note in
:func:`SabrInterpolation._fit`.

Documented divergences from C++:

* **Optimizer.** QuantLib uses the projected Levenberg-Marquardt from
  ``ql/math/optimization/levenbergmarquardt.hpp`` plus a bespoke
  inverse-direct re-parameterisation in ``SABRSpecs::{direct,inverse}``
  that maps a constrained 4-vector onto an unconstrained R^4 search
  space. SciPy's ``least_squares(method='trf')`` uses native box
  constraints + a trust-region-reflective LM variant; the two solvers
  visit different points but converge to the same SABR fit on
  well-behaved slices. Cross-validation against C++ via the L9-C probe
  is at LOOSE tier on recovered SABR parameters.
* **Multi-start sampling.** C++ runs up to ``maxGuesses`` (default 50)
  re-initialisations from a Halton-style low-discrepancy sequence; we
  fit once from the user-supplied or default initial guess. This is
  adequate for the realistic 5-strike slices used in Phase 9 testing.
* **Vega weighting.** The C++ ``weight()`` callback calls
  ``blackFormulaStdDevDerivative`` per (strike, vol) pair. PQuantLib
  implements the equivalent weighting; see the inline note on the
  ``_vega_weights`` helper. For the Phase 9 tests we exercise both
  arms (vega-weighted on / off) and round-trip-recover the input vols.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.math.interpolations.sabr_formula import shifted_sabr_volatility
from pquantlib.termstructures.volatility.volatility_type import VolatilityType

# Reasonable default initial guesses for unconstrained params (mirrors
# the C++ ``SABRSpecs::defaultValues`` and Hagan's typical starting
# points). For ``alpha`` we adopt the C++ rule that adapts to ``beta``.
_BETA_DEFAULT: Final[float] = 0.5
_NU_DEFAULT: Final[float] = 0.6324555320336759  # sqrt(0.4) — C++ default
_RHO_DEFAULT: Final[float] = 0.0
_ALPHA_DEFAULT_ATM_VOL: Final[float] = 0.2

# Box bounds for the trust-region-reflective solver. Matches the C++
# constraints expressed indirectly via ``SABRSpecs::direct``: ``alpha
# > eps1``, ``beta`` in [eps1, 1-eps1], ``nu > eps1``, ``rho`` in
# ``(-eps2, +eps2)`` with ``eps2 = .9999``. We pin slightly inside the
# open intervals to keep the solver in feasible territory.
_EPS1: Final[float] = 1.0e-7
_EPS2: Final[float] = 0.9999


def _default_alpha(beta: float, forward: float, shift: float) -> float:
    """C++ ``SABRSpecs::defaultValues`` alpha initialisation.

    # C++ parity: sabrinterpolation.hpp:69-76.
    """
    if beta < 0.9999:
        return _ALPHA_DEFAULT_ATM_VOL * (forward + shift) ** (1.0 - beta)
    return _ALPHA_DEFAULT_ATM_VOL


class SabrInterpolation:
    """Fit SABR (alpha, beta, nu, rho) to a strike-vol slice.

    # C++ parity: ``SABRInterpolation`` (sabrinterpolation.hpp:150-194).

    The constructor stores the inputs + runs the optimisation. Use
    :meth:`alpha`, :meth:`beta`, :meth:`nu`, :meth:`rho`,
    :meth:`rms_error` and :meth:`max_error` to inspect the fit, and
    call the instance like ``interp(strike)`` to evaluate the fitted
    SABR vol at any strike.

    Args:
        strikes: x-axis (strikes), sorted ascending.
        volatilities: y-axis (market vols).
        expiry_time: option expiry in year fractions.
        forward: ATM forward.
        alpha, beta, nu, rho: SABR parameter initial values. Each is
            optional; when ``None`` the C++ ``defaultValues`` rule is
            applied.
        alpha_is_fixed, beta_is_fixed, nu_is_fixed, rho_is_fixed: if
            ``True``, the parameter is pinned at its initial value and
            excluded from the optimisation.
        vega_weighted: if ``True``, weight the per-strike residual by
            the Black vega at the ATM forward. Matches C++
            ``vegaWeighted`` flag.
        shift: shifted-lognormal shift (default 0).
        volatility_type: ``ShiftedLognormal`` (default) or ``Normal``.
        max_nfev: passed to ``scipy.optimize.least_squares``; matches
            the C++ ``maxIterations`` budget (default 100).
    """

    def __init__(
        self,
        strikes: Sequence[float],
        volatilities: Sequence[float],
        expiry_time: float,
        forward: float,
        alpha: float | None = None,
        beta: float | None = None,
        nu: float | None = None,
        rho: float | None = None,
        alpha_is_fixed: bool = False,
        beta_is_fixed: bool = False,
        nu_is_fixed: bool = False,
        rho_is_fixed: bool = False,
        vega_weighted: bool = False,
        shift: float = 0.0,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        max_nfev: int = 1000,
    ) -> None:
        qassert.require(len(strikes) >= 2, "SabrInterpolation needs at least 2 strikes")
        qassert.require(
            len(strikes) == len(volatilities),
            "strikes and volatilities must have same length",
        )
        qassert.require(expiry_time >= 0.0, "expiry_time must be non-negative")

        # C++ parity: `defaultValues` rule applies if any input is None.
        beta_init = beta if beta is not None else _BETA_DEFAULT
        alpha_init = alpha if alpha is not None else _default_alpha(beta_init, forward, shift)
        nu_init = nu if nu is not None else _NU_DEFAULT
        rho_init = rho if rho is not None else _RHO_DEFAULT

        self._strikes: np.ndarray = np.ascontiguousarray(strikes, dtype=np.float64)
        self._volatilities: np.ndarray = np.ascontiguousarray(volatilities, dtype=np.float64)
        self._expiry_time: float = expiry_time
        self._forward: float = forward
        self._shift: float = shift
        self._volatility_type: VolatilityType = volatility_type
        self._vega_weighted: bool = vega_weighted

        self._is_fixed: list[bool] = [
            alpha_is_fixed,
            beta_is_fixed,
            nu_is_fixed,
            rho_is_fixed,
        ]
        self._initial: list[float] = [alpha_init, beta_init, nu_init, rho_init]

        # Fit and stash final params + diagnostics.
        self._alpha: float = alpha_init
        self._beta: float = beta_init
        self._nu: float = nu_init
        self._rho: float = rho_init
        self._rms_error: float = 0.0
        self._max_error: float = 0.0
        self._converged: bool = False
        self._fit(max_nfev=max_nfev)

    # --- fit ---------------------------------------------------------

    def _vega_weights(self) -> np.ndarray:
        """Per-strike Black-vega weights (sqrt-vega, like C++).

        # C++ parity: ``SABRSpecs::weight`` (sabrinterpolation.hpp:131-135).
        Each residual is multiplied by sqrt(vega) so the cost function
        becomes a vega-weighted RMS error. We approximate vega via a
        simple Black vega using the *initial* vol per strike (which is
        the market vol); this matches the C++ semantics that vega is
        evaluated against the market input, not the iterative fit.
        """
        # Black vega at ATM (per strike) = K * sqrt(T) * phi(d1).
        # We use a flat ATM proxy weight for the strike-spread family;
        # this is consistent with C++'s use of the per-pair black
        # vega derivative.
        # For numerical stability we just normalise by sum-to-1.
        from math import exp, log, pi, sqrt  # noqa: PLC0415 — local for speed
        n = len(self._strikes)
        weights = np.zeros(n, dtype=np.float64)
        sqrt_t = sqrt(self._expiry_time) if self._expiry_time > 0 else 1.0
        for i in range(n):
            k = self._strikes[i]
            v = self._volatilities[i]
            std_dev = v * sqrt_t
            if std_dev <= 0.0:
                weights[i] = 1.0
                continue
            f = self._forward + self._shift
            ks = k + self._shift
            if ks <= 0.0 or f <= 0.0:
                weights[i] = 1.0
                continue
            d1 = (log(f / ks) + 0.5 * std_dev * std_dev) / std_dev
            phi = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)
            vega = f * sqrt_t * phi
            weights[i] = max(vega, 1e-12)
        # Normalise to keep residuals comparable across slices.
        total = float(np.sum(weights))
        if total > 0.0:
            weights /= total
        else:
            weights[:] = 1.0 / n
        # Return sqrt(weight) so |sqrt(w) * r|^2 = w * r^2.
        return np.sqrt(weights)

    def _residuals(self, free_params: np.ndarray) -> np.ndarray:
        # Reconstruct the full 4-vector by interleaving free + fixed.
        params = list(self._initial)
        j = 0
        for i, fixed in enumerate(self._is_fixed):
            if not fixed:
                params[i] = float(free_params[j])
                j += 1
        alpha, beta, nu, rho = params
        # The TRF bounds keep us in feasible territory, but solver can
        # transiently visit edges; clamp to be safe.
        alpha = max(alpha, _EPS1)
        beta = min(max(beta, _EPS1), 1.0 - _EPS1)
        nu = max(nu, _EPS1)
        rho = min(max(rho, -_EPS2), _EPS2)
        model_vols = np.empty_like(self._strikes)
        for i, k in enumerate(self._strikes):
            model_vols[i] = shifted_sabr_volatility(
                float(k), self._forward, self._expiry_time,
                alpha, beta, nu, rho, self._shift, self._volatility_type,
            )
        r = model_vols - self._volatilities
        if self._vega_weighted:
            r = r * self._vega_weights()
        return r

    def _fit(self, *, max_nfev: int) -> None:
        free_initial: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        # alpha
        if not self._is_fixed[0]:
            free_initial.append(self._initial[0])
            lower.append(_EPS1)
            upper.append(np.inf)
        # beta
        if not self._is_fixed[1]:
            free_initial.append(self._initial[1])
            lower.append(_EPS1)
            upper.append(1.0 - _EPS1)
        # nu
        if not self._is_fixed[2]:
            free_initial.append(self._initial[2])
            lower.append(_EPS1)
            upper.append(np.inf)
        # rho
        if not self._is_fixed[3]:
            free_initial.append(self._initial[3])
            lower.append(-_EPS2)
            upper.append(_EPS2)

        if not free_initial:
            # Everything fixed — just evaluate residuals at the initial.
            r = self._residuals(np.array([], dtype=np.float64))
            self._update_diagnostics(r)
            self._converged = True
            return

        # scipy.optimize.least_squares is untyped in upstream stubs;
        # we capture the OptimizeResult into ``Any`` so pyright doesn't
        # try to introspect its (unknown) ``.x`` / ``.fun`` / ``.success``
        # attributes.
        result: Any = least_squares(  # pyright: ignore[reportUnknownVariableType]
            self._residuals,
            np.array(free_initial, dtype=np.float64),
            bounds=(np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)),
            method="trf",
            max_nfev=max_nfev,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        # Pull final params back into the full 4-vector.
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
        self._alpha, self._beta, self._nu, self._rho = params
        self._converged = bool(
            result.success  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        self._update_diagnostics(np.asarray(
            result.fun,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            dtype=np.float64,
        ))

    def _update_diagnostics(self, residuals: np.ndarray) -> None:
        if residuals.size == 0:
            self._rms_error = 0.0
            self._max_error = 0.0
            return
        self._rms_error = float(np.sqrt(np.mean(residuals * residuals)))
        self._max_error = float(np.max(np.abs(residuals)))

    # --- public API --------------------------------------------------

    def alpha(self) -> float:
        return self._alpha

    def beta(self) -> float:
        return self._beta

    def nu(self) -> float:
        return self._nu

    def rho(self) -> float:
        return self._rho

    def expiry(self) -> float:
        return self._expiry_time

    def forward(self) -> float:
        return self._forward

    def shift(self) -> float:
        return self._shift

    def rms_error(self) -> float:
        return self._rms_error

    def max_error(self) -> float:
        return self._max_error

    def converged(self) -> bool:
        return self._converged

    def value(self, strike: float) -> float:
        """Evaluate the fitted SABR vol at ``strike``."""
        return shifted_sabr_volatility(
            strike, self._forward, self._expiry_time,
            self._alpha, self._beta, self._nu, self._rho,
            self._shift, self._volatility_type,
        )

    def __call__(self, strike: float) -> float:
        return self.value(strike)
