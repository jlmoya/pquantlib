"""ZabrInterpolation — fits ZABR (alpha, beta, nu, rho, gamma) to a strike-vol slice.

# C++ parity: ql/math/interpolations/zabrinterpolation.hpp (v1.42.1).

The C++ class wires the generic ``XABRInterpolationImpl`` template
through ``ZabrSpecs<Evaluation>``: a 5-parameter constrained
optimisation using QuantLib's own projected ``LevenbergMarquardt``
plus a multi-start "max-guesses" grid (50 randomized restarts by
default). PQuantLib delegates the optimisation to
``scipy.optimize.least_squares(method='trf', bounds=...)`` which is
QuantLib's de-facto equivalent for this kind of bound-constrained
non-linear least squares — see the divergence note in
:func:`ZabrInterpolation._fit`. Mirrors the L9-C SabrInterpolation port
extended with the fifth ZABR parameter gamma.

Documented divergences from C++:

* **Optimizer.** Same divergence pattern as :class:`SabrInterpolation`:
  QuantLib uses the projected Levenberg-Marquardt
  (``ql/math/optimization/levenbergmarquardt.hpp``) plus the bespoke
  ``ZabrSpecs::{direct,inverse}`` re-parameterisation that maps the
  constrained 5-vector onto an unconstrained R^5 search space; PQuantLib
  uses ``least_squares(method='trf')`` with native box constraints.
  Different solver paths, equivalent fits on well-behaved slices.

  **gamma-inclusive ZABR fits.** The ``y(strike) → x(K)`` transform
  uses ``solve_ivp(method='RK45')`` for gamma != 1 — see
  :mod:`zabr_formula`. Each scipy least-squares step calls the ODE
  integrator per strike, so the gamma-free fit is materially more
  expensive than the SABR fit. Cross-validation against the C++ W2-A
  probe is at LOOSE tier on recovered parameters; the fitted vols
  themselves recover to LOOSE tier at every strike.

* **Multi-start sampling.** C++ runs up to ``maxGuesses`` (default 50)
  re-initialisations from a Halton-style low-discrepancy sequence
  passed through ``ZabrSpecs::guess``. PQuantLib mirrors this via the
  ``max_guesses`` kwarg (default 1, single-start; matches L9-C/L10-A
  SabrInterpolation pattern).

* **Vega weighting.** Same Black-vega per-pair weighting as
  :class:`SabrInterpolation`.

* **Box bounds for gamma.** C++ ``ZabrSpecs::direct`` clamps gamma to
  ``(0, 1.9)`` via ``y[4] = (atan(x[4])/pi + 0.5) * 1.9`` — that's the
  effective constraint. PQuantLib uses native scipy bounds ``[eps,
  1.9]`` to mirror this.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.math.interpolations.zabr_formula import (
    ZabrEvaluation,
    zabr_volatility,
)

_BETA_DEFAULT: Final[float] = 0.5
_NU_DEFAULT: Final[float] = 0.6324555320336759  # sqrt(0.4) — C++ default
_RHO_DEFAULT: Final[float] = 0.0
_GAMMA_DEFAULT: Final[float] = 1.0
_ALPHA_DEFAULT_ATM_VOL: Final[float] = 0.2

# Box bounds matching the C++ ``ZabrSpecs::direct`` images:
#   alpha > eps1, beta in [eps1, 1-eps1], nu in [eps1, 5],
#   rho in (-eps2, +eps2), gamma in [eps1, 1.9].
_EPS1: Final[float] = 1.0e-7
_EPS2: Final[float] = 0.9999
_GAMMA_UPPER: Final[float] = 1.9
_NU_UPPER: Final[float] = 5.0


def _default_alpha(beta: float, forward: float) -> float:
    """C++ ``ZabrSpecs::defaultValues`` alpha initialisation.

    # C++ parity: zabrinterpolation.hpp:42-48 — same rule as SABR's
    # ``defaultValues`` but no shift parameter (ZABR is unshifted).
    """
    if beta < 0.9999:
        return _ALPHA_DEFAULT_ATM_VOL * (forward ** (1.0 - beta))
    return _ALPHA_DEFAULT_ATM_VOL


class ZabrInterpolation:
    """Fit ZABR (alpha, beta, nu, rho, gamma) to a strike-vol slice.

    # C++ parity: ``ZabrInterpolation<Evaluation>`` (zabrinterpolation.hpp:121).

    The constructor stores the inputs + runs the optimisation. Use
    :meth:`alpha`, :meth:`beta`, :meth:`nu`, :meth:`rho`, :meth:`gamma`,
    :meth:`rms_error` and :meth:`max_error` to inspect the fit, and
    call the instance like ``interp(strike)`` to evaluate the fitted
    ZABR vol at any strike.

    Args:
        strikes: x-axis (strikes), sorted ascending.
        volatilities: y-axis (market vols).
        expiry_time: option expiry in year fractions.
        forward: ATM forward.
        alpha, beta, nu, rho, gamma: ZABR parameter initial values. Each
            is optional; when ``None`` the C++ ``defaultValues`` rule is
            applied.
        alpha_is_fixed, beta_is_fixed, nu_is_fixed, rho_is_fixed,
            gamma_is_fixed: if ``True``, the parameter is pinned at its
            initial value and excluded from the optimisation.
        vega_weighted: if ``True``, weight per-strike residuals by Black
            vega at the ATM forward. Matches C++ ``vegaWeighted`` flag.
        evaluation: ZABR evaluation mode used to compute model vols
            during the fit; defaults to ``ShortMaturityLognormal``. The
            FD modes raise ``LibraryException``.
        max_nfev: passed to ``scipy.optimize.least_squares``; matches
            the C++ ``maxIterations`` budget. Default 1000.
        max_guesses: number of Halton-distributed initial guesses to
            try. When set above 1 the constructor samples
            ``max_guesses - 1`` additional starting points from
            :class:`HaltonRsg` (over the free param subspace), runs the
            optimisation from each, and keeps the lowest-RMS fit.
            Default 1 = single-start (back-compat).
        multi_start_seed: seed for the Halton multi-start RNG (default
            42); only consulted when ``max_guesses > 1``.
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
        gamma: float | None = None,
        alpha_is_fixed: bool = False,
        beta_is_fixed: bool = False,
        nu_is_fixed: bool = False,
        rho_is_fixed: bool = False,
        gamma_is_fixed: bool = False,
        vega_weighted: bool = False,
        evaluation: ZabrEvaluation = ZabrEvaluation.ShortMaturityLognormal,
        max_nfev: int = 1000,
        max_guesses: int = 1,
        multi_start_seed: int = 42,
    ) -> None:
        qassert.require(
            len(strikes) >= 2, "ZabrInterpolation needs at least 2 strikes"
        )
        qassert.require(
            len(strikes) == len(volatilities),
            "strikes and volatilities must have same length",
        )
        qassert.require(expiry_time >= 0.0, "expiry_time must be non-negative")

        beta_init = beta if beta is not None else _BETA_DEFAULT
        alpha_init = alpha if alpha is not None else _default_alpha(beta_init, forward)
        nu_init = nu if nu is not None else _NU_DEFAULT
        rho_init = rho if rho is not None else _RHO_DEFAULT
        gamma_init = gamma if gamma is not None else _GAMMA_DEFAULT

        self._strikes: np.ndarray = np.ascontiguousarray(strikes, dtype=np.float64)
        self._volatilities: np.ndarray = np.ascontiguousarray(
            volatilities, dtype=np.float64
        )
        self._expiry_time: float = expiry_time
        self._forward: float = forward
        self._evaluation: ZabrEvaluation = evaluation
        self._vega_weighted: bool = vega_weighted

        self._is_fixed: list[bool] = [
            alpha_is_fixed,
            beta_is_fixed,
            nu_is_fixed,
            rho_is_fixed,
            gamma_is_fixed,
        ]
        self._initial: list[float] = [
            alpha_init, beta_init, nu_init, rho_init, gamma_init,
        ]

        self._alpha: float = alpha_init
        self._beta: float = beta_init
        self._nu: float = nu_init
        self._rho: float = rho_init
        self._gamma: float = gamma_init
        self._rms_error: float = 0.0
        self._max_error: float = 0.0
        self._converged: bool = False
        if max_guesses <= 1:
            self._fit(max_nfev=max_nfev)
        else:
            self._fit_multi_start(
                max_nfev=max_nfev,
                max_guesses=max_guesses,
                seed=multi_start_seed,
            )

    # --- fit ---------------------------------------------------------

    def _vega_weights(self) -> np.ndarray:
        """Per-strike Black-vega weights — same shape/semantics as L9-C SABR."""
        from math import exp, log, pi, sqrt  # noqa: PLC0415
        n = len(self._strikes)
        weights = np.zeros(n, dtype=np.float64)
        sqrt_t = sqrt(self._expiry_time) if self._expiry_time > 0 else 1.0
        for i in range(n):
            k = float(self._strikes[i])
            v = float(self._volatilities[i])
            std_dev = v * sqrt_t
            if std_dev <= 0.0 or k <= 0.0 or self._forward <= 0.0:
                weights[i] = 1.0
                continue
            d1 = (log(self._forward / k) + 0.5 * std_dev * std_dev) / std_dev
            phi = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)
            vega = self._forward * sqrt_t * phi
            weights[i] = max(vega, 1e-12)
        total = float(np.sum(weights))
        if total > 0.0:
            weights /= total
        else:
            weights[:] = 1.0 / n
        return np.sqrt(weights)

    def _eval_vols_at(
        self, alpha: float, beta: float, nu: float, rho: float, gamma: float,
    ) -> np.ndarray:
        """Evaluate the ZABR vol at each strike with the given params."""
        # Clamp to feasible territory in case TRF visits a boundary.
        alpha = max(alpha, _EPS1)
        beta = min(max(beta, _EPS1), 1.0 - _EPS1)
        nu = min(max(nu, _EPS1), _NU_UPPER)
        rho = min(max(rho, -_EPS2), _EPS2)
        gamma = min(max(gamma, _EPS1), _GAMMA_UPPER)
        model_vols = np.empty_like(self._strikes)
        for i, k in enumerate(self._strikes):
            model_vols[i] = zabr_volatility(
                float(k), self._forward, self._expiry_time,
                alpha, beta, nu, rho, gamma, mode=self._evaluation,
            )
        return model_vols

    def _residuals(self, free_params: np.ndarray) -> np.ndarray:
        params = list(self._initial)
        j = 0
        for i, fixed in enumerate(self._is_fixed):
            if not fixed:
                params[i] = float(free_params[j])
                j += 1
        alpha, beta, nu, rho, gamma = params
        model_vols = self._eval_vols_at(alpha, beta, nu, rho, gamma)
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
            upper.append(_NU_UPPER)
        # rho
        if not self._is_fixed[3]:
            free_initial.append(self._initial[3])
            lower.append(-_EPS2)
            upper.append(_EPS2)
        # gamma
        if not self._is_fixed[4]:
            free_initial.append(self._initial[4])
            lower.append(_EPS1)
            upper.append(_GAMMA_UPPER)

        if not free_initial:
            r = self._residuals(np.array([], dtype=np.float64))
            self._update_diagnostics(r)
            self._converged = True
            return

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
        self._alpha, self._beta, self._nu, self._rho, self._gamma = params
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

    def _fit_multi_start(self, *, max_nfev: int, max_guesses: int, seed: int) -> None:
        """Halton multi-start — same pattern as :class:`SabrInterpolation`."""
        from pquantlib.math.randomnumbers.halton import HaltonRsg  # noqa: PLC0415

        # Free-axis bounding box.
        axes: list[tuple[float, float]] = []
        if not self._is_fixed[0]:
            axes.append((_EPS1, 0.5))
        if not self._is_fixed[1]:
            axes.append((_EPS1, 1.0 - _EPS1))
        if not self._is_fixed[2]:
            axes.append((_EPS1, _NU_UPPER))
        if not self._is_fixed[3]:
            axes.append((-_EPS2, _EPS2))
        if not self._is_fixed[4]:
            axes.append((_EPS1, _GAMMA_UPPER))

        # First pass: the user-supplied (or default-rule) initial.
        self._fit(max_nfev=max_nfev)
        best_rms = self._rms_error
        best_params = (
            self._alpha, self._beta, self._nu, self._rho, self._gamma,
        )
        best_converged = self._converged

        if not axes:
            return

        rsg = HaltonRsg(
            dimensionality=len(axes), seed=seed, random_start=True, random_shift=False,
        )

        for _ in range(max_guesses - 1):
            sample = rsg.next_sequence().value
            params = list(self._initial)
            j = 0
            for i, fixed in enumerate(self._is_fixed):
                if fixed:
                    continue
                lo, hi = axes[j]
                params[i] = lo + (hi - lo) * float(sample[j])
                j += 1
            self._initial = params
            try:
                self._fit(max_nfev=max_nfev)
            except Exception:
                continue
            if self._rms_error < best_rms:
                best_rms = self._rms_error
                best_params = (
                    self._alpha, self._beta, self._nu, self._rho, self._gamma,
                )
                best_converged = self._converged

        (
            self._alpha, self._beta, self._nu, self._rho, self._gamma,
        ) = best_params
        self._rms_error = best_rms
        self._converged = best_converged
        residuals = self._residuals_at_full(best_params)
        self._update_diagnostics(residuals)

    def _residuals_at_full(
        self, params: tuple[float, float, float, float, float],
    ) -> np.ndarray:
        """Evaluate residuals at the full 5-vector."""
        alpha, beta, nu, rho, gamma = params
        model_vols = self._eval_vols_at(alpha, beta, nu, rho, gamma)
        r = model_vols - self._volatilities
        if self._vega_weighted:
            r = r * self._vega_weights()
        return r

    # --- public API --------------------------------------------------

    def alpha(self) -> float:
        return self._alpha

    def beta(self) -> float:
        return self._beta

    def nu(self) -> float:
        return self._nu

    def rho(self) -> float:
        return self._rho

    def gamma(self) -> float:
        return self._gamma

    def expiry(self) -> float:
        return self._expiry_time

    def forward(self) -> float:
        return self._forward

    def rms_error(self) -> float:
        return self._rms_error

    def max_error(self) -> float:
        return self._max_error

    def converged(self) -> bool:
        return self._converged

    def evaluation(self) -> ZabrEvaluation:
        return self._evaluation

    def value(self, strike: float) -> float:
        """Evaluate the fitted ZABR vol at ``strike``."""
        return zabr_volatility(
            strike, self._forward, self._expiry_time,
            self._alpha, self._beta, self._nu, self._rho, self._gamma,
            mode=self._evaluation,
        )

    def __call__(self, strike: float) -> float:
        return self.value(strike)
