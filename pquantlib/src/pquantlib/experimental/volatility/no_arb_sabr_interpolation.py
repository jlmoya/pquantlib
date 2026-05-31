"""NoArbSabrInterpolation — fit no-arbitrage SABR to a strike-vol slice.

# C++ parity: ql/experimental/volatility/noarbsabrinterpolation.hpp (v1.42.1).

Fits the 4 no-arbitrage SABR parameters ``(alpha, beta, nu, rho)`` to a
market strike-vol slice, mirroring the L9-C :class:`SabrInterpolation`
surface but evaluating the model vol via :func:`no_arb_sabr_volatility`
(which prices the Doust no-arb terminal density and inverts Black).

The C++ class wires the generic ``XABRInterpolationImpl`` through
``NoArbSabrSpecs``. PQuantLib delegates the optimisation to
``scipy.optimize.least_squares(method='trf')`` (see the
:class:`SabrInterpolation` docstring for the optimiser-divergence
rationale).

Parameter bounds differ from plain SABR — the no-arb model constrains
``sigmaI = alpha * forward^(beta-1)`` to ``[0.05, 1.0]`` (rather than
``alpha`` directly), ``beta`` to ``[0.01, 0.99]``, ``nu`` to
``[0.01, 0.80]`` and ``rho`` to ``[-0.99, 0.99]`` (see
``detail::NoArbSabrModel`` and ``NoArbSabrSpecs::guess``). The
``defaultValues`` adjustment that nudges ``alpha`` into the admissible
``sigmaI`` band is reproduced.

Because each model evaluation prices + integrates the no-arb density,
the fit is materially more expensive than the SABR or SVI fit; the
multi-start guess count therefore defaults to a modest value.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.experimental.volatility.no_arb_sabr import (
    BETA_MAX,
    BETA_MIN,
    NU_MAX,
    NU_MIN,
    RHO_MAX,
    RHO_MIN,
    SIGMA_I_MAX,
    SIGMA_I_MIN,
    no_arb_sabr_volatility,
)

_EPS: Final[float] = 0.000001


def _default_alpha(forward: float, beta: float) -> float:
    """Plain-SABR ``defaultValues`` alpha, then nudged into sigmaI band.

    # C++ parity: ``NoArbSabrSpecs::defaultValues`` (noarbsabrinterpolation.hpp:41-73)
    # which calls ``SABRSpecs::defaultValues`` then adjusts alpha so
    # ``sigmaI = alpha*F^(beta-1)`` lands within [sigmaI_min, sigmaI_max].
    """
    alpha = 0.2 * forward ** (1.0 - beta) if beta < 0.9999 else 0.2
    sigma_i = alpha * forward ** (beta - 1.0)
    if sigma_i < SIGMA_I_MIN:
        alpha = SIGMA_I_MIN * (1.0 + _EPS) / forward ** (beta - 1.0)
    elif sigma_i > SIGMA_I_MAX:
        alpha = SIGMA_I_MAX * (1.0 - _EPS) / forward ** (beta - 1.0)
    return alpha


class NoArbSabrInterpolation:
    """Fit no-arbitrage SABR ``(alpha, beta, nu, rho)`` to a strike-vol slice.

    # C++ parity: ``NoArbSabrInterpolation`` (noarbsabrinterpolation.hpp:193-238).

    Args:
        strikes: x-axis (strikes), ascending; length >= 2.
        volatilities: y-axis market vols.
        expiry_time: option expiry ``tau`` in year fractions (positive).
        forward: ATM forward (positive).
        alpha, beta, nu, rho: initial values; ``None`` uses the C++
            ``defaultValues`` rule (beta=0.5, alpha nudged into sigmaI
            band, nu=sqrt(0.4) clamped to [nu_min,nu_max], rho=0).
        alpha_is_fixed .. rho_is_fixed: pin a parameter during the fit.
        vega_weighted: vega-weight residuals (Black vega at ATM).
        max_nfev: ``least_squares`` budget.
        max_guesses: Halton multi-start count (default 1 — single start,
            since each no-arb model evaluation is expensive). Set above 1
            for the multi-modal robustness the C++ ``maxGuesses=50`` path
            provides.
        multi_start_seed: seed for the Halton multi-start RNG.
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
        max_nfev: int = 1000,
        max_guesses: int = 1,
        multi_start_seed: int = 42,
    ) -> None:
        qassert.require(
            len(strikes) >= 2, "NoArbSabrInterpolation needs at least 2 strikes"
        )
        qassert.require(
            len(strikes) == len(volatilities),
            "strikes and volatilities must have same length",
        )
        qassert.require(expiry_time > 0.0, "expiry_time must be positive")

        beta_init = beta if beta is not None else 0.5
        alpha_init = alpha if alpha is not None else _default_alpha(forward, beta_init)
        nu_init = nu if nu is not None else min(max(0.6324555320336759, NU_MIN), NU_MAX)
        rho_init = rho if rho is not None else 0.0

        self._strikes: np.ndarray = np.ascontiguousarray(strikes, dtype=np.float64)
        self._volatilities: np.ndarray = np.ascontiguousarray(
            volatilities, dtype=np.float64
        )
        self._expiry_time: float = expiry_time
        self._forward: float = forward
        self._vega_weighted: bool = vega_weighted

        self._is_fixed: list[bool] = [
            alpha_is_fixed,
            beta_is_fixed,
            nu_is_fixed,
            rho_is_fixed,
        ]
        self._initial: list[float] = [alpha_init, beta_init, nu_init, rho_init]

        self._alpha: float = alpha_init
        self._beta: float = beta_init
        self._nu: float = nu_init
        self._rho: float = rho_init
        self._rms_error: float = 0.0
        self._max_error: float = 0.0
        self._converged: bool = False
        if max_guesses <= 1:
            self._fit(max_nfev=max_nfev)
        else:
            self._fit_multi_start(
                max_nfev=max_nfev, max_guesses=max_guesses, seed=multi_start_seed
            )

    # --- fit ------------------------------------------------------------

    def _vega_weights(self) -> np.ndarray:
        from math import exp, log, pi, sqrt  # noqa: PLC0415

        n = len(self._strikes)
        weights = np.zeros(n, dtype=np.float64)
        sqrt_t = sqrt(self._expiry_time)
        for i in range(n):
            k = float(self._strikes[i])
            v = float(self._volatilities[i])
            std_dev = v * sqrt_t
            if std_dev <= 0.0 or k <= 0.0 or self._forward <= 0.0:
                weights[i] = 1.0
                continue
            d1 = (log(self._forward / k) + 0.5 * std_dev * std_dev) / std_dev
            phi = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)
            weights[i] = max(self._forward * sqrt_t * phi, 1e-12)
        total = float(np.sum(weights))
        if total > 0.0:
            weights /= total
        else:
            weights[:] = 1.0 / n
        return np.sqrt(weights)

    def _clamp(self, params: list[float]) -> tuple[float, float, float, float]:
        """Project ``(alpha, beta, nu, rho)`` into the admissible region.

        Beta + nu + rho are clamped to their bounds; alpha is nudged so
        ``sigmaI = alpha*F^(beta-1)`` stays within [sigmaI_min, sigmaI_max].
        """
        alpha, beta, nu, rho = params
        beta = min(max(beta, BETA_MIN), BETA_MAX)
        nu = min(max(nu, NU_MIN), NU_MAX)
        rho = min(max(rho, RHO_MIN), RHO_MAX)
        sigma_i = alpha * self._forward ** (beta - 1.0)
        if sigma_i < SIGMA_I_MIN:
            alpha = SIGMA_I_MIN * (1.0 + _EPS) / self._forward ** (beta - 1.0)
        elif sigma_i > SIGMA_I_MAX:
            alpha = SIGMA_I_MAX * (1.0 - _EPS) / self._forward ** (beta - 1.0)
        return alpha, beta, nu, rho

    def _model_vols(self, params: list[float]) -> np.ndarray:
        alpha, beta, nu, rho = self._clamp(params)
        out = np.empty_like(self._strikes)
        for i, strike in enumerate(self._strikes):
            out[i] = no_arb_sabr_volatility(
                float(strike), self._forward, self._expiry_time, alpha, beta, nu, rho
            )
        return out

    def _residuals(self, free_params: np.ndarray) -> np.ndarray:
        params = list(self._initial)
        j = 0
        for i, fixed in enumerate(self._is_fixed):
            if not fixed:
                params[i] = float(free_params[j])
                j += 1
        r = self._model_vols(params) - self._volatilities
        if self._vega_weighted:
            r = r * self._vega_weights()
        return r

    def _free_bounds(self) -> tuple[list[float], list[float], list[float]]:
        """Free initial + lower/upper bounds for the TRF solver.

        Bounds are expressed on ``(alpha, beta, nu, rho)`` directly; the
        ``alpha`` band is derived from the sigmaI band at the *initial*
        beta (a conservative proxy — ``_clamp`` enforces the exact band
        per residual evaluation).
        """
        beta0 = self._initial[1]
        alpha_lo = SIGMA_I_MIN / self._forward ** (beta0 - 1.0)
        alpha_hi = SIGMA_I_MAX / self._forward ** (beta0 - 1.0)
        free_initial: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        if not self._is_fixed[0]:
            free_initial.append(min(max(self._initial[0], alpha_lo), alpha_hi))
            lower.append(alpha_lo)
            upper.append(alpha_hi)
        if not self._is_fixed[1]:
            free_initial.append(self._initial[1])
            lower.append(BETA_MIN)
            upper.append(BETA_MAX)
        if not self._is_fixed[2]:
            free_initial.append(self._initial[2])
            lower.append(NU_MIN)
            upper.append(NU_MAX)
        if not self._is_fixed[3]:
            free_initial.append(self._initial[3])
            lower.append(RHO_MIN)
            upper.append(RHO_MAX)
        return free_initial, lower, upper

    def _fit(self, *, max_nfev: int) -> None:
        free_initial, lower, upper = self._free_bounds()
        if not free_initial:
            self._update_diagnostics(self._residuals(np.array([], dtype=np.float64)))
            self._store_params(list(self._initial))
            self._converged = True
            return

        result: Any = least_squares(  # pyright: ignore[reportUnknownVariableType]
            self._residuals,
            np.array(free_initial, dtype=np.float64),
            bounds=(np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)),
            method="trf",
            max_nfev=max_nfev,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
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
        self._store_params(list(self._clamp(params)))
        self._converged = bool(
            result.success  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        )
        self._update_diagnostics(
            np.asarray(
                result.fun,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                dtype=np.float64,
            )
        )

    def _store_params(self, params: list[float]) -> None:
        self._alpha, self._beta, self._nu, self._rho = params

    def _update_diagnostics(self, residuals: np.ndarray) -> None:
        if residuals.size == 0:
            self._rms_error = 0.0
            self._max_error = 0.0
            return
        self._rms_error = float(np.sqrt(np.mean(residuals * residuals)))
        self._max_error = float(np.max(np.abs(residuals)))

    def _fit_multi_start(  # noqa: PLR0915 — direct port of C++ guess+restart loop
        self, *, max_nfev: int, max_guesses: int, seed: int
    ) -> None:
        """Halton multi-start over the (sigmaI, beta, nu, rho) box.

        # C++ parity: ``NoArbSabrSpecs::guess`` samples sigmaI / beta /
        # nu / rho uniformly from their admissible bands via the Halton
        # generator; the outer loop keeps the best fit over maxGuesses.
        """
        from pquantlib.math.randomnumbers.halton import HaltonRsg  # noqa: PLC0415

        # Free-axis bands matching NoArbSabrSpecs::guess.
        axes: list[tuple[float, float]] = []
        if not self._is_fixed[0]:
            axes.append((SIGMA_I_MIN, SIGMA_I_MAX))  # sampled in sigmaI space
        if not self._is_fixed[1]:
            axes.append((BETA_MIN, BETA_MAX))
        if not self._is_fixed[2]:
            axes.append((NU_MIN, NU_MAX))
        if not self._is_fixed[3]:
            axes.append((RHO_MIN, RHO_MAX))

        self._fit(max_nfev=max_nfev)
        best_rms = self._rms_error
        best_params = [self._alpha, self._beta, self._nu, self._rho]
        best_converged = self._converged
        best_max = self._max_error

        if not axes:
            return

        rsg = HaltonRsg(
            dimensionality=len(axes), seed=seed,
            random_start=True, random_shift=False,
        )

        for _ in range(max_guesses - 1):
            sample = rsg.next_sequence().value
            params = list(self._initial)
            # First resolve the beta sample (free axis #1 if not fixed),
            # because the alpha-from-sigmaI map depends on it.
            beta_local = params[1]
            if not self._is_fixed[1]:
                jb = _free_index(self._is_fixed, 1)
                lo_b, hi_b = axes[jb]
                beta_local = lo_b + (hi_b - lo_b) * float(sample[jb])
                params[1] = beta_local
            j = 0
            for i, fixed in enumerate(self._is_fixed):
                if fixed:
                    continue
                lo, hi = axes[j]
                u = float(sample[j])
                if i == 0:
                    # axis is in sigmaI space → map to alpha via beta.
                    sigma_i = lo + (hi - lo) * u
                    params[0] = sigma_i / self._forward ** (beta_local - 1.0)
                elif i != 1:
                    params[i] = lo + (hi - lo) * u
                j += 1
            self._initial = params
            try:
                self._fit(max_nfev=max_nfev)
            except (ValueError, FloatingPointError):
                continue
            if self._rms_error < best_rms:
                best_rms = self._rms_error
                best_params = [self._alpha, self._beta, self._nu, self._rho]
                best_converged = self._converged
                best_max = self._max_error

        self._store_params(best_params)
        self._rms_error = best_rms
        self._max_error = best_max
        self._converged = best_converged

    # --- public API -----------------------------------------------------

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

    def rms_error(self) -> float:
        return self._rms_error

    def max_error(self) -> float:
        return self._max_error

    def converged(self) -> bool:
        return self._converged

    def value(self, strike: float) -> float:
        return no_arb_sabr_volatility(
            strike, self._forward, self._expiry_time,
            self._alpha, self._beta, self._nu, self._rho,
        )

    def __call__(self, strike: float) -> float:
        return self.value(strike)


def _free_index(is_fixed: list[bool], target: int) -> int:
    """Index of parameter ``target`` within the free-param subsequence."""
    j = 0
    for i in range(target):
        if not is_fixed[i]:
            j += 1
    return j


__all__ = ["NoArbSabrInterpolation"]
