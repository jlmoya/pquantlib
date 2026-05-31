"""SVI (Stochastic-Volatility-Inspired) raw parameterization + fit.

# C++ parity: ql/experimental/volatility/sviinterpolation.hpp (v1.42.1).

Gatheral's raw SVI total-variance slice:

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

where ``k = log(strike / forward)`` is the log-moneyness and ``w`` is
the total implied variance. The Black (lognormal) volatility is then
``sqrt(max(0, w / tau))``.

:func:`svi_volatility` evaluates the raw slice directly (no fitting).
:class:`SviInterpolation` fits the 5 SVI parameters ``(a, b, sigma,
rho, m)`` to a market strike-vol slice via
``scipy.optimize.least_squares`` — the same delegation pattern as the
L9-C :class:`SabrInterpolation` (see that module's docstring for the
full optimizer-divergence rationale). The SVI no-arbitrage parameter
constraints from ``detail::checkSviParameters`` are enforced as box
bounds + an explicit feasibility check.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from pquantlib import qassert

# SVI no-arb epsilons matching C++ SviSpecs::eps1()/eps2().
_EPS1: Final[float] = 0.000001
_EPS2: Final[float] = 0.999999


def check_svi_parameters(
    a: float,
    b: float,
    sigma: float,
    rho: float,
    m: float,
    tte: float,
) -> None:
    """Validate raw-SVI no-arbitrage parameter constraints.

    # C++ parity: ``detail::checkSviParameters`` (sviinterpolation.hpp:35-47).
    """
    qassert.require(b >= 0.0, f"b ({b}) must be non negative")
    qassert.require(abs(rho) < 1.0, f"rho ({rho}) must be in (-1,1)")
    qassert.require(sigma > 0.0, f"sigma ({sigma}) must be positive")
    qassert.require(
        a + b * sigma * math.sqrt(1.0 - rho * rho) >= 0.0,
        f"a + b sigma sqrt(1-rho^2) (a={a}, b={b}, sigma={sigma}, "
        f"rho={rho}) must be non negative",
    )
    qassert.require(
        b * (1.0 + abs(rho)) <= 4.0,
        f"b(1+|rho|) must be less than or equal to 4, (b={b}, rho={rho})",
    )


def svi_total_variance(
    a: float,
    b: float,
    sigma: float,
    rho: float,
    m: float,
    k: float,
) -> float:
    """Raw-SVI total variance at log-moneyness ``k``.

    # C++ parity: ``detail::sviTotalVariance`` (sviinterpolation.hpp:49-53).
    """
    return a + b * (rho * (k - m) + math.sqrt((k - m) * (k - m) + sigma * sigma))


def svi_volatility(
    strike: float,
    forward: float,
    expiry: float,
    a: float,
    b: float,
    sigma: float,
    rho: float,
    m: float,
) -> float:
    """Raw-SVI Black (lognormal) volatility at ``strike``.

    # C++ parity: ``SviSmileSection::volatilityImpl``
    # (svismilesection.cpp:48-55) — log-moneyness against the forward,
    # raw SVI total variance, then ``sqrt(max(0, w / tau))``.

    Args:
        strike: option strike (clamped to ``>= 1e-6`` as in C++).
        forward: ATM forward (positive).
        expiry: option expiry in year fractions (positive).
        a, b, sigma, rho, m: raw SVI parameters.
    """
    k = math.log(max(strike, 1e-6) / forward)
    total_variance = svi_total_variance(a, b, sigma, rho, m, k)
    return math.sqrt(max(0.0, total_variance / expiry))


class SviInterpolation:
    """Fit raw-SVI ``(a, b, sigma, rho, m)`` to a strike-vol slice.

    # C++ parity: ``SviInterpolation`` (sviinterpolation.hpp:144-188)
    # wired through the generic ``XABRInterpolationImpl`` + ``SviSpecs``.

    Mirrors the :class:`SabrInterpolation` surface: constructor fits +
    stores params, accessors expose the fit, ``interp(strike)``
    evaluates the fitted SVI vol.

    Args:
        strikes: x-axis (strikes), ascending; length >= 5 for a unique
            5-parameter fit (fewer is accepted but under-determined).
        volatilities: y-axis market vols.
        expiry_time: option expiry in year fractions (positive).
        forward: ATM forward.
        a, b, sigma, rho, m: SVI initial values; ``None`` uses the C++
            ``SviSpecs::defaultValues`` rule.
        a_is_fixed .. m_is_fixed: pin a parameter at its initial value.
        vega_weighted: vega-weight residuals (Black vega at ATM).
        max_nfev: ``least_squares`` budget.
        max_guesses: number of Halton-distributed initial guesses to try
            (default 50, matching the C++ ``maxGuesses`` default for SVI;
            the raw-SVI total-variance slice has many local minima, so a
            single TRF start frequently diverges). Each start runs the
            optimisation and the lowest-RMS fit is kept.
        multi_start_seed: seed for the Halton multi-start RNG (default 42).
    """

    def __init__(
        self,
        strikes: Sequence[float],
        volatilities: Sequence[float],
        expiry_time: float,
        forward: float,
        a: float | None = None,
        b: float | None = None,
        sigma: float | None = None,
        rho: float | None = None,
        m: float | None = None,
        a_is_fixed: bool = False,
        b_is_fixed: bool = False,
        sigma_is_fixed: bool = False,
        rho_is_fixed: bool = False,
        m_is_fixed: bool = False,
        vega_weighted: bool = False,
        max_nfev: int = 1000,
        max_guesses: int = 50,
        multi_start_seed: int = 42,
    ) -> None:
        qassert.require(
            len(strikes) >= 2, "SviInterpolation needs at least 2 strikes"
        )
        qassert.require(
            len(strikes) == len(volatilities),
            "strikes and volatilities must have same length",
        )
        qassert.require(expiry_time > 0.0, "expiry_time must be positive")

        self._strikes: np.ndarray = np.ascontiguousarray(strikes, dtype=np.float64)
        self._volatilities: np.ndarray = np.ascontiguousarray(
            volatilities, dtype=np.float64
        )
        self._expiry_time: float = expiry_time
        self._forward: float = forward
        self._vega_weighted: bool = vega_weighted

        # C++ ``SviSpecs::defaultValues`` rule (sviinterpolation.hpp:59-80).
        sigma_init = sigma if sigma is not None else 0.1
        rho_init = rho if rho is not None else -0.4
        m_init = m if m is not None else 0.0
        b_init = b if b is not None else 2.0 / (1.0 + abs(rho_init))
        if a is not None:
            a_init = a
        else:
            a_init = max(
                0.20 * 0.20 * expiry_time
                - b_init * (
                    rho_init * (-m_init)
                    + math.sqrt((-m_init) * (-m_init) + sigma_init * sigma_init)
                ),
                -b_init * sigma_init * math.sqrt(1.0 - rho_init * rho_init) + _EPS1,
            )

        self._is_fixed: list[bool] = [
            a_is_fixed,
            b_is_fixed,
            sigma_is_fixed,
            rho_is_fixed,
            m_is_fixed,
        ]
        self._initial: list[float] = [a_init, b_init, sigma_init, rho_init, m_init]

        self._a: float = a_init
        self._b: float = b_init
        self._sigma: float = sigma_init
        self._rho: float = rho_init
        self._m: float = m_init
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

    def _model_vols(self, params: list[float]) -> np.ndarray:
        a, b, sigma, rho, m = params
        # Keep the SVI slice feasible (no-arb constraints); the bounds
        # below keep the solver here, but clamp defensively.
        b = max(b, 0.0)
        sigma = max(sigma, _EPS1)
        rho = min(max(rho, -_EPS2), _EPS2)
        out = np.empty_like(self._strikes)
        for i, strike in enumerate(self._strikes):
            out[i] = svi_volatility(
                float(strike), self._forward, self._expiry_time, a, b, sigma, rho, m
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

    def _fit(self, *, max_nfev: int) -> None:
        free_initial: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        # a (>= small lower bound; can be slightly negative per SVI).
        if not self._is_fixed[0]:
            free_initial.append(self._initial[0])
            lower.append(-np.inf)
            upper.append(np.inf)
        # b (>= 0).
        if not self._is_fixed[1]:
            free_initial.append(self._initial[1])
            lower.append(0.0)
            upper.append(np.inf)
        # sigma (> 0).
        if not self._is_fixed[2]:
            free_initial.append(self._initial[2])
            lower.append(_EPS1)
            upper.append(np.inf)
        # rho (|rho| < 1).
        if not self._is_fixed[3]:
            free_initial.append(self._initial[3])
            lower.append(-_EPS2)
            upper.append(_EPS2)
        # m (unbounded).
        if not self._is_fixed[4]:
            free_initial.append(self._initial[4])
            lower.append(-np.inf)
            upper.append(np.inf)

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
        self._store_params(params)
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
        self._a, self._b, self._sigma, self._rho, self._m = params

    def _update_diagnostics(self, residuals: np.ndarray) -> None:
        if residuals.size == 0:
            self._rms_error = 0.0
            self._max_error = 0.0
            return
        self._rms_error = float(np.sqrt(np.mean(residuals * residuals)))
        self._max_error = float(np.max(np.abs(residuals)))

    def _fit_multi_start(self, *, max_nfev: int, max_guesses: int, seed: int) -> None:
        """Halton multi-start: run ``max_guesses`` fits, keep best RMS.

        # C++ parity: ``XABRInterpolationImpl::calculate`` outer loop that
        # resamples the initial guess via a Halton low-discrepancy
        # sequence (``SviSpecs::guess``) and restarts the optimiser up to
        # ``maxGuesses`` times, retaining the best fit. The raw-SVI slice
        # is multi-modal so this is load-bearing (a single start from the
        # ``defaultValues`` point frequently lands in a bad local min).
        # PQuantLib draws starts from the same :class:`HaltonRsg`
        # low-discrepancy generator and maps each sample into the
        # data-derived feasible box.
        """
        from pquantlib.math.randomnumbers.halton import HaltonRsg  # noqa: PLC0415

        # Data-derived bounding box per free SVI axis. Total-variance
        # scale sets the range for ``a``; log-moneyness range sets ``m``
        # and an upper bound for ``sigma``.
        ks = np.log(np.maximum(self._strikes, 1e-6) / self._forward)
        k_lo = float(np.min(ks))
        k_hi = float(np.max(ks))
        w = self._volatilities * self._volatilities * self._expiry_time
        w_max = float(np.max(w)) if w.size else 1.0
        k_span = max(k_hi - k_lo, 0.1)

        axes: list[tuple[float, float]] = []
        if not self._is_fixed[0]:
            axes.append((-2.0 * w_max, 2.0 * w_max))  # a
        if not self._is_fixed[1]:
            axes.append((0.0, 4.0))  # b
        if not self._is_fixed[2]:
            axes.append((_EPS1, 2.0 * k_span))  # sigma
        if not self._is_fixed[3]:
            axes.append((-_EPS2, _EPS2))  # rho
        if not self._is_fixed[4]:
            axes.append((k_lo - 0.5 * k_span, k_hi + 0.5 * k_span))  # m

        # First pass: user-supplied / default-rule initial.
        self._fit(max_nfev=max_nfev)
        best_rms = self._rms_error
        best_params = [self._a, self._b, self._sigma, self._rho, self._m]
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
            except (ValueError, FloatingPointError):
                continue
            if self._rms_error < best_rms:
                best_rms = self._rms_error
                best_params = [self._a, self._b, self._sigma, self._rho, self._m]
                best_converged = self._converged
                best_max = self._max_error

        self._store_params(best_params)
        self._rms_error = best_rms
        self._max_error = best_max
        self._converged = best_converged

    # --- public API -----------------------------------------------------

    def a(self) -> float:
        return self._a

    def b(self) -> float:
        return self._b

    def sigma(self) -> float:
        return self._sigma

    def rho(self) -> float:
        return self._rho

    def m(self) -> float:
        return self._m

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
        return svi_volatility(
            strike, self._forward, self._expiry_time,
            self._a, self._b, self._sigma, self._rho, self._m,
        )

    def __call__(self, strike: float) -> float:
        return self.value(strike)


__all__ = [
    "SviInterpolation",
    "check_svi_parameters",
    "svi_total_variance",
    "svi_volatility",
]
