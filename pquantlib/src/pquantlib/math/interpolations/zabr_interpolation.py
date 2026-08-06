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

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.zabr_formula import (
    ZabrEvaluation,
    zabr_volatility,
)
from pquantlib.pricingengines.black_formula import black_formula_std_dev_derivative

if TYPE_CHECKING:
    from pquantlib.termstructures.volatility.zabr_smile_section import ZabrSmileSection

#: C++ ``Null<Real>()`` — the sentinel a caller passes for "use the default".
#: Same value as ``sabr_interpolation.NULL_REAL``.
NULL_REAL: Final[float] = float(np.finfo(np.float32).max)

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


class ZabrSpecs:
    """The ZABR model policy the C++ XABR template is instantiated with.

    # C++ parity: ``struct detail::ZabrSpecs<Evaluation>``
    # (zabrinterpolation.hpp:36-118).

    ``XABRInterpolation<Model>`` is generic over a "specs" type supplying the
    parameter count, the default/initial values, the residual weights and —
    the substantive part — a bijection between the constrained parameter box
    and unconstrained R^5, so that an *unconstrained* optimiser can be used.
    :meth:`direct` maps R^5 into the box, :meth:`inverse` maps back.

    Same standing as :class:`pquantlib.math.interpolations.sabr_interpolation.SABRSpecs`:
    :class:`ZabrInterpolation` does not route its optimisation through this
    reparameterisation (it uses scipy's native box constraints instead — see
    the module docstring), so these methods exist as a faithful,
    cross-validated transcription rather than as the live calibration path.
    They are what any port of the C++ optimiser arm needs.

    Unlike ``SABRSpecs``, ZABR takes no shift: C++ ``ZabrSpecs`` ignores
    ``addParams`` everywhere, including in :meth:`weight`.

    ``Evaluation`` is a C++ template parameter; here it is a constructor
    argument, consulted only by :meth:`instance`.
    """

    __slots__ = ("_evaluation",)

    def __init__(
        self, evaluation: ZabrEvaluation = ZabrEvaluation.ShortMaturityLognormal
    ) -> None:
        self._evaluation: ZabrEvaluation = evaluation

    def dimension(self) -> int:
        """Number of model parameters (5). C++ ``dimension``."""
        return 5

    def eps(self) -> float:
        """Optimiser epsilon. C++ ``eps``."""
        return 0.000001

    def eps1(self) -> float:
        """Lower clamp used by :meth:`direct` / :meth:`inverse`. C++ ``eps1``."""
        return 0.0000001

    def eps2(self) -> float:
        """Rho saturation level. C++ ``eps2``."""
        return 0.9999

    def dilation_factor(self) -> float:
        """Unused in the live formulas; kept for parity. C++ ``dilationFactor``."""
        return 0.001

    def default_values(
        self,
        params: list[float],
        param_is_fixed: Sequence[bool],
        forward: float,
        expiry_time: float,
        add_params: Sequence[float],
    ) -> None:
        """Fill any ``NULL_REAL`` slot of ``params`` in place.

        # C++ parity: ``defaultValues`` (zabrinterpolation.hpp:39-53).

        Order matters: beta is defaulted first because alpha's default reads
        it. ``expiry_time``, ``param_is_fixed`` and ``add_params`` are
        accepted and ignored, as in C++.
        """
        del param_is_fixed, expiry_time, add_params
        if params[1] == NULL_REAL:
            params[1] = 0.5
        if params[0] == NULL_REAL:
            # adapt alpha to beta level
            params[0] = 0.2 * (
                math.pow(forward, 1.0 - params[1]) if params[1] < 0.9999 else 1.0
            )
        if params[2] == NULL_REAL:
            params[2] = math.sqrt(0.4)
        if params[3] == NULL_REAL:
            params[3] = 0.0
        if params[4] == NULL_REAL:
            params[4] = 1.0

    def guess(
        self,
        values: Array,
        param_is_fixed: Sequence[bool],
        forward: float,
        expiry_time: float,
        r: Sequence[float],
        add_params: Sequence[float],
    ) -> None:
        """Seed ``values`` in place from the low-discrepancy draws ``r``.

        # C++ parity: ``guess`` (zabrinterpolation.hpp:54-71).

        ``r`` is consumed by a single running index in the order
        beta, alpha, nu, rho, gamma — *not* the parameter order — and a fixed
        parameter consumes nothing. Getting that order wrong silently
        reshuffles a multi-start search.

        Note the alpha branch reads ``values[1]`` even when beta is fixed and
        was therefore never written by this call; C++ does the same, so the
        incoming contents of ``values[1]`` matter. Reproduced verbatim.
        """
        del expiry_time, add_params
        j = 0
        if not param_is_fixed[1]:
            values[1] = (1.0 - 2e-6) * r[j] + 1e-6
            j += 1
        if not param_is_fixed[0]:
            values[0] = (1.0 - 2e-6) * r[j] + 1e-6  # lognormal vol guess
            j += 1
            # adapt this to beta level
            if values[1] < 0.999:
                values[0] *= math.pow(forward, 1.0 - float(values[1]))
        if not param_is_fixed[2]:
            values[2] = 1.5 * r[j] + 1e-6
            j += 1
        if not param_is_fixed[3]:
            values[3] = (2.0 * r[j] - 1.0) * (1.0 - 1e-6)
            j += 1
        if not param_is_fixed[4]:
            values[4] = r[j] * 2.0

    def inverse(
        self,
        y: Array,
        param_is_fixed: Sequence[bool],
        params: Sequence[float],
        forward: float,
    ) -> Array:
        """Constrained ``y`` -> unconstrained ``x``.

        # C++ parity: ``inverse`` (zabrinterpolation.hpp:77-88).

        Inverse of :meth:`direct`. Alpha switches to a linear arm above
        ``25 + eps1`` so the map stays well conditioned far out; beta goes
        through ``sqrt(-log(.))``, nu and gamma through ``tan``, and rho
        through ``asin(./eps2)``.
        """
        del param_is_fixed, params, forward
        eps1 = self.eps1()
        x = np.zeros(5, dtype=np.float64)
        y0 = float(y[0])
        x[0] = math.sqrt(y0 - eps1) if y0 < 25.0 + eps1 else (y0 - eps1 + 25.0) / 10.0
        x[1] = math.sqrt(-math.log(float(y[1])))
        x[2] = math.tan(math.pi * (float(y[2]) / 5.0 - 0.5))
        x[3] = math.asin(float(y[3]) / self.eps2())
        x[4] = math.tan(math.pi * (float(y[4]) / 1.9 - 0.5))
        return x

    def direct(
        self,
        x: Array,
        param_is_fixed: Sequence[bool],
        params: Sequence[float],
        forward: float,
    ) -> Array:
        """Unconstrained ``x`` -> constrained ``y``.

        # C++ parity: ``direct`` (zabrinterpolation.hpp:89-104).

        Note the saturating arms: beyond ``|x0| = 5`` alpha grows linearly
        rather than quadratically, beta collapses to ``eps1`` once ``|x1|``
        reaches ``sqrt(-log(eps1))``, and rho saturates at ``+/- eps2``
        beyond ``|x3| = 2.5 pi``. Those are the branches a port is tempted to
        drop as unreachable; the optimiser does reach them. ``atan`` caps nu
        at 5.0 and gamma at 1.9.
        """
        del param_is_fixed, params, forward
        eps1 = self.eps1()
        eps2 = self.eps2()
        y = np.zeros(5, dtype=np.float64)
        x0 = float(x[0])
        y[0] = x0 * x0 + eps1 if abs(x0) < 5.0 else (10.0 * abs(x0) - 25.0) + eps1
        x1 = float(x[1])
        y[1] = math.exp(-(x1 * x1)) if abs(x1) < math.sqrt(-math.log(eps1)) else eps1
        # limit nu to 5.00
        y[2] = (math.atan(float(x[2])) / math.pi + 0.5) * 5.0
        x3 = float(x[3])
        y[3] = (
            eps2 * math.sin(x3)
            if abs(x3) < 2.5 * math.pi
            else eps2 * (1.0 if x3 > 0.0 else -1.0)
        )
        # limit gamma to 1.9
        y[4] = (math.atan(float(x[4])) / math.pi + 0.5) * 1.9
        return y

    def weight(
        self,
        strike: float,
        forward: float,
        std_dev: float,
        add_params: Sequence[float],
    ) -> float:
        """Per-strike residual weight — the Black vega wrt std dev.

        # C++ parity: ``weight`` (zabrinterpolation.hpp:105-108). Unlike the
        # SABR specs, no displacement is passed: C++ calls
        # ``blackFormulaStdDevDerivative(strike, forward, stdDev, 1.0)``.
        """
        del add_params
        return black_formula_std_dev_derivative(strike, forward, std_dev, 1.0)

    def instance(
        self,
        t: float,
        forward: float,
        params: Sequence[float],
        add_params: Sequence[float],
    ) -> ZabrSmileSection:
        """Build the bound model.

        C++ ``instance`` / ``typedef ZabrSmileSection<Evaluation> type``
        (zabrinterpolation.hpp:109-115).
        """
        del add_params
        # Deferred import: ZabrSmileSection lives under termstructures, which
        # already depends on math.interpolations. C++ has the same edge
        # (zabrinterpolation.hpp includes zabrsmilesection.hpp); importing it
        # at module scope here would close the cycle in Python.
        from pquantlib.termstructures.volatility.zabr_smile_section import (  # noqa: PLC0415
            ZabrSmileSection as _ZabrSmileSection,
        )

        alpha, beta, nu, rho, gamma = (float(p) for p in params[:5])
        return _ZabrSmileSection(
            forward=forward,
            zabr_params=(alpha, beta, nu, rho, gamma),
            exercise_time=t,
            evaluation=self._evaluation,
        )


class Zabr:
    """ZABR interpolation factory and traits.

    # C++ parity: ``template<class Evaluation> class Zabr``
    # (zabrinterpolation.hpp:169-215).

    Holds the fit configuration and stamps out a
    :class:`ZabrInterpolation` per strike/vol slice, so a caller can
    configure the smile model once and hand the factory to something that
    interpolates many slices. ``global_`` is the C++ ``static const bool
    global`` traits flag that tells the interpolated-curve machinery this
    interpolation is fitted over all points at once rather than piecewise.

    Mirrors :class:`pquantlib.math.interpolations.sabr_interpolation.SABR`
    with the fifth ZABR parameter gamma.
    """

    #: C++ ``static const bool global = true``.
    global_: Final[bool] = True

    def __init__(
        self,
        t: float,
        forward: float,
        alpha: float,
        beta: float,
        nu: float,
        rho: float,
        gamma: float,
        alpha_is_fixed: bool,
        beta_is_fixed: bool,
        nu_is_fixed: bool,
        rho_is_fixed: bool,
        gamma_is_fixed: bool,
        vega_weighted: bool = False,
        end_criteria: Any = None,
        optimization_method: Any = None,
        error_accept: float = 0.0020,
        use_max_error: bool = False,
        max_guesses: int = 50,
        evaluation: ZabrEvaluation = ZabrEvaluation.ShortMaturityLognormal,
    ) -> None:
        self._t: float = t
        self._forward: float = forward
        self._alpha: float = alpha
        self._beta: float = beta
        self._nu: float = nu
        self._rho: float = rho
        self._gamma: float = gamma
        self._alpha_is_fixed: bool = alpha_is_fixed
        self._beta_is_fixed: bool = beta_is_fixed
        self._nu_is_fixed: bool = nu_is_fixed
        self._rho_is_fixed: bool = rho_is_fixed
        self._gamma_is_fixed: bool = gamma_is_fixed
        self._vega_weighted: bool = vega_weighted
        # C++ takes ext::shared_ptr<EndCriteria> / <OptimizationMethod>
        # pass-throughs; PQuantLib's fitter is fixed at the scipy TRF arm, so
        # they are accepted and unused (same treatment as SABR).
        self._end_criteria: Any = end_criteria
        self._optimization_method: Any = optimization_method
        self._error_accept: float = error_accept
        self._use_max_error: bool = use_max_error
        self._max_guesses: int = max_guesses
        self._evaluation: ZabrEvaluation = evaluation

    def interpolate(
        self, strikes: Sequence[float], volatilities: Sequence[float]
    ) -> ZabrInterpolation:
        """Fit a :class:`ZabrInterpolation` to one strike/vol slice.

        # C++ parity: ``Zabr::interpolate`` (zabrinterpolation.hpp:197-206).
        """
        return ZabrInterpolation(
            strikes,
            volatilities,
            self._t,
            self._forward,
            alpha=self._alpha,
            beta=self._beta,
            nu=self._nu,
            rho=self._rho,
            gamma=self._gamma,
            alpha_is_fixed=self._alpha_is_fixed,
            beta_is_fixed=self._beta_is_fixed,
            nu_is_fixed=self._nu_is_fixed,
            rho_is_fixed=self._rho_is_fixed,
            gamma_is_fixed=self._gamma_is_fixed,
            vega_weighted=self._vega_weighted,
            evaluation=self._evaluation,
            max_guesses=self._max_guesses,
        )


__all__ = [
    "NULL_REAL",
    "Zabr",
    "ZabrInterpolation",
    "ZabrSpecs",
]
