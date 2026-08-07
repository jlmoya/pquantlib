"""NoArbSabrInterpolation — fit no-arbitrage SABR to a strike-vol slice.

# C++ parity: ql/experimental/volatility/noarbsabrinterpolation.hpp (v1.43).

Fits the 4 no-arbitrage SABR parameters ``(alpha, beta, nu, rho)`` to a
market strike-vol slice, mirroring the L9-C :class:`SabrInterpolation`
surface but evaluating the model vol via :func:`no_arb_sabr_volatility`
(which prices the Doust no-arb terminal density and inverts Black).

The C++ class wires the generic ``XABRInterpolationImpl`` through
``NoArbSabrSpecs``, ported here as :class:`NoArbSabrSpecs`.
:class:`NoArbSabr` is the interpolation factory/traits class. PQuantLib
delegates the optimisation to ``scipy.optimize.least_squares(method='trf')``
(see the :class:`SabrInterpolation` docstring for the optimiser-divergence
rationale); the delegation is cross-validated against the C++ fitted curve
by blocks ``D3``/``D4`` of
``references/v143/experimental/volatility.json``.

Parameter bounds differ from plain SABR — the no-arb model constrains
``sigmaI = alpha * forward^(beta-1)`` to ``[0.05, 1.0]`` (rather than
``alpha`` directly), ``beta`` to ``[0.01, 0.99]``, ``nu`` to
``[0.01, 0.80]`` and ``rho`` to ``[-0.99, 0.99]`` (see
``detail::NoArbSabrModel`` and :meth:`NoArbSabrSpecs.guess`). The
``defaultValues`` adjustment that nudges ``alpha`` into the admissible
``sigmaI`` band is reproduced.

Because each model evaluation prices + integrates the no-arb density,
the fit is materially more expensive than the SABR or SVI fit; the
multi-start guess count therefore defaults to a modest value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

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
from pquantlib.math.array import Array
from pquantlib.math.interpolations.sabr_interpolation import (
    NULL_REAL,
    SABRSpecs,
    as_null,
    xabr_interpolation_error,
)
from pquantlib.pricingengines.black_formula import black_formula_std_dev_derivative

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime
    from pquantlib.experimental.volatility.no_arb_sabr_smile_section import (
        NoArbSabrSmileSection,
    )

_EPS: Final[float] = 0.000001


class NoArbSabrSpecs:
    """The no-arb-SABR model policy the C++ XABR template is instantiated with.

    # C++ parity: ``struct detail::NoArbSabrSpecs``
    # (noarbsabrinterpolation.hpp:38-190).

    Same role as :class:`SABRSpecs` — parameter count, defaults, multi-start
    guess rule, residual weights and the constrained<->unconstrained
    bijection — but the box is the Doust admissible region, and the alpha
    axis is parameterised through ``sigmaI = alpha * forward^(beta - 1)``
    rather than through alpha directly. That coupling is why
    :meth:`direct` and :meth:`default_values` have to adjust *beta* when
    alpha is pinned outside the sigmaI band.

    PQuantLib's :class:`NoArbSabrInterpolation` does not currently route its
    optimisation through this reparameterisation (it uses scipy's native box
    constraints instead), so these methods exist as a faithful,
    cross-validated transcription rather than as the live calibration path.

    Every method is stateless; the class is instantiable to mirror the C++
    call sites (``Model().direct(...)``).
    """

    __slots__ = ()

    def dimension(self) -> int:
        """Number of model parameters (4). C++ ``dimension``."""
        return 4

    def eps(self) -> float:
        """Relative nudge used when snapping sigmaI into its band. C++ ``eps``."""
        return 0.000001

    def default_values(
        self,
        params: list[float],
        param_is_fixed: list[bool],
        forward: float,
        expiry_time: float,
        add_params: Sequence[float],
    ) -> None:
        """Fill any ``NULL_REAL`` slot of ``params`` in place, then snap sigmaI.

        # C++ parity: ``defaultValues`` (noarbsabrinterpolation.hpp:41-73).

        Runs :meth:`SABRSpecs.default_values` first, then checks
        ``sigmaI = alpha * F^(beta - 1)`` against the Doust band and repairs
        it: by rescaling alpha when alpha is free, otherwise by rescaling
        beta when beta is free, otherwise not at all (the model constructor
        raises later — C++ says so in a comment and does the same).

        # C++ parity note: ``sigmaI`` is computed ONCE, before both branches,
        # so the ``> sigmaI_max`` test still sees the pre-repair value. It
        # cannot be both below the floor and above the cap, so the ordering
        # is unobservable — but it is reproduced verbatim.
        #
        # C++ parity note: the beta repair is ``1 + log(bound/alpha)/log(F)``
        # with NO clamp to [beta_min, beta_max]; for a small forward it
        # readily returns a beta outside the admissible range (e.g. -0.1156
        # for alpha=0.001, F=0.03). Reproduced as-is.
        """
        SABRSpecs().default_values(params, param_is_fixed, forward, expiry_time, add_params)
        eps = self.eps()
        sigma_i = params[0] * math.pow(forward, params[1] - 1.0)
        if sigma_i < SIGMA_I_MIN:
            if not param_is_fixed[0]:
                params[0] = SIGMA_I_MIN * (1.0 + eps) / math.pow(forward, params[1] - 1.0)
            elif not param_is_fixed[1]:
                params[1] = 1.0 + math.log(SIGMA_I_MIN * (1.0 + eps) / params[0]) / math.log(
                    forward
                )
        if sigma_i > SIGMA_I_MAX:
            if not param_is_fixed[0]:
                params[0] = SIGMA_I_MAX * (1.0 - eps) / math.pow(forward, params[1] - 1.0)
            elif not param_is_fixed[1]:
                params[1] = 1.0 + math.log(SIGMA_I_MAX * (1.0 - eps) / params[0]) / math.log(
                    forward
                )

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

        # C++ parity: ``guess`` (noarbsabrinterpolation.hpp:74-102).

        ``r`` is consumed by a single running index in the order
        beta, alpha, nu, rho — *not* the parameter order — and a fixed
        parameter consumes nothing. The alpha draw is made in sigmaI space,
        squeezed by ``(1 - eps)`` and offset by ``eps/2`` to stay strictly
        inside the band, then divided by ``F^(beta - 1)`` using the beta
        that was *just* written.
        """
        del expiry_time, add_params
        eps = self.eps()
        j = 0
        if not param_is_fixed[1]:
            values[1] = BETA_MIN + (BETA_MAX - BETA_MIN) * r[j]
            j += 1
        if not param_is_fixed[0]:
            sigma_i = SIGMA_I_MIN + (SIGMA_I_MAX - SIGMA_I_MIN) * r[j]
            j += 1
            sigma_i *= 1.0 - eps
            sigma_i += eps / 2.0
            values[0] = sigma_i / math.pow(forward, float(values[1]) - 1.0)
        if not param_is_fixed[2]:
            values[2] = NU_MIN + (NU_MAX - NU_MIN) * r[j]
            j += 1
        if not param_is_fixed[3]:
            values[3] = RHO_MIN + (RHO_MAX - RHO_MIN) * r[j]

    def inverse(
        self,
        y: Array,
        param_is_fixed: Sequence[bool],
        params: Sequence[float],
        forward: float,
    ) -> Array:
        """Constrained ``y`` -> unconstrained ``x``.

        # C++ parity: ``inverse`` (noarbsabrinterpolation.hpp:103-128).

        Each axis is the ``tan`` inverse of the ``atan`` map in
        :meth:`direct`. Note the alpha axis inverts ``sigmaI``, not alpha.

        # C++ parity note: the alpha branch writes ``- M_PI/2`` where beta,
        # nu and rho write ``+ M_PI/2``. The two spellings produce the same
        # number because ``tan`` has period pi, so the asymmetry is cosmetic;
        # it is reproduced verbatim rather than normalised.
        """
        del param_is_fixed, params
        x = np.zeros(4, dtype=np.float64)
        y0, y1, y2, y3 = (float(y[i]) for i in range(4))
        half_pi = math.pi / 2.0
        x[1] = math.tan((y1 - BETA_MIN) / (BETA_MAX - BETA_MIN) * math.pi + half_pi)
        x[0] = math.tan(
            (y0 * math.pow(forward, y1 - 1.0) - SIGMA_I_MIN)
            / (SIGMA_I_MAX - SIGMA_I_MIN)
            * math.pi
            - half_pi
        )
        x[2] = math.tan((y2 - NU_MIN) / (NU_MAX - NU_MIN) * math.pi + half_pi)
        x[3] = math.tan((y3 - RHO_MIN) / (RHO_MAX - RHO_MIN) * math.pi + half_pi)
        return x

    def direct(
        self,
        x: Array,
        param_is_fixed: Sequence[bool],
        params: Sequence[float],
        forward: float,
    ) -> Array:
        """Unconstrained ``x`` -> constrained ``y``.

        # C++ parity: ``direct`` (noarbsabrinterpolation.hpp:129-179).

        Order is load-bearing: beta is resolved first because the alpha axis
        divides by ``F^(beta - 1)``. When alpha is pinned, beta is then
        *overwritten* by the log-ratio repair if the implied sigmaI leaves
        the band — discarding whatever beta the transform just produced.

        # C++ parity note: the repair is unclamped, exactly as in
        # :meth:`default_values`, and can return a beta outside
        # ``[beta_min, beta_max]``.
        """
        eps = self.eps()
        half_pi = math.pi / 2.0
        y = np.zeros(4, dtype=np.float64)
        x0, x1, x2, x3 = (float(x[i]) for i in range(4))
        if param_is_fixed[1]:
            y[1] = params[1]
        else:
            y[1] = BETA_MIN + (BETA_MAX - BETA_MIN) * (math.atan(x1) + half_pi) / math.pi
        # alpha is carried through sigmaI; if alpha is pinned we have to
        # check beta is admissible and adjust if need be.
        if param_is_fixed[0]:
            y[0] = params[0]
            sigma_i = float(y[0]) * math.pow(forward, float(y[1]) - 1.0)
            if sigma_i < SIGMA_I_MIN:
                y[1] = 1.0 + math.log(SIGMA_I_MIN * (1.0 + eps) / float(y[0])) / math.log(forward)
            if sigma_i > SIGMA_I_MAX:
                y[1] = 1.0 + math.log(SIGMA_I_MAX * (1.0 - eps) / float(y[0])) / math.log(forward)
        else:
            sigma_i = (
                SIGMA_I_MIN
                + (SIGMA_I_MAX - SIGMA_I_MIN) * (math.atan(x0) + half_pi) / math.pi
            )
            y[0] = sigma_i / math.pow(forward, float(y[1]) - 1.0)
        if param_is_fixed[2]:
            y[2] = params[2]
        else:
            y[2] = NU_MIN + (NU_MAX - NU_MIN) * (math.atan(x2) + half_pi) / math.pi
        if param_is_fixed[3]:
            y[3] = params[3]
        else:
            y[3] = RHO_MIN + (RHO_MAX - RHO_MIN) * (math.atan(x3) + half_pi) / math.pi
        return y

    def weight(
        self,
        strike: float,
        forward: float,
        std_dev: float,
        add_params: Sequence[float],
    ) -> float:
        """Per-strike residual weight — the Black vega wrt std dev.

        # C++ parity: ``weight`` (noarbsabrinterpolation.hpp:180-183). As in
        # ``SviSpecs::weight`` no displacement is passed; no-arb SABR has no
        # shift (a non-zero shift is rejected by the interpolation ctor).
        """
        del add_params
        return black_formula_std_dev_derivative(strike, forward, std_dev, 1.0)

    def instance(
        self,
        t: float,
        forward: float,
        params: Sequence[float],
        add_params: Sequence[float],
    ) -> NoArbSabrSmileSection:
        """Build the bound model.

        # C++ parity: ``instance`` / ``typedef NoArbSabrWrapper type``
        # (noarbsabrinterpolation.hpp:36, 184-189) where
        # ``typedef NoArbSabrSmileSection NoArbSabrWrapper``.
        """
        del add_params
        from pquantlib.experimental.volatility.no_arb_sabr_smile_section import (  # noqa: PLC0415
            NoArbSabrSmileSection,
        )

        alpha, beta, nu, rho = (float(v) for v in params)
        return NoArbSabrSmileSection(
            forward=forward, sabr_params=(alpha, beta, nu, rho), exercise_time=t
        )


class NoArbSabrInterpolation:
    """Fit no-arbitrage SABR ``(alpha, beta, nu, rho)`` to a strike-vol slice.

    # C++ parity: ``NoArbSabrInterpolation`` (noarbsabrinterpolation.hpp:193-238).

    Args:
        strikes: x-axis (strikes), ascending; length >= 2.
        volatilities: y-axis market vols.
        expiry_time: option expiry ``tau`` in year fractions (positive).
        forward: ATM forward (positive).
        alpha, beta, nu, rho: initial values; ``None`` (or the C++
            ``Null<Real>()`` sentinel :data:`NULL_REAL`) uses the
            :meth:`NoArbSabrSpecs.default_values` rule (beta=0.5, alpha
            nudged into the sigmaI band, nu=sqrt(0.4), rho=0). As in C++
            ``XABRCoeffHolder``, a null parameter is forced free even if
            its ``*_is_fixed`` flag is set.
        alpha_is_fixed .. rho_is_fixed: pin a parameter during the fit.
        vega_weighted: vega-weight residuals (Black vega wrt std dev,
            :meth:`NoArbSabrSpecs.weight`).
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

        self._strikes: np.ndarray = np.ascontiguousarray(strikes, dtype=np.float64)
        self._volatilities: np.ndarray = np.ascontiguousarray(
            volatilities, dtype=np.float64
        )
        self._expiry_time: float = expiry_time
        self._forward: float = forward
        self._vega_weighted: bool = vega_weighted

        # C++ parity: ``XABRCoeffHolder`` (xabrinterpolation.hpp:70-74) only
        # honours a ``paramIsFixed`` flag when the parameter is NOT null.
        raw: list[float] = [as_null(alpha), as_null(beta), as_null(nu), as_null(rho)]
        requested_fixed = [alpha_is_fixed, beta_is_fixed, nu_is_fixed, rho_is_fixed]
        self._is_fixed: list[bool] = [
            requested_fixed[i] and raw[i] != NULL_REAL for i in range(4)
        ]
        params = list(raw)
        NoArbSabrSpecs().default_values(params, self._is_fixed, forward, expiry_time, ())
        self._initial: list[float] = params

        self._alpha: float = params[0]
        self._beta: float = params[1]
        self._nu: float = params[2]
        self._rho: float = params[3]
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

    def _weights(self) -> np.ndarray:
        """The normalised residual weights, exactly as C++ builds them.

        # C++ parity: ``XABRInterpolationImpl::update`` (xabrinterpolation.hpp:
        # 132-159). Non-vega-weighted the vector is flat ``1/n``; vega-weighted
        # it is :meth:`NoArbSabrSpecs.weight` normalised to sum 1.
        """
        n = len(self._strikes)
        if not self._vega_weighted:
            return np.full(n, 1.0 / n, dtype=np.float64)
        specs = NoArbSabrSpecs()
        weights = np.empty(n, dtype=np.float64)
        for i in range(n):
            vol = float(self._volatilities[i])
            std_dev = math.sqrt(vol * vol * self._expiry_time)
            weights[i] = specs.weight(float(self._strikes[i]), self._forward, std_dev, ())
        return weights / float(np.sum(weights))

    def interpolation_weights(self) -> np.ndarray:
        """The normalised per-strike residual weights.

        # C++ parity: ``NoArbSabrInterpolation::interpolationWeights``
        # (noarbsabrinterpolation.hpp:229-231).
        """
        return self._weights()

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

    def _raw_residuals(self, params: list[float]) -> np.ndarray:
        """Unweighted ``model(k_i) - market_i``. C++ ``value(*x) - *y``."""
        return self._model_vols(params) - self._volatilities

    def _residuals(self, free_params: np.ndarray) -> np.ndarray:
        params = list(self._initial)
        j = 0
        for i, fixed in enumerate(self._is_fixed):
            if not fixed:
                params[i] = float(free_params[j])
                j += 1
        r = self._raw_residuals(params)
        if self._vega_weighted:
            # C++ ``interpolationErrors`` scales each residual by sqrt(w).
            r = r * np.sqrt(self._weights())
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
            # C++ parity: "there is nothing to optimize" branch
            # (xabrinterpolation.hpp:161-169) — error_/maxError_ are still
            # evaluated, and XABREndCriteria_ stays EndCriteria::None.
            self._store_params(list(self._initial))
            self._update_diagnostics()
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
        self._update_diagnostics()

    def _store_params(self, params: list[float]) -> None:
        self._alpha, self._beta, self._nu, self._rho = params

    def _update_diagnostics(self) -> None:
        """Recompute ``error_`` / ``maxError_`` at the stored parameters.

        # C++ parity: ``interpolationError`` / ``interpolationMaxError``
        # (xabrinterpolation.hpp:246-262). Both start from the UNWEIGHTED
        # residual; only ``error_`` applies the weights, and it divides by
        # ``n - 1``. ``maxError_`` never sees the weights.
        """
        residuals = self._raw_residuals([self._alpha, self._beta, self._nu, self._rho])
        if residuals.size == 0:
            self._rms_error = 0.0
            self._max_error = 0.0
            return
        self._rms_error = xabr_interpolation_error(residuals, self._weights())
        self._max_error = float(np.max(np.abs(residuals)))

    def _fit_multi_start(  # noqa: PLR0915 — direct port of C++ guess+restart loop
        self, *, max_nfev: int, max_guesses: int, seed: int
    ) -> None:
        """Halton multi-start over the (sigmaI, beta, nu, rho) box.

        # C++ parity: ``XABRInterpolationImpl::calculate``
        # (xabrinterpolation.hpp:183-230) resamples from
        # ``HaltonRsg(freeParameters, 42)`` fed through
        # :meth:`NoArbSabrSpecs.guess`, which draws sigmaI / beta / nu / rho
        # uniformly from their admissible bands; the outer loop keeps the
        # best fit over ``maxGuesses``.
        #
        # C++ parity note — the restart SEQUENCE is NOT reproduced: see the
        # same note on :meth:`SviInterpolation._fit_multi_start`. PQuantLib's
        # ``HaltonRsg`` seeds its ``randomStart`` offsets from
        # ``numpy.random.default_rng`` where C++ uses
        # ``MersenneTwisterUniformRng``, so the draws differ from the first
        # element on. The band *geometry* below is the C++ one.
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
        """The weighted fit error.

        # C++ parity: ``NoArbSabrInterpolation::rmsError`` ->
        # ``coeffs().error_`` = ``interpolationError()``. Despite the name
        # this is ``sqrt(n * sum_i w_i e_i^2 / (n - 1))``, not
        # ``sqrt(mean(e^2))``.
        """
        return self._rms_error

    def max_error(self) -> float:
        """The largest UNWEIGHTED absolute residual.

        # C++ parity: ``NoArbSabrInterpolation::maxError`` ->
        # ``interpolationMaxError()``, which ignores the weights even when
        # ``vegaWeighted`` is on.
        """
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


class NoArbSabr:
    """No-arbitrage SABR interpolation factory and traits.

    # C++ parity: ``class NoArbSabr`` (noarbsabrinterpolation.hpp:241-285).

    Holds the fit configuration and stamps out a
    :class:`NoArbSabrInterpolation` per strike/vol slice. ``global_`` is the
    C++ ``static const bool global`` traits flag telling the
    interpolated-curve machinery this interpolation is fitted over all points
    at once rather than piecewise.

    Not to be confused with :class:`NoArbSabrModel` (``noarbsabr.hpp``),
    which is the Doust terminal-density model this interpolation calibrates;
    the C++ names really are ``NoArbSabr`` (factory) and ``NoArbSabrModel``
    (numerics).

    Note the C++ default for ``vega_weighted`` differs between ``NoArbSabr``
    (``false``) and ``NoArbSabrInterpolation`` (``true``); the factory
    default is reproduced here.

    Args:
        t: option expiry in year fractions.
        forward: ATM forward.
        alpha, beta, nu, rho: initial values. Pass :data:`NULL_REAL` (or
            ``None``) for "use the :meth:`NoArbSabrSpecs.default_values`
            rule", which is what the C++ ``Null<Real>()`` sentinel means.
        alpha_is_fixed .. rho_is_fixed: pin a parameter during the fit.
        vega_weighted: vega-weight the residuals.
        end_criteria / optimization_method / error_accept / use_max_error:
            C++ pass-throughs; PQuantLib's fitter is fixed at the scipy TRF
            arm, so they are accepted and unused (same treatment as
            :class:`Svi` / ``SABR``).
        max_guesses: multi-start restart count. C++ defaults to 50; each
            no-arb evaluation prices + integrates the terminal density, so
            that default is expensive by construction.
    """

    #: C++ ``static const bool global = true``.
    global_: Final[bool] = True

    def __init__(
        self,
        t: float,
        forward: float,
        alpha: float | None,
        beta: float | None,
        nu: float | None,
        rho: float | None,
        alpha_is_fixed: bool,
        beta_is_fixed: bool,
        nu_is_fixed: bool,
        rho_is_fixed: bool,
        vega_weighted: bool = False,
        end_criteria: Any = None,
        optimization_method: Any = None,
        error_accept: float = 0.0020,
        use_max_error: bool = False,
        max_guesses: int = 50,
    ) -> None:
        self._t: float = t
        self._forward: float = forward
        self._alpha: float | None = alpha
        self._beta: float | None = beta
        self._nu: float | None = nu
        self._rho: float | None = rho
        self._alpha_is_fixed: bool = alpha_is_fixed
        self._beta_is_fixed: bool = beta_is_fixed
        self._nu_is_fixed: bool = nu_is_fixed
        self._rho_is_fixed: bool = rho_is_fixed
        self._vega_weighted: bool = vega_weighted
        self._end_criteria: Any = end_criteria
        self._optimization_method: Any = optimization_method
        self._error_accept: float = error_accept
        self._use_max_error: bool = use_max_error
        self._max_guesses: int = max_guesses

    def interpolate(
        self, strikes: Sequence[float], volatilities: Sequence[float]
    ) -> NoArbSabrInterpolation:
        """Fit a :class:`NoArbSabrInterpolation` to one strike/vol slice.

        # C++ parity: ``NoArbSabr::interpolate``
        # (noarbsabrinterpolation.hpp:264-271).
        """
        return NoArbSabrInterpolation(
            strikes,
            volatilities,
            self._t,
            self._forward,
            alpha=self._alpha,
            beta=self._beta,
            nu=self._nu,
            rho=self._rho,
            alpha_is_fixed=self._alpha_is_fixed,
            beta_is_fixed=self._beta_is_fixed,
            nu_is_fixed=self._nu_is_fixed,
            rho_is_fixed=self._rho_is_fixed,
            vega_weighted=self._vega_weighted,
            max_guesses=self._max_guesses,
        )


def _free_index(is_fixed: list[bool], target: int) -> int:
    """Index of parameter ``target`` within the free-param subsequence."""
    j = 0
    for i in range(target):
        if not is_fixed[i]:
            j += 1
    return j


__all__ = ["NoArbSabr", "NoArbSabrInterpolation", "NoArbSabrSpecs"]
