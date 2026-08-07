"""SABRInterpolation — fits SABR (alpha, beta, nu, rho) to a strike-vol slice.

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
  re-initialisations from a Halton-style low-discrepancy sequence.
  PQuantLib mirrors this via the L10-A ``max_guesses`` kwarg (default
  1): when set above 1, additional starts are sampled from
  :class:`HaltonRsg` over the *free* (non-fixed) parameter subspace and
  the best-RMS fit is retained.
* **Vega weighting.** The C++ ``weight()`` callback calls
  ``blackFormulaStdDevDerivative`` per (strike, vol) pair. PQuantLib
  implements the equivalent weighting; see the inline note on the
  ``_vega_weights`` helper. For the Phase 9 tests we exercise both
  arms (vega-weighted on / off) and round-trip-recover the input vols.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.sabr_formula import (
    shifted_sabr_volatility,
    validate_sabr_parameters,
)
from pquantlib.pricingengines.black_formula import black_formula_std_dev_derivative
from pquantlib.termstructures.volatility.volatility_type import VolatilityType

# C++ ``Null<Real>()`` is ``std::numeric_limits<float>::max()`` — the sentinel
# SABRSpecs::defaultValues tests against (ql/utilities/null.hpp).
NULL_REAL: Final[float] = float(np.finfo(np.float32).max)


def as_null(value: float | None) -> float:
    """Map the Pythonic ``None`` onto the C++ ``Null<Real>()`` sentinel.

    # C++ parity: none — bridges Python's ``None`` onto the sentinel the XABR
    # ``*Specs`` classes test against (ql/utilities/null.hpp).
    """
    return NULL_REAL if value is None else float(value)


def xabr_interpolation_error(
    residuals: npt.NDArray[np.float64], weights: npt.NDArray[np.float64]
) -> float:
    """RMS-like fit error every XABR interpolation reports.

    # C++ parity: ``XABRInterpolationImpl::interpolationError``
    # (xabrinterpolation.hpp:246-251 @ v1.43)::
    #
    #     squaredError = sum_i w_i * e_i^2
    #     error        = sqrt(n * squaredError / (n == 1 ? 1 : n - 1))
    #
    # Note the ``n - 1`` denominator: this is NOT ``sqrt(mean(e^2))``. With
    # the default flat weights ``w_i = 1/n`` it collapses to
    # ``sqrt(sum(e^2) / (n - 1))``, which is larger than the plain RMS by
    # ``sqrt(n / (n - 1))``.
    """
    n = residuals.size
    if n == 0:
        return 0.0
    squared_error = float(np.sum(residuals * residuals * weights))
    return math.sqrt(n * squared_error / (1 if n == 1 else n - 1))


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


class SABRWrapper:
    """Bind (t, forward, params, shift) and evaluate the SABR smile.

    # C++ parity: ``class detail::SABRWrapper`` (sabrinterpolation.hpp:41-62).

    The model instance the XABR machinery holds while calibrating: it
    validates its inputs once at construction and then answers
    ``volatility(strike, volatility_type)`` from the closed form.

    Args:
        t: option expiry, in year fractions.
        forward: the ATM forward.
        params: ``[alpha, beta, nu, rho]``.
        add_params: model extras; ``add_params[0]`` is the shift when
            present, otherwise the shift is 0.
    """

    __slots__ = ("_forward", "_params", "_shift", "_t")

    def __init__(
        self,
        t: float,
        forward: float,
        params: Sequence[float],
        add_params: Sequence[float] = (),
    ) -> None:
        self._t: float = t
        self._forward: float = forward
        self._params: list[float] = list(params)
        self._shift: float = 0.0 if len(add_params) == 0 else float(add_params[0])
        qassert.require(
            self._forward + self._shift > 0.0,
            f"forward+shift must be positive: {self._forward} with shift "
            f"{self._shift} not allowed",
        )
        validate_sabr_parameters(
            self._params[0], self._params[1], self._params[2], self._params[3]
        )

    def volatility(self, x: float, volatility_type: VolatilityType) -> float:
        """SABR volatility at strike ``x``.

        # C++ parity: sabrinterpolation.hpp:53-56.
        """
        return shifted_sabr_volatility(
            x,
            self._forward,
            self._t,
            self._params[0],
            self._params[1],
            self._params[2],
            self._params[3],
            self._shift,
            volatility_type,
        )


class SABRSpecs:
    """The SABR model policy the C++ XABR template is instantiated with.

    # C++ parity: ``struct detail::SABRSpecs`` (sabrinterpolation.hpp:64-142).

    ``XABRInterpolation<Model>`` is generic over a "specs" type supplying the
    parameter count, the default/initial values, the residual weights and —
    the substantive part — a bijection between the constrained parameter box
    and unconstrained R^4, so that an *unconstrained* optimiser can be used.
    :meth:`direct` maps R^4 into the box, :meth:`inverse` maps back.

    PQuantLib's :class:`SabrInterpolation` does not currently route its
    optimisation through this reparameterisation (it uses scipy's native box
    constraints instead — see the module docstring), so these methods exist
    as a faithful, cross-validated transcription rather than as the live
    calibration path. They are what any port of the C++ optimiser arm needs.

    Every method is stateless; the class is instantiable to mirror the C++
    call sites (``Model().direct(...)``).
    """

    __slots__ = ()

    def dimension(self) -> int:
        """Number of model parameters (4). C++ ``dimension``."""
        return 4

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
        param_is_fixed: list[bool],
        forward: float,
        expiry_time: float,
        add_params: Sequence[float],
    ) -> None:
        """Fill any ``NULL_REAL`` slot of ``params`` in place.

        # C++ parity: ``defaultValues`` (sabrinterpolation.hpp:66-81).

        Order matters: beta is defaulted first because alpha's default reads
        it. ``expiry_time`` and ``param_is_fixed`` are accepted and ignored,
        as in C++.
        """
        del param_is_fixed, expiry_time
        shift = 0.0 if len(add_params) == 0 else float(add_params[0])
        if params[1] == NULL_REAL:
            params[1] = 0.5
        if params[0] == NULL_REAL:
            # adapt alpha to beta level
            params[0] = 0.2 * (
                math.pow(forward + shift, 1.0 - params[1]) if params[1] < 0.9999 else 1.0
            )
        if params[2] == NULL_REAL:
            params[2] = math.sqrt(0.4)
        if params[3] == NULL_REAL:
            params[3] = 0.0

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

        # C++ parity: ``guess`` (sabrinterpolation.hpp:82-99).

        ``r`` is consumed by a single running index in the order
        beta, alpha, nu, rho — *not* the parameter order — and a fixed
        parameter consumes nothing. Getting that order wrong silently
        reshuffles a multi-start search.
        """
        del expiry_time
        shift = 0.0 if len(add_params) == 0 else float(add_params[0])
        j = 0
        if not param_is_fixed[1]:
            values[1] = (1.0 - 2e-6) * r[j] + 1e-6
            j += 1
        if not param_is_fixed[0]:
            values[0] = (1.0 - 2e-6) * r[j] + 1e-6  # lognormal vol guess
            j += 1
            # adapt this to beta level
            if values[1] < 0.999:
                values[0] *= math.pow(forward + shift, 1.0 - float(values[1]))
        if not param_is_fixed[2]:
            values[2] = 1.5 * r[j] + 1e-6
            j += 1
        if not param_is_fixed[3]:
            values[3] = (2.0 * r[j] - 1.0) * (1.0 - 1e-6)

    def inverse(
        self,
        y: Array,
        param_is_fixed: Sequence[bool],
        params: Sequence[float],
        forward: float,
    ) -> Array:
        """Constrained ``y`` -> unconstrained ``x``.

        # C++ parity: ``inverse`` (sabrinterpolation.hpp:103-114).

        Inverse of :meth:`direct`. The alpha/nu branches switch to a linear
        arm above ``25 + eps1`` so the map stays well conditioned far out;
        beta goes through ``sqrt(-log(.))`` and rho through ``asin(./eps2)``.
        """
        del param_is_fixed, params, forward
        eps1 = self.eps1()
        x = np.zeros(4, dtype=np.float64)
        y0 = float(y[0])
        x[0] = math.sqrt(y0 - eps1) if y0 < 25.0 + eps1 else (y0 - eps1 + 25.0) / 10.0
        # C++ keeps the commented-out arctan alternative for beta; the live
        # branch is the sqrt(-log(.)) one.
        x[1] = math.sqrt(-math.log(float(y[1])))
        y2 = float(y[2])
        x[2] = math.sqrt(y2 - eps1) if y2 < 25.0 + eps1 else (y2 - eps1 + 25.0) / 10.0
        x[3] = math.asin(float(y[3]) / self.eps2())
        return x

    def direct(
        self,
        x: Array,
        param_is_fixed: Sequence[bool],
        params: Sequence[float],
        forward: float,
    ) -> Array:
        """Unconstrained ``x`` -> constrained ``y``.

        # C++ parity: ``direct`` (sabrinterpolation.hpp:115-130).

        Note the saturating arms: beyond ``|x| = 5`` alpha and nu grow
        linearly rather than quadratically, beta collapses to ``eps1`` once
        ``|x1|`` reaches ``sqrt(-log(eps1))``, and rho saturates at
        ``+/- eps2`` beyond ``|x3| = 2.5 pi``. Those are the branches a port
        is tempted to drop as unreachable; the optimiser does reach them.
        """
        del param_is_fixed, params, forward
        eps1 = self.eps1()
        eps2 = self.eps2()
        y = np.zeros(4, dtype=np.float64)
        x0 = float(x[0])
        y[0] = x0 * x0 + eps1 if abs(x0) < 5.0 else (10.0 * abs(x0) - 25.0) + eps1
        x1 = float(x[1])
        y[1] = math.exp(-(x1 * x1)) if abs(x1) < math.sqrt(-math.log(eps1)) else eps1
        x2 = float(x[2])
        y[2] = x2 * x2 + eps1 if abs(x2) < 5.0 else (10.0 * abs(x2) - 25.0) + eps1
        x3 = float(x[3])
        y[3] = (
            eps2 * math.sin(x3)
            if abs(x3) < 2.5 * math.pi
            else eps2 * (1.0 if x3 > 0.0 else -1.0)
        )
        return y

    def weight(
        self,
        strike: float,
        forward: float,
        std_dev: float,
        add_params: Sequence[float],
    ) -> float:
        """Per-strike residual weight — the Black vega wrt std dev.

        # C++ parity: ``weight`` (sabrinterpolation.hpp:131-135). Note C++
        # indexes ``addParams[0]`` unconditionally here, so a caller that
        # reaches ``weight`` must supply the shift.
        """
        return black_formula_std_dev_derivative(
            strike, forward, std_dev, 1.0, float(add_params[0])
        )

    def instance(
        self,
        t: float,
        forward: float,
        params: Sequence[float],
        add_params: Sequence[float],
    ) -> SABRWrapper:
        """Build the bound model. C++ ``instance`` / ``typedef SABRWrapper type``."""
        return SABRWrapper(t, forward, params, add_params)


class SABRInterpolation:
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
        alpha_is_fixed: bool = False,
        beta_is_fixed: bool = False,
        nu_is_fixed: bool = False,
        rho_is_fixed: bool = False,
        vega_weighted: bool = False,
        shift: float = 0.0,
        volatility_type: VolatilityType = VolatilityType.ShiftedLognormal,
        max_nfev: int = 1000,
        max_guesses: int = 1,
        multi_start_seed: int = 42,
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

    def _weights(self) -> np.ndarray:
        """The normalised residual weights, exactly as C++ builds them.

        # C++ parity: ``XABRInterpolationImpl`` constructor
        # (xabrinterpolation.hpp:178-179) seeds ``weights_`` with a FLAT
        # ``1/n``; ``update()`` (hpp:142-159) replaces it with
        # ``SABRSpecs::weight`` normalised to sum 1 when ``vegaWeighted``.
        #
        # The flat 1/n is not cosmetic: ``interpolationError`` multiplies by
        # it, so a port that treats the unweighted case as "no weights"
        # reports an error larger by sqrt(n).
        """
        n = len(self._strikes)
        if not self._vega_weighted:
            return np.full(n, 1.0 / n, dtype=np.float64)
        # _vega_weights() returns sqrt(w); square it back.
        sqrt_w = self._vega_weights()
        return sqrt_w * sqrt_w

    def _raw_residuals(
        self, params: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Unweighted ``model(k_i) - market_i``. C++ ``value(*x) - *y``."""
        alpha, beta, nu, rho = params
        model_vols = np.empty_like(self._strikes)
        for i, k in enumerate(self._strikes):
            model_vols[i] = shifted_sabr_volatility(
                float(k), self._forward, self._expiry_time,
                alpha, beta, nu, rho, self._shift, self._volatility_type,
            )
        return model_vols - self._volatilities

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
            # Everything fixed — C++ short-circuits identically
            # (xabrinterpolation.hpp:161-168 "there is nothing to optimize").
            self._alpha, self._beta, self._nu, self._rho = self._initial
            self._update_diagnostics()
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
        self._update_diagnostics()

    def _update_diagnostics(self) -> None:
        """Recompute ``error_`` / ``maxError_`` at the stored parameters.

        # C++ parity: ``XABRInterpolationImpl::interpolationError`` /
        # ``interpolationMaxError`` (xabrinterpolation.hpp:270-285), which
        # ``calculate()`` assigns to ``error_`` / ``maxError_`` on BOTH exits
        # (hpp:166-167 for the nothing-to-optimise branch, hpp:233-234 after
        # the fit). ``SABRInterpolation::rmsError()`` / ``maxError()`` are
        # just those two members (sabrinterpolation.hpp:183-184).
        #
        # Both start from the UNWEIGHTED residual; only the RMS applies the
        # weights, and it divides by ``n - 1``, not ``n``. This module already
        # carried the correct :func:`xabr_interpolation_error` helper — it was
        # used by no_arb_sabr_interpolation.py and svi_interpolation.py but
        # never by this class, which computed ``sqrt(mean(r^2))`` off the
        # scipy residual vector instead. Pinned against C++ by
        # tests/math/interpolations/test_abcd_error_formula.py.
        """
        residuals = self._raw_residuals(
            (self._alpha, self._beta, self._nu, self._rho)
        )
        if residuals.size == 0:
            self._rms_error = 0.0
            self._max_error = 0.0
            return
        self._rms_error = xabr_interpolation_error(residuals, self._weights())
        self._max_error = float(np.max(np.abs(residuals)))

    def _fit_multi_start(self, *, max_nfev: int, max_guesses: int, seed: int) -> None:
        """Halton multi-start: run ``max_guesses`` fits, keep best RMS.

        # C++ parity: ``XABRCoeffHolder::reset()`` + the outer loop in
        # ``XABRInterpolationImpl::calculate`` that resamples and
        # restarts the optimisation up to ``maxGuesses`` times. The
        # underlying initial-guess sampler in C++ is a Halton-style
        # low-discrepancy sequence over the unconstrained R^4 space;
        # PQuantLib draws from the same low-discrepancy generator (via
        # :class:`HaltonRsg`) but maps each sample directly into the
        # bound-constrained search box used by ``trf``.
        """
        from pquantlib.math.randomnumbers.halton import HaltonRsg  # noqa: PLC0415

        # Free-axis bounding box (per param):
        #   alpha ∈ [eps, 0.5]   beta ∈ [eps, 1-eps]
        #   nu    ∈ [eps, 10]    rho  ∈ [-eps2, eps2]
        # We map a unit-cube Halton sample u ∈ [0,1)^free_dim into
        # this box per axis. Beta is intentionally bracketed below 1
        # to avoid the singular β=1 limit; alpha gets a generous
        # upper bound (0.5 covers reasonable atm vol ranges); nu's
        # 10 upper bound mirrors C++ ``QL_MAX_REAL``-clamped resamples
        # in practice (typical fits hit nu in (0.1, 5)).
        axes: list[tuple[float, float]] = []
        if not self._is_fixed[0]:
            axes.append((_EPS1, 0.5))
        if not self._is_fixed[1]:
            axes.append((_EPS1, 1.0 - _EPS1))
        if not self._is_fixed[2]:
            axes.append((_EPS1, 10.0))
        if not self._is_fixed[3]:
            axes.append((-_EPS2, _EPS2))

        # First pass: the user-supplied (or default-rule) initial.
        self._fit(max_nfev=max_nfev)
        best_rms = self._rms_error
        best_params = (self._alpha, self._beta, self._nu, self._rho)
        best_converged = self._converged

        if not axes:
            return

        rsg = HaltonRsg(
            dimensionality=len(axes), seed=seed, random_start=True, random_shift=False,
        )

        for _ in range(max_guesses - 1):
            sample = rsg.next_sequence().value
            # Map sample to per-axis ranges + write back into the
            # full 4-vector, leaving fixed params alone.
            params = list(self._initial)
            j = 0
            for i, fixed in enumerate(self._is_fixed):
                if fixed:
                    continue
                lo, hi = axes[j]
                params[i] = lo + (hi - lo) * float(sample[j])
                j += 1
            # Reset the initial-guess buffer used by ``_fit`` then
            # re-run the optimisation.
            self._initial = params
            try:
                self._fit(max_nfev=max_nfev)
            except Exception:
                # Bad initial; skip this restart.
                continue
            if self._rms_error < best_rms:
                best_rms = self._rms_error
                best_params = (self._alpha, self._beta, self._nu, self._rho)
                best_converged = self._converged

        # Restore the best-found fit.
        self._alpha, self._beta, self._nu, self._rho = best_params
        self._rms_error = best_rms
        self._converged = best_converged
        # Refresh max_error from best_params.
        self._update_diagnostics()

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


# The class carries the C++ spelling, ``SABRInterpolation``. PQuantLib
# originally landed it as ``SabrInterpolation`` and a good deal of the library
# imports that name, so it stays as the alias.
SabrInterpolation = SABRInterpolation


class SABR:
    """SABR interpolation factory and traits.

    # C++ parity: ``class SABR`` (sabrinterpolation.hpp:198-245).

    Holds the fit configuration and stamps out a
    :class:`SabrInterpolation` per strike/vol slice, so a caller can
    configure the smile model once and hand the factory to something that
    interpolates many slices. ``global`` is the C++ static traits flag that
    tells the interpolated-curve machinery this interpolation is fitted over
    all points at once rather than piecewise.

    Note the C++ default for ``vega_weighted`` differs between ``SABR``
    (``false``) and ``SABRInterpolation`` (``true``); the factory default is
    reproduced here.
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
        shift: float = 0.0,
    ) -> None:
        self._t: float = t
        self._forward: float = forward
        self._alpha: float = alpha
        self._beta: float = beta
        self._nu: float = nu
        self._rho: float = rho
        self._alpha_is_fixed: bool = alpha_is_fixed
        self._beta_is_fixed: bool = beta_is_fixed
        self._nu_is_fixed: bool = nu_is_fixed
        self._rho_is_fixed: bool = rho_is_fixed
        self._vega_weighted: bool = vega_weighted
        # C++ takes ext::shared_ptr<EndCriteria> / <OptimizationMethod>
        # pass-throughs; PQuantLib's fitter is fixed at the scipy TRF arm, so
        # they are accepted and unused (same treatment as AbcdInterpolation).
        self._end_criteria: Any = end_criteria
        self._optimization_method: Any = optimization_method
        self._error_accept: float = error_accept
        self._use_max_error: bool = use_max_error
        self._max_guesses: int = max_guesses
        self._shift: float = shift

    def interpolate(
        self, strikes: Sequence[float], volatilities: Sequence[float]
    ) -> SabrInterpolation:
        """Fit a :class:`SabrInterpolation` to one strike/vol slice.

        # C++ parity: ``SABR::interpolate`` (sabrinterpolation.hpp:222-230).
        """
        return SabrInterpolation(
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
            shift=self._shift,
            max_guesses=self._max_guesses,
        )


__all__ = [
    "NULL_REAL",
    "SABR",
    "SABRInterpolation",
    "SABRSpecs",
    "SABRWrapper",
    "SabrInterpolation",
    "as_null",
    "xabr_interpolation_error",
]
