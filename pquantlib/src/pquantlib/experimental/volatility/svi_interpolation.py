"""SVI (Stochastic-Volatility-Inspired) raw parameterization + fit.

# C++ parity: ql/experimental/volatility/sviinterpolation.hpp (v1.43).

Gatheral's raw SVI total-variance slice:

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

where ``k = log(strike / forward)`` is the log-moneyness and ``w`` is
the total implied variance. The Black (lognormal) volatility is then
``sqrt(max(0, w / tau))``.

:func:`svi_volatility` evaluates the raw slice directly (no fitting).
:class:`SviSpecs` is the model policy the C++ XABR template is
instantiated with. :class:`SviInterpolation` fits the 5 SVI parameters
``(a, b, sigma, rho, m)`` to a market strike-vol slice via
``scipy.optimize.least_squares`` — the same delegation pattern as the
L9-C :class:`SabrInterpolation` (see that module's docstring for the
full optimizer-divergence rationale). :class:`Svi` is the interpolation
factory/traits class. The SVI no-arbitrage parameter constraints from
``detail::checkSviParameters`` are enforced as box bounds + an explicit
feasibility check.

Documented divergences from C++:

* **Optimizer.** C++ runs ``LevenbergMarquardt(1e-8, 1e-8, 1e-8)`` in
  the *unconstrained* coordinates produced by
  :meth:`SviSpecs.inverse`, mapping back through :meth:`SviSpecs.direct`
  on every cost evaluation (which is what makes every visited point
  feasible by construction). PQuantLib runs
  ``least_squares(method='trf')`` on the constrained parameters with
  native box bounds. Cross-validated at the *fitted curve* level
  against the C++ probe (``B3``/``B4`` of
  ``references/v143/experimental/volatility.json``): on a noiseless
  slice both solvers land on the same global optimum.
* **Multi-start sampling.** C++ resamples restart points from
  ``HaltonRsg(freeParameters, 42)`` fed through
  :meth:`SviSpecs.guess`. PQuantLib's :class:`HaltonRsg` reproduces the
  deterministic Halton core but *not* the C++ random-start offsets
  (C++ seeds them from ``MersenneTwisterUniformRng``, PQuantLib from
  ``numpy.random.default_rng``), so the restart *sequence* cannot be
  reproduced; PQuantLib samples its own data-derived box instead.
  :meth:`SviSpecs.guess` itself is cross-validated against the C++
  Halton draws recorded in the reference.
* **Error metric.** ``rms_error`` / ``max_error`` follow the C++
  ``XABRInterpolationImpl`` definitions — see :meth:`SviInterpolation.rms_error`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.interpolations.sabr_interpolation import (
    NULL_REAL,
    as_null,
    xabr_interpolation_error,
)
from pquantlib.pricingengines.black_formula import black_formula_std_dev_derivative

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime
    from pquantlib.experimental.volatility.svi_smile_section import SviSmileSection

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


class SviSpecs:
    """The SVI model policy the C++ XABR template is instantiated with.

    # C++ parity: ``struct detail::SviSpecs`` (sviinterpolation.hpp:57-140).

    ``XABRInterpolationImpl<Model>`` is generic over a "specs" type supplying
    the parameter count, the default/initial values, the multi-start guess
    rule, the residual weights and — the substantive part — a bijection
    between the constrained SVI box and unconstrained R^5, so that an
    *unconstrained* optimiser can be used. :meth:`direct` maps R^5 into the
    box, :meth:`inverse` maps back.

    The bijection is load-bearing, not decorative: for any ``x`` in R^5,
    :meth:`direct` yields ``sigma > 0``, ``|rho| < 1``,
    ``b (1 + |rho|) < 4`` and ``a + b sigma sqrt(1 - rho^2) = eps1 + x0^2 > 0``
    — i.e. every point the C++ optimiser can visit already satisfies
    :func:`check_svi_parameters`.

    PQuantLib's :class:`SviInterpolation` does not currently route its
    optimisation through this reparameterisation (it uses scipy's native box
    constraints instead — see the module docstring), so these methods exist
    as a faithful, cross-validated transcription rather than as the live
    calibration path. They are what any port of the C++ optimiser arm needs.

    Every method is stateless; the class is instantiable to mirror the C++
    call sites (``Model().direct(...)``).
    """

    __slots__ = ()

    def dimension(self) -> int:
        """Number of model parameters (5). C++ ``dimension``."""
        return 5

    def eps1(self) -> float:
        """Lower clamp used by :meth:`direct` / :meth:`inverse`. C++ ``eps1``."""
        return 0.000001

    def eps2(self) -> float:
        """Rho / b saturation level. C++ ``eps2``."""
        return 0.999999

    def default_values(
        self,
        params: list[float],
        param_is_fixed: list[bool],
        forward: float,
        expiry_time: float,
        add_params: Sequence[float],
    ) -> None:
        """Fill any ``NULL_REAL`` slot of ``params`` in place.

        # C++ parity: ``defaultValues`` (sviinterpolation.hpp:59-80).

        Order matters and is *not* the parameter order: sigma, rho and m are
        defaulted first because ``b``'s default reads rho, and ``a``'s default
        reads b, sigma, rho and m. ``forward``, ``param_is_fixed`` and
        ``add_params`` are accepted and ignored, as in C++.
        """
        del param_is_fixed, forward, add_params
        if params[2] == NULL_REAL:
            params[2] = 0.1
        if params[3] == NULL_REAL:
            params[3] = -0.4
        if params[4] == NULL_REAL:
            params[4] = 0.0
        if params[1] == NULL_REAL:
            params[1] = 2.0 / (1.0 + abs(params[3]))
        if params[0] == NULL_REAL:
            params[0] = max(
                0.20 * 0.20 * expiry_time
                - params[1]
                * (
                    params[3] * (-params[4])
                    + math.sqrt((-params[4]) * (-params[4]) + params[2] * params[2])
                ),
                -params[1] * params[2] * math.sqrt(1.0 - params[3] * params[3]) + self.eps1(),
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

        # C++ parity: ``guess`` (sviinterpolation.hpp:81-97).

        ``r`` is consumed by a single running index in the order
        sigma, rho, m, b, a — *not* the parameter order — and a fixed
        parameter consumes nothing. Getting that order wrong silently
        reshuffles a multi-start search. ``b`` reads the freshly drawn rho
        and ``a`` reads the freshly drawn b/sigma/rho, so the write order is
        load-bearing too.
        """
        del forward, add_params
        j = 0
        if not param_is_fixed[2]:
            values[2] = r[j] + self.eps1()
            j += 1
        if not param_is_fixed[3]:
            values[3] = (2.0 * r[j] - 1.0) * self.eps2()
            j += 1
        if not param_is_fixed[4]:
            values[4] = 2.0 * r[j] - 1.0
            j += 1
        if not param_is_fixed[1]:
            values[1] = r[j] * 4.0 / (1.0 + abs(float(values[3]))) * self.eps2()
            j += 1
        if not param_is_fixed[0]:
            values[0] = r[j] * expiry_time - self.eps2() * (
                float(values[1])
                * float(values[2])
                * math.sqrt(1.0 - float(values[3]) * float(values[3]))
            )

    def inverse(
        self,
        y: Array,
        param_is_fixed: Sequence[bool],
        params: Sequence[float],
        forward: float,
    ) -> Array:
        """Constrained ``y`` -> unconstrained ``x``.

        # C++ parity: ``inverse`` (sviinterpolation.hpp:98-109).

        Inverse of :meth:`direct`. Note C++ ignores ``paramIsFixed`` here —
        the fixed slots are inverted anyway and then discarded by the
        projection, so this is *not* a left inverse of ``direct`` when
        anything is fixed.
        """
        del param_is_fixed, params, forward
        eps1 = self.eps1()
        eps2 = self.eps2()
        x = np.zeros(5, dtype=np.float64)
        y0, y1, y2, y3, y4 = (float(y[i]) for i in range(5))
        x[2] = math.sqrt(y2 - eps1)
        x[3] = math.asin(y3 / eps2)
        x[4] = y4
        x[1] = math.tan(y1 / 4.0 * (1.0 + abs(y3)) / eps2 * math.pi - math.pi / 2.0)
        x[0] = math.sqrt(y0 - eps1 + y1 * y2 * math.sqrt(1.0 - y3 * y3))
        return x

    def direct(
        self,
        x: Array,
        param_is_fixed: Sequence[bool],
        params: Sequence[float],
        forward: float,
    ) -> Array:
        """Unconstrained ``x`` -> constrained ``y``.

        # C++ parity: ``direct`` (sviinterpolation.hpp:112-129).

        The ``paramIsFixed`` arms for ``a`` and ``b`` copy ``params[k]``
        straight through and short-circuit the transform — and ``a``'s
        transform reads the *resulting* ``y[1]``, so pinning ``b`` changes
        ``a``'s value for the same ``x``.
        """
        del forward
        eps1 = self.eps1()
        eps2 = self.eps2()
        y = np.zeros(5, dtype=np.float64)
        x0, x1, x2, x3, x4 = (float(x[i]) for i in range(5))
        y[2] = x2 * x2 + eps1
        y[3] = math.sin(x3) * eps2
        y[4] = x4
        y[1] = (
            params[1]
            if param_is_fixed[1]
            else (math.atan(x1) + math.pi / 2.0)
            / math.pi
            * eps2
            * 4.0
            / (1.0 + abs(float(y[3])))
        )
        y[0] = (
            params[0]
            if param_is_fixed[0]
            else eps1
            + x0 * x0
            - float(y[1]) * float(y[2]) * math.sqrt(1.0 - float(y[3]) * float(y[3]))
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

        # C++ parity: ``weight`` (sviinterpolation.hpp:130-133). Unlike
        # ``SABRSpecs::weight`` this passes no displacement, so ``addParams``
        # is unused; SVI has no shift.
        """
        del add_params
        return black_formula_std_dev_derivative(strike, forward, std_dev, 1.0)

    def instance(
        self,
        t: float,
        forward: float,
        params: Sequence[float],
        add_params: Sequence[float],
    ) -> SviSmileSection:
        """Build the bound model.

        # C++ parity: ``instance`` / ``typedef SviWrapper type``
        # (sviinterpolation.hpp:55, 134-139) where
        # ``typedef SviSmileSection SviWrapper``.
        """
        del add_params
        from pquantlib.experimental.volatility.svi_smile_section import (  # noqa: PLC0415
            SviSmileSection,
        )

        a, b, sigma, rho, m = (float(v) for v in params)
        return SviSmileSection(
            forward=forward, svi_params=(a, b, sigma, rho, m), exercise_time=t
        )


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
        a, b, sigma, rho, m: SVI initial values; ``None`` (or the C++
            ``Null<Real>()`` sentinel :data:`NULL_REAL`) uses the C++
            :meth:`SviSpecs.default_values` rule. As in C++
            ``XABRCoeffHolder``, a null parameter is forced free even if
            its ``*_is_fixed`` flag is set.
        a_is_fixed .. m_is_fixed: pin a parameter at its initial value.
        vega_weighted: vega-weight residuals (Black vega wrt std dev,
            :meth:`SviSpecs.weight`).
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

        # C++ parity: ``XABRCoeffHolder`` (xabrinterpolation.hpp:70-74) only
        # honours a ``paramIsFixed`` flag when the corresponding parameter is
        # NOT null — a null parameter is always optimised.
        raw: list[float] = [as_null(a), as_null(b), as_null(sigma), as_null(rho), as_null(m)]
        requested_fixed = [a_is_fixed, b_is_fixed, sigma_is_fixed, rho_is_fixed, m_is_fixed]
        self._is_fixed: list[bool] = [
            requested_fixed[i] and raw[i] != NULL_REAL for i in range(5)
        ]
        params = list(raw)
        SviSpecs().default_values(params, self._is_fixed, forward, expiry_time, ())
        self._initial: list[float] = params

        self._a: float = params[0]
        self._b: float = params[1]
        self._sigma: float = params[2]
        self._rho: float = params[3]
        self._m: float = params[4]
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

    def _weights(self) -> np.ndarray:
        """The normalised residual weights, exactly as C++ builds them.

        # C++ parity: ``XABRInterpolationImpl::update`` (xabrinterpolation.hpp:
        # 132-159). Non-vega-weighted the vector is flat ``1/n``; vega-weighted
        # it is ``SviSpecs::weight(strike, forward, sqrt(vol^2 * t))``
        # normalised to sum 1. C++ divides by the raw sum with no guard, and
        # so do we.
        """
        n = len(self._strikes)
        if not self._vega_weighted:
            return np.full(n, 1.0 / n, dtype=np.float64)
        specs = SviSpecs()
        weights = np.empty(n, dtype=np.float64)
        for i in range(n):
            vol = float(self._volatilities[i])
            std_dev = math.sqrt(vol * vol * self._expiry_time)
            weights[i] = specs.weight(float(self._strikes[i]), self._forward, std_dev, ())
        return weights / float(np.sum(weights))

    def interpolation_weights(self) -> np.ndarray:
        """The normalised per-strike residual weights.

        # C++ parity: ``SviInterpolation::interpolationWeights``
        # (sviinterpolation.hpp:179-181).
        """
        return self._weights()

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
        self._update_diagnostics()

    def _store_params(self, params: list[float]) -> None:
        self._a, self._b, self._sigma, self._rho, self._m = params

    def _update_diagnostics(self) -> None:
        """Recompute ``error_`` / ``maxError_`` at the stored parameters.

        # C++ parity: ``interpolationError`` / ``interpolationMaxError``
        # (xabrinterpolation.hpp:246-262). Both are evaluated from the
        # UNWEIGHTED residual; only ``error_`` then applies the weights, and
        # it divides by ``n - 1``. ``maxError_`` never sees the weights.
        """
        residuals = self._raw_residuals(
            [self._a, self._b, self._sigma, self._rho, self._m]
        )
        if residuals.size == 0:
            self._rms_error = 0.0
            self._max_error = 0.0
            return
        self._rms_error = xabr_interpolation_error(residuals, self._weights())
        self._max_error = float(np.max(np.abs(residuals)))

    def _fit_multi_start(self, *, max_nfev: int, max_guesses: int, seed: int) -> None:
        """Halton multi-start: run ``max_guesses`` fits, keep best RMS.

        # C++ parity: ``XABRInterpolationImpl::calculate`` outer loop
        # (xabrinterpolation.hpp:183-230) resamples the initial guess from
        # ``HaltonRsg(freeParameters, 42)`` fed through
        # :meth:`SviSpecs.guess`, restarts the optimiser up to
        # ``maxGuesses`` times, and retains the best fit. The raw-SVI slice
        # is multi-modal so this is load-bearing (a single start from the
        # ``defaultValues`` point frequently lands in a bad local min).
        #
        # C++ parity note — the restart SEQUENCE is NOT reproduced.
        # PQuantLib's :class:`HaltonRsg` matches C++ on the deterministic
        # Halton core but not on the ``randomStart`` offsets: C++ draws them
        # from ``RandomSequenceGenerator<MersenneTwisterUniformRng>``
        # (haltonrsg.cpp:46-53) while PQuantLib draws them from
        # ``numpy.random.default_rng`` (PCG64). Verified against the C++
        # probe: for ``HaltonRsg(4, 42)`` C++ emits
        # 0.8993458733893931 first, PQuantLib 0.06794139184057713.
        # Until that is fixed the C++ restart points are unreachable, so
        # this loop samples its own data-derived box instead. What IS
        # cross-validated is :meth:`SviSpecs.guess` applied to the C++
        # Halton draws recorded in the reference.
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
        """The weighted fit error.

        # C++ parity: ``SviInterpolation::rmsError`` -> ``coeffs().error_``
        # = ``interpolationError()``. Despite the name this is
        # ``sqrt(n * sum_i w_i e_i^2 / (n - 1))``, not ``sqrt(mean(e^2))``.
        """
        return self._rms_error

    def max_error(self) -> float:
        """The largest UNWEIGHTED absolute residual.

        # C++ parity: ``SviInterpolation::maxError`` ->
        # ``interpolationMaxError()`` (xabrinterpolation.hpp:253-262), which
        # ignores the weights even when ``vegaWeighted`` is on.
        """
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


class Svi:
    """Svi interpolation factory and traits.

    # C++ parity: ``class Svi`` (sviinterpolation.hpp:191-238).

    Holds the fit configuration and stamps out a :class:`SviInterpolation`
    per strike/vol slice, so a caller can configure the smile model once and
    hand the factory to something that interpolates many slices. ``global_``
    is the C++ ``static const bool global`` traits flag telling the
    interpolated-curve machinery this interpolation is fitted over all points
    at once rather than piecewise.

    Note the C++ default for ``vega_weighted`` differs between ``Svi``
    (``false``) and ``SviInterpolation`` (``true``); the factory default is
    reproduced here.

    Args:
        t: option expiry in year fractions.
        forward: ATM forward.
        a, b, sigma, rho, m: SVI initial values. Pass :data:`NULL_REAL` (or
            ``None``) for "use the :meth:`SviSpecs.default_values` rule",
            which is what the C++ ``Null<Real>()`` sentinel means.
        a_is_fixed .. m_is_fixed: pin a parameter during the fit.
        vega_weighted: vega-weight the residuals.
        end_criteria / optimization_method: C++ takes
            ``ext::shared_ptr<EndCriteria>`` / ``<OptimizationMethod>``
            pass-throughs; PQuantLib's fitter is fixed at the scipy TRF arm,
            so they are accepted and unused (same treatment as ``SABR``).
        error_accept: C++ ``errorAccept`` — the multi-start loop stops early
            once the error drops below it. Accepted and unused for the same
            reason.
        use_max_error: C++ ``useMaxError`` — selects maxError rather than
            rmsError as the multi-start comparison key. Accepted and unused.
        max_guesses: multi-start restart count.
    """

    #: C++ ``static const bool global = true``.
    global_: Final[bool] = True

    def __init__(
        self,
        t: float,
        forward: float,
        a: float | None,
        b: float | None,
        sigma: float | None,
        rho: float | None,
        m: float | None,
        a_is_fixed: bool,
        b_is_fixed: bool,
        sigma_is_fixed: bool,
        rho_is_fixed: bool,
        m_is_fixed: bool,
        vega_weighted: bool = False,
        end_criteria: Any = None,
        optimization_method: Any = None,
        error_accept: float = 0.0020,
        use_max_error: bool = False,
        max_guesses: int = 50,
    ) -> None:
        self._t: float = t
        self._forward: float = forward
        self._a: float | None = a
        self._b: float | None = b
        self._sigma: float | None = sigma
        self._rho: float | None = rho
        self._m: float | None = m
        self._a_is_fixed: bool = a_is_fixed
        self._b_is_fixed: bool = b_is_fixed
        self._sigma_is_fixed: bool = sigma_is_fixed
        self._rho_is_fixed: bool = rho_is_fixed
        self._m_is_fixed: bool = m_is_fixed
        self._vega_weighted: bool = vega_weighted
        self._end_criteria: Any = end_criteria
        self._optimization_method: Any = optimization_method
        self._error_accept: float = error_accept
        self._use_max_error: bool = use_max_error
        self._max_guesses: int = max_guesses

    def interpolate(
        self, strikes: Sequence[float], volatilities: Sequence[float]
    ) -> SviInterpolation:
        """Fit a :class:`SviInterpolation` to one strike/vol slice.

        # C++ parity: ``Svi::interpolate`` (sviinterpolation.hpp:216-224).
        """
        return SviInterpolation(
            strikes,
            volatilities,
            self._t,
            self._forward,
            a=self._a,
            b=self._b,
            sigma=self._sigma,
            rho=self._rho,
            m=self._m,
            a_is_fixed=self._a_is_fixed,
            b_is_fixed=self._b_is_fixed,
            sigma_is_fixed=self._sigma_is_fixed,
            rho_is_fixed=self._rho_is_fixed,
            m_is_fixed=self._m_is_fixed,
            vega_weighted=self._vega_weighted,
            max_guesses=self._max_guesses,
        )


__all__ = [
    "Svi",
    "SviInterpolation",
    "SviSpecs",
    "check_svi_parameters",
    "svi_total_variance",
    "svi_volatility",
]
