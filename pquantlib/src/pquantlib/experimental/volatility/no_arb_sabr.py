"""NoArbSabrModel — Doust (2012) no-arbitrage SABR.

# C++ parity: ql/experimental/volatility/noarbsabr.{hpp,cpp} (v1.42.1).

Reference: Paul Doust, *No-arbitrage SABR*, The Journal of Computational
Finance, Volume 15/Number 3, Spring 2012 (3-31).

The no-arbitrage SABR model replaces the Hagan 2002 implied-vol
expansion with a directly-specified (and arbitrage-free) terminal
density ``p(f)`` for the forward, plus an *absorption probability*
``absProb`` at the zero boundary. Option prices are then computed by
numerically integrating the payoff against that density:

    optionPrice(K) = (1 - absProb) * ∫ max(f-K, 0) p(f) df / ∫ p(f) df

The absorption probability is *tabulated*: it is read from a 1,209,600-
entry Monte-Carlo grid (``sabrabsprob`` in the C++ source) indexed by
``(beta, nu, rho, sigmaI, tau)`` and multilinearly interpolated by
:class:`D0Interpolator`.

PQuantLib divergences:

* **Tabulated absorption grid.** The C++ source literal
  ``noarbsabrabsprobs.cpp`` is a 7.9 MB array that cannot be hand-ported.
  We ship the *identical* integer grid (extracted byte-for-byte from the
  v1.42.1 build via the W6-A probe) as a gzip-compressed numpy asset at
  ``data/noarbsabr_absprob.npy.gz`` and mmap it on first use. The grid
  is reshaped to ``(10, 8, 7, 18, 120)`` matching the C++ flat index
  ``tau + (sigmaI + (rho + (nu + beta*8)*7)*18)*120``. The
  :class:`D0Interpolator` multilinear blend reproduces the C++
  absorption fractions bit-for-bit (validated against the explicit
  ``testAbsorptionMatrix`` checkpoints).
* **Special functions.** C++ uses ``boost::math::gamma_q`` /
  ``gamma_q_inv`` (regularised upper incomplete gamma + its inverse)
  and ``modifiedBesselFunction_i_exponentiallyWeighted``. PQuantLib
  delegates to ``scipy.special.gammaincc`` / ``gammainccinv`` and
  ``scipy.special.ive`` (exponentially-scaled modified Bessel I), which
  are the de-facto equivalents.
* **Integration.** C++ uses ``GaussLobattoIntegral`` over the density.
  PQuantLib uses the ported :class:`GaussLobattoIntegral` for identical
  adaptive Gauss-Lobatto quadrature, keeping the same accuracy / max-
  iteration budget (1e-7 / 10000).
* **Forward-adjustment solve.** C++ runs a ``Brent`` 1-D solve to nudge
  the model forward onto the external forward. PQuantLib uses the ported
  :class:`Brent`. When the solve fails (admissible-but-unadjustable
  parameter sets, see the C++ note) both fall back to the unadjusted
  forward.
"""

from __future__ import annotations

import gzip
import io
import math
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import gammaincc, gammainccinv, ive  # type: ignore[import-untyped]

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.math.integrals.lobatto import GaussLobattoIntegral
from pquantlib.math.solvers1d.brent import Brent

# --- parameter bounds (C++ detail::NoArbSabrModel namespace) --------------

BETA_MIN = 0.01
BETA_MAX = 0.99
EXPIRY_TIME_MAX = 30.0
SIGMA_I_MIN = 0.05
SIGMA_I_MAX = 1.00
NU_MIN = 0.01
NU_MAX = 0.80
RHO_MIN = -0.99
RHO_MAX = 0.99
# cutoff for phi(d0)/tau: if beta=0.99, d0 is below 1e-14 above this.
_PHI_BY_TAU_CUTOFF = 124.587
# number of MC simulations behind the tabulated absorption probabilities.
NSIM = 2500000.0
# small probability used for extrapolation of beta towards 1.
_TINY_PROB = 1e-5
# minimum strike used for normal-case integration.
_STRIKE_MIN = 1e-6
# accuracy + max iterations for numerical integration.
_I_ACCURACY = 1e-7
_I_MAX_ITERATIONS = 10000
# accuracy when adjusting the model forward to match the given forward.
_FORWARD_ACCURACY = 1e-6
# step for searching the model forward in the Newton/Brent algorithm.
_FORWARD_SEARCH_STEP = 0.0010
# lower bound for density evaluation.
_DENSITY_LOWER_BOUND = 1e-50
# threshold to identify a zero density.
_DENSITY_THRESHOLD = 1e-100
# Smallest positive double (C++ QL_MIN_POSITIVE_REAL).
_MIN_POSITIVE_REAL = 2.2250738585072014e-308

# --- absorption grid axes (C++ D0Interpolator constructor) ----------------

_TAU_G: list[float] = [
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
    3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25,
    6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5,
    9.75, 10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25,
    12.5, 12.75, 13.0, 13.25, 13.5, 13.75, 14.0, 14.25, 14.5, 14.75, 15.0,
    15.25, 15.5, 15.75, 16.0, 16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75,
    18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5,
    20.75, 21.0, 21.25, 21.5, 21.75, 22.0, 22.25, 22.5, 22.75, 23.0, 23.25,
    23.5, 23.75, 24.0, 24.25, 24.5, 24.75, 25.0, 25.25, 25.5, 25.75, 26.0,
    26.25, 26.5, 26.75, 27.0, 27.25, 27.5, 27.75, 28.0, 28.25, 28.5, 28.75,
    29.0, 29.25, 29.5, 29.75, 30.0,
]
# sigmaI grid is *descending* in the C++ source.
_SIGMA_I_G: list[float] = [
    1.0, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.27, 0.24, 0.21,
    0.18, 0.15, 0.125, 0.1, 0.075, 0.05,
]
# rho grid is *descending*.
_RHO_G: list[float] = [0.75, 0.50, 0.25, 0.00, -0.25, -0.50, -0.75]
_NU_G: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
_BETA_G: list[float] = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

_ASSET_PATH = (
    Path(__file__).parent / "data" / "noarbsabr_absprob.npy.gz"
)


@lru_cache(maxsize=1)
def _absorption_grid() -> np.ndarray:
    """Load + cache the absorption-probability grid (uint32 MC counts).

    Shape ``(10, 8, 7, 18, 120)`` over ``(beta, nu, rho, sigmaI, tau)``.
    Values are raw MC absorption counts; divide by :data:`NSIM` to get
    the absorption fraction.
    """
    with gzip.open(_ASSET_PATH, "rb") as f:
        arr = np.load(io.BytesIO(f.read()), allow_pickle=False)
    if arr.shape != (10, 8, 7, 18, 120):
        raise LibraryException(
            f"noarbsabr absorption grid has unexpected shape {arr.shape}"
        )
    return arr


class D0Interpolator:
    """Multilinear interpolation of the tabulated absorption probability.

    # C++ parity: ``detail::D0Interpolator`` (noarbsabr.cpp:186-329).

    Calling the instance returns ``d0`` — the absorption probability
    fraction in ``[0, 1)`` — obtained by:

    1. Mapping ``sigmaI = alpha * forward^(beta-1)`` and ``(tau, rho,
       nu, beta)`` to fractional positions on the tabulated axes.
    2. Reading the 32 grid corners, converting each tabulated count to a
       ``phi`` value via the regularised inverse-incomplete-gamma
       transform, and blending them multilinearly in ``phi`` space.
    3. Mapping the blended ``phi`` back to ``d0`` via the forward
       regularised incomplete gamma.

    The boundary extrapolation rules (beta -> 1, nu == 0, rho outside
    [-0.75, 0.75], tau < 0.25) follow the C++ source exactly.
    """

    def __init__(
        self,
        forward: float,
        expiry_time: float,
        alpha: float,
        beta: float,
        nu: float,
        rho: float,
    ) -> None:
        self._forward = forward
        self._expiry_time = expiry_time
        self._alpha = alpha
        self._beta = beta
        self._nu = nu
        self._rho = rho
        self._gamma = 1.0 / (2.0 * (1.0 - beta))
        self._sigma_i = alpha * forward ** (beta - 1.0)

    def _phi(self, d0: float) -> float:
        # C++: gamma_q_inv(gamma_, d0) * expiryTime_, with cutoff guard.
        if d0 < 1e-14:
            return _PHI_BY_TAU_CUTOFF * self._expiry_time
        return float(gammainccinv(self._gamma, d0)) * self._expiry_time

    def _d0(self, phi: float) -> float:
        # C++: gamma_q(gamma_, max(0, phi/expiryTime_)).
        return float(gammaincc(self._gamma, max(0.0, phi / self._expiry_time)))

    def __call__(self) -> float:
        grid = _absorption_grid()
        tau = self._expiry_time
        sigma_i = self._sigma_i
        rho = self._rho
        nu = self._nu
        beta = self._beta

        # tau axis (ascending): upper_bound then clamp/shift like C++.
        tau_ind = bisect_right(_TAU_G, tau)
        if tau_ind == len(_TAU_G):
            tau_ind -= 1
        expiry_tmp = tau
        if tau_ind == 0:
            tau_ind += 1
            expiry_tmp = _TAU_G[0]
        tau_l = (expiry_tmp - _TAU_G[tau_ind - 1]) / (
            _TAU_G[tau_ind] - _TAU_G[tau_ind - 1]
        )

        # sigmaI axis (descending): mirror C++ reverse upper_bound.
        sigma_ind = len(_SIGMA_I_G) - _upper_bound_desc(_SIGMA_I_G, sigma_i)
        if sigma_ind == 0:
            sigma_ind += 1
        sigma_l = (sigma_i - _SIGMA_I_G[sigma_ind - 1]) / (
            _SIGMA_I_G[sigma_ind] - _SIGMA_I_G[sigma_ind - 1]
        )

        # rho axis (descending).
        rho_ind = len(_RHO_G) - _upper_bound_desc(_RHO_G, rho)
        if rho_ind == 0:
            rho_ind += 1
        if rho_ind == len(_RHO_G):
            rho_ind -= 1
        rho_l = (rho - _RHO_G[rho_ind - 1]) / (
            _RHO_G[rho_ind] - _RHO_G[rho_ind - 1]
        )

        # nu axis (ascending). For nu=0 we know phi = 0.5*z_F^2.
        nu_ind = bisect_right(_NU_G, nu)
        if nu_ind == len(_NU_G):
            nu_ind -= 1
        tmp_nu_g = _NU_G[nu_ind - 1] if nu_ind > 0 else 0.0
        nu_l = (nu - tmp_nu_g) / (_NU_G[nu_ind] - tmp_nu_g)

        # beta axis (ascending). For beta=1 we know phi = 0.0.
        beta_ind = bisect_right(_BETA_G, beta)
        tmp_beta_g = 1.0 if beta_ind == len(_BETA_G) else _BETA_G[beta_ind]
        beta_l = (beta - _BETA_G[beta_ind - 1]) / (
            tmp_beta_g - _BETA_G[beta_ind - 1]
        )

        phi_res = 0.0
        for i_tau in (-1, 0):
            w_tau = (1.0 - tau_l) if i_tau == -1 else tau_l
            for i_sigma in (-1, 0):
                w_sigma = (1.0 - sigma_l) if i_sigma == -1 else sigma_l
                for i_rho in (-1, 0):
                    w_rho = (1.0 - rho_l) if i_rho == -1 else rho_l
                    for i_nu in (-1, 0):
                        w_nu = (1.0 - nu_l) if i_nu == -1 else nu_l
                        for i_beta in (-1, 0):
                            w_beta = (1.0 - beta_l) if i_beta == -1 else beta_l
                            if i_nu == -1 and nu_ind == 0:
                                # 0.5 * z_F^2 (nu == 0 limit).
                                phi_tmp = 0.5 / (
                                    sigma_i * sigma_i
                                    * (1.0 - beta) * (1.0 - beta)
                                )
                            elif i_beta == 0 and beta_ind == len(_BETA_G):
                                # beta -> 1 extrapolation to tiny_prob.
                                phi_tmp = self._phi(_TINY_PROB)
                            else:
                                count = int(
                                    grid[
                                        beta_ind + i_beta,
                                        nu_ind + i_nu,
                                        rho_ind + i_rho,
                                        sigma_ind + i_sigma,
                                        tau_ind + i_tau,
                                    ]
                                )
                                phi_tmp = self._phi(count / NSIM)
                            phi_res += (
                                phi_tmp * w_tau * w_sigma * w_rho * w_nu * w_beta
                            )
        return self._d0(phi_res)


def _upper_bound_desc(grid: list[float], value: float) -> int:
    """Index of first element ``< value`` in a *descending* list.

    Mirrors ``std::upper_bound(rbegin, rend, value) - rbegin`` used by
    the C++ D0Interpolator for the descending sigmaI / rho axes: it
    walks the reversed (ascending) view and returns how many reversed
    elements are ``<= value``.
    """
    # Reversed view is ascending; upper_bound on it = count of elems <= value.
    n = len(grid)
    count = 0
    for i in range(n - 1, -1, -1):
        if grid[i] <= value:
            count += 1
        else:
            break
    return count


class NoArbSabrModel:
    """Doust no-arbitrage SABR terminal-density model.

    # C++ parity: ``NoArbSabrModel`` (noarbsabr.{hpp,cpp}).

    Args:
        expiry_time: option expiry ``tau`` in year fractions; must be in
            ``(0, 30]``.
        forward: external forward; must be positive.
        alpha, beta, nu, rho: SABR parameters. Bounds (matching C++):
            ``beta`` in ``[0.01, 0.99]``, ``sigmaI = alpha*F^(beta-1)``
            in ``[0.05, 1.0]``, ``nu`` in ``[0.01, 0.80]``, ``rho`` in
            ``[-0.99, 0.99]``.
    """

    def __init__(
        self,
        expiry_time: float,
        forward: float,
        alpha: float,
        beta: float,
        nu: float,
        rho: float,
    ) -> None:
        qassert.require(
            0.0 < expiry_time <= EXPIRY_TIME_MAX,
            f"expiryTime ({expiry_time}) out of bounds",
        )
        qassert.require(forward > 0.0, f"forward ({forward}) must be positive")
        qassert.require(
            BETA_MIN <= beta <= BETA_MAX, f"beta ({beta}) out of bounds"
        )
        sigma_i = alpha * forward ** (beta - 1.0)
        qassert.require(
            SIGMA_I_MIN <= sigma_i <= SIGMA_I_MAX,
            f"sigmaI = alpha*forward^(beta-1.0) ({sigma_i}) out of bounds, "
            f"alpha={alpha} beta={beta} forward={forward}",
        )
        qassert.require(NU_MIN <= nu <= NU_MAX, f"nu ({nu}) out of bounds")
        qassert.require(RHO_MIN <= rho <= RHO_MAX, f"rho ({rho}) out of bounds")

        self._expiry_time = expiry_time
        self._external_forward = forward
        self._alpha = alpha
        self._beta = beta
        self._nu = nu
        self._rho = rho
        self._forward = forward
        self._numerical_forward = forward
        self._numerical_integral_over_p = 1.0

        # Determine a region sufficient for integration in the normal case.
        fmin = forward
        fmax = forward
        tmp = self._p(fmax)
        while tmp > max(_I_ACCURACY / max(1.0, fmax - fmin), _DENSITY_THRESHOLD):
            fmax *= 2.0
            tmp = self._p(fmax)
        tmp = self._p(fmin)
        while tmp > max(_I_ACCURACY / max(1.0, fmax - fmin), _DENSITY_THRESHOLD):
            fmin *= 0.5
            tmp = self._p(fmin)
        fmin = max(_STRIKE_MIN, fmin)
        self._fmin = fmin
        self._fmax = fmax
        qassert.require(
            fmax > fmin, "could not find a reasonable integration domain"
        )

        self._integrator = GaussLobattoIntegral(
            _I_MAX_ITERATIONS, _I_ACCURACY
        )

        d0 = D0Interpolator(
            self._forward, expiry_time, alpha, beta, nu, rho
        )
        self._abs_prob = d0()

        # Adjust the model forward onto the external forward via Brent.
        try:
            brent = Brent()
            start = math.sqrt(self._external_forward - _STRIKE_MIN)
            tmp_root = brent.solve(
                self._forward_error,
                _FORWARD_ACCURACY,
                start,
                min(_FORWARD_SEARCH_STEP, start / 2.0),
            )
            self._forward = tmp_root * tmp_root + _STRIKE_MIN
        except LibraryException:
            # C++ parity: ``catch (Error&)`` — fall back to the
            # unadjusted forward when the root-search fails (admissible-
            # but-unadjustable parameter sets, see noarbsabr.hpp note).
            self._forward = self._external_forward

        d = self._forward_error(math.sqrt(self._forward - _STRIKE_MIN))
        self._numerical_forward = d + self._external_forward

    # --- density --------------------------------------------------------

    def _p(self, f: float) -> float:
        """Terminal density ``p(f)`` (unnormalised).

        # C++ parity: ``NoArbSabrModel::p`` (noarbsabr.cpp:143-182).
        """
        beta = self._beta
        nu = self._nu
        rho = self._rho
        alpha = self._alpha
        if f < _DENSITY_LOWER_BOUND or self._forward < _DENSITY_LOWER_BOUND:
            return 0.0

        f_omb = f ** (1.0 - beta)
        big_f_omb = self._forward ** (1.0 - beta)

        zf = f_omb / (alpha * (1.0 - beta))
        big_zf = big_f_omb / (alpha * (1.0 - beta))
        z = big_zf - zf

        jmzf = math.sqrt(1.0 + 2.0 * rho * nu * zf + nu * nu * zf * zf)
        jz = math.sqrt(1.0 - 2.0 * rho * nu * z + nu * nu * z * z)

        xz = math.log((jz - rho + nu * z) / (1.0 - rho)) / nu
        bp_b = beta / big_f_omb
        kappa1 = (
            0.125 * nu * nu * (2.0 - 3.0 * rho * rho)
            - 0.25 * rho * nu * alpha * bp_b
        )
        gamma = 1.0 / (2.0 * (1.0 - beta))
        sqrt_omr = math.sqrt(1.0 - rho * rho)
        h = (
            0.5 * beta * rho / ((1.0 - beta) * jmzf * jmzf)
            * (
                nu * zf * math.log(zf * jz / big_zf)
                + (1.0 + rho * nu * zf) / sqrt_omr
                * (
                    math.atan((nu * z - rho) / sqrt_omr)
                    + math.atan(rho / sqrt_omr)
                )
            )
        )

        res = (
            jz ** (-1.5)
            / (alpha * f ** beta * self._expiry_time)
            * zf ** (1.0 - gamma)
            * big_zf ** gamma
            * math.exp(-(xz * xz) / (2.0 * self._expiry_time) + (h + kappa1 * self._expiry_time))
            * float(ive(gamma, big_zf * zf / self._expiry_time))
        )
        return res

    def density(self, strike: float) -> float:
        """Normalised risk-neutral density at ``strike``.

        # C++ parity: ``NoArbSabrModel::density`` (noarbsabr.hpp:107-109).
        """
        return (
            self._p(strike) * (1.0 - self._abs_prob)
            / self._numerical_integral_over_p
        )

    # --- pricing --------------------------------------------------------

    def _forward_error(self, forward: float) -> float:
        # C++: forward_ = forward^2 + strike_min; integrate p; price-fwd.
        self._forward = forward * forward + _STRIKE_MIN
        self._numerical_integral_over_p = self._integrator(
            self._p, self._fmin, self._fmax
        )
        return self.option_price(0.0) - self._external_forward

    def option_price(self, strike: float) -> float:
        """Undiscounted call price ``E[max(f-K, 0)]`` under the model.

        # C++ parity: ``NoArbSabrModel::optionPrice`` (noarbsabr.cpp:116-123).
        """
        if self._p(max(self._forward, strike)) < _DENSITY_THRESHOLD:
            return 0.0

        def integrand(f: float) -> float:
            return max(f - strike, 0.0) * self._p(f)

        return (1.0 - self._abs_prob) * (
            self._integrator(integrand, strike, max(self._fmax, 2.0 * strike))
            / self._numerical_integral_over_p
        )

    def digital_option_price(self, strike: float) -> float:
        """Undiscounted digital call price ``P(f > K)`` under the model.

        # C++ parity: ``NoArbSabrModel::digitalOptionPrice``
        # (noarbsabr.cpp:125-134).
        """
        if strike < _MIN_POSITIVE_REAL:
            return 1.0
        if self._p(max(self._forward, strike)) < _DENSITY_THRESHOLD:
            return 0.0
        return (1.0 - self._abs_prob) * (
            self._integrator(self._p, strike, max(self._fmax, 2.0 * strike))
            / self._numerical_integral_over_p
        )

    # --- inspectors -----------------------------------------------------

    def forward(self) -> float:
        return self._external_forward

    def numerical_forward(self) -> float:
        return self._numerical_forward

    def expiry_time(self) -> float:
        return self._expiry_time

    def alpha(self) -> float:
        return self._alpha

    def beta(self) -> float:
        return self._beta

    def nu(self) -> float:
        return self._nu

    def rho(self) -> float:
        return self._rho

    def absorption_probability(self) -> float:
        return self._abs_prob


def no_arb_sabr_volatility(
    strike: float,
    forward: float,
    expiry: float,
    alpha: float,
    beta: float,
    nu: float,
    rho: float,
) -> float:
    """No-arbitrage SABR implied (lognormal) volatility at ``strike``.

    # C++ parity: ``NoArbSabrSmileSection::volatilityImpl``
    # (noarbsabrsmilesection.cpp:76-98) — builds the model, prices the
    # OTM option, and inverts Black to an implied vol, falling back to
    # the Hagan 2002 closed form when the inversion fails.

    Args:
        strike: option strike.
        forward: ATM forward (positive).
        expiry: option expiry ``tau`` in year fractions.
        alpha, beta, nu, rho: SABR parameters (see
            :class:`NoArbSabrModel` for bounds).
    """
    from pquantlib.math.interpolations.sabr_formula import (  # noqa: PLC0415
        sabr_volatility,
    )
    from pquantlib.payoffs import OptionType  # noqa: PLC0415
    from pquantlib.pricingengines.black_formula import (  # noqa: PLC0415
        black_formula_implied_std_dev,
    )

    model = NoArbSabrModel(expiry, forward, alpha, beta, nu, rho)
    implied_vol = 0.0
    try:
        option_type = OptionType.Call if strike >= forward else OptionType.Put
        price = model.option_price(strike)
        if option_type == OptionType.Put:
            # put-call parity: put = call - (forward - strike).
            price = price - (forward - strike)
        implied_vol = black_formula_implied_std_dev(
            option_type, strike, forward, price, 1.0
        ) / math.sqrt(expiry)
    except (LibraryException, ValueError, ZeroDivisionError):
        implied_vol = 0.0
    if implied_vol == 0.0:
        implied_vol = sabr_volatility(
            strike, forward, expiry, alpha, beta, nu, rho
        )
    return implied_vol


__all__ = [
    "D0Interpolator",
    "NoArbSabrModel",
    "no_arb_sabr_volatility",
]
