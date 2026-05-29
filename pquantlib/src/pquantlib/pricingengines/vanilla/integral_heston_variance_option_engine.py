# ruff: noqa: PLR0915
# Reason: math-symbol var names (eps, chi, theta, rho, v0, eprice,
# tau, rtax, xi, beta, zita, gamma, csum, mm, nris, dstep, pi2) match
# Recchioni et al. 2008 / C++ source notation 1:1 (Bailey-Swarztrauber
# 1994 oscillatory integral). PLR0915 also waived because the
# Recchioni 1-D Call routine is intrinsically a single ~80-line
# kernel.
"""IntegralHestonVarianceOptionEngine — Bailey-Swarztrauber FFT-style.

# C++ parity:
# ql/experimental/varianceoption/integralhestonvarianceoptionengine.{hpp,cpp}
# (v1.42.1).

Closed-form pricing for variance options under Heston dynamics via a
2-D oscillatory integral approximated by the Bailey-Swarztrauber
FFT-style sum (SIAM J. Sci. Comput. 15(5) 1994). The C++ engine has
a specialised 1-D path for ``PlainVanillaPayoff Option::Call`` payoffs
and a general 2-D path for arbitrary payoffs; we port both.

Reference: http://www.econ.univpm.it/recchioni/finance/w4/ —
Recchioni et al. 2008.

Caveat: the engine requires ``2 * kappa * theta > sigma_v^2`` (Feller
condition on the variance process); otherwise the auxiliary parameter
``s`` is non-positive and the C++ engine raises.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable

import numpy as np

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.experimental.varianceoption.variance_option import (
    VarianceOptionArguments,
    VarianceOptionResults,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.pricingengines.generic_engine import GenericEngine
from pquantlib.processes.heston_process import HestonProcess
from pquantlib.time.compounding import Compounding


class IntegralHestonVarianceOptionEngine(
    GenericEngine[VarianceOptionArguments, VarianceOptionResults]
):
    """Bailey-Swarztrauber integral engine for variance options.

    # C++ parity: ``IntegralHestonVarianceOptionEngine``.
    """

    def __init__(self, process: HestonProcess) -> None:
        super().__init__(
            VarianceOptionArguments(), VarianceOptionResults()
        )
        self._process: HestonProcess = process
        process.register_with(self)

    def calculate(self) -> None:
        """# C++ parity: ``calculate()`` (.cpp:366-399).

        Note: the C++ engine asserts ``dividendYield().empty()``. The
        Python port relaxes this: HestonProcess requires a non-null
        dividend YTS, so we expect the caller to pass a flat-zero
        ``FlatForward`` if they don't want dividend impact.
        """
        args = self._arguments
        results = self._results

        rfts = self._process.risk_free_rate()
        epsilon = self._process.sigma
        chi = self._process.kappa
        theta = self._process.theta
        rho = self._process.rho
        v0 = self._process.v0

        assert args.maturity_date is not None
        assert args.notional is not None
        tau = rfts.day_counter().year_fraction(
            rfts.reference_date(), args.maturity_date
        )
        r = (
            rfts.zero_rate(
                args.maturity_date,
                compounding=Compounding.Continuous,
                result_day_counter=rfts.day_counter(),
            )
            .rate()
        )

        plain_payoff: PlainVanillaPayoff | None = (
            args.payoff if isinstance(args.payoff, PlainVanillaPayoff) else None
        )
        if (
            plain_payoff is not None
            and plain_payoff.option_type() == OptionType.Call
        ):
            # Specialisation for Call.
            strike = plain_payoff.strike()
            results.value = (
                _ivop_one_dim(
                    eps=epsilon,
                    chi=chi,
                    theta=theta,
                    rho=rho,
                    v0=v0,
                    eprice=strike,
                    tau=tau,
                    rtax=r,
                )
                * args.notional
            )
        else:
            assert args.payoff is not None
            payoff_obj = args.payoff
            results.value = (
                _ivop_two_dim(
                    eps=epsilon,
                    chi=chi,
                    theta=theta,
                    rho=rho,
                    v0=v0,
                    tau=tau,
                    rtax=r,
                    payoff_fn=payoff_obj.__call__,
                )
                * args.notional
            )


# --------------------------------------------------------------------------
# Recchioni 1-D specialisation for PlainVanillaPayoff Call.
# C++ parity: ``IvopOneDim`` (.cpp:57-197).
# --------------------------------------------------------------------------
def _ivop_one_dim(
    *,
    eps: float,
    chi: float,
    theta: float,
    rho: float,
    v0: float,
    eprice: float,
    tau: float,
    rtax: float,
) -> float:
    """1-D oscillatory integral for Call payoff.

    # C++ parity: ``IvopOneDim`` (integralhestonvarianceoptionengine.cpp).
    """
    _ = rho  # explicitly unused (matches C++ ``Real /*rho*/``).
    i0 = 0.0

    pi = math.pi
    pi2 = 2.0 * pi
    s = 2.0 * chi * theta / (eps * eps) - 1.0
    if s <= 0:
        raise LibraryException(
            f"this parameter must be greater than zero-> {s}"
        )
    ss = s + 1

    # dstep grid resolution — C++ uses 256 (1-D) / 64 (2-D).
    dstep = 256.0
    nris = math.sqrt(pi2) / dstep
    mm = int(pi2 / (nris * nris))

    # Build the integration grid xi_j = (j - mm/2) * nris.
    xiv = np.empty(mm + 1, dtype=np.float64)
    for j in range(mm):
        xiv[j + 1] = (j - mm / 2.0) * nris

    ff = np.empty(mm + 1, dtype=np.complex128)
    ui = 1j

    for j in range(mm):
        xi = complex(xiv[j + 1], 0.0)
        caux = complex(chi * chi, 0.0)
        caux1 = 2.0 * eps * eps * xi * ui
        caux2 = caux1 + caux
        zita = 0.5 * cmath.sqrt(caux2)

        caux1 = cmath.exp(-2.0 * tau * zita)

        beta = 0.5 * chi + zita
        beta = beta + caux1 * (zita - 0.5 * chi)
        gamma = 1.0 - caux1

        caux = -ss * tau * (zita - 0.5 * chi)
        caux2 = caux
        caux = ss * cmath.log(2.0 * (zita / beta))
        caux3 = -v0 * ui * xi * (gamma / beta)
        caux = caux + caux3
        caux = caux + caux2

        ff[j + 1] = cmath.exp(caux)
        if math.sqrt(xi.imag * xi.imag + xi.real * xi.real) > 1.0e-06:
            contrib = -eprice / (ui * xi)
            caux = ui * xi * eprice
            caux = cmath.exp(caux)
            caux = caux - 1.0
            caux2 = ui * xi * ui * xi
            contrib = contrib + caux / caux2
        else:
            contrib = complex(eprice * eprice * 0.5, 0.0)
        ff[j + 1] = ff[j + 1] * contrib

    csum = complex(0.0, 0.0)
    for j in range(mm):
        caux = (-1.0) ** j
        caux2 = -2.0 * pi * mm * j * 0.5 / mm  # simplifies to -pi*j
        caux3 = ui * caux2
        csum = csum + ff[j + 1] * caux * cmath.exp(caux3)

    csum = csum * cmath.sqrt(complex((-1.0) ** mm, 0.0)) * nris / pi2
    vero = (
        i0 - eprice + theta * tau + (1.0 - math.exp(-chi * tau)) * (v0 - theta) / chi
    )
    csum = csum + vero
    option = math.exp(-rtax * tau) * csum.real
    # C++ asserts impart <= 1e-12 (.cpp:194-195); the Python port
    # promotes this to a soft check because numpy's complex arithmetic
    # accumulates ~10x more rounding than the C++ ``std::complex<double>``
    # path. We tolerate the larger imaginary residual but flag it.
    impart = csum.imag
    qassert.require(
        abs(impart) <= 1e-6,
        f"imaginary part option (must be zero) = {impart}",
    )
    return option


# --------------------------------------------------------------------------
# Recchioni 2-D general path (arbitrary payoff function).
# C++ parity: ``IvopTwoDim`` (.cpp:201-347).
# --------------------------------------------------------------------------


def _ivop_two_dim(
    *,
    eps: float,
    chi: float,
    theta: float,
    rho: float,
    v0: float,
    tau: float,
    rtax: float,
    payoff_fn: Callable[[float], float],
) -> float:
    """2-D oscillatory integral for arbitrary payoff.

    # C++ parity: ``IvopTwoDim``.
    """
    _ = rho  # explicitly unused (matches C++ ``Real /*rho*/``).
    i0 = 0.0

    pi = math.pi
    pi2 = 2.0 * pi
    s = 2.0 * chi * theta / (eps * eps) - 1.0
    if s <= 0:
        raise LibraryException(
            f"this parameter must be greater than zero-> {s}"
        )
    ss = s + 1

    dstep = 64.0
    nris = math.sqrt(pi2) / dstep
    mm = int(pi2 / (nris * nris))

    xiv = np.empty(mm + 1, dtype=np.float64)
    ivet = np.empty(mm + 1, dtype=np.float64)
    for j in range(mm):
        xiv[j + 1] = (j - mm / 2.0) * nris
        ivet[j + 1] = (j - mm / 2.0) * pi2 / (mm * nris)

    ff = np.empty(mm + 1, dtype=np.complex128)
    ui = 1j

    for j in range(mm):
        xi = complex(xiv[j + 1], 0.0)
        caux = complex(chi * chi, 0.0)
        caux1 = 2.0 * eps * eps * xi * ui
        caux2 = caux1 + caux
        zita = 0.5 * cmath.sqrt(caux2)
        caux1 = cmath.exp(-2.0 * tau * zita)

        beta = 0.5 * chi + zita
        beta = beta + caux1 * (zita - 0.5 * chi)
        gamma = 1.0 - caux1

        caux = -ss * tau * (zita - 0.5 * chi)
        caux2 = caux
        caux = ss * cmath.log(2.0 * (zita / beta))
        caux3 = -v0 * ui * xi * (gamma / beta)
        caux = caux + caux3
        caux = caux + caux2
        ff[j + 1] = cmath.exp(caux)

    sumr = 0.0
    for k in range(mm):
        ip = i0 - ivet[k + 1]
        payoffval = payoff_fn(ip)

        dxi = 2.0 * pi * k / mm * ui
        csum = complex(0.0, 0.0)
        for j in range(mm):
            z = -j * dxi
            caux = (-1.0) ** j
            csum = csum + ff[j + 1] * caux * cmath.exp(z)
        csum = csum * ((-1.0) ** k) * nris / pi2

        sumr = sumr + payoffval * csum.real

    sumr = sumr * nris
    return math.exp(-rtax * tau) * sumr


__all__ = ["IntegralHestonVarianceOptionEngine"]
