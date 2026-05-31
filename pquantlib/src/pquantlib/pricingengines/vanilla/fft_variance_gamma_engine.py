"""FFTVarianceGammaEngine — Carr-Madan FFT engine under a VG process.

# C++ parity: ql/experimental/variancegamma/fftvariancegammaengine.{hpp,cpp}
# (v1.42.1).

Specialises :class:`FFTEngine` for a :class:`VarianceGammaProcess`. The
log-spot characteristic function is the Variance-Gamma one::

    omega  = log(1 - theta*nu - sigma^2*nu/2) / nu
    phi(u) = exp(i u (log S + omega t)) * (div_disc / rf_disc)^(i u)
             * (1 - i theta nu u + sigma^2 nu u^2 / 2)^(-t/nu)
"""

from __future__ import annotations

import cmath
import math
from typing import final

from pquantlib.experimental.variancegamma.variance_gamma_process import (
    VarianceGammaProcess,
)
from pquantlib.pricingengines.vanilla.fft_engine import FFTEngine
from pquantlib.time.date import Date


@final
class FFTVarianceGammaEngine(FFTEngine):
    """FFT engine for vanilla options under a Variance-Gamma process.

    # C++ parity: ``class FFTVarianceGammaEngine : public FFTEngine`` in
    # fftvariancegammaengine.hpp:39.

    Parameters
    ----------
    process
        The :class:`VarianceGammaProcess`.
    log_strike_spacing
        FFT log-strike grid spacing (default 0.001; matches C++).
    """

    def __init__(
        self,
        process: VarianceGammaProcess,
        log_strike_spacing: float = 0.001,
    ) -> None:
        super().__init__(process, log_strike_spacing)
        self._dividend_discount: float = 1.0
        self._risk_free_discount: float = 1.0
        self._t: float = 0.0
        self._sigma: float = process.sigma()
        self._nu: float = process.nu()
        self._theta: float = process.theta()

    def clone(self) -> FFTVarianceGammaEngine:
        # C++ parity: FFTVarianceGammaEngine::clone.
        assert isinstance(self._process, VarianceGammaProcess)
        return FFTVarianceGammaEngine(self._process, self._lambda)

    def precalculate_expiry(self, d: Date) -> None:
        """Cache discounts + VG params for expiry ``d``.

        # C++ parity: FFTVarianceGammaEngine::precalculateExpiry.
        """
        process = self._process
        assert isinstance(process, VarianceGammaProcess)
        self._dividend_discount = process.dividend_yield().discount(d)
        self._risk_free_discount = process.risk_free_rate().discount(d)

        rfdc = process.risk_free_rate().day_counter()
        self._t = rfdc.year_fraction(process.risk_free_rate().reference_date(), d)

        self._sigma = process.sigma()
        self._nu = process.nu()
        self._theta = process.theta()

    def complex_fourier_transform(self, u: complex) -> complex:
        """Variance-Gamma log-spot characteristic function.

        # C++ parity: FFTVarianceGammaEngine::complexFourierTransform.
        """
        s = self._process.x0()
        i1 = 1j
        omega = (
            math.log(
                1.0 - self._theta * self._nu - self._sigma * self._sigma * self._nu / 2.0
            )
            / self._nu
        )
        phi = cmath.exp(i1 * u * (math.log(s) + omega * self._t)) * (
            self._dividend_discount / self._risk_free_discount
        ) ** (i1 * u)
        phi = phi * (
            (
                1.0
                - i1 * self._theta * self._nu * u
                + self._sigma * self._sigma * self._nu * u * u / 2.0
            )
            ** (-self._t / self._nu)
        )
        return phi

    def discount_factor(self, d: Date) -> float:
        # C++ parity: FFTVarianceGammaEngine::discountFactor.
        process = self._process
        assert isinstance(process, VarianceGammaProcess)
        return process.risk_free_rate().discount(d)

    def dividend_yield(self, d: Date) -> float:
        # C++ parity: FFTVarianceGammaEngine::dividendYield.
        process = self._process
        assert isinstance(process, VarianceGammaProcess)
        return process.dividend_yield().discount(d)


__all__ = ["FFTVarianceGammaEngine"]
