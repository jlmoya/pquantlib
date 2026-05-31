"""FFTVanillaEngine — Carr-Madan FFT engine under a Black-Scholes process.

# C++ parity: ql/experimental/variancegamma/fftvanillaengine.{hpp,cpp}
# (v1.42.1).

Specialises :class:`FFTEngine` for a
:class:`GeneralizedBlackScholesProcess` with a *constant* Black vol. The
log-spot characteristic function is the Black-Scholes one::

    phi(u) = exp(i u (log S - var t / 2) - var u^2 t / 2)
             * (div_disc / rf_disc)^(i u)

with ``var = sigma^2`` read from the (required) ``BlackConstantVol``
surface.
"""

from __future__ import annotations

import cmath
import math
from typing import final

from pquantlib import qassert
from pquantlib.pricingengines.vanilla.fft_engine import FFTEngine
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.time.date import Date


@final
class FFTVanillaEngine(FFTEngine):
    """FFT engine for vanilla options under a Black-Scholes process.

    # C++ parity: ``class FFTVanillaEngine : public FFTEngine`` in
    # fftvanillaengine.hpp:39.

    Parameters
    ----------
    process
        The :class:`GeneralizedBlackScholesProcess` (constant-vol).
    log_strike_spacing
        FFT log-strike grid spacing (default 0.001; matches C++).
    """

    def __init__(
        self,
        process: GeneralizedBlackScholesProcess,
        log_strike_spacing: float = 0.001,
    ) -> None:
        super().__init__(process, log_strike_spacing)
        self._dividend_discount: float = 1.0
        self._risk_free_discount: float = 1.0
        self._t: float = 0.0
        self._var: float = 0.0

    def clone(self) -> FFTVanillaEngine:
        # C++ parity: FFTVanillaEngine::clone.
        assert isinstance(self._process, GeneralizedBlackScholesProcess)
        return FFTVanillaEngine(self._process, self._lambda)

    def precalculate_expiry(self, d: Date) -> None:
        """Cache discounts + the constant variance for expiry ``d``.

        # C++ parity: FFTVanillaEngine::precalculateExpiry.
        """
        process = self._process
        assert isinstance(process, GeneralizedBlackScholesProcess)
        self._dividend_discount = process.dividend_yield().discount(d)
        self._risk_free_discount = process.risk_free_rate().discount(d)

        rfdc = process.risk_free_rate().day_counter()
        self._t = rfdc.year_fraction(process.risk_free_rate().reference_date(), d)

        const_vol = process.black_volatility()
        qassert.require(
            isinstance(const_vol, BlackConstantVol), "Constant volatility required"
        )
        assert isinstance(const_vol, BlackConstantVol)
        vol = const_vol.black_vol_at_time(0.0, 0.0)
        self._var = vol * vol

    def complex_fourier_transform(self, u: complex) -> complex:
        """Black-Scholes log-spot characteristic function.

        # C++ parity: FFTVanillaEngine::complexFourierTransform.
        """
        i1 = 1j
        s = self._process.x0()
        phi = cmath.exp(
            i1 * u * (math.log(s) - (self._var * self._t) / 2.0)
            - (self._var * u * u * self._t) / 2.0
        )
        phi = phi * (self._dividend_discount / self._risk_free_discount) ** (i1 * u)
        return phi

    def discount_factor(self, d: Date) -> float:
        # C++ parity: FFTVanillaEngine::discountFactor.
        process = self._process
        assert isinstance(process, GeneralizedBlackScholesProcess)
        return process.risk_free_rate().discount(d)

    def dividend_yield(self, d: Date) -> float:
        # C++ parity: FFTVanillaEngine::dividendYield.
        process = self._process
        assert isinstance(process, GeneralizedBlackScholesProcess)
        return process.dividend_yield().discount(d)


__all__ = ["FFTVanillaEngine"]
