"""GBSMRNDCalculator — risk-neutral terminal density for a GBSM process.

# C++ parity: ql/methods/finitedifferences/utilities/gbsmrndcalculator.{hpp,cpp}
# (v1.42.1).

Computes the risk-neutral terminal probability density (pdf), cumulative
distribution (cdf) and its inverse (invcdf) for the underlying of a
``GeneralizedBlackScholesProcess`` with a (possibly strike-dependent)
Black volatility surface.

The cdf is the Breeden-Litzenberger relation applied to the Black-Scholes
call/put price with a vol-skew correction:

    F(k, t) = 1 + (dC/dK + vega * d(vol)/dK) / df_riskfree    (forward <= k)
    F(k, t) =     (dP/dK + vega * d(vol)/dK) / df_riskfree    (forward >  k)

where ``dC/dK`` is the ``BlackCalculator`` strike sensitivity and the
``vega * dvol/dK`` term accounts for the local-vol skew. The pdf is a
central finite difference of the cdf; invcdf brackets a guess from the
ATM lognormal quantile and solves ``cdf(k) = q`` via Brent.

The Python port has no ``RiskNeutralDensityCalculator`` abstract base
(that interface is a single header with three pure-virtuals); the three
methods are implemented directly. This is the density calculator the
NormalCLVModel / SquareRootCLVModel use for their CDF / inverse-CDF.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, final

from pquantlib import qassert
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.solvers1d.brent import Brent
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.black_calculator import BlackCalculator

if TYPE_CHECKING:
    from pquantlib.processes.generalized_black_scholes_process import (
        GeneralizedBlackScholesProcess,
    )

_INV_CUM_NORMAL = InverseCumulativeNormal()


@final
class GBSMRNDCalculator:
    """Black-Scholes-Merton risk-neutral terminal density calculator.

    # C++ parity: ``class GBSMRNDCalculator : public
    # RiskNeutralDensityCalculator`` in gbsmrndcalculator.hpp:34-44.
    """

    __slots__ = ("_process",)

    def __init__(self, process: GeneralizedBlackScholesProcess) -> None:
        self._process: GeneralizedBlackScholesProcess = process

    def pdf(self, k: float, t: float) -> float:
        # C++ parity: gbsmrndcalculator.cpp:37-41 — central difference of cdf.
        dk = 1e-3 * k
        return (self.cdf(k + dk, t) - self.cdf(k - dk, t)) / (2.0 * dk)

    def cdf(self, k: float, t: float) -> float:
        # C++ parity: gbsmrndcalculator.cpp:43-72.
        vol_ts = self._process.black_volatility()
        dk = 1e-3 * k
        dvol_dk = (
            vol_ts.black_vol_at_time(t, k + dk, True)
            - vol_ts.black_vol_at_time(t, k - dk, True)
        ) / (2.0 * dk)

        d_r = self._process.risk_free_rate().discount(t, True)
        d_d = self._process.dividend_yield().discount(t, True)

        forward = self._process.x0() * d_d / d_r
        std_dev = math.sqrt(vol_ts.black_variance_at_time(t, k, True))

        if forward <= k:
            calc = BlackCalculator.from_type_strike(
                OptionType.Call, k, forward, std_dev, d_r
            )
            return 1.0 + (calc.strike_sensitivity() + calc.vega(t) * dvol_dk) / d_r
        calc = BlackCalculator.from_type_strike(
            OptionType.Put, k, forward, std_dev, d_r
        )
        return (calc.strike_sensitivity() + calc.vega(t) * dvol_dk) / d_r

    def invcdf(self, q: float, t: float) -> float:
        # C++ parity: gbsmrndcalculator.cpp:74-101.
        fwd = (
            self._process.x0()
            / self._process.risk_free_rate().discount(t, True)
            * self._process.dividend_yield().discount(t, True)
        )

        atm_variance = math.sqrt(
            self._process.black_volatility().black_variance_at_time(t, fwd, True)
        )
        atm_x = _INV_CUM_NORMAL(q)
        guess = fwd * math.exp(atm_variance * atm_x)

        lower = guess
        while guess / lower < 65535.0 and self.cdf(lower, t) > q:
            lower *= 0.5

        upper = guess
        while upper / guess < 65535.0 and self.cdf(upper, t) < q:
            upper *= 2.0

        qassert.require(
            guess / lower < 65535.0 and upper / guess < 65535.0,
            f"Could not find a start interval with ({lower}, {upper}) -> "
            f"({self.cdf(lower, t)}, {self.cdf(upper, t)})",
        )

        solver = Brent()
        return solver.solve(
            lambda kk: self.cdf(kk, t) - q,
            1e-10,
            0.5 * (lower + upper),
            lower,
            upper,
        )


__all__ = ["GBSMRNDCalculator"]
