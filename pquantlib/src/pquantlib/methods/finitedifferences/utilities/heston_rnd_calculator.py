"""HestonRNDCalculator — terminal density of ``ln(S)`` under the Heston model.

# C++ parity: ql/methods/finitedifferences/utilities/hestonrndcalculator.{hpp,cpp}
# (v1.43).

Dragulescu-Yakovenko (2002) closed form. The density is the inverse Fourier
transform of ``phi(p_x)``; C++ maps the semi-infinite ``p_x`` integral onto
``(0, 1]`` with ``u = -log(x)/c_inf`` and runs adaptive Gauss-Lobatto over
that unit interval, where

    c_inf = min(10, max(1e-4, sqrt(1-rho^2)/sigma)) * (v0 + kappa theta t)

``pdf`` integrates ``Re(phi(u)/(x c_inf))``; ``cdf`` integrates
``Re(phi(u)/((x c_inf) i u))`` and adds 1/2; ``invcdf`` warm-starts from the
lognormal quantile at the model's expected volatility and refines with Brent.

The helper structs C++ hides in an anonymous namespace (``HestonParams``,
``CpxPv_Helper``) have no separate identity here — the parameters live on the
calculator and the two integrands are closures, which is the same code with
one fewer indirection.
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING, final

from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.integrals.lobatto import GaussLobattoIntegral
from pquantlib.methods.finitedifferences.utilities.bsm_rnd_calculator import (
    BSMRNDCalculator,
)
from pquantlib.methods.finitedifferences.utilities.risk_neutral_density_calculator import (
    InvCDFHelper,
    RiskNeutralDensityCalculator,
)
from pquantlib.processes.black_scholes_merton_process import BlackScholesMertonProcess
from pquantlib.termstructures.volatility.equity_fx.black_constant_vol import (
    BlackConstantVol,
)
from pquantlib.time.calendars.null_calendar import NullCalendar

if TYPE_CHECKING:
    from pquantlib.processes.heston_process import HestonProcess

_TWO_PI: float = 2.0 * math.pi


@final
class HestonRNDCalculator(RiskNeutralDensityCalculator):
    """Risk-neutral terminal density of ``x = ln(S)`` under Heston.

    # C++ parity: ``class HestonRNDCalculator : public RiskNeutralDensityCalculator``.
    """

    __slots__ = (
        "_integration_eps",
        "_max_integration_iterations",
        "_process",
        "_x0",
    )

    def __init__(
        self,
        heston_process: HestonProcess,
        integration_eps: float = 1e-6,
        max_integration_iterations: int = 10000,
    ) -> None:
        self._process: HestonProcess = heston_process
        self._x0: float = math.log(heston_process.s0().value())
        self._integration_eps: float = integration_eps
        self._max_integration_iterations: int = max_integration_iterations

    # --- Dragulescu-Yakovenko characteristic function -------------------

    def _c_inf(self, t: float) -> float:
        """# C++ parity: ``CpxPv_Helper`` ctor — the Fourier-domain scale."""
        p = self._process
        return min(10.0, max(0.0001, math.sqrt(1.0 - p.rho ** 2) / p.sigma)) * (
            p.v0 + p.kappa * p.theta * t
        )

    def _phi(self, p_x: float, x: float, t: float) -> complex:
        """# C++ parity: ``CpxPv_Helper::phi``."""
        p = self._process
        sigma = p.sigma
        sigma2 = sigma * sigma
        g = complex(p.kappa, p.rho * sigma * p_x)
        o = cmath.sqrt(g * g + sigma2 * complex(p_x * p_x, -p_x))
        gamma = (g - o) / (g + o)

        return 2.0 * cmath.exp(
            complex(0.0, p_x * x)
            - p.v0
            * complex(p_x * p_x, -p_x)
            / (g + o * (1.0 + cmath.exp(-o * t)) / (1.0 - cmath.exp(-o * t)))
            + p.kappa
            * p.theta
            / sigma2
            * ((g - o) * t - 2.0 * cmath.log((1.0 - gamma * cmath.exp(-o * t)) / (1.0 - gamma)))
        )

    def _transform_phi(self, u: float, x: float, t: float, c_inf: float) -> complex:
        """# C++ parity: ``CpxPv_Helper::transformPhi``."""
        if u < QL_EPSILON:
            return 0j
        p_x = -math.log(u) / c_inf
        return self._phi(p_x, x, t) / (u * c_inf)

    def _p0(self, u: float, x: float, t: float, c_inf: float) -> float:
        """# C++ parity: ``CpxPv_Helper::p0``."""
        if u < QL_EPSILON:
            return 0.0
        p_x = max(QL_EPSILON, -math.log(u) / c_inf)
        return (self._phi(p_x, x, t) / ((u * c_inf) * complex(0.0, p_x))).real

    def _x_t(self, x: float, t: float) -> float:
        """# C++ parity: ``HestonRNDCalculator::x_t``."""
        dr = self._process.risk_free_rate().discount(t)
        dq = self._process.dividend_yield().discount(t)
        return x - self._x0 + math.log(dr / dq)

    # --- RiskNeutralDensityCalculator ----------------------------------

    def pdf(self, x: float, t: float, /) -> float:
        """# C++ parity: ``HestonRNDCalculator::pdf``."""
        xt = self._x_t(x, t)
        c_inf = self._c_inf(t)
        integral = GaussLobattoIntegral(
            self._max_integration_iterations, 0.1 * self._integration_eps
        )
        return integral(lambda u: self._transform_phi(u, xt, t, c_inf).real, 0.0, 1.0) / _TWO_PI

    def cdf(self, x: float, t: float, /) -> float:
        """# C++ parity: ``HestonRNDCalculator::cdf``."""
        xt = self._x_t(x, t)
        c_inf = self._c_inf(t)
        integral = GaussLobattoIntegral(
            self._max_integration_iterations, 0.1 * self._integration_eps
        )
        return integral(lambda u: self._p0(u, xt, t, c_inf), 0.0, 1.0) / _TWO_PI + 0.5

    def invcdf(self, p: float, t: float, /) -> float:
        """# C++ parity: ``HestonRNDCalculator::invcdf``.

        Warm start = the BSM quantile at the model's expected volatility
        ``sqrt(theta + (v0-theta)(1-exp(-kappa t))/(t kappa))``, then Brent on
        the Heston cdf.
        """
        proc = self._process
        v0 = proc.v0
        kappa = proc.kappa
        theta = proc.theta

        exp_vol = math.sqrt(theta + (v0 - theta) * (1.0 - math.exp(-kappa * t)) / (t * kappa))

        rts = proc.risk_free_rate()
        bsm_process = BlackScholesMertonProcess(
            x0=proc.s0(),
            dividend_ts=proc.dividend_yield(),
            risk_free_ts=rts,
            black_vol_ts=BlackConstantVol(
                reference_date=rts.reference_date(),
                calendar=NullCalendar(),
                day_counter=rts.day_counter(),
                volatility=exp_vol,
            ),
        )
        guess = BSMRNDCalculator(bsm_process).invcdf(p, t)

        return InvCDFHelper(
            self, guess, 0.1 * self._integration_eps, self._max_integration_iterations
        ).inverse_cdf(p, t)


__all__ = ["HestonRNDCalculator"]
