"""Heston variance-direction meshers.

# C++ parity: ql/methods/finitedifferences/meshers/fdmhestonvariancemesher.{hpp,cpp}
# (v1.43).

``FdmHestonVarianceMesher`` builds the variance axis by pooling, over
``tAvgSteps`` maturities, the quantiles of the exact non-central chi-square
transition law of the CIR variance, then averaging the pooled multiset into
``size`` buckets. If any of that raises (typically the inverse chi-square
running out of evaluations) C++ falls back to a plain uniform grid around
``theta``; the fallback is part of the contract and is reproduced.

``volaEstimate()`` is a Gauss-Lobatto integral of ``sqrt(v)`` against the
pooled probability grid, scaled by ``max(1, sigma/kappa)^{1.5}``.
``FdmHestonLocalVolatilityVarianceMesher`` reuses that grid verbatim and only
rescales ``volaEstimate()`` by the running mean of the leverage function.

The C++ file-local ``interpolated_volatility`` functor (a
``LinearInterpolation`` whose ``operator()`` returns ``sqrt`` of the
interpolated value, *with extrapolation enabled*) is spelled here as a
closure; the ``allow_extrapolation=True`` is load-bearing because Gauss-Lobatto
evaluates slightly outside the knot range.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np

from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.chi_square_distribution import (
    InverseNonCentralCumulativeChiSquareDistribution,
)
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.distributions.non_central_chi_square_distribution import (
    NonCentralCumulativeChiSquareDistribution,
)
from pquantlib.math.integrals.lobatto import GaussLobattoIntegral
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher

if TYPE_CHECKING:
    from pquantlib.processes.heston_process import HestonProcess
    from pquantlib.termstructures.volatility.equity_fx.local_vol_term_structure import (
        LocalVolTermStructure,
    )

_INV_CUM_NORMAL = InverseCumulativeNormal()


def _interpolated_volatility(
    p_grid: Sequence[float], v_grid: Sequence[float]
) -> Callable[[float], float]:
    """# C++ parity: the anonymous-namespace ``interpolated_volatility`` functor."""
    interp = LinearInterpolation(
        np.asarray(p_grid, dtype=np.float64), np.asarray(v_grid, dtype=np.float64)
    )
    return lambda x: math.sqrt(interp(x, allow_extrapolation=True))


class FdmHestonVarianceMesher(Fdm1dMesher):
    """Variance-direction mesher for the Heston model.

    # C++ parity: ``class FdmHestonVarianceMesher : public Fdm1dMesher``.
    """

    def __init__(  # noqa: PLR0915 — one C++ constructor, kept as one function
        self,
        size: int,
        process: HestonProcess,
        maturity: float,
        t_avg_steps: int = 10,
        epsilon: float = 0.0001,
        mixing_factor: float = 1.0,
    ) -> None:
        super().__init__(size)

        v_grid = [0.0] * size
        p_grid = [0.0] * size
        mixed_sigma = process.sigma * mixing_factor
        kappa = process.kappa
        theta = process.theta
        v0 = process.v0
        df = 4.0 * theta * kappa / (mixed_sigma * mixed_sigma)

        try:
            # C++ uses std::multiset<pair<Real,Real>>, i.e. sorted with
            # duplicates kept and ordered lexicographically by (first, second).
            grid: list[tuple[float, float]] = []

            for step in range(1, t_avg_steps + 1):
                t = (maturity * step) / t_avg_steps
                ncp = (
                    4.0
                    * kappa
                    * math.exp(-kappa * t)
                    / (mixed_sigma * mixed_sigma * (1.0 - math.exp(-kappa * t)))
                    * v0
                )
                k = mixed_sigma * mixed_sigma * (1.0 - math.exp(-kappa * t)) / (4.0 * kappa)

                q_min = 0.0  # v_min = 0.0
                q_max = max(
                    v0,
                    k * InverseNonCentralCumulativeChiSquareDistribution(df, ncp, 100, 1e-8)(
                        1.0 - epsilon
                    ),
                )

                min_v_step = (q_max - q_min) / (50 * size)
                p = 0.0
                v_tmp = q_min
                bisect.insort(grid, (q_min, epsilon))

                for i in range(1, size):
                    ps = (1.0 - epsilon - p) / (size - i)
                    p += ps
                    tmp = k * InverseNonCentralCumulativeChiSquareDistribution(
                        df, ncp, 100, 1e-8
                    )(p)
                    vx = max(v_tmp + min_v_step, tmp)
                    p = NonCentralCumulativeChiSquareDistribution(df, ncp)(vx / k)
                    v_tmp = vx
                    bisect.insort(grid, (vx, p))

            if len(grid) != size * t_avg_steps:
                raise LibraryException("something wrong with the grid size")

            tp = grid
            for i in range(size):
                b = (i * len(tp)) // size
                e = ((i + 1) * len(tp)) // size
                for j in range(b, e):
                    v_grid[i] += tp[j][0] / (e - b)
                    p_grid[i] += tp[j][1] / (e - b)
        except LibraryException:
            # C++ parity: `catch (const Error&) { // use default mesh }`.
            vol = mixed_sigma * math.sqrt(theta / (2.0 * kappa))
            mean = theta
            upper_bound = max(v0 + 4.0 * vol, mean + 4.0 * vol)
            lower_bound = max(0.0, min(v0 - 4.0 * vol, mean - 4.0 * vol))
            for i in range(size):
                p_grid[i] = i / (size - 1.0)
                v_grid[i] = lower_bound + i * (upper_bound - lower_bound) / (size - 1.0)

        skew_hint = max(1.0, mixed_sigma / kappa) if kappa != 0.0 else 1.0

        p_grid.sort()
        self._vola_estimate: float = GaussLobattoIntegral(100000, 1e-4)(
            _interpolated_volatility(p_grid, v_grid), p_grid[0], p_grid[-1]
        ) * math.pow(skew_hint, 1.5)

        for i in range(1, len(v_grid)):
            if v_grid[i - 1] <= v0 <= v_grid[i]:
                if abs(v_grid[i - 1] - v0) < abs(v_grid[i] - v0):
                    v_grid[i - 1] = v0
                else:
                    v_grid[i] = v0

        for i in range(size):
            self._locations[i] = v_grid[i]
        for i in range(size - 1):
            self._dminus[i + 1] = self._dplus[i] = v_grid[i + 1] - v_grid[i]
        self._dplus[-1] = math.nan
        self._dminus[0] = math.nan

    def vola_estimate(self) -> float:
        """# C++ parity: ``Real volaEstimate() const``."""
        return self._vola_estimate


class FdmHestonLocalVolatilityVarianceMesher(Fdm1dMesher):
    """Heston variance mesher whose vol estimate accounts for a leverage function.

    # C++ parity: ``class FdmHestonLocalVolatilityVarianceMesher :
    # public Fdm1dMesher``.
    """

    def __init__(
        self,
        size: int,
        process: HestonProcess,
        leverage_fct: LocalVolTermStructure | None,
        maturity: float,
        t_avg_steps: int = 10,
        epsilon: float = 0.0001,
        mixing_factor: float = 1.0,
    ) -> None:
        super().__init__(size)

        mesher = FdmHestonVarianceMesher(
            size, process, maturity, t_avg_steps, epsilon, mixing_factor
        )
        for i in range(size):
            self._dplus[i] = mesher.dplus(i)
            self._dminus[i] = mesher.dminus(i)
            self._locations[i] = mesher.location(i)

        self._vola_estimate: float = mesher.vola_estimate()

        if leverage_fct is not None:
            # C++ uses a boost accumulator whose running mean is read *inside*
            # the loop, so step l sees the mean of the l values pushed so far.
            acc: list[float] = [leverage_fct.local_vol_at_time(0.0, process.s0().value(), True)]

            s0 = process.s0().value()
            r_ts = process.risk_free_rate()
            q_ts = process.dividend_yield()

            for step in range(1, t_avg_steps + 1):
                t = (maturity * step) / t_avg_steps
                vol = self._vola_estimate * (sum(acc) / len(acc))
                fwd = s0 * q_ts.discount(t) / r_ts.discount(t)

                s_avg_steps = 50
                u = [0.0] * s_avg_steps
                sig = [0.0] * s_avg_steps
                for i in range(s_avg_steps):
                    u[i] = epsilon + ((1.0 - 2.0 * epsilon) / (s_avg_steps - 1)) * i
                    x = _INV_CUM_NORMAL(u[i])
                    gf = x * vol * math.sqrt(t)
                    f = fwd * math.exp(gf)
                    sig[i] = leverage_fct.local_vol_at_time(t, f, True) ** 2

                leverage_avg = GaussLobattoIntegral(10000, 1e-4)(
                    _interpolated_volatility(u, sig), u[0], u[-1]
                ) / (1.0 - 2.0 * epsilon)
                acc.append(leverage_avg)

            self._vola_estimate *= sum(acc) / len(acc)

    def vola_estimate(self) -> float:
        """# C++ parity: ``Real volaEstimate() const``."""
        return self._vola_estimate


__all__ = ["FdmHestonLocalVolatilityVarianceMesher", "FdmHestonVarianceMesher"]
