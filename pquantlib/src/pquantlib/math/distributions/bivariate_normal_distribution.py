"""Bivariate cumulative normal distribution — Drezner 1978 and West 2004.

# C++ parity: ql/math/distributions/bivariatenormaldistribution.{hpp,cpp}
#             (v1.43).

**Divergence fixed here.** Before this commit both classes were one scipy
delegation to ``scipy.stats.multivariate_normal.cdf``, documented as being
"at-or-above the precision of either C++ variant". Cross-validated against
the v1.43 probe (``references/v143/math/distributions.json``) that is false:
scipy's 2-D CDF is Genz's randomized lattice rule, whose default stopping
rule is an *absolute* tolerance around 1e-5..1e-8, so it has an absolute
noise floor of roughly 5.5e-17 regardless of the answer's size. At
``rho = -0.95, a = -6, b = 2`` the true value is ``1.55e-42`` and scipy
returned ``5.55e-17`` — 25 orders of magnitude out. Anything that inverts or
differentiates the bivariate CDF in a tail (barrier and two-asset payoffs do)
was reading noise.

Both C++ algorithms are transcribed instead:

* :class:`BivariateCumulativeNormalDistributionDr78` — Drezner (1978) via
  Haug, a 5x5 tabulated Gauss-Hermite-style double sum plus an explicit case
  analysis over the signs of ``a``, ``b`` and ``rho``. Six decimal places, by
  design; the C++ docstring says so and the port must not "improve" it.
* :class:`BivariateCumulativeNormalDistributionWe04DP` — West (2004) /
  Genz (2004) hybrid, near double precision, and the C++ ``typedef`` target
  for ``BivariateCumulativeNormalDistribution``. Its quadrature is
  :class:`~pquantlib.math.integrals.tabulated_gauss_legendre.TabulatedGaussLegendre`
  at order 6 / 12 / 20 selected on ``|rho|``, so the order thresholds are part
  of the answer.
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.integrals.tabulated_gauss_legendre import TabulatedGaussLegendre

# C++ parity: bivariatenormaldistribution.cpp:30-44 — Drezner's 5-point
# weights and abscissae.
_DR78_X: Final[tuple[float, ...]] = (
    0.24840615,
    0.39233107,
    0.21141819,
    0.03324666,
    0.00082485334,
)
_DR78_Y: Final[tuple[float, ...]] = (
    0.10024215,
    0.48281397,
    1.06094980,
    1.77972940,
    2.66976040000,
)


class BivariateCumulativeNormalDistributionDr78:
    """``P[X <= a, Y <= b]`` by Drezner's 1978 algorithm (six decimal places).

    # C++ parity: ``class BivariateCumulativeNormalDistributionDr78`` —
    # bivariatenormaldistribution.hpp:53-60,
    # bivariatenormaldistribution.cpp:46-107.
    """

    __slots__ = ("_rho", "_rho2")

    def __init__(self, rho: float) -> None:
        qassert.require(rho >= -1.0, f"rho must be >= -1.0 ({rho} not allowed)")
        qassert.require(rho <= 1.0, f"rho must be <= 1.0 ({rho} not allowed)")
        self._rho: float = float(rho)
        self._rho2: float = float(rho) * float(rho)

    def __call__(self, a: float, b: float) -> float:
        # C++ parity: bivariatenormaldistribution.cpp:58-106.
        cum_normal_dist = CumulativeNormalDistribution()
        cum_norm_dist_a = cum_normal_dist(a)
        cum_norm_dist_b = cum_normal_dist(b)
        max_cum_norm_dist_ab = max(cum_norm_dist_a, cum_norm_dist_b)
        min_cum_norm_dist_ab = min(cum_norm_dist_a, cum_norm_dist_b)

        if 1.0 - max_cum_norm_dist_ab < 1e-15:
            return min_cum_norm_dist_ab

        if min_cum_norm_dist_ab < 1e-15:
            return min_cum_norm_dist_ab

        a1 = a / math.sqrt(2.0 * (1.0 - self._rho2))
        b1 = b / math.sqrt(2.0 * (1.0 - self._rho2))

        if a <= 0.0 and b <= 0 and self._rho <= 0:
            total = 0.0
            for i in range(5):
                for j in range(5):
                    total += (
                        _DR78_X[i]
                        * _DR78_X[j]
                        * math.exp(
                            a1 * (2.0 * _DR78_Y[i] - a1)
                            + b1 * (2.0 * _DR78_Y[j] - b1)
                            + 2.0 * self._rho * (_DR78_Y[i] - a1) * (_DR78_Y[j] - b1)
                        )
                    )
            return math.sqrt(1.0 - self._rho2) / math.pi * total
        if a <= 0 and b >= 0 and self._rho >= 0:
            biv_cum_normal_dist = BivariateCumulativeNormalDistributionDr78(-self._rho)
            return cum_norm_dist_a - biv_cum_normal_dist(a, -b)
        if a >= 0.0 and b <= 0.0 and self._rho >= 0.0:
            biv_cum_normal_dist = BivariateCumulativeNormalDistributionDr78(-self._rho)
            return cum_norm_dist_b - biv_cum_normal_dist(-a, b)
        if a >= 0.0 and b >= 0.0 and self._rho <= 0.0:
            return cum_norm_dist_a + cum_norm_dist_b - 1.0 + self(-a, -b)
        if a * b * self._rho > 0.0:
            denom = math.sqrt(a * a - 2.0 * self._rho * a * b + b * b)
            rho1 = (self._rho * a - b) * (1.0 if a > 0.0 else -1.0) / denom
            biv_cum_normal_dist = BivariateCumulativeNormalDistributionDr78(rho1)

            rho2 = (self._rho * b - a) * (1.0 if b > 0.0 else -1.0) / denom
            cbnd2 = BivariateCumulativeNormalDistributionDr78(rho2)

            delta = (1.0 - (1.0 if a > 0.0 else -1.0) * (1.0 if b > 0.0 else -1.0)) / 4.0

            return biv_cum_normal_dist(a, 0.0) + cbnd2(b, 0.0) - delta
        return qassert.fail("case not handled")


class _Eqn3:
    """Integrand of eqn (3), Genz 2004.

    # C++ parity: anonymous-namespace ``class eqn3`` —
    # bivariatenormaldistribution.cpp:113-125.
    """

    __slots__ = ("_asr", "_hk", "_hs")

    def __init__(self, h: float, k: float, asr: float) -> None:
        self._hk: float = h * k
        self._asr: float = asr
        self._hs: float = (h * h + k * k) / 2

    def __call__(self, x: float) -> float:
        sn = math.sin(self._asr * (-x + 1) * 0.5)
        return math.exp((sn * self._hk - self._hs) / (1.0 - sn * sn))


class _Eqn6:
    """Integrand of eqn (6), Genz 2004.

    # C++ parity: anonymous-namespace ``class eqn6`` —
    # bivariatenormaldistribution.cpp:127-147.
    """

    __slots__ = ("_a", "_bs", "_c", "_d", "_hk")

    def __init__(self, a: float, c: float, d: float, bs: float, hk: float) -> None:
        self._a: float = a
        self._c: float = c
        self._d: float = d
        self._bs: float = bs
        self._hk: float = hk

    def __call__(self, x: float) -> float:
        xs = self._a * (-x + 1)
        xs = math.fabs(xs * xs)
        rs = math.sqrt(1 - xs)
        asr = -(self._bs / xs + self._hk) / 2
        if asr > -100.0:
            return (
                self._a
                * math.exp(asr)
                * (
                    math.exp(-self._hk * (1 - rs) / (2 * (1 + rs))) / rs
                    - (1 + self._c * xs * (1 + self._d * xs))
                )
            )
        return 0.0


class BivariateCumulativeNormalDistributionWe04DP:
    """``P[X <= x, Y <= y]`` by West (2004) / Genz (2004) — near double precision.

    # C++ parity: ``class BivariateCumulativeNormalDistributionWe04DP`` —
    # bivariatenormaldistribution.hpp:85-93,
    # bivariatenormaldistribution.cpp:150-260.
    """

    __slots__ = ("_correlation", "_cumnorm")

    def __init__(self, rho: float) -> None:
        qassert.require(rho >= -1.0, f"rho must be >= -1.0 ({rho} not allowed)")
        qassert.require(rho <= 1.0, f"rho must be <= 1.0 ({rho} not allowed)")
        self._correlation: float = float(rho)
        self._cumnorm: CumulativeNormalDistribution = CumulativeNormalDistribution()

    def __call__(self, x: float, y: float) -> float:
        # C++ parity: bivariatenormaldistribution.cpp:167-259.
        gauss_legendre_quad = TabulatedGaussLegendre(20)
        if math.fabs(self._correlation) < 0.3:
            gauss_legendre_quad.set_order(6)
        elif math.fabs(self._correlation) < 0.75:
            gauss_legendre_quad.set_order(12)

        h = -x
        k = -y
        hk = h * k
        bvn = 0.0

        if math.fabs(self._correlation) < 0.925:
            if math.fabs(self._correlation) > 0:
                asr = math.asin(self._correlation)
                f3 = _Eqn3(h, k, asr)
                bvn = gauss_legendre_quad(f3)
                bvn *= asr * (0.25 / math.pi)
            bvn += self._cumnorm(-h) * self._cumnorm(-k)
        else:
            if self._correlation < 0:
                k *= -1
                hk *= -1
            # C++ parity: bivariatenormaldistribution.cpp:213. At |rho| == 1
            # exactly, this guard skips the whole Genz series block and only
            # the closing correction below survives — which IS the comonotone
            # / countermonotone limit: N(min(a, b)) at rho = +1, and
            # max(0, N(a) + N(b) - 1) at rho = -1. Both endpoints are reached
            # by real engines (both partial-time lookback engines build the
            # degenerate copula when the lookback window meets the option's
            # own window), and the constructor admits the closed interval, so
            # this branch is load-bearing, not a formality. No special case is
            # written for it here on purpose: the transcription already gives
            # C++'s answer. Cross-validated against all 84 ``bvn_rhop1_*`` /
            # ``bvn_rhom1_*`` cases of
            # ``references/v143/inst/lookbackvarswap.json`` (max rel err
            # 4.7e-16); see test_bivariate_normal_distribution.py.
            if math.fabs(self._correlation) < 1:
                ass = (1 - self._correlation) * (1 + self._correlation)
                a = math.sqrt(ass)
                bs = (h - k) * (h - k)
                c = (4 - hk) / 8
                d = (12 - hk) / 16
                asr = -(bs / ass + hk) / 2
                if asr > -100:
                    bvn = (
                        a
                        * math.exp(asr)
                        * (1 - c * (bs - ass) * (1 - d * bs / 5) / 3 + c * d * ass * ass / 5)
                    )
                if -hk < 100:
                    b = math.sqrt(bs)
                    # C++ spells the constant 2.506628274631 out rather than
                    # using sqrt(2*pi) = 2.5066282746310002; keep the literal.
                    bvn -= (
                        math.exp(-hk / 2)
                        * 2.506628274631
                        * self._cumnorm(-b / a)
                        * b
                        * (1 - c * bs * (1 - d * bs / 5) / 3)
                    )
                a /= 2
                f6 = _Eqn6(a, c, d, bs, hk)
                bvn += gauss_legendre_quad(f6)
                bvn /= -2.0 * math.pi

            if self._correlation > 0:
                bvn += self._cumnorm(-max(h, k))
            else:
                bvn *= -1
                if k > h:
                    # evaluate cumnorm where it is most precise, that is in
                    # the lower tail because of double accuracy around 0.0 vs
                    # around 1.0
                    if h >= 0:
                        bvn += self._cumnorm(-h) - self._cumnorm(-k)
                    else:
                        bvn += self._cumnorm(k) - self._cumnorm(h)
        return bvn


# C++ parity: bivariatenormaldistribution.hpp:96-97 —
# ``typedef BivariateCumulativeNormalDistributionWe04DP
#          BivariateCumulativeNormalDistribution;``
BivariateCumulativeNormalDistribution = BivariateCumulativeNormalDistributionWe04DP


__all__ = [
    "BivariateCumulativeNormalDistribution",
    "BivariateCumulativeNormalDistributionDr78",
    "BivariateCumulativeNormalDistributionWe04DP",
]
