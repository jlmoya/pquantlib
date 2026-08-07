"""OperatorSplittingSpreadEngine — Chi-Fai Lo (2015) operator splitting.

# C++ parity:
# ql/pricingengines/basket/operatorsplittingspreadengine.{hpp,cpp} (v1.43),
# ``QuantLib::OperatorSplittingSpreadEngine``.

Chi-Fai Lo, *Pricing Spread Options by the Operator Splitting Method*,
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2429696

A Strang-splitting correction on top of Kirk's approximation. Derives from
:class:`SpreadBlackScholesVanillaEngine`, so only the protected
``(f1, f2, strike, type, var1, var2, df)`` hook is implemented.

Structure of the C++ ``calculate``, reproduced verbatim below:

1. ``kirkCallNPV`` — Kirk's call value at ``sig_m``.
2. ``oPlt`` — the first-order operator-splitting correction. ``Order.First``
   returns ``kirkCallNPV + 0.5 * oPlt``.
3. ``Order.Second`` adds ``0.125 * ooPlt``, and ``ooPlt`` has **two** closed
   forms:
   * a degenerate one when ``rs = (rho*vol1 - sig2)**2 < QL_EPSILON**0.625``,
     because the generic expression divides by powers of ``e = rho*vol1 - sig2``
     and would return NaN there;
   * the generic Mathematica-derived expression otherwise.
   The probe drives ``rs`` to exactly zero (``os_degenerate_*``), so a port
   that only transcribes the generic branch fails outright rather than
   silently losing accuracy.
4. A put is never computed from a put formula: ``callPutParityPrice`` returns
   ``callPrice - df * (f1 - f2 - k)``.

The long ``ooPlt`` expressions are transcribed symbol-for-symbol from
``operatorsplittingspreadengine.cpp``; the single-letter locals are C++'s and
are kept so the two files can be diffed line by line.
"""

from __future__ import annotations

import math
from enum import IntEnum

import numpy as np

from pquantlib import qassert
from pquantlib.math.distributions.cumulative_normal_distribution import (
    CumulativeNormalDistribution,
)
from pquantlib.math.distributions.normal_distribution import NormalDistribution
from pquantlib.payoffs import OptionType
from pquantlib.pricingengines.basket.spread_black_scholes_vanilla_engine import (
    SpreadBlackScholesVanillaEngine,
)
from pquantlib.processes.generalized_black_scholes_process import (
    GeneralizedBlackScholesProcess,
)

_QL_EPSILON: float = float(np.finfo(np.float64).eps)
_M_SQRTPI: float = math.sqrt(math.pi)
_M_SQRT2: float = math.sqrt(2.0)


def _squared(x: float) -> float:
    # C++ parity: ql/math/functional.hpp ``squared``.
    return x * x


class OperatorSplittingSpreadEngine(SpreadBlackScholesVanillaEngine):
    """Operator-splitting spread engine on two assets.

    # C++ parity: ``OperatorSplittingSpreadEngine``
    # (operatorsplittingspreadengine.hpp:38-52).

    Args:
        process1: leg 1.
        process2: leg 2.
        correlation: instantaneous correlation between the two legs.
        order: expansion order. C++ default is ``Order.Second``.
    """

    class Order(IntEnum):
        """# C++ parity: ``enum Order {First, Second}`` (…hpp:40)."""

        First = 0
        Second = 1

    def __init__(
        self,
        process1: GeneralizedBlackScholesProcess,
        process2: GeneralizedBlackScholesProcess,
        correlation: float,
        order: Order = Order.Second,
    ) -> None:
        # C++ parity: operatorsplittingspreadengine.cpp:28-35.
        super().__init__(process1, process2, correlation)
        self._order: OperatorSplittingSpreadEngine.Order = order

    def _calculate(  # noqa: PLR0915  (faithful transcription of one long C++ function)
        self,
        f1: float,
        f2: float,
        strike: float,
        option_type: OptionType,
        variance1: float,
        variance2: float,
        df: float,
    ) -> float:
        """# C++ parity: ``calculate`` (operatorsplittingspreadengine.cpp:37-183)."""
        k = strike
        rho = self._rho

        def call_put_parity_price(call_price: float) -> float:
            # C++ parity: the callPutParityPrice lambda (…cpp:41-46).
            if option_type == OptionType.Call:
                return call_price
            return call_price - df * (f1 - f2 - k)

        vol1 = math.sqrt(variance1)
        vol2 = math.sqrt(variance2)
        sig2 = vol2 * f2 / (f2 + k)
        sig_m = math.sqrt(variance1 + sig2 * (sig2 - 2 * rho * vol1))

        d1 = (math.log(f1) - math.log(f2 + k)) / sig_m + 0.5 * sig_m
        d2 = d1 - sig_m

        n_cum = CumulativeNormalDistribution()

        kirk_call_npv = df * (f1 * n_cum(d1) - (f2 + k) * n_cum(d2))

        vs = vol2 / (sig_m * sig_m)
        rs = _squared(rho * vol1 - sig2)

        o_plt = (
            -sig2
            * sig2
            * k
            * df
            * NormalDistribution()(d2)
            * vs
            * (
                -d2 * rs / sig2
                - 0.5
                * vs
                * sig_m
                * k
                / (f2 + k)
                * (rs * d1 * d2 + (1 - rho * rho) * variance1)
            )
        )

        if self._order == OperatorSplittingSpreadEngine.Order.First:
            return call_put_parity_price(kirk_call_npv + 0.5 * o_plt)

        qassert.require(
            self._order == OperatorSplittingSpreadEngine.Order.Second,
            "unknown approximation type",
        )

        r2 = f2 + k
        r1 = f1 / r2
        vol12 = vol1 * vol1
        vol22 = vol2 * vol2
        vol23 = vol22 * vol2

        if rs < math.pow(_QL_EPSILON, 0.625):
            # Degenerate branch: e = rho*vol1 - sig2 is (numerically) zero, so
            # the generic expression below would divide by zero.
            # C++ parity: operatorsplittingspreadengine.cpp:95-112.
            vol24 = vol22 * vol22
            vol26 = vol22 * vol24
            k2 = k * k
            r22 = r2 * r2
            r24 = r22 * r22
            kmr22 = _squared(k - r2)
            ln_r1 = math.log(r1)

            oo_plt = (
                -0.0625
                * (
                    k2
                    * kmr22
                    * vol26
                    * (
                        -8 * r22 * r24 * (7 * k2 - 7 * k * r2 + r22) * vol12 * vol12
                        + kmr22
                        * r24
                        * vol12
                        * (-112 * k * r2 + 16 * r22 + k2 * (124 + 3 * vol12))
                        * vol22
                        - 2
                        * kmr22
                        * kmr22
                        * r22
                        * (-28 * k * r2 + 4 * r22 + k2 * (34 + 3 * vol12))
                        * vol24
                        + 3 * k2 * kmr22 * kmr22 * kmr22 * vol26
                        - 4
                        * k
                        * (k - r2)
                        * r24
                        * ln_r1
                        * (
                            -4 * r22 * vol12
                            + 4 * kmr22 * vol22
                            + 3 * k * (k - r2) * vol22 * ln_r1
                        )
                    )
                )
                / (
                    math.exp(
                        _squared(-(r22 * vol12) + kmr22 * vol22 + 2 * r22 * ln_r1)
                        / (8 * r24 * vol12 - 8 * kmr22 * r22 * vol22)
                    )
                    * _M_SQRTPI
                    * _M_SQRT2
                    * r22
                    * r24
                    * r2
                    * _squared(r22 * vol12 - kmr22 * vol22)
                    * math.sqrt(vol12 - (kmr22 * vol22) / r22)
                )
            )

            return call_put_parity_price(kirk_call_npv + 0.5 * o_plt + 0.125 * oo_plt)

        # Generic second-order branch.
        # C++ parity: operatorsplittingspreadengine.cpp:114-180. The
        # single-letter names are C++'s; keeping them makes the two files
        # diffable term by term.
        f2_ = f2
        f22 = f2_ * f2_
        f23 = f22 * f2_
        f24 = f22 * f22

        i_r2 = 1.0 / r2
        i_r22 = i_r2 * i_r2
        i_r23 = i_r22 * i_r2
        i_r24 = i_r22 * i_r22
        a = vol12 - 2 * f2_ * i_r2 * rho * vol1 * vol2 + f22 * i_r22 * vol22
        a2 = a * a
        b = a / 2 + math.log(r1)
        b2 = b * b
        c = math.sqrt(a)
        d = b / c
        e = rho * vol1 - f2_ * i_r2 * vol2
        e2 = e * e
        f = d - c
        g = (
            -2 * i_r2 * rho * vol1 * vol2
            + 2 * f2_ * i_r22 * rho * vol1 * vol2
            + 2 * f2_ * i_r22 * vol22
            - 2 * f22 * i_r23 * vol22
        )
        h = rho * rho
        j = 1 - h
        iat = 1 / c
        l = b * iat - c  # noqa: E741  (C++ local name)
        m = f * (1 - (r2 * rho * vol1) / (f2_ * vol2)) - (
            e * i_r2 * k * (d * l + (j * vol12) / (e * e)) * vol2
        ) / (2.0 * c)
        n = (iat * (1 - (r2 * rho * vol1) / (f2_ * vol2))) / r1 - (
            e * i_r2 * k * ((f * iat) / r1 + b / (a * r1)) * vol2
        ) / (2.0 * c)
        o = df * math.exp(-0.5 * f * f)
        p = d * l + (j * vol12) / (e * e)
        q = (-2 * j * vol12 * (-(i_r2 * vol2) + f2_ * i_r22 * vol2)) / (e * e * e)
        s = q - (b2 * g) / (2.0 * a2) - (b * f * g) / (2.0 * a * c) + (f * g) / (2.0 * c)
        u = f * (-((rho * vol1) / (f2_ * vol2)) + (r2 * rho * vol1) / (f22 * vol2))
        v = -0.5 * (b * g * (1 - (r2 * rho * vol1) / (f2_ * vol2))) / (a * c)
        w = (3 * g * g) / (4.0 * a2 * c) - (
            4 * i_r22 * rho * vol1 * vol2
            - 4 * f2_ * i_r23 * rho * vol1 * vol2
            + 2 * i_r22 * vol22
            - 8 * f2_ * i_r23 * vol22
            + 6 * f22 * i_r24 * vol22
        ) / (2.0 * a * c)
        x = (
            u
            + v
            + (e * g * i_r2 * k * p * vol2) / (4.0 * a * c)
            + (e * i_r22 * k * p * vol2) / (2.0 * c)
            - (e * i_r2 * k * s * vol2) / (2.0 * c)
            - (i_r2 * k * p * vol2 * (-(i_r2 * vol2) + f2_ * i_r22 * vol2)) / (2.0 * c)
        )
        y = (4 * i_r22 - 4 * f2_ * i_r23) * rho * vol1 * vol2 + (
            2 * i_r22 - 8 * f2_ * i_r23 + 6 * f22 * i_r24
        ) * vol22
        z = (
            4 * i_r22 * rho * vol1 * vol2
            - 4 * f2_ * i_r23 * rho * vol1 * vol2
            + 2 * i_r22 * vol22
            - 8 * f2_ * i_r23 * vol22
            + 6 * f22 * i_r24 * vol22
        )

        oo_plt = (
            k
            * o
            * vol23
            * (
                -2
                * c
                * b2
                * e2
                * e
                * (-1 + f * f)
                * f23
                * f24
                * g
                * g
                * i_r22
                * m
                * vol23
                + 2 * b2 * e2 * e2 * f23 * f24 * g * g * i_r2 * i_r22 * k * vol22 * vol22
                + 2
                * a
                * b
                * e2
                * e
                * f23
                * f22
                * g
                * i_r22
                * vol2
                * (-8 * e2 * f2_ * i_r2 * k * vol22 + 7 * f * f22 * g * m * vol22)
                - a
                * c
                * e2
                * e
                * f23
                * f22
                * g
                * i_r22
                * vol2
                * (
                    4
                    * e
                    * f2_
                    * vol2
                    * (-2 * b * (-1 + f * f) * m + e * f * i_r2 * k * vol2)
                    + f22
                    * g
                    * (16 * m + e * (2 * f + 3 * b * iat) * i_r2 * k * vol2)
                    * vol22
                )
                - 4
                * a2
                * a
                * c
                * e2
                * (
                    e2
                    * f22
                    * vol2
                    * (
                        4 * f22 * iat * i_r22 * r2 * rho * vol1
                        + 8 * f23 * i_r22 * n * r1 * vol2
                        - 4 * f24 * 3 * i_r23 * n * r1 * vol2
                        - f23
                        * i_r22
                        * (4 * iat * rho * vol1 + f22 * i_r2 * k * p * vol23 * w)
                    )
                    + 4
                    * f23
                    * f22
                    * vol22
                    * vol22
                    * (
                        i_r22 * (-2 * f2_ * i_r2 + 3 * f22 * i_r22) * m
                        + f22 * (2 * i_r2 - 3 * f2_ * i_r22) * i_r23 * m
                        + f22 * i_r22 * (-i_r2 + f2_ * i_r22) * x
                    )
                    + 2
                    * e
                    * f22
                    * (
                        2 * f24 * f2_ * i_r24 * n * r1 * vol23
                        + 2 * f * f2_ * f22 * i_r22 * rho * vol1 * vol22
                        - 2 * f * f22 * i_r22 * r2 * rho * vol1 * vol22
                        - b * f24 * i_r22 * r2 * rho * vol1 * vol22 * w
                        - 2
                        * f24
                        * vol2
                        * (
                            i_r23 * n * r1 * vol22
                            + 4 * i_r23 * m * vol22
                            - 2 * i_r22 * vol22 * x
                        )
                        + f23
                        * (
                            2 * i_r22 * m * vol23
                            + 6 * f22 * i_r24 * m * vol23
                            + b * f22 * i_r22 * vol23 * w
                            - 4 * f22 * i_r23 * vol23 * x
                        )
                    )
                )
                + 2
                * a2
                * c
                * e2
                * f23
                * f22
                * vol2
                * (
                    8 * f22 * g * i_r22 * (-i_r2 + f2_ * i_r22) * m * vol23
                    + e2
                    * i_r22
                    * vol2
                    * (8 * f2_ * g * n * r1 + b * f22 * iat * i_r2 * k * vol22 * (y - z))
                    + 4
                    * e
                    * vol22
                    * (
                        4 * f2_ * g * i_r22 * m
                        + f22 * (-4 * g * i_r23 * m + 2 * g * i_r22 * x + i_r22 * m * z)
                    )
                )
                + 2
                * a2
                * a
                * f22
                * (
                    -4 * e2 * e2 * e * f * f24 * iat * i_r24 * k * vol23
                    + 8
                    * e
                    * f2_
                    * f24
                    * i_r23
                    * (-i_r22 + f2_ * i_r23)
                    * j
                    * k
                    * vol12
                    * vol23
                    * vol22
                    + 12
                    * f2_
                    * f24
                    * i_r23
                    * _squared(i_r2 - f2_ * i_r22)
                    * j
                    * k
                    * vol12
                    * vol23
                    * vol23
                    + e2
                    * e2
                    * f2_
                    * vol22
                    * (
                        2
                        * f24
                        * i_r22
                        * k
                        * vol22
                        * (2 * (i_r23 * p - i_r22 * s) + b2 * iat * i_r2 * w)
                        + f
                        * (
                            4 * f22 * i_r22 * (4 * m + f22 * iat * i_r2 * i_r22 * k * vol22)
                            - 4 * f23 * (6 * i_r23 * m + iat * i_r24 * k * vol22 - 2 * i_r22 * x)
                            + f24 * i_r23 * k * vol22 * (2 * b * w + iat * y)
                        )
                    )
                    - 2
                    * e2
                    * e
                    * f22
                    * i_r22
                    * (
                        4 * f * f22 * (i_r2 - f2_ * i_r22) * m * vol23
                        + f22
                        * vol22
                        * (
                            f2_
                            * vol2
                            * (
                                2
                                * k
                                * (
                                    f2_ * i_r24 * p
                                    + f2_ * i_r24 * p
                                    + i_r22 * s
                                    - i_r23 * (2 * p + f2_ * s)
                                )
                                * vol22
                                + y
                                - z
                            )
                            + r2 * rho * vol1 * (-y + z)
                        )
                    )
                )
                - 2
                * a2
                * e2
                * f23
                * (
                    2
                    * e2
                    * e
                    * f23
                    * i_r22
                    * k
                    * (2 * b * i_r22 + g * (-1 + f * iat) * i_r2)
                    * vol23
                    + 4
                    * b
                    * f
                    * f22
                    * f22
                    * g
                    * i_r22
                    * (-i_r2 + f2_ * i_r22)
                    * m
                    * vol22
                    * vol22
                    + 2
                    * e2
                    * f22
                    * i_r22
                    * vol2
                    * (
                        2 * b * f2_ * i_r2 * (i_r2 - f2_ * i_r22) * k * vol23
                        + g
                        * (
                            2 * r2 * rho * vol1
                            + 2 * f2_ * (-1 + 3 * f * m + b * f * n * r1) * vol2
                            + f22 * k * (-(i_r22 * p) + i_r2 * s) * vol23
                        )
                    )
                    + e
                    * vol22
                    * (
                        f2_
                        * f22
                        * g
                        * i_r22
                        * (
                            g * r2 * rho * vol1
                            + f2_ * g * (-1 + f * m) * vol2
                            + 2 * f2_ * i_r2 * (-i_r2 + f2_ * i_r22) * k * p * vol23
                        )
                        + 2
                        * b
                        * (
                            2 * f2_ * f22 * g * i_r22 * rho * vol1
                            - 2 * f22 * g * i_r22 * r2 * rho * vol1
                            + 4 * f * f23 * g * i_r22 * m * vol2
                            + f
                            * f24
                            * vol2
                            * (
                                -4 * g * i_r23 * m
                                + 2 * g * i_r22 * x
                                + i_r22 * m * z
                            )
                        )
                    )
                )
            )
        ) / (16.0 * a2 * a2 * c * e2 * f23 * _M_SQRT2 * _M_SQRTPI * vol2)

        return call_put_parity_price(kirk_call_npv + 0.5 * o_plt + 0.125 * oo_plt)


__all__ = ["OperatorSplittingSpreadEngine"]
