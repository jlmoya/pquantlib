"""Bivariate cumulative Student t-distribution (Dunnett & Sobel 1954).

# C++ parity: ql/math/distributions/bivariatestudenttdistribution.{hpp,cpp}
#             (v1.43).

The even-``n`` and odd-``n`` cases are structurally different closed forms —
equations (10) and (11) of the paper — not two limits of one expression, so
both branches are transcribed. The ``arctan`` helper is *not* ``math.atan2``:
C++ shifts the principal value into ``[0, 2*pi)``, and the shift changes the
answer wherever the argument lands in the third or fourth quadrant.
"""

from __future__ import annotations

import math
from typing import Final

# C++ parity: bivariatestudenttdistribution.cpp:24 — the guard below which
# f_x collapses to 0 (the |rho| == 1 limit).
_EPSILON: Final[float] = 1.0e-8
_M_TWOPI: Final[float] = 2.0 * math.pi


def _sign(val: float) -> float:
    # C++ parity: bivariatestudenttdistribution.cpp:26-29.
    if val == 0.0:
        return 0.0
    return -1.0 if val < 0.0 else 1.0


def _arctan(x: float, y: float) -> float:
    """``atan2`` shifted into ``[0, 2*pi)``.

    # C++ parity: bivariatestudenttdistribution.cpp:34-37.
    """
    res = math.atan2(x, y)
    return res if res >= 0.0 else res + 2 * math.pi


def _f_x(m: float, h: float, k: float, rho: float) -> float:
    """Function ``x(m, h, k)`` from the top of page 155.

    # C++ parity: bivariatestudenttdistribution.cpp:40-47.
    """
    un_cor = 1 - rho * rho
    sub = math.pow(h - rho * k, 2)
    denom = sub + un_cor * (m + k * k)
    if denom < _EPSILON:
        return 0.0  # limit case for rho = +/-1.0
    return sub / (sub + un_cor * (m + k * k))


def _p_n(h: float, k: float, n: int, rho: float) -> float:  # noqa: PLR0915 — one closed form per parity class; splitting them would break the line-for-line correspondence with the C++
    """The bivariate Student-t CDF itself.

    # C++ parity: ``P_n`` — bivariatestudenttdistribution.cpp:50-160.
    """
    un_cor = 1.0 - rho * rho

    div = 4 * math.sqrt(n * math.pi)
    x_hk = _f_x(n, h, k, rho)
    x_kh = _f_x(n, k, h, rho)
    div_h = 1 + h * h / n
    div_k = 1 + k * k / n
    sgn_hk = _sign(h - rho * k)
    sgn_kh = _sign(k - rho * h)

    if n % 2 == 0:
        # n is even, equation (10).
        # C++ parity: bivariatestudenttdistribution.cpp:62-105.
        # first line of (10)
        res = _arctan(math.sqrt(un_cor), -rho) / _M_TWOPI

        # second line of (10)
        dg_m = 2 * (1 - x_hk)  # multiplier for dgj
        gj_m = sgn_hk * 2 / math.pi  # multiplier for g_j
        # initializations for j = 1:
        f_j = math.sqrt(math.pi / div_k)
        g_j = 1 + gj_m * _arctan(math.sqrt(x_hk), math.sqrt(1 - x_hk))
        total = f_j * g_j
        if n >= 4:
            # different formulas for j = 2:
            f_j *= 0.5 / div_k
            dgj = gj_m * math.sqrt(x_hk * (1 - x_hk))
            g_j += dgj
            total += f_j * g_j
            # and then the loop for the rest of the j's:
            for j in range(3, n // 2 + 1):
                f_j *= (j - 1.5) / (j - 1) / div_k
                dgj *= (j - 2) / (2 * j - 3) * dg_m
                g_j += dgj
                total += f_j * g_j
        res += k / div * total

        # third line of (10)
        dg_m = 2 * (1 - x_kh)
        gj_m = sgn_kh * 2 / math.pi
        # initializations for j = 1:
        f_j = math.sqrt(math.pi / div_h)
        g_j = 1 + gj_m * _arctan(math.sqrt(x_kh), math.sqrt(1 - x_kh))
        total = f_j * g_j
        if n >= 4:
            # different formulas for j = 2:
            f_j *= 0.5 / div_h
            dgj = gj_m * math.sqrt(x_kh * (1 - x_kh))
            g_j += dgj
            total += f_j * g_j
            # and then the loop for the rest of the j's:
            for j in range(3, n // 2 + 1):
                f_j *= (j - 1.5) / (j - 1) / div_h
                dgj *= (j - 2) / (2 * j - 3) * dg_m
                g_j += dgj
                total += f_j * g_j
        res += h / div * total
        return res

    # n is odd, equation (11).
    # C++ parity: bivariatestudenttdistribution.cpp:107-157.
    # first line of (11)
    hk = h * k
    hkcn = hk + rho * n
    sqrt_expr = math.sqrt(h * h - 2 * rho * hk + k * k + n * un_cor)
    res = (
        _arctan(
            math.sqrt(float(n)) * (-(h + k) * hkcn - (hk - n) * sqrt_expr),
            (hk - n) * hkcn - n * (h + k) * sqrt_expr,
        )
        / _M_TWOPI
    )

    if n > 1:
        # second line of (11)
        mult = (1 - x_hk) / 2
        # initializations for j = 1:
        f_j = 2 / math.sqrt(math.pi) / div_k
        dgj = sgn_hk * math.sqrt(x_hk)
        g_j = 1 + dgj
        total = f_j * g_j
        # and then the loop for the rest of the j's:
        for j in range(2, (n - 1) // 2 + 1):
            f_j *= (j - 1) / (j - 0.5) / div_k
            dgj *= (2 * j - 3) / (j - 1) * mult
            g_j += dgj
            total += f_j * g_j
        res += k / div * total

        # third line of (11)
        mult = (1 - x_kh) / 2
        # initializations for j = 1:
        f_j = 2 / math.sqrt(math.pi) / div_h
        dgj = sgn_kh * math.sqrt(x_kh)
        g_j = 1 + dgj
        total = f_j * g_j
        # and then the loop for the rest of the j's:
        for j in range(2, (n - 1) // 2 + 1):
            f_j *= (j - 1) / (j - 0.5) / div_h
            dgj *= (2 * j - 3) / (j - 1) * mult
            g_j += dgj
            total += f_j * g_j
        res += h / div * total

    return res


class BivariateCumulativeStudentDistribution:
    """``P[X <= x, Y <= y]`` for a bivariate Student-t.

    # C++ parity: ``class BivariateCumulativeStudentDistribution`` —
    # bivariatestudenttdistribution.hpp:33-45,
    # bivariatestudenttdistribution.cpp:163-173.

    Parameters
    ----------
    n: degrees of freedom.
    rho: correlation.
    """

    __slots__ = ("_n", "_rho")

    def __init__(self, n: int, rho: float) -> None:
        self._n: int = n
        self._rho: float = float(rho)

    def __call__(self, x: float, y: float) -> float:
        return _p_n(x, y, self._n, self._rho)


__all__ = ["BivariateCumulativeStudentDistribution"]
