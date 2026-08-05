"""Tabulated Gauss-Legendre quadrature on ``[-1, 1]``.

# C++ parity: ql/math/integrals/gaussianquadratures.hpp (v1.43) —
#             ``class TabulatedGaussLegendre``; tables in
#             gaussianquadratures.cpp:129-200.

Only four orders exist — 6, 7, 12 and 20 — and each stores just the
non-negative half of the abscissae, exploiting the symmetry of the rule. The
odd order (7) stores the zero node first and weights it once; the even orders
weight every stored node twice, at ``+x`` and ``-x``. The summation order is
part of the answer, so it is reproduced rather than replaced with a dot
product.

The tables are Abramowitz & Stegun's, given to 15 decimal places — that is
the accuracy of the rule's nodes, and it is why this class is not
interchangeable with ``numpy.polynomial.legendre.leggauss`` (which computes
the nodes to full double precision from the recurrence). ``BivariateCumulative
NormalDistributionWe04DP`` is defined in terms of *these* numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from pquantlib import qassert

# C++ parity: gaussianquadratures.cpp:132-200. Each entry maps the requested
# order to (abscissae, weights); n_ is implicit in the tuple length.
_TABLES: Final[dict[int, tuple[tuple[float, ...], tuple[float, ...]]]] = {
    6: (
        (0.238619186083197, 0.661209386466265, 0.932469514203152),
        (0.467913934572691, 0.360761573048139, 0.171324492379170),
    ),
    7: (
        (0.000000000000000, 0.405845151377397, 0.741531185599394, 0.949107912342759),
        (0.417959183673469, 0.381830050505119, 0.279705391489277, 0.129484966168870),
    ),
    12: (
        (
            0.125233408511469,
            0.367831498998180,
            0.587317954286617,
            0.769902674194305,
            0.904117256370475,
            0.981560634246719,
        ),
        (
            0.249147045813403,
            0.233492536538355,
            0.203167426723066,
            0.160078328543346,
            0.106939325995318,
            0.047175336386512,
        ),
    ),
    20: (
        (
            0.076526521133497,
            0.227785851141645,
            0.373706088715420,
            0.510867001950827,
            0.636053680726515,
            0.746331906460151,
            0.839116971822219,
            0.912234428251326,
            0.963971927277914,
            0.993128599185095,
        ),
        (
            0.152753387130726,
            0.149172986472604,
            0.142096109318382,
            0.131688638449177,
            0.118194531961518,
            0.101930119817240,
            0.083276741576704,
            0.062672048334109,
            0.040601429800387,
            0.017614007139152,
        ),
    ),
}


class TabulatedGaussLegendre:
    """Fixed-order Gauss-Legendre rule over ``[-1, 1]``.

    # C++ parity: ``class TabulatedGaussLegendre`` —
    # gaussianquadratures.hpp:270-320.
    """

    __slots__ = ("_n", "_order", "_w", "_x")

    def __init__(self, n: int = 20) -> None:
        self._order: int = 0
        self._x: tuple[float, ...] = ()
        self._w: tuple[float, ...] = ()
        self._n: int = 0
        self.set_order(n)

    def set_order(self, order: int) -> None:
        """Select one of the four tabulated orders.

        # C++ parity: ``TabulatedGaussLegendre::order(Size)`` —
        # gaussianquadratures.cpp:110-127.
        """
        table = _TABLES.get(order)
        if table is None:
            qassert.fail(f"order {order} not supported")
        self._order = order
        self._x, self._w = table
        self._n = len(self._x)

    def order(self) -> int:
        # C++ parity: gaussianquadratures.hpp:297.
        return self._order

    def __call__(self, f: Callable[[float], float]) -> float:
        # C++ parity: gaussianquadratures.hpp:274-295. For odd order the
        # stored node 0 is the zero abscissa and contributes once; every
        # other stored node contributes at +x and -x, in that order.
        is_order_odd = self._order & 1

        if is_order_odd:
            qassert.require(self._n > 0, "assume at least 1 point in quadrature")
            val = self._w[0] * f(self._x[0])
            start_idx = 1
        else:
            val = 0.0
            start_idx = 0

        for i in range(start_idx, self._n):
            val += self._w[i] * f(self._x[i])
            val += self._w[i] * f(-self._x[i])
        return val


__all__ = ["TabulatedGaussLegendre"]
