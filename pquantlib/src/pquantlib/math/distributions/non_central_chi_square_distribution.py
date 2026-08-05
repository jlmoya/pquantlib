"""Non-central chi-square cumulative distribution.

# C++ parity: ql/math/distributions/chisquaredistribution.{hpp,cpp} (v1.43) —
#             ``class NonCentralCumulativeChiSquareDistribution``.

**Divergence fixed here.** Before this commit the port delegated to
``scipy.stats.ncx2.cdf`` on the argument that Boost's continued fraction and
the C++ series "typically match to 1e-12 or better". Cross-validated against
the v1.43 probe (``references/v143/math/distributions.json``), that premise
does not hold: at ``df=30, ncp=100, x=3`` the C++ series returns
``8.3315e-32`` and scipy returns ``1.0367e-30`` — a factor of 12. The two are
not the same computation and never were.

The reason is structural, not a bug on either side. The C++ series is Ding's
recurrence with an *absolute* error bound: it stops as soon as
``bound = t*x/f_x_2n <= errmax`` with ``errmax = 1e-12``, so in the far tail
it stops after the leading term and its answer is only meaningful to 1e-12
absolute. Boost's continued fraction is accurate relatively, all the way
down. Both are defensible; only one of them is QuantLib.

Since C++ v1.43 is the source of truth, this module now transcribes the
series — including ``errmax``, the 10000-iteration cap, the
``QL_FAIL("didn't converge")`` exit and the (effectively unreachable)
large-``f2`` initialisation branch. Callers that want relative accuracy in
the far tail should reach for ``scipy.stats.ncx2`` explicitly rather than
have it silently substituted here.
"""

from __future__ import annotations

import math
from typing import Final

from pquantlib import qassert
from pquantlib.math.constants import QL_EPSILON
from pquantlib.math.distributions.gamma_function import GammaFunction

# C++ parity: chisquaredistribution.cpp:38-39.
_ERRMAX: Final[float] = 1e-12
_ITRMAX: Final[int] = 10000


class NonCentralCumulativeChiSquareDistribution:
    """Cumulative distribution function of a non-central chi-square.

    # C++ parity: ``class NonCentralCumulativeChiSquareDistribution`` —
    # chisquaredistribution.hpp:41-49, chisquaredistribution.cpp:34-95.

    Parameters
    ----------
    df: degrees of freedom (``df_`` in C++).
    ncp: non-centrality parameter (``ncp_`` in C++).
    """

    __slots__ = ("_df", "_ncp")

    def __init__(self, df: float, ncp: float) -> None:
        self._df: float = float(df)
        self._ncp: float = float(ncp)

    def __call__(self, x: float) -> float:
        """Return ``P[X <= x]`` for ``X ~ chi^2_{df}(ncp)``.

        # C++ parity: chisquaredistribution.cpp:34-95, transcribed
        # statement for statement. The C++ body is written with two nested
        # ``for(;;)`` loops and a ``goto L10`` into the middle of the inner
        # one. The equivalent structure below relies on one observation:
        # ``L10`` is only ever reached with ``flag == True`` or with
        # ``n > itrmax`` (which exits immediately), so once control reaches
        # ``L10`` it never returns to the outer loop. That makes the two
        # phases separable — a term-at-a-time phase while ``f_x_2n <= 0``,
        # then a bound-checked phase.
        """
        if x <= 0.0:
            return 0.0

        df = self._df
        lam = 0.5 * self._ncp

        u = math.exp(-lam)
        v = u
        x2 = 0.5 * x
        f2 = 0.5 * df
        f_x_2n = df - x

        t = 0.0
        if f2 * QL_EPSILON > 0.125 and math.fabs(x2 - f2) < math.sqrt(QL_EPSILON) * f2:
            # C++ parity: chisquaredistribution.cpp:50-53. ``t`` is still 0
            # here, so this reduces to exp(2)/sqrt(2*pi*(f2+1)); the guard
            # needs f2 > 0.125/QL_EPSILON ~ 5.6e14 and is unreachable in
            # practice. Kept verbatim.
            t = math.exp((1 - t) * (2 - t / (f2 + 1))) / math.sqrt(2.0 * math.pi * (f2 + 1.0))
        else:
            t = math.exp(f2 * math.log(x2) - x2 - GammaFunction().log_value(f2 + 1))

        ans = v * t

        n = 1
        f_2n = df + 2.0
        f_x_2n += 2.0

        # Phase 1 — C++ outer ``for(;;)``: while ``f_x_2n <= 0`` the inner
        # loop runs exactly one term and breaks back out to re-test.
        while True:
            if f_x_2n > 0.0:
                # C++ sets ``flag = true`` and jumps to L10 here. The flag has
                # no other reader once the two phases are separated, so it is
                # folded into the control flow rather than kept as state.
                break
            u *= lam / n
            v += u
            t *= x / f_2n
            ans += v * t
            n += 1
            f_2n += 2.0
            f_x_2n += 2.0
            # C++: ``if (!flag && n <= itrmax) break;`` — the flag is false
            # for the whole of this phase, so the only fall-through to L10 is
            # the iteration cap.
            if n > _ITRMAX:
                break

        # Phase 2 — C++ label ``L10`` onwards.
        bound: float
        while True:
            # f_x_2n is strictly positive on every path that gets here except
            # the iteration-cap escape, where C++ would divide by zero and
            # produce an infinite bound; math.inf reproduces that rather than
            # raising ZeroDivisionError.
            bound = t * x / f_x_2n if f_x_2n != 0.0 else math.inf
            if bound <= _ERRMAX or n > _ITRMAX:
                break
            u *= lam / n
            v += u
            t *= x / f_2n
            ans += v * t
            n += 1
            f_2n += 2.0
            f_x_2n += 2.0
            # C++ repeats ``if (!flag && n <= itrmax) break;`` here. Phase 1
            # only reaches this point having set the flag (or with
            # ``n > itrmax``, which the bound test above has already caught),
            # so the test can never fire and is omitted rather than
            # transcribed into a branch that would be dead but wrong.

        if bound > _ERRMAX:
            qassert.fail("didn't converge")
        return ans


__all__ = ["NonCentralCumulativeChiSquareDistribution"]
