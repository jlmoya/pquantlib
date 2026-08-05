"""Richardson extrapolation — sequence acceleration in the step size.

# C++ parity: ql/math/richardsonextrapolation.{hpp,cpp} (v1.43).

For ``f(dh) = f_0 + alpha * dh^n + O(dh^{n+1})``, extrapolate to ``dh -> 0``.

Two overloads, with genuinely different content:

* known order ``n`` — one closed-form combination of ``f(dh)`` and ``f(dh/t)``;
* unknown order — estimate ``n`` first, by scanning ``k`` upward from 0.05 in
  steps of 0.1 until a sign change brackets a root of the two-scaling-factor
  consistency equation, then Brent on that bracket to 1e-8. The scan, its step
  and its 15.1 cut-off are all part of the answer: a different bracketing
  strategy finds a different root of the same equation when there is more than
  one, and gives up in different places.

``f`` is evaluated once in the constructor (at ``delta_h``) and cached, so a
stateful or expensive ``f`` sees exactly the C++ call pattern.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from pquantlib import qassert
from pquantlib.math.solvers1d.brent import Brent


class _RichardsonEqn:
    """Consistency equation whose root is the order of convergence.

    # C++ parity: anonymous-namespace ``class RichardsonEqn`` —
    # richardsonextrapolation.cpp:30-44.
    """

    __slots__ = ("_fdelta_h", "_fs", "_ft", "_s", "_t")

    def __init__(self, fh: float, ft: float, fs: float, t: float, s: float) -> None:
        self._fdelta_h: float = fh
        self._ft: float = ft
        self._fs: float = fs
        self._t: float = t
        self._s: float = s

    def __call__(self, k: float) -> float:
        return (self._ft + (self._ft - self._fdelta_h) / (math.pow(self._t, k) - 1.0)) - (
            self._fs + (self._fs - self._fdelta_h) / (math.pow(self._s, k) - 1.0)
        )


class RichardsonExtrapolation:
    """Extrapolate ``f(delta_h)`` to ``delta_h -> 0``.

    # C++ parity: ``class RichardsonExtrapolation`` —
    # richardsonextrapolation.hpp:42-62, richardsonextrapolation.cpp:47-92.

    Parameters
    ----------
    f: function to be extrapolated.
    delta_h: step size.
    n: order of convergence, if known; ``None`` mirrors C++ ``Null<Real>()``.
    """

    __slots__ = ("_delta_h", "_f", "_fdelta_h", "_n")

    def __init__(
        self,
        f: Callable[[float], float],
        delta_h: float,
        n: float | None = None,
    ) -> None:
        self._delta_h: float = delta_h
        # C++ evaluates f(delta_h) in the member-initialiser list.
        self._fdelta_h: float = f(delta_h)
        self._n: float | None = n
        self._f: Callable[[float], float] = f

    def __call__(self, t: float = 2.0) -> float:
        """Extrapolation for a known order of convergence.

        # C++ parity: richardsonextrapolation.cpp:57-66.
        """
        qassert.require(t > 1, "scaling factor must be greater than 1")
        qassert.require(self._n is not None, "order of convergence must be known")
        assert self._n is not None  # narrowing for the type checker

        tk = math.pow(t, self._n)
        return (tk * self._f(self._delta_h / t) - self._fdelta_h) / (tk - 1.0)

    def with_unknown_order(self, t: float, s: float) -> float:
        """Extrapolation for an unknown order of convergence.

        # C++ parity: the two-argument ``operator()(Real t, Real s)`` —
        # richardsonextrapolation.cpp:68-91. Python cannot overload on arity,
        # so the second form gets its own name.
        """
        qassert.require(t > 1 and s > 1, "scaling factors must be greater than 1")
        qassert.require(t > s, "t must be greater than s")

        ft = self._f(self._delta_h / t)
        fs = self._f(self._delta_h / s)

        eqn = _RichardsonEqn(self._fdelta_h, ft, fs, t, s)

        step = 0.1
        left = 0.05
        fr = eqn(left + step)
        fl = eqn(left)
        while fr * fl > 0.0 and left < 15.1:
            left += step
            fl = fr
            fr = eqn(left + step)

        qassert.require(left < 15.1, "could not estimate the order of convergence")

        k = Brent().solve(eqn, 1e-8, left + 0.5 * step, left, left + step)

        ts = math.pow(s, k)

        return (ts * fs - self._fdelta_h) / (ts - 1.0)


__all__ = ["RichardsonExtrapolation"]
