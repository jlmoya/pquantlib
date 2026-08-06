"""BlackVolTimeExtrapolation — time-extrapolation policies for Black vol.

# C++ parity: ql/termstructures/volatility/equityfx/blackvoltimeextrapolation.hpp +
#             blackvoltimeextrapolation.cpp (v1.43).

Static strategy holder used by Black volatility term structures when a
query lands past the last quoted maturity. Three policies:

- ``FlatVolatility`` — hold the *volatility* of the last node flat, i.e.
  grow the variance linearly through the origin::

      var(t) = max(var(t_last), 0) / t_last * t

- ``UseInterpolator`` — let the underlying curve/surface extrapolate
  however it likes, floored at zero.
- ``LinearVariance`` — straight line through the *variance* at the last
  two nodes, extended past the last one.

The two public entry points mirror the two C++ overloads of
``BlackVolTimeExtrapolation::extrapolatedVariance``: one for a
strike-dependent surface, one for an ATM curve. Python has no overload
resolution, so the names carry the discriminant.

Faithfulness note — the ``LinearVariance`` branch of the *curve* overload
carries a ``times.size() >= 2`` precondition that the *surface* overload
does NOT (blackvoltimeextrapolation.cpp:86 vs :74). The asymmetry is kept
rather than smoothed over, because adding the guard to the surface form
would be a silent behavioural change. With a single time C++ then reads
``times[N-2] == times[SIZE_MAX]`` (undefined behaviour); Python's negative
index instead yields ``t1 == t2 == times[0]``, so the call fails
deterministically on ``_linear_extrapolation``'s "times must be sorted"
precondition. Neither language returns a number, which is the property
that matters.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum

from pquantlib import qassert


class BlackVolTimeExtrapolation:
    """Time-extrapolation strategies for Black volatility term structures."""

    class Type(IntEnum):
        """Time-extrapolation strategy.

        # C++ parity: ``BlackVolTimeExtrapolation::Type``. Declaration-order
        # values: FlatVolatility=0, UseInterpolator=1, LinearVariance=2.
        """

        FlatVolatility = 0
        """Flat extrapolation of the latest available volatility."""

        UseInterpolator = 1
        """Delegate extrapolation to the underlying curve or surface."""

        LinearVariance = 2
        """Linear extrapolation of variance from the last two nodes."""

    @staticmethod
    def extrapolated_variance_surface(
        extrapolation_type: BlackVolTimeExtrapolation.Type,
        t: float,
        strike: float,
        times: Sequence[float],
        variance_surface: Callable[[float, float], float],
    ) -> float:
        """Extrapolate a strike-dependent surface to ``(t, strike)``.

        # C++ parity: ``extrapolatedVariance(Type, Time, Real,
        # const std::vector<Time>&, const std::function<Real(Time, Real)>&)``.
        """
        if extrapolation_type == BlackVolTimeExtrapolation.Type.FlatVolatility:
            t_last = times[-1]
            return max(variance_surface(t_last, strike), 0.0) / t_last * t
        if extrapolation_type == BlackVolTimeExtrapolation.Type.UseInterpolator:
            return max(variance_surface(t, strike), 0.0)
        if extrapolation_type == BlackVolTimeExtrapolation.Type.LinearVariance:
            n = len(times)
            t1, t2 = times[n - 2], times[n - 1]
            return _linear_extrapolation(
                t, t1, t2, variance_surface(t1, strike), variance_surface(t2, strike)
            )
        qassert.fail("unknown extrapolation type")

    @staticmethod
    def extrapolated_variance_curve(
        extrapolation_type: BlackVolTimeExtrapolation.Type,
        t: float,
        times: Sequence[float],
        variance_curve: Callable[[float], float],
    ) -> float:
        """Extrapolate an ATM (strike-independent) variance curve to ``t``.

        # C++ parity: ``extrapolatedVariance(Type, Time,
        # const std::vector<Time>&, const std::function<Real(Time)>&)``.
        """
        if extrapolation_type == BlackVolTimeExtrapolation.Type.FlatVolatility:
            t_last = times[-1]
            return max(variance_curve(t_last), 0.0) / t_last * t
        if extrapolation_type == BlackVolTimeExtrapolation.Type.UseInterpolator:
            return max(variance_curve(t), 0.0)
        if extrapolation_type == BlackVolTimeExtrapolation.Type.LinearVariance:
            qassert.require(
                len(times) >= 2,
                "at least two times required for volatility extrapolation",
            )
            n = len(times)
            t1, t2 = times[n - 2], times[n - 1]
            return _linear_extrapolation(
                t, t1, t2, variance_curve(t1), variance_curve(t2)
            )
        qassert.fail("unknown extrapolation type")


def _linear_extrapolation(
    t: float, t1: float, t2: float, v1: float, v2: float
) -> float:
    """Straight line through ``(t1, v1)`` and ``(t2, v2)``, evaluated at ``t``.

    # C++ parity: the anonymous-namespace ``linearExtrapolation`` in
    # blackvoltimeextrapolation.cpp:29-36, preconditions included.
    """
    qassert.require(t > 0.0, "t must be greater than 0.0")
    qassert.require(t > t2, "t must be greater than times[1]")
    qassert.require(t2 > t1, "times must be sorted")
    qassert.require(v2 >= v1, "variances must be non-decreasing")
    return v1 + (t - t1) * (v2 - v1) / (t2 - t1)


__all__ = ["BlackVolTimeExtrapolation"]
