"""AbcdFunction — Rebonato abcd instantaneous-volatility functional.

# C++ parity: ql/termstructures/volatility/abcd.{hpp,cpp} (v1.42.1) —
#             ``AbcdFunction`` (a thin subclass of ``AbcdMathFunction``).

The Rebonato (2003) functional form for instantaneous volatility:

.. math::

   f(T - t) = [a + b (T - t)] e^{-c (T - t)} + d

This helper provides the covariance / variance integrals used by the
concrete market models (``AbcdVol`` and ``PiecewiseConstantAbcdVariance``):

- ``covariance(t, T, S)`` — instantaneous covariance ``f(T-t) f(S-t)``.
- ``covariance(t1, t2, T, S)`` — integral of the instantaneous covariance
  between ``t1`` and ``t2`` for ``T``- and ``S``-fixing rates, via the
  closed-form ``primitive``.
- ``variance(t_min, t_max, T)`` — ``covariance(t_min, t_max, T, T)``.

# C++ parity note: pquantlib already has ``abcd_value`` (the
# ``AbcdMathFunction::operator()``) in
# ``math.interpolations.abcd_interpolation`` and a fitting-only
# ``AbcdCalibration``; neither exposes the covariance/variance integrals, so
# this module ports the integral surface from ``AbcdFunction`` directly. The
# overloaded C++ ``covariance`` is split into two distinctly-named Python
# methods (``instantaneous_covariance`` and ``covariance``) to avoid
# argument-count overloading.

The ``primitive`` closed form is the indefinite integral
``int f(T-t) f(S-t) dt`` (with the ``c == 0`` degenerate branch handled
separately), copied verbatim from the C++ ``AbcdFunction::primitive``.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.math.interpolations.abcd_interpolation import abcd_value, validate_abcd


class AbcdFunction:
    """Rebonato abcd instantaneous-volatility functional + integrals.

    # C++ parity: abcd.hpp/.cpp ``AbcdFunction``.
    """

    def __init__(
        self,
        a: float = -0.06,
        b: float = 0.17,
        c: float = 0.54,
        d: float = 0.17,
    ) -> None:
        validate_abcd(a, b, c, d)
        # C++ parity: AbcdMathFunction stores a_, b_, c_, d_.
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def __call__(self, t: float) -> float:
        """Evaluate ``f(t) = (a + b t) e^{-c t} + d`` (0 for ``t < 0``).

        # C++ parity: AbcdMathFunction::operator().
        """
        return abcd_value(t, self._a, self._b, self._c, self._d)

    # --- covariance / variance --------------------------------------------

    def instantaneous_covariance(self, u: float, t: float, s: float) -> float:
        """Instantaneous covariance ``f(T-u) f(S-u)`` at time ``u``.

        # C++ parity: AbcdFunction::covariance(Time t, Time T, Time S) /
        # instantaneousCovariance.
        """
        return self(t - u) * self(s - u)

    def covariance(self, t1: float, t2: float, t: float, s: float) -> float:
        """Integral of the instantaneous covariance over ``[t1, t2]``.

        # C++ parity: AbcdFunction::covariance(Time t1, Time t2, Time T, Time S).
        """
        qassert.require(
            t1 <= t2,
            f"integrations bounds ({t1},{t2}) are in reverse order",
        )
        cut_off = min(s, t)
        if t1 >= cut_off:
            return 0.0
        cut_off = min(t2, cut_off)
        return self.primitive(cut_off, t, s) - self.primitive(t1, t, s)

    def variance(self, t_min: float, t_max: float, t: float) -> float:
        """Variance over ``[t_min, t_max]`` of the ``T``-fixing rate.

        # C++ parity: AbcdFunction::variance.
        """
        return self.covariance(t_min, t_max, t, t)

    def volatility(self, t_min: float, t_max: float, t: float) -> float:
        """Average volatility over ``[t_min, t_max]`` of the ``T``-fixing rate.

        # C++ parity: AbcdFunction::volatility.
        """
        if t_max == t_min:
            return self.instantaneous_volatility(t_max, t)
        qassert.require(t_max > t_min, "tMax must be > tMin")
        return math.sqrt(self.variance(t_min, t_max, t) / (t_max - t_min))

    def instantaneous_volatility(self, u: float, t: float) -> float:
        """Instantaneous volatility ``f(T-u)`` at time ``u``.

        # C++ parity: AbcdFunction::instantaneousVolatility.
        """
        return math.sqrt(self.instantaneous_covariance(u, t, t))

    # --- primitive (indefinite covariance integral) -----------------------

    def primitive(self, t: float, t_fix: float, s: float) -> float:
        """Indefinite integral ``int f(T-t) f(S-t) dt`` evaluated at ``t``.

        # C++ parity: AbcdFunction::primitive. ``t_fix`` is the C++ ``T``.
        """
        a = self._a
        b = self._b
        c = self._c
        d = self._d
        # C++ parity: uses the C++ arg names T (=t_fix) and S (=s).
        big_t = t_fix
        if big_t < t or s < t:
            return 0.0

        if close(c, 0.0):
            v = a + d
            return t * (
                v * v
                + v * b * s
                + v * b * big_t
                - v * b * t
                + b * b * s * big_t
                - 0.5 * b * b * t * (s + big_t)
                + b * b * t * t / 3.0
            )

        k1 = math.exp(c * t)
        k2 = math.exp(c * s)
        k3 = math.exp(c * big_t)

        return (
            b * b * (
                -1
                - 2 * c * c * s * big_t
                - c * (s + big_t)
                + k1 * k1 * (1 + c * (s + big_t - 2 * t) + 2 * c * c * (s - t) * (big_t - t))
            )
            + 2 * c * c * (
                2 * d * a * (k2 + k3) * (k1 - 1)
                + a * a * (k1 * k1 - 1)
                + 2 * c * d * d * k2 * k3 * t
            )
            + 2 * b * c * (
                a * (-1 - c * (s + big_t) + k1 * k1 * (1 + c * (s + big_t - 2 * t)))
                - 2 * d * (
                    k3 * (1 + c * s)
                    + k2 * (1 + c * big_t)
                    - k1 * k3 * (1 + c * (s - t))
                    - k1 * k2 * (1 + c * (big_t - t))
                )
            )
        ) / (4 * c * c * c * k2 * k3)
