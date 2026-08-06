"""Lévy-flight (Pareto Type I) distribution.

# C++ parity: ql/experimental/math/levyflightdistribution.hpp (v1.42.1)
# (Copyright 2015 Andres Hernandez).

The Lévy-flight distribution has pdf

.. math::
    p(x) = \\frac{\\alpha\\, x_m^{\\alpha}}{x^{\\alpha+1}}, \\quad x \\ge x_m

with parameter :math:`\\alpha > 0`. The classic Lévy flight fixes
:math:`x_m = 1` and :math:`0 < \\alpha < 2` (infinite variance); the
general Pareto Type I admits any :math:`\\alpha > 0`, which the C++
implementation (and this port) allow.

Random variates are produced by inverse transform:
:math:`x = x_m\\, u^{-1/\\alpha}` for :math:`u \\sim U(0, 1)`.

The C++ ``operator()(Engine&)`` draws ``u`` from
``std::uniform_real_distribution<Real>(0,1)(eng)`` over a
``std::mt19937``. That is **two** engine words combined by
``generate_canonical``, not QuantLib's one-word ``nextReal``, so the
uniform source is part of the contract and not an implementation
detail: driving the same transform off a QuantLib ``nextReal`` produces
a different sequence. The port therefore takes a
:class:`~pquantlib.experimental.math.std_random.StdMt19937` and goes
through :class:`~pquantlib.experimental.math.std_random.StdUniformRealDistribution`,
which reproduces the C++ stream exactly (cross-validated in
``migration-harness/references/v143/experimental/pso.json``, block B).
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.experimental.math.std_random import (
    StdMt19937,
    StdUniformRealDistribution,
)


class LevyFlightDistribution:
    """Lévy-flight / Pareto-Type-I distribution.

    # C++ parity: ``class LevyFlightDistribution`` in
    # ql/experimental/math/levyflightdistribution.hpp:51-148 (v1.42.1).

    Parameters
    ----------
    xm:
        Lower-support / scale parameter (``x >= xm``). Default 1.0.
    alpha:
        Tail exponent (must be > 0). Default 1.0.
    """

    __slots__ = ("_alpha", "_uniform", "_xm")

    def __init__(self, xm: float = 1.0, alpha: float = 1.0) -> None:
        qassert.require(alpha > 0.0, "alpha must be larger than 0")
        self._xm: float = xm
        self._alpha: float = alpha
        self._uniform: StdUniformRealDistribution = StdUniformRealDistribution(0.0, 1.0)

    @property
    def xm(self) -> float:
        """Scale parameter ``x_m``."""
        return self._xm

    @property
    def alpha(self) -> float:
        """Tail exponent ``alpha``."""
        return self._alpha

    def min(self) -> float:
        """Smallest value the distribution can produce (``x_m``).

        # C++ parity: levyflightdistribution.hpp:101-102.
        """
        return self._xm

    def pdf(self, x: float) -> float:
        """Probability density at ``x`` (0 for ``x < x_m``).

        # C++ parity: ``operator()(Real x)``
        # levyflightdistribution.hpp:119-123.
        """
        if x < self._xm:
            return 0.0
        return self._alpha * (self._xm / x) ** self._alpha / x

    def __call__(self, engine: StdMt19937) -> float:
        """Draw a random variate via inverse transform off ``engine``.

        # C++ parity: ``template<class Engine> operator()(Engine&)``
        # levyflightdistribution.hpp:128-131 — ``xm * u^{-1/alpha}`` with
        # ``u`` from ``std::uniform_real_distribution<Real>(0,1)``.
        """
        u = self._uniform(engine)
        return self._xm * u ** (-1.0 / self._alpha)
