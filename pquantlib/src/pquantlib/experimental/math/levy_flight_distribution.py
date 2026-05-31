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

Divergence (documented): the C++ ``operator()(Engine&)`` draws ``u``
from ``std::uniform_real_distribution<Real>(0,1)(eng)`` over a
``std::mt19937`` engine. PQuantLib drives the inverse transform off a
``MersenneTwisterUniformRng`` (the QuantLib MT wrapper, whose
``next_real()`` is bit-identical to C++ ``mt.nextReal()``). The
*transform* ``xm * u^{-1/alpha}`` is identical on both sides; only the
uniform-draw source differs from ``std::mt19937`` — a deliberate choice
because the IsotropicRandomWalk / firefly consumers in QuantLib already
mix QuantLib-MT and std::mt19937 streams, and PQuantLib standardises on
the QuantLib MT throughout this cluster for reproducibility.
"""

from __future__ import annotations

from typing import Protocol

from pquantlib import qassert


class _UniformEngine(Protocol):
    """Structural type for a uniform source exposing ``next_real()``.

    ``MersenneTwisterUniformRng`` satisfies this; the inverse transform
    only needs a uniform draw in (0, 1).
    """

    def next_real(self) -> float: ...


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

    __slots__ = ("_alpha", "_xm")

    def __init__(self, xm: float = 1.0, alpha: float = 1.0) -> None:
        qassert.require(alpha > 0.0, "alpha must be larger than 0")
        self._xm: float = xm
        self._alpha: float = alpha

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

    def __call__(self, rng: _UniformEngine) -> float:
        """Draw a random variate via inverse transform off ``rng``.

        # C++ parity: ``template<class Engine> operator()(Engine&)``
        # levyflightdistribution.hpp:128-132 — ``xm * u^{-1/alpha}``.

        ``rng`` must expose ``next_real() -> float`` returning a uniform
        in (0, 1) (e.g. ``MersenneTwisterUniformRng``). See the module
        docstring for the uniform-source divergence note.
        """
        u = rng.next_real()
        return self._xm * u ** (-1.0 / self._alpha)
