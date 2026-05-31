"""FarlieGumbelMorgensternCopulaRng — FGM copula random-number generator.

# C++ parity: ql/experimental/math/farliegumbelmorgensterncopularng.hpp
# @ v1.42.1 (099987f0).

Wraps a uniform RNG and emits 2-vectors ``[u1, u2]`` of uniforms with the
Farlie-Gumbel-Morgenstern dependence structure. The parameter ``theta`` must lie
in ``[-1, 1]``.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.experimental.math.copula_sample import VectorSample
from pquantlib.math.randomnumbers.random_number_generator import (
    RandomNumberGenerator,
)


class FarlieGumbelMorgensternCopulaRng:
    """Farlie-Gumbel-Morgenstern copula random-number generator."""

    __slots__ = ("_theta", "_uniform_generator")

    def __init__(self, uniform_generator: RandomNumberGenerator, theta: float) -> None:
        qassert.require(
            -1.0 <= theta <= 1.0, f"theta ({theta}) must be in [-1,1]"
        )
        self._uniform_generator = uniform_generator
        self._theta = theta

    def next(self) -> VectorSample:
        """Draw a correlated 2-vector ``[u1, u2]`` of uniforms."""
        v1 = self._uniform_generator.next()
        v2 = self._uniform_generator.next()
        theta = self._theta
        u1 = v1.value
        a = theta * (2.0 * u1 - 1.0)
        b = (1.0 - theta * (2.0 * u1 - 1.0)) ** 2.0 + 4.0 * theta * v2.value * (
            2.0 * u1 - 1.0
        )
        u2 = (2.0 * v2.value) / (math.sqrt(b) - a)
        return VectorSample([u1, u2], v1.weight * v2.weight)
