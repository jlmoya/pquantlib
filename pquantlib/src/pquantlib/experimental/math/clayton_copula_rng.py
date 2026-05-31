"""ClaytonCopulaRng — Clayton copula random-number generator.

# C++ parity: ql/experimental/math/claytoncopularng.hpp @ v1.42.1 (099987f0).

Wraps a uniform RNG and emits 2-vectors ``[u1, u2]`` of uniforms with the
Clayton-copula dependence structure (conditional-distribution sampling). The
parameter ``theta`` must satisfy ``theta >= -1`` and ``theta != 0``.
"""

from __future__ import annotations

from pquantlib import qassert
from pquantlib.experimental.math.copula_sample import VectorSample
from pquantlib.math.randomnumbers.random_number_generator import (
    RandomNumberGenerator,
)


class ClaytonCopulaRng:
    """Clayton copula random-number generator."""

    __slots__ = ("_theta", "_uniform_generator")

    def __init__(self, uniform_generator: RandomNumberGenerator, theta: float) -> None:
        qassert.require(
            theta >= -1.0, f"theta ({theta}) must be greater or equal to -1"
        )
        qassert.require(theta != 0.0, f"theta ({theta}) must be different from 0")
        self._uniform_generator = uniform_generator
        self._theta = theta

    def next(self) -> VectorSample:
        """Draw a correlated 2-vector ``[u1, u2]`` of uniforms."""
        v1 = self._uniform_generator.next()
        v2 = self._uniform_generator.next()
        theta = self._theta
        u1 = v1.value
        u2 = (
            v1.value**-theta * (v2.value ** (-theta / (theta + 1.0)) - 1.0) + 1.0
        ) ** (-1.0 / theta)
        return VectorSample([u1, u2], v1.weight * v2.weight)
