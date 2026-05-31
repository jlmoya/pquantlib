"""PolarStudentTRng — polar-method Student-t random-number generator.

# C++ parity: ql/experimental/math/polarstudenttrng.hpp @ v1.42.1 (099987f0).

Polar-transformation Student-t generator (Bailey 1994; the variant from
*Random Number Generation and Monte Carlo Methods*, Springer 2003, p. 185).
Using uniforms remapped to ``[-1, 1]`` it avoids the extra draw for the sign.

# C++ parity divergence: the C++ template is parameterised on a uniform RNG
# ``URNG`` and offers two constructors — one taking ``(degFreedom, seed)`` that
# builds the RNG internally, and one taking ``(degFreedom, urng)``. The Python
# port keeps the second (inject the uniform generator); the seed-only form is
# obtained by passing a freshly-seeded ``MersenneTwisterUniformRng``. This keeps
# the class RNG-agnostic without binding it to one concrete generator.

Warning: do not use with a low-discrepancy sequence generator.
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.math.randomnumbers.random_number_generator import (
    RandomNumberGenerator,
    Sample,
)


class PolarStudentTRng:
    """Polar-method Student-t random-number generator."""

    __slots__ = ("_deg_freedom", "_uniform_generator")

    def __init__(
        self, deg_freedom: float, uniform_generator: RandomNumberGenerator
    ) -> None:
        qassert.require(
            deg_freedom > 0, "Invalid degrees of freedom parameter."
        )
        self._deg_freedom = deg_freedom
        self._uniform_generator = uniform_generator

    def next(self) -> Sample:
        """Draw a single Student-t variate."""
        while True:
            # samples remapped to [-1, 1]:
            v = 2.0 * self._uniform_generator.next().value - 1.0
            u = 2.0 * self._uniform_generator.next().value - 1.0
            r_sqr = v * v + u * u
            if r_sqr < 1.0:
                break
        value = u * math.sqrt(
            self._deg_freedom
            * (r_sqr ** (-2.0 / self._deg_freedom) - 1.0)
            / r_sqr
        )
        return Sample(value=value, weight=1.0)
