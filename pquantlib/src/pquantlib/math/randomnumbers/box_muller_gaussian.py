"""Box-Muller Gaussian RNG over a uniform RNG.

# C++ parity: ql/math/randomnumbers/boxmullergaussianrng.hpp (v1.42.1) —
# the templated ``BoxMullerGaussianRng<RNG>``.

The C++ implementation uses Marsaglia's polar variant of Box-Muller:
two uniforms in (-1, 1) sampled until ``r = x1^2 + x2^2`` is in (0, 1),
then ``ratio = sqrt(-2 * ln(r) / r)`` converts the pair to two
independent standard normals. The samples are emitted on alternating
``next()`` calls — the first call returns ``x1 * ratio`` and caches
``x2 * ratio`` for the second call.

The Python ``BoxMullerGaussianRng`` is a PEP 695 generic over the
uniform-RNG type for static-type narrowing; the stored attribute is
the uniform RNG instance itself, mirroring C++ value-semantics.
"""

from __future__ import annotations

import math

from pquantlib.math.randomnumbers.random_number_generator import RandomNumberGenerator, Sample


class BoxMullerGaussianRng[Urng: RandomNumberGenerator]:
    """Box-Muller Gaussian RNG over a uniform RNG.

    # C++ parity: ``BoxMullerGaussianRng<RNG>`` in
    # ql/math/randomnumbers/boxmullergaussianrng.hpp (v1.42.1).

    The PEP 695 ``[Urng: RandomNumberGenerator]`` upper-bound mirrors
    the C++ duck-typed template parameter at the type-checker level
    while preserving structural compatibility at runtime.
    """

    __slots__ = (
        "_first_weight",
        "_return_first",
        "_second_value",
        "_uniform_generator",
        "_weight",
    )

    def __init__(self, uniform_generator: Urng) -> None:
        # C++ parity: hpp:61 — value-stored uniformGenerator_.
        self._uniform_generator: Urng = uniform_generator
        # Mutable state — C++ ``mutable``; Python plain attrs.
        self._return_first: bool = True
        self._second_value: float = 0.0
        self._first_weight: float = 0.0
        self._weight: float = 0.0

    def next(self) -> Sample:
        """One standard-normal sample with weight = product of uniform weights."""
        # C++ parity: hpp:64-89.
        if self._return_first:
            # Marsaglia polar rejection: keep sampling until r in (0, 1).
            while True:
                s1 = self._uniform_generator.next()
                x1 = s1.value * 2.0 - 1.0
                self._first_weight = s1.weight
                s2 = self._uniform_generator.next()
                x2 = s2.value * 2.0 - 1.0
                second_weight = s2.weight
                r = x1 * x1 + x2 * x2
                if r < 1.0 and r != 0.0:
                    break
            ratio = math.sqrt(-2.0 * math.log(r) / r)
            first_value = x1 * ratio
            self._second_value = x2 * ratio
            self._weight = self._first_weight * second_weight
            self._return_first = False
            return Sample(value=first_value, weight=self._weight)
        # Cached half — flip back and emit the cached second value.
        self._return_first = True
        return Sample(value=self._second_value, weight=self._weight)

    def dimension(self) -> int:
        """Scalar RNG — dimension is always 1."""
        return 1
