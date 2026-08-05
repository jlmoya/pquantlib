"""Central-limit Gaussian random number generator.

# C++ parity: ql/math/randomnumbers/centrallimitgaussianrng.hpp (v1.43) —
# ``template <class RNG> class CLGaussianRng``.

Uses the fact that the sum of twelve uniform deviates on (-1/2, 1/2) is
approximately standard normal: sum twelve draws on (0, 1) and subtract 6.
Cheap, and visibly wrong in the tails — the support is exactly (-6, 6) — but
it is what C++ ships and what a port has to reproduce.
"""

from __future__ import annotations

from pquantlib.math.randomnumbers.random_number_generator import (
    RandomNumberGenerator,
    Sample,
)


class CLGaussianRng[Rng: RandomNumberGenerator]:
    """Twelve-uniform central-limit Gaussian generator.

    # C++ parity: ``CLGaussianRng<RNG>``
    # (centrallimitgaussianrng.hpp:44-56).
    """

    __slots__ = ("_uniform_generator",)

    def __init__(self, uniform_generator: Rng) -> None:
        # C++ parity: centrallimitgaussianrng.hpp:58-60.
        self._uniform_generator: Rng = uniform_generator

    def next(self) -> Sample:
        """One approximately-standard-normal sample.

        # C++ parity: ``next`` (centrallimitgaussianrng.hpp:62-72). The
        # accumulation order — start at -6.0 and add each draw in turn — is
        # load-bearing for bit-exactness and is kept verbatim.
        """
        gauss_point = -6.0
        gauss_weight = 1.0
        for _ in range(12):
            sample = self._uniform_generator.next()
            gauss_point += sample.value
            gauss_weight *= sample.weight
        return Sample(value=gauss_point, weight=gauss_weight)

    def dimension(self) -> int:
        """Scalar generator — dimension is always 1."""
        return 1


__all__ = ["CLGaussianRng"]
