"""Random sequence generator over a pseudo-random number generator.

# C++ parity: ql/math/randomnumbers/randomsequencegenerator.hpp (v1.43) —
# ``template<class RNG> class RandomSequenceGenerator``.

Stacks ``dimensionality`` consecutive scalar draws into one vector and
multiplies their weights together. The C++ warning applies verbatim: do not
use this with a low-discrepancy generator — consecutive draws of an LDS are
not independent coordinates, so slicing them this way destroys the very
equidistribution the sequence was built for.
"""

from __future__ import annotations

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.random_number_generator import (
    RandomNumberGenerator,
    SequenceSample,
)


class RandomSequenceGenerator[Rng: RandomNumberGenerator = MersenneTwisterUniformRng]:
    """Vector-valued generator built from a scalar one.

    # C++ parity: ``RandomSequenceGenerator<RNG>``
    # (randomsequencegenerator.hpp:51-92).

    Two constructions mirror the C++ overloads:

    * ``RandomSequenceGenerator(dim, rng)`` wraps an existing generator;
    * ``RandomSequenceGenerator.from_seed(dim, seed)`` builds a
      ``MersenneTwisterUniformRng`` — the C++ ``explicit`` overload, which is
      only well-formed for RNGs constructible from a ``BigNatural`` seed and
      is therefore a named constructor here rather than an overload.

    Note the C++ asymmetry, reproduced as-is: only the two-argument
    constructor checks ``dimensionality > 0``.
    """

    __slots__ = ("_dim", "_int32_sequence", "_rng", "_sequence", "_weight")

    def __init__(self, dimensionality: int, rng: Rng) -> None:
        # C++ parity: randomsequencegenerator.hpp:55-62.
        qassert.require(dimensionality > 0, "dimensionality must be greater than 0")
        self._dim: int = dimensionality
        self._rng: Rng = rng
        self._sequence: Array = np.zeros(dimensionality, dtype=np.float64)
        self._weight: float = 1.0
        self._int32_sequence: list[int] = [0] * dimensionality

    @staticmethod
    def from_seed(
        dimensionality: int, seed: int = 0
    ) -> RandomSequenceGenerator[MersenneTwisterUniformRng]:
        """Build over a ``MersenneTwisterUniformRng``.

        # C++ parity: ``explicit RandomSequenceGenerator(Size, BigNatural)``
        # (randomsequencegenerator.hpp:64-68) — the default ``RNG`` of every
        # QuantLib trait that uses this class is the Mersenne Twister.
        Seed 0 defers to the clock-seeded ``SeedGenerator``, as in C++.

        Built through ``__new__`` rather than ``__init__`` because the C++
        seed constructor, unlike the RNG one, does *not* check
        ``dimensionality > 0``.
        """
        gen: RandomSequenceGenerator[MersenneTwisterUniformRng] = (
            RandomSequenceGenerator.__new__(RandomSequenceGenerator)
        )
        gen._dim = dimensionality
        gen._rng = MersenneTwisterUniformRng(seed)
        gen._sequence = np.zeros(dimensionality, dtype=np.float64)
        gen._weight = 1.0
        gen._int32_sequence = [0] * dimensionality
        return gen

    def next_sequence(self) -> SequenceSample:
        """Next ``dimensionality``-vector and its accumulated weight.

        # C++ parity: ``nextSequence`` (randomsequencegenerator.hpp:70-79).
        """
        weight = 1.0
        seq = self._sequence
        for i in range(self._dim):
            sample = self._rng.next()
            seq[i] = sample.value
            weight *= sample.weight
        self._weight = weight
        return SequenceSample(value=seq.copy(), weight=weight)

    def next_int32_sequence(self) -> list[int]:
        """Next vector of raw 32-bit integers from the underlying generator.

        # C++ parity: ``nextInt32Sequence``
        # (randomsequencegenerator.hpp:80-85). Requires the wrapped RNG to
        # expose ``next_int32`` (C++ requires ``nextInt32``).
        """
        next_int32 = getattr(self._rng, "next_int32", None)
        qassert.require(
            callable(next_int32),
            f"{type(self._rng).__name__} does not provide next_int32",
        )
        assert next_int32 is not None  # pyright narrowing aid
        for i in range(self._dim):
            self._int32_sequence[i] = int(next_int32())
        return list(self._int32_sequence)

    def last_sequence(self) -> SequenceSample:
        """Most recent draw.

        # C++ parity: ``lastSequence`` (randomsequencegenerator.hpp:86-88).
        """
        return SequenceSample(value=self._sequence.copy(), weight=self._weight)

    def dimension(self) -> int:
        """Output vector dimension.

        # C++ parity: ``dimension`` (randomsequencegenerator.hpp:89).
        """
        return self._dim


__all__ = ["RandomSequenceGenerator"]
