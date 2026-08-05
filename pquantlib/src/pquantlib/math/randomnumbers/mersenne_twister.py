"""Mersenne Twister MT19937 uniform random number generator.

# C++ parity: ql/math/randomnumbers/mt19937uniformrng.hpp +
# mt19937uniformrng.cpp (v1.43) — the QuantLib wrapper around the
# canonical 1997-2002 Matsumoto/Nishimura implementation.

Period 2^19937 - 1, 32-bit output via Tempering, 624-word state.
Bit-identical sequence to the C++ reference for any nonzero seed.

The 32-bit arithmetic is done with explicit ``& 0xFFFFFFFF`` masks at
every word-store site to match the C++ ``unsigned long`` (32-bit
truncation) behavior portably under Python's arbitrary-precision ints.

Both C++ constructors are here: the single-seed one and the
``std::vector<unsigned long>`` one (Matsumoto's ``init_by_array``), exposed
as :meth:`MersenneTwisterUniformRng.from_seeds`. The latter is what
``SeedGenerator`` builds its generator from, so it is not optional.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from pquantlib.math.randomnumbers.random_number_generator import Sample

_N: Final[int] = 624  # state size
_M: Final[int] = 397  # shift size
_MATRIX_A: Final[int] = 0x9908B0DF
_UPPER_MASK: Final[int] = 0x80000000  # most significant w-r bits
_LOWER_MASK: Final[int] = 0x7FFFFFFF  # least significant r bits
_MASK32: Final[int] = 0xFFFFFFFF
_INV_2_POW_32: Final[float] = 1.0 / 4294967296.0


class MersenneTwisterUniformRng:
    """MT19937 uniform RNG over (0.0, 1.0).

    # C++ parity: ``MersenneTwisterUniformRng`` in
    # ql/math/randomnumbers/mt19937uniformrng.{hpp,cpp} (v1.43).

    Construct with a nonzero seed for deterministic sequences. Seed 0 means
    "pick a seed from ``SeedGenerator``", which is clock-derived and therefore
    not reproducible — exactly as documented in C++ ("if the given seed is 0,
    a random seed will be chosen based on clock()").
    """

    __slots__ = ("_mt", "_mti")

    def __init__(self, seed: int) -> None:
        # Mutable state. C++ marks both as ``mutable``; Python plain attrs.
        self._mt: list[int] = [0] * _N
        self._mti: int = _N
        self._seed_initialization(seed)

    @classmethod
    def from_seeds(cls, seeds: Sequence[int]) -> MersenneTwisterUniformRng:
        """Seed from a key array (Matsumoto's ``init_by_array``).

        # C++ parity: ``MersenneTwisterUniformRng(const std::vector<unsigned
        # long>&)`` (mt19937uniformrng.cpp:104-123).
        """
        rng = cls.__new__(cls)
        rng._mt = [0] * _N
        rng._mti = _N
        rng._seed_initialization(19650218)
        mt = rng._mt
        n_seeds = len(seeds)
        i = 1
        j = 0
        k = _N if n_seeds < _N else n_seeds
        while k:
            mt[i] = (
                (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525)) + seeds[j] + j
            ) & _MASK32
            i += 1
            j += 1
            if i >= _N:
                mt[0] = mt[_N - 1]
                i = 1
            if j >= n_seeds:
                j = 0
            k -= 1
        for _ in range(_N - 1, 0, -1):
            mt[i] = ((mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941)) - i) & _MASK32
            i += 1
            if i >= _N:
                mt[0] = mt[_N - 1]
                i = 1
        # MSB is 1, assuring a non-zero initial array.
        mt[0] = _UPPER_MASK
        return rng

    def _seed_initialization(self, seed: int) -> None:
        # C++ parity: mt19937uniformrng.cpp:86-100.
        if seed == 0:
            # C++ parity: mt19937uniformrng.cpp:88 — seed 0 defers to the
            # clock-seeded SeedGenerator singleton. Imported here rather than
            # at module scope because SeedGenerator is built out of this very
            # class (C++ has the same cycle via the .cpp include).
            from pquantlib.math.randomnumbers.seed_generator import (  # noqa: PLC0415
                SeedGenerator,
            )

            seed = SeedGenerator.instance().get()
        self._mt[0] = seed & _MASK32
        for mti in range(1, _N):
            prev = self._mt[mti - 1]
            self._mt[mti] = (1812433253 * (prev ^ (prev >> 30)) + mti) & _MASK32
        self._mti = _N

    def _twist(self) -> None:
        # C++ parity: mt19937uniformrng.cpp:125-143.
        mt = self._mt
        # mag01[x] = x * MATRIX_A  for x = 0, 1
        for kk in range(_N - _M):
            y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
            mt[kk] = mt[kk + _M] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
        for kk in range(_N - _M, _N - 1):
            y = (mt[kk] & _UPPER_MASK) | (mt[kk + 1] & _LOWER_MASK)
            mt[kk] = mt[kk + (_M - _N)] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
        y = (mt[_N - 1] & _UPPER_MASK) | (mt[0] & _LOWER_MASK)
        mt[_N - 1] = mt[_M - 1] ^ (y >> 1) ^ (_MATRIX_A if y & 1 else 0)
        self._mti = 0

    def next_int32(self) -> int:
        """Return a random integer in ``[0, 0xFFFFFFFF]`` (C++ ``nextInt32``)."""
        if self._mti == _N:
            self._twist()
        y = self._mt[self._mti]
        self._mti += 1
        # Tempering — every ``&= _MASK32`` matches the C++ ``unsigned long``
        # truncation that Python's arbitrary-precision int does not perform.
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y &= _MASK32
        y ^= (y << 15) & 0xEFC60000
        y &= _MASK32
        y ^= y >> 18
        return y & _MASK32

    def next_real(self) -> float:
        """Return a random uniform in (0.0, 1.0) — C++ ``nextReal``."""
        # C++ parity: mt19937uniformrng.hpp:57.
        return (float(self.next_int32()) + 0.5) * _INV_2_POW_32

    def next(self) -> Sample:
        """One sample with weight 1.0."""
        return Sample(value=self.next_real(), weight=1.0)

    def dimension(self) -> int:
        """Scalar RNG — dimension is always 1."""
        return 1
