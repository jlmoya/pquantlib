"""Mersenne Twister MT19937 uniform random number generator.

# C++ parity: ql/math/randomnumbers/mt19937uniformrng.hpp +
# mt19937uniformrng.cpp (v1.42.1) — the QuantLib wrapper around the
# canonical 1997-2002 Matsumoto/Nishimura implementation.

Period 2^19937 - 1, 32-bit output via Tempering, 624-word state.
Bit-identical sequence to the C++ reference for any nonzero seed.

The 32-bit arithmetic is done with explicit ``& 0xFFFFFFFF`` masks at
every word-store site to match the C++ ``unsigned long`` (32-bit
truncation) behavior portably under Python's arbitrary-precision ints.
"""

from __future__ import annotations

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
    # ql/math/randomnumbers/mt19937uniformrng.{hpp,cpp} (v1.42.1).

    Construct with a nonzero seed for deterministic sequences. Seed 0 is
    rejected (the C++ version falls back to ``SeedGenerator``, which is
    deferred to a later cluster — pquantlib refuses seed 0 explicitly
    rather than silently picking a clock-derived seed). Multi-word
    seeding (``vector<unsigned long>`` ctor in C++) is deferred until a
    consumer requires it.
    """

    __slots__ = ("_mt", "_mti")

    def __init__(self, seed: int) -> None:
        # Diverge from C++ default-seed-0 behavior: pquantlib does not
        # yet have ``SeedGenerator``. Require an explicit nonzero seed.
        if seed == 0:
            raise ValueError(
                "MersenneTwisterUniformRng requires nonzero seed "
                "(C++ SeedGenerator clock fallback not yet ported)"
            )
        # Mutable state. C++ marks both as ``mutable``; Python plain attrs.
        self._mt: list[int] = [0] * _N
        self._mti: int = _N
        self._seed_initialization(seed)

    def _seed_initialization(self, seed: int) -> None:
        # C++ parity: mt19937uniformrng.cpp:86-100.
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
