"""Xoshiro256** uniform random number generator.

# C++ parity: ql/math/randomnumbers/xoshiro256starstaruniformrng.hpp +
# xoshiro256starstaruniformrng.cpp (v1.42.1) — Blackman & Vigna's 2018
# xoshiro256** reference C implementation, wrapped in a QuantLib class.

Period 2^256 - 1, 64-bit output, four 64-bit state words. The state is
seeded by running a SplitMix64 (Vigna 2015) from the user seed for four
iterations — this is what the spec calls "seed expansion".

Floating-point output uses ``(nextInt64() >> 11 + 0.5) / 2^53`` so that
the entire 53-bit mantissa is filled randomly. Bit-identical sequences
to the C++ reference for any nonzero seed.
"""

from __future__ import annotations

from typing import Final

from pquantlib.math.randomnumbers.random_number_generator import Sample

_MASK64: Final[int] = (1 << 64) - 1
_INV_2_POW_53: Final[float] = 1.0 / (1 << 53)
_SPLITMIX_ADD: Final[int] = 0x9E3779B97F4A7C15
_SPLITMIX_MUL1: Final[int] = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL2: Final[int] = 0x94D049BB133111EB


class _SplitMix64:
    """SplitMix64 — Sebastiano Vigna 2015, used as Xoshiro seed expander.

    # C++ parity: anonymous-namespace ``SplitMix64`` class in
    # ql/math/randomnumbers/xoshiro256starstaruniformrng.cpp:39-51 (v1.42.1).
    """

    __slots__ = ("_x",)

    def __init__(self, x: int) -> None:
        self._x: int = x & _MASK64

    def next(self) -> int:
        self._x = (self._x + _SPLITMIX_ADD) & _MASK64
        z = self._x
        z = ((z ^ (z >> 30)) * _SPLITMIX_MUL1) & _MASK64
        z = ((z ^ (z >> 27)) * _SPLITMIX_MUL2) & _MASK64
        return z ^ (z >> 31)


def _rotl(x: int, k: int) -> int:
    """64-bit left-rotate."""
    return ((x << k) | (x >> (64 - k))) & _MASK64


class Xoshiro256StarStarUniformRng:
    """xoshiro256** uniform RNG over (0.0, 1.0).

    # C++ parity: ``Xoshiro256StarStarUniformRng`` in
    # ql/math/randomnumbers/xoshiro256starstaruniformrng.{hpp,cpp}
    # (v1.42.1).

    Construction from a single ``seed`` runs ``_SplitMix64`` four times
    to fill ``s0..s3``. The direct-state constructor
    ``Xoshiro256StarStarUniformRng.from_state(s0, s1, s2, s3)`` mirrors
    the C++ four-arg ctor for use cases that already hold a 256-bit
    state (e.g. resuming from a snapshot).

    Seed 0 is rejected (C++ falls back to ``SeedGenerator``, deferred).
    The all-zero direct state is also rejected — xoshiro is degenerate
    at zero (always returns 0).
    """

    __slots__ = ("_s0", "_s1", "_s2", "_s3")

    def __init__(self, seed: int) -> None:
        if seed == 0:
            raise ValueError(
                "Xoshiro256StarStarUniformRng requires nonzero seed "
                "(C++ SeedGenerator clock fallback not yet ported)"
            )
        # C++ parity: cpp:54-60 — four SplitMix64 outputs from the seed.
        sm = _SplitMix64(seed)
        self._s0: int = sm.next()
        self._s1: int = sm.next()
        self._s2: int = sm.next()
        self._s3: int = sm.next()

    @classmethod
    def from_state(cls, s0: int, s1: int, s2: int, s3: int) -> Xoshiro256StarStarUniformRng:
        """Construct directly from a 256-bit state (4 x 64-bit words).

        # C++ parity: cpp:62-66 — the four-arg ctor.

        The four words must not all be zero (the xoshiro family is
        degenerate at the zero state). Otherwise no constraints.
        """
        if s0 == 0 and s1 == 0 and s2 == 0 and s3 == 0:
            raise ValueError("xoshiro256** is degenerate at all-zero state; choose nonzero seeds")
        instance = cls.__new__(cls)
        instance._s0 = s0 & _MASK64
        instance._s1 = s1 & _MASK64
        instance._s2 = s2 & _MASK64
        instance._s3 = s3 & _MASK64
        return instance

    def next_int64(self) -> int:
        """Return a random integer in ``[0, 0xFFFFFFFFFFFFFFFF]`` — C++ ``nextInt64``."""
        # C++ parity: xoshiro256starstaruniformrng.hpp:76-91.
        result = (_rotl((self._s1 * 5) & _MASK64, 7) * 9) & _MASK64
        t = (self._s1 << 17) & _MASK64
        self._s2 ^= self._s0
        self._s3 ^= self._s1
        self._s1 ^= self._s2
        self._s0 ^= self._s3
        self._s2 ^= t
        self._s3 = _rotl(self._s3, 45)
        return result

    def next_real(self) -> float:
        """Return a random uniform in (0.0, 1.0) — C++ ``nextReal``."""
        # C++ parity: hpp:73 — ``(Real(nextInt64() >> 11) + 0.5) / (1 << 53)``.
        return (float(self.next_int64() >> 11) + 0.5) * _INV_2_POW_53

    def next(self) -> Sample:
        """One sample with weight 1.0."""
        return Sample(value=self.next_real(), weight=1.0)

    def dimension(self) -> int:
        """Scalar RNG — dimension is always 1."""
        return 1
