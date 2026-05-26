"""L'Ecuyer combined LCG with Bays-Durham shuffle.

# C++ parity: ql/math/randomnumbers/lecuyeruniformrng.hpp +
# lecuyeruniformrng.cpp (v1.42.1) — QuantLib's port of ``ran2`` from
# Numerical Recipes in C (Press et al., 2nd ed., Section 7.1).

Two parallel linear-congruential streams ``temp1`` (m1=2147483563,
a1=40014) and ``temp2`` (m2=2147483399, a2=40692) are combined via a
32-element Bays-Durham shuffle table for cycle-length extension. The
output is then clamped strictly below 1.0 by ``maxRandom = 1 -
machine-epsilon``.

The two LCG steps use Schrage's multiplication algorithm (``a*(t - k*q) -
k*r`` with ``k = t // q``) to avoid 32-bit overflow on the C++ side.
Python's arbitrary-precision ints make this safe regardless, but we
keep the structure verbatim for bit-identical sequence output.
"""

from __future__ import annotations

import sys
from typing import Final

from pquantlib.math.randomnumbers.random_number_generator import Sample

_M1: Final[int] = 2147483563
_A1: Final[int] = 40014
_Q1: Final[int] = 53668
_R1: Final[int] = 12211
_M2: Final[int] = 2147483399
_A2: Final[int] = 40692
_Q2: Final[int] = 52774
_R2: Final[int] = 3791
_BUFFER_SIZE: Final[int] = 32
# C++ ``int(1 + m1 / bufferSize) = int(1 + (m1 - 1) / bufferSize)`` — both
# evaluate to 67108862 under integer division.
_BUFFER_NORMALIZER: Final[int] = 67108862
# C++ uses ``long double`` for ``maxRandom``. Python ``float`` is IEEE
# binary64; ``sys.float_info.epsilon`` is the binary64 epsilon, which
# equals what C++ ``QL_EPSILON`` resolves to in the QuantLib build.
_MAX_RANDOM: Final[float] = 1.0 - sys.float_info.epsilon


class LecuyerUniformRng:
    """L'Ecuyer's combined LCG with Bays-Durham shuffle.

    # C++ parity: ``LecuyerUniformRng`` in
    # ql/math/randomnumbers/lecuyeruniformrng.{hpp,cpp} (v1.42.1).

    Seed 0 is rejected (the C++ version falls back to ``SeedGenerator``,
    which is deferred to a later cluster — pquantlib refuses seed 0
    explicitly rather than silently picking a clock-derived seed).
    """

    __slots__ = ("_buffer", "_temp1", "_temp2", "_y")

    def __init__(self, seed: int) -> None:
        if seed == 0:
            raise ValueError(
                "LecuyerUniformRng requires nonzero seed (C++ SeedGenerator clock fallback not yet ported)"
            )
        # Mutable state. C++ marks all as ``mutable``; Python plain attrs.
        self._temp1: int = seed
        self._temp2: int = seed
        self._buffer: list[int] = [0] * _BUFFER_SIZE
        # Load the shuffle table (after 8 warm-ups), exactly mirroring C++.
        # C++ parity: lecuyeruniformrng.cpp:46-55.
        for j in range(_BUFFER_SIZE + 7, -1, -1):
            k = self._temp1 // _Q1
            self._temp1 = _A1 * (self._temp1 - k * _Q1) - k * _R1
            if self._temp1 < 0:
                self._temp1 += _M1
            if j < _BUFFER_SIZE:
                self._buffer[j] = self._temp1
        self._y: int = self._buffer[0]

    def next(self) -> Sample:
        """One sample uniformly drawn from (0.0, 1.0) with weight 1.0."""
        # C++ parity: lecuyeruniformrng.cpp:58-84.
        k = self._temp1 // _Q1
        # Schrage's method for ``temp1 = (a1 * temp1) % m1``.
        self._temp1 = _A1 * (self._temp1 - k * _Q1) - k * _R1
        if self._temp1 < 0:
            self._temp1 += _M1
        k = self._temp2 // _Q2
        # Schrage's method for ``temp2 = (a2 * temp2) % m2``.
        self._temp2 = _A2 * (self._temp2 - k * _Q2) - k * _R2
        if self._temp2 < 0:
            self._temp2 += _M2
        # Shuffle-table index in [0, bufferSize - 1].
        j = self._y // _BUFFER_NORMALIZER
        # Combine the two streams.
        self._y = self._buffer[j] - self._temp2
        self._buffer[j] = self._temp1
        if self._y < 1:
            self._y += _M1 - 1
        result = self._y / float(_M1)
        # Users don't expect endpoint values — clamp strictly below 1.
        result = min(result, _MAX_RANDOM)
        return Sample(value=result, weight=1.0)

    def dimension(self) -> int:
        """Scalar RNG — dimension is always 1."""
        return 1
