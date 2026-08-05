"""Burley (2020) Owen-scrambled Sobol' low-discrepancy sequence.

# C++ parity: ql/math/randomnumbers/burley2020sobolrsg.{hpp,cpp} (v1.43)
#             ``class Burley2020SobolRsg``.

Brent Burley, "Practical Hash-based Owen Scrambling", Journal of Computer
Graphics Techniques 9(4), 2020. Two scrambles are applied per draw:

1. the *sample index* itself is Owen-scrambled, and the underlying
   ``SobolRsg`` (built with ``use_gray_code=False`` so it can be addressed by
   an arbitrary index) is asked for that sample;
2. each coordinate is Owen-scrambled with a seed derived from a Boost-style
   ``hash_combine`` chain, one chain per group of four dimensions.

Both scrambles are ``nested_uniform_scramble`` = bit-reverse, Laine-Karras
permutation, bit-reverse. The C++ comment is emphatic that "the results depend
a lot on the details of the hash_combine() function that is used", and pins
Boost 1.83's ``hash``/``hash_mix``; those exact 64-bit constants and the exact
mixing order are transcribed below.

.. rubric:: Divergence repaired

Through 2026-08 this class subclassed the scipy-backed ``SobolRsg`` and
enabled ``scipy.stats.qmc.Sobol(scramble=True)``, whose docstring conceded
that the result was "**not** bit-identical to the C++ Burley2020 sequence
(different hash, but same statistical properties)" and tested only the
*properties* of the output. scipy implements Matousek's LMS+shift, which is a
different scramble of a different base sequence — not Burley 2020. This module
now transcribes burley2020sobolrsg.cpp and is cross-validated bit-exactly.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.sobol_rsg import DirectionIntegers, SobolRsg

_MASK32: Final[int] = 0xFFFFFFFF
_MASK64: Final[int] = (1 << 64) - 1

# C++ parity: burley2020sobolrsg.cpp:97 — the Boost 1.83 hash_mix constant.
_HASH_MIX_M: Final[int] = 0x0E9846AF9B1A615D
# C++ parity: burley2020sobolrsg.cpp:112 — hash_combine's golden-ratio addend.
_HASH_COMBINE_ADDEND: Final[int] = 0x9E3779B9

# C++ parity: burley2020sobolrsg.cpp:82-88 — the Laine-Karras multipliers.
_LK_MULTIPLIERS: Final[tuple[int, ...]] = (
    0x6C50B47C,
    0xB82F1E52,
    0xC7AFE638,
    0x8D22F6E6,
)


# C++ parity: ``bitReverseTable`` (burley2020sobolrsg.cpp:56-74) — byte b
# reversed. Built here rather than pasted: the C++ table is itself generated
# (http://graphics.stanford.edu/~seander/bithacks.html#BitReverseTable) and the
# construction is checkable at a glance, unlike 256 magic numbers.
_BIT_REVERSE_TABLE: Final[tuple[int, ...]] = tuple(
    int(f"{b:08b}"[::-1], 2) for b in range(256)
)


def _reverse_bits(x: int) -> int:
    """Reverse the 32 bits of ``x``.

    # C++ parity: ``reverseBits`` (burley2020sobolrsg.cpp:76-79).
    """
    t = _BIT_REVERSE_TABLE
    return (
        (t[x & 0xFF] << 24)
        | (t[(x >> 8) & 0xFF] << 16)
        | (t[(x >> 16) & 0xFF] << 8)
        | t[(x >> 24) & 0xFF]
    )


def _laine_karras_permutation(x: int, seed: int) -> int:
    """# C++ parity: ``laine_karras_permutation`` (cpp:81-88)."""
    x = (x + seed) & _MASK32
    for m in _LK_MULTIPLIERS:
        x ^= (x * m) & _MASK32
    return x & _MASK32


def _nested_uniform_scramble(x: int, seed: int) -> int:
    """# C++ parity: ``nested_uniform_scramble`` (cpp:90-95)."""
    return _reverse_bits(_laine_karras_permutation(_reverse_bits(x), seed))


def _local_hash_mix(x: int) -> int:
    """# C++ parity: ``local_hash_mix`` (cpp:96-104) == Boost ``hash_mix``."""
    x &= _MASK64
    x ^= x >> 32
    x = (x * _HASH_MIX_M) & _MASK64
    x ^= x >> 32
    x = (x * _HASH_MIX_M) & _MASK64
    x ^= x >> 28
    return x


def _local_hash(v: int) -> int:
    """# C++ parity: ``local_hash`` (cpp:106-111) == Boost ``hash<uint64>``."""
    seed = 0
    seed = ((v >> 32) + _local_hash_mix(seed)) & _MASK64
    seed = ((v & 0xFFFFFFFF) + _local_hash_mix(seed)) & _MASK64
    return seed


def _local_hash_combine(x: int, v: int) -> int:
    """# C++ parity: ``local_hash_combine`` (cpp:113-115)."""
    return _local_hash_mix((x + _HASH_COMBINE_ADDEND + _local_hash(v)) & _MASK64)


class Burley2020SobolRsg:
    """Owen-scrambled Sobol' sequence (Burley 2020).

    # C++ parity: ``Burley2020SobolRsg`` (burley2020sobolrsg.hpp:36-60).

    Args:
        dimensionality: output vector dimension.
        seed: forwarded to the underlying ``SobolRsg`` (C++ default 42).
        direction_integers: free-direction-integer family (C++ default
            ``Jaeckel``).
        scramble_seed: seeds the ``MersenneTwisterUniformRng`` that produces
            one scramble seed per group of four dimensions (C++ default 43).
    """

    __slots__ = (
        "_dim",
        "_group4_seeds",
        "_integer_sequence",
        "_next_sequence_counter",
        "_sequence",
        "_sobol",
    )

    def __init__(
        self,
        dimensionality: int,
        seed: int = 42,
        direction_integers: DirectionIntegers = DirectionIntegers.Jaeckel,
        scramble_seed: int = 43,
    ) -> None:
        # C++ parity: burley2020sobolrsg.cpp:27-37.
        self._dim: int = dimensionality
        self._integer_sequence: list[int] = [0] * dimensionality
        self._sequence: Array = np.zeros(dimensionality, dtype=np.float64)
        # reset(): the underlying generator must NOT use the Gray code —
        # Burley addresses it by a scrambled (arbitrary) index.
        self._sobol: SobolRsg = SobolRsg(dimensionality, seed, direction_integers, False)
        self._next_sequence_counter: int = 0
        mt = MersenneTwisterUniformRng(scramble_seed)
        self._group4_seeds: list[int] = [
            mt.next_int32() & _MASK32 for _ in range((dimensionality - 1) // 4 + 1)
        ]

    def skip_to(self, n: int) -> list[int]:
        """Position the generator so the next draw is sample ``n``.

        # C++ parity: ``Burley2020SobolRsg::skipTo`` (cpp:45-50) — it draws
        # sample ``n`` and then rewinds the counter, so the returned vector is
        # also what the next ``next_int32_sequence`` call produces.
        """
        self._next_sequence_counter = n
        self.next_int32_sequence()
        self._next_sequence_counter = (self._next_sequence_counter - 1) & _MASK32
        return list(self._integer_sequence)

    def next_int32_sequence(self) -> list[int]:
        """Next scrambled draw as raw 32-bit integers.

        # C++ parity: ``Burley2020SobolRsg::nextInt32Sequence``
        # (cpp:118-136).
        """
        n = _nested_uniform_scramble(self._next_sequence_counter, self._group4_seeds[0])
        seq = self._sobol.skip_to(n)
        out = self._integer_sequence
        out[:] = seq
        i = 0
        group = 0
        while True:
            seed = self._group4_seeds[group]
            group += 1
            g = 0
            while g < 4 and i < self._dim:
                seed = _local_hash_combine(seed, g)
                out[i] = _nested_uniform_scramble(out[i], seed & _MASK32)
                g += 1
                i += 1
            if i >= self._dim:
                break
        self._next_sequence_counter = (self._next_sequence_counter + 1) & _MASK32
        qassert.require(
            self._next_sequence_counter != 0,
            "Burley2020SobolRsg::nextInt32Sequence(): period exceeded",
        )
        return list(out)

    def next_sequence(self) -> Array:
        """Next draw, normalised into ``(0, 1)``.

        # C++ parity: ``Burley2020SobolRsg::nextSequence`` (cpp:138-145).
        Note the normalisation is ``(v + 0.5) / 2^32`` here, *not* the
        ``v * 2^-32`` that plain ``SobolRsg`` uses.
        """
        v = self.next_int32_sequence()
        seq = self._sequence
        for k in range(self._dim):
            seq[k] = (float(v[k]) + 0.5) / 4294967296.0
        return seq.copy()

    def last_sequence(self) -> Array:
        """Most recent normalised draw.

        # C++ parity: ``Burley2020SobolRsg::lastSequence``.
        """
        return self._sequence.copy()

    def dimension(self) -> int:
        """Output vector dimension.

        # C++ parity: ``Burley2020SobolRsg::dimension``.
        """
        return self._dim


__all__ = ["Burley2020SobolRsg"]
