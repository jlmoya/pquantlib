"""Faure low-discrepancy sequence generator.

# C++ parity: ql/math/randomnumbers/faurersg.{hpp,cpp} (v1.43) —
# ``class FaureRsg``.

Faure sequences in base ``b`` = the smallest prime >= dimensionality, built
from Pascal-triangle generator matrices mod ``b`` and walked with a ``b``-ary
Gray code (Thiemard; ACM Algorithms 647 and 659).

Two C++ properties are easy to "fix" by accident and are kept exactly:

* ``integerSequence_`` is **cumulative**. ``generateNextIntSequence`` does
  ``integerSequence_[i] += ...`` and never clears the vector, so the emitted
  integers grow without bound and the reals are the running total divided by
  the normalisation factor. That is the sequence C++ produces.
* ``mbit_`` is ``log(LONG_MAX) / log(base)`` — a *platform-dependent* count,
  because ``std::numeric_limits<long int>::max()`` is 2^63-1 on LP64 (Linux,
  macOS) and 2^31-1 on Windows. The reference build is LP64, so 2^63-1 is
  what this port uses; the constant is named so the assumption is visible
  rather than buried in a call to ``sys.maxsize``.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array

#: ``std::numeric_limits<long int>::max()`` on the LP64 reference build.
_LONG_MAX: Final[int] = (1 << 63) - 1

#: # C++ parity: ``firstPrimes`` (ql/math/primenumbers.cpp:37-41) — the seed
#: of ``PrimeNumbers``' lazily-grown sieve.
_FIRST_PRIMES: Final[tuple[int, ...]] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)


def _prime(absolute_index: int) -> int:
    """The ``absolute_index``-th prime, counting ``get(0) == 2``.

    # C++ parity: ``PrimeNumbers::get`` (ql/math/primenumbers.cpp:45-56).
    ``PrimeNumbers`` is a lazily-grown static vector; here the growth is a
    plain trial-division loop, which produces the same primes in the same
    order — the only observable of that class.
    """
    primes: list[int] = list(_FIRST_PRIMES)
    while len(primes) <= absolute_index:
        m = primes[-1]
        while True:
            m += 2
            limit = int(math.sqrt(m))
            if all(m % p for p in primes if p <= limit):
                break
        primes.append(m)
    return primes[absolute_index]


class FaureRsg:
    """Faure low-discrepancy sequence.

    # C++ parity: ``FaureRsg`` (faurersg.hpp:48-79, faurersg.cpp:26-128).
    """

    __slots__ = (
        "_add_one",
        "_bary",
        "_base",
        "_dim",
        "_gray",
        "_integer_sequence",
        "_mbit",
        "_normalization_factor",
        "_pascal3d",
        "_pow_base",
        "_sequence",
    )

    def __init__(self, dimensionality: int) -> None:
        # C++ parity: faurersg.cpp:27-104.
        qassert.require(dimensionality > 0, "dimensionality must be greater than 0")
        self._dim: int = dimensionality
        self._sequence: Array = np.zeros(dimensionality, dtype=np.float64)
        self._integer_sequence: list[int] = [0] * dimensionality

        # base is the lowest prime number >= dimensionality_
        base = 2
        k = 1
        while base < dimensionality:
            base = _prime(k)
            k += 1
        self._base: int = base

        mbit = int(math.log(float(_LONG_MAX)) / math.log(float(base)))
        self._mbit: int = mbit
        self._gray: list[list[int]] = [[0] * (mbit + 1) for _ in range(dimensionality)]
        self._bary: list[int] = [0] * (mbit + 1)

        # setMatrixValues(): powBase_[i][j] = (j - base) * base^(mbit-1-i)
        pow_base = [[0] * (2 * base - 1) for _ in range(mbit)]
        pow_base[mbit - 1][base] = 1
        for i2 in range(mbit - 2, -1, -1):
            pow_base[i2][base] = pow_base[i2 + 1][base] * base
        for ii in range(mbit):
            for j1 in range(base + 1, 2 * base - 1):
                pow_base[ii][j1] = pow_base[ii][j1 - 1] + pow_base[ii][base]
            for j2 in range(base - 1, -1, -1):
                pow_base[ii][j2] = pow_base[ii][j2 + 1] - pow_base[ii][base]
        self._pow_base: list[list[int]] = pow_base

        self._add_one: list[int] = [(j + 1) % base for j in range(base)]

        # setPascalMatrix(): pascal3D[k][dimension][i]
        pascal3d: list[list[list[int]]] = []
        for k2 in range(mbit):
            mm = [[0] * (k2 + 1) for _ in range(dimensionality + 1)]
            pascal3d.append(mm)
            pascal3d[k2][0][k2] = 1
            pascal3d[k2][1][0] = 1
            pascal3d[k2][1][k2] = 1
        for k2 in range(2, mbit):
            for i in range(1, k2):
                p1 = pascal3d[k2 - 1][1][i - 1]
                p2 = pascal3d[k2 - 1][1][i]
                pascal3d[k2][1][i] = (p1 + p2) % base
        fact = 1
        for j in range(2, dimensionality):
            for kk in range(mbit - 1, -1, -1):
                diag = mbit - kk - 1
                fact = 1 if diag == 0 else (fact * j) % base
                for ii in range(kk + 1):
                    pascal3d[diag + ii][j][ii] = (fact * pascal3d[diag + ii][1][ii]) % base
        self._pascal3d: list[list[list[int]]] = pascal3d

        self._normalization_factor: float = float(base) * float(pow_base[0][base])

    def _generate_next_int_sequence(self) -> None:
        """Advance the b-ary Gray code and accumulate the integer sequence.

        # C++ parity: ``generateNextIntSequence``
        # (faurersg.cpp:106-126).
        """
        bary = self._bary
        add_one = self._add_one
        base = self._base
        bit = 0
        bary[bit] = add_one[bary[bit]]
        while bary[bit] == 0:
            bit += 1
            bary[bit] = add_one[bary[bit]]
        qassert.require(bit != self._mbit, "Error processing Faure sequence.")

        gray = self._gray
        pascal3d = self._pascal3d
        pow_base = self._pow_base
        seq = self._integer_sequence
        for i in range(self._dim):
            gi = gray[i]
            for j in range(bit + 1):
                tmp = gi[j]
                gi[j] = (pascal3d[bit][i][j] + tmp) % base
                g1 = gi[j]
                g2 = base - 1 + g1 - tmp
                seq[i] += pow_base[j][g2]

    def next_int_sequence(self) -> list[int]:
        """Next raw integer sequence (cumulative — see the module docstring).

        # C++ parity: ``nextIntSequence`` (faurersg.hpp:52-55).
        """
        self._generate_next_int_sequence()
        return list(self._integer_sequence)

    def last_int_sequence(self) -> list[int]:
        """Most recent integer sequence.

        # C++ parity: ``lastIntSequence`` (faurersg.hpp:56-58).
        """
        return list(self._integer_sequence)

    def next_sequence(self) -> Array:
        """Next point, normalised by ``base * base^(mbit-1)``.

        # C++ parity: ``nextSequence`` (faurersg.hpp:59-64).
        """
        self._generate_next_int_sequence()
        seq = self._sequence
        for i in range(self._dim):
            seq[i] = self._integer_sequence[i] / self._normalization_factor
        return seq.copy()

    def last_sequence(self) -> Array:
        """Most recent normalised point.

        # C++ parity: ``lastSequence`` (faurersg.hpp:65).
        """
        return self._sequence.copy()

    def dimension(self) -> int:
        """Output vector dimension.

        # C++ parity: ``dimension`` (faurersg.hpp:66).
        """
        return self._dim


__all__ = ["FaureRsg"]
