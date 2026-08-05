"""Sobol low-discrepancy sequence generator.

# C++ parity: ql/math/randomnumbers/sobolrsg.{hpp,cpp} (v1.43)
#             ``class SobolRsg``.

A Gray-code counter and bitwise operations generate the sequence: the
``n``-th draw XORs a single direction integer into the previous one, using
the position of the rightmost zero bit of ``n`` (Antonov-Saleev). Setting
``use_gray_code=False`` switches to the plain generating integer
:math:`\\gamma(n) = n`, which is what ``Burley2020SobolRsg`` needs because it
addresses the sequence by an arbitrary scrambled index.

Ten families of free direction integers are supported, exactly as in C++:
``Unit``, ``Jaeckel`` (the default), ``SobolLevitan``,
``SobolLevitanLemieux``, ``JoeKuoD5``/``D6``/``D7`` and ``Kuo``/``Kuo2``/
``Kuo3``. Dimensions beyond a family's tabulated range are initialised from a
``MersenneTwisterUniformRng`` seeded with ``seed`` — which is the *only* use
of ``seed``; for dimensions inside the tabulated range the sequence is fully
deterministic and seed-independent.

.. rubric:: Divergence repaired

Through 2026-08 this module delegated to ``scipy.stats.qmc.Sobol`` on the
premise that "scipy internally uses Joe-Kuo direction numbers, the most
widely used modern family". The premise does not survive contact with the
C++: QuantLib's default is **Jaeckel**, its Gray-code counter starts at
draw 1 rather than at the origin, and none of the other nine families exist
in scipy at all. A scipy-backed ``SobolRsg`` therefore produced a different
sequence from C++ for every dimensionality above 2, silently, and everything
built on top of it (``SobolBrownianGenerator``, ``Burley2020SobolRsg``,
``LowDiscrepancy``) inherited the divergence. This module now transcribes
sobolrsg.cpp and is cross-validated bit-exactly against it.

.. rubric:: Deliberate divergence

C++ returns ``const&`` into the generator's own buffer from ``nextSequence``,
``nextInt32Sequence``, ``skipTo`` and ``lastSequence``, so two "different"
samples held at once are silently the same object. The Python port returns a
fresh array/list from each of them; every caller that was correct in C++ is
still correct, and the aliasing trap is gone.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Final

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.sobol_tables import (
    alt_primitive_polynomials,
    direction_integer_initializers,
    primitive_polynomials,
)

#: Number of primitive polynomials modulo two compiled into the library, and
#: therefore the largest supported dimensionality.
#: # C++ parity: ``PPMT_MAX_DIM`` == ``N_PRIMITIVES_UP_TO_DEGREE_18``.
PPMT_MAX_DIM: Final[int] = 21200

#: # C++ parity: ``maxAltDegree`` (sobolrsg.cpp:35).
_MAX_ALT_DEGREE: Final[int] = 52

_MASK32: Final[int] = 0xFFFFFFFF
#: # C++ parity: ``0.5 / (1UL << 31)`` (sobolrsg.hpp:139).
_SOBOL_NORM: Final[float] = 0.5 / (1 << 31)
#: ``M_LN2`` as C++ spells it in ``skipTo``.
_M_LN2: Final[float] = 0.693147180559945309417232121458176568


class DirectionIntegers(IntEnum):
    """Free-direction-integer family.

    # C++ parity: ``SobolRsg::DirectionIntegers`` (sobolrsg.hpp:112-115).
    The integer values match the C++ enumerators' declaration order.
    """

    Unit = 0
    Jaeckel = 1
    SobolLevitan = 2
    SobolLevitanLemieux = 3
    JoeKuoD5 = 4
    JoeKuoD6 = 5
    JoeKuoD7 = 6
    Kuo = 7
    Kuo2 = 8
    Kuo3 = 9


#: Which tabulated resource backs each family (``Unit`` is computed, not read).
_FAMILY_FILE: Final[dict[DirectionIntegers, str]] = {
    DirectionIntegers.Jaeckel: "jaeckel",
    DirectionIntegers.SobolLevitan: "sobol_levitan",
    DirectionIntegers.SobolLevitanLemieux: "lemieux",
    DirectionIntegers.JoeKuoD5: "joe_kuo_d5",
    DirectionIntegers.JoeKuoD6: "joe_kuo_d6",
    DirectionIntegers.JoeKuoD7: "joe_kuo_d7",
    DirectionIntegers.Kuo: "kuo",
    DirectionIntegers.Kuo2: "kuo2",
    DirectionIntegers.Kuo3: "kuo3",
}

#: Families generated against the *alternative* ordering of the low-degree
#: primitive polynomials. # C++ parity: sobolrsg.cpp:78500-78503.
_USES_ALT_POLYNOMIALS: Final[frozenset[DirectionIntegers]] = frozenset(
    {
        DirectionIntegers.Kuo,
        DirectionIntegers.Kuo2,
        DirectionIntegers.Kuo3,
        DirectionIntegers.SobolLevitan,
        DirectionIntegers.SobolLevitanLemieux,
    }
)


class SobolRsg:
    """Sobol' low-discrepancy sequence over ``(0, 1)^d``.

    # C++ parity: ``SobolRsg`` (sobolrsg.hpp:110-151, sobolrsg.cpp:78477-78848).

    Args:
        dimensionality: output vector dimension, ``1 <= d <= PPMT_MAX_DIM``.
        seed: seeds the ``MersenneTwisterUniformRng`` that fills in free
            direction integers for dimensions past the tabulated range of the
            chosen family. Unused below that range.
        direction_integers: free-direction-integer family (default
            ``Jaeckel``, as in C++).
        use_gray_code: ``True`` (default) uses the Antonov-Saleev Gray-code
            counter; ``False`` recomputes each draw from the plain counter,
            which is what ``Burley2020SobolRsg`` requires.
    """

    __slots__ = (
        "_dim",
        "_direction_integers",
        "_first_draw",
        "_integer_sequence",
        "_sequence",
        "_sequence_counter",
        "_use_gray_code",
    )

    def __init__(
        self,
        dimensionality: int,
        seed: int = 0,
        direction_integers: DirectionIntegers = DirectionIntegers.Jaeckel,
        use_gray_code: bool = True,
    ) -> None:
        # C++ parity: sobolrsg.cpp:78486-78491.
        qassert.require(dimensionality > 0, "dimensionality must be greater than 0")
        qassert.require(
            dimensionality <= PPMT_MAX_DIM,
            f"dimensionality {dimensionality} exceeds the number of available "
            f"primitive polynomials modulo two ({PPMT_MAX_DIM})",
        )
        self._dim: int = dimensionality
        self._use_gray_code: bool = use_gray_code
        self._sequence_counter: int = 0
        self._first_draw: bool = True
        self._integer_sequence: list[int] = [0] * dimensionality
        self._sequence: Array = np.zeros(dimensionality, dtype=np.float64)

        degree, ppmt = self._polynomials(dimensionality, direction_integers)

        # 32 direction integers per dimension, stored [bit][dimension] exactly
        # as C++ stores directionIntegers_[32][dimensionality_].
        # C++ parity: sobolrsg.cpp:78552-78554 — the degenerate first dimension
        # has no free direction integers.
        di: list[list[int]] = [[0] * dimensionality for _ in range(32)]
        for j in range(32):
            di[j][0] = 1 << (32 - j - 1)

        max_tabulated = self._fill_tabulated(di, degree, direction_integers)
        self._fill_random(di, degree, max_tabulated, seed)
        self._fill_recurrence(di, degree, ppmt)
        self._direction_integers: list[list[int]] = di

        # C++ parity: sobolrsg.cpp:78766-78771 — with the Gray code the first
        # draw is precomputed in the constructor.
        if use_gray_code:
            for k in range(dimensionality):
                self._integer_sequence[k] = di[0][k]

    # -- construction helpers ---------------------------------------------

    @staticmethod
    def _polynomials(
        dimensionality: int, direction_integers: DirectionIntegers
    ) -> tuple[list[int], list[int]]:
        """Per-dimension polynomial degree and encoded coefficients.

        # C++ parity: sobolrsg.cpp:78494-78541.
        """
        alt = direction_integers in _USES_ALT_POLYNOMIALS
        alt_table = alt_primitive_polynomials()
        table = primitive_polynomials()

        degree = [0] * dimensionality
        ppmt = [0] * dimensionality
        current_degree = 1
        index = 0
        k = 1

        alt_degree = _MAX_ALT_DEGREE if alt else 0
        while k < min(dimensionality, alt_degree):
            row = alt_table[current_degree - 1]
            if index >= len(row):
                # C++ encodes the end of a degree block with -1 and rolls over.
                current_degree += 1
                index = 0
                row = alt_table[current_degree - 1]
            ppmt[k] = row[index]
            degree[k] = current_degree
            k += 1
            index += 1

        while k < dimensionality:
            row = table[current_degree - 1]
            if index >= len(row):
                current_degree += 1
                index = 0
                row = table[current_degree - 1]
            ppmt[k] = row[index]
            degree[k] = current_degree
            k += 1
            index += 1

        return degree, ppmt

    def _fill_tabulated(
        self,
        di: list[list[int]],
        degree: list[int],
        direction_integers: DirectionIntegers,
    ) -> int:
        """Seed the free direction integers from the tabulated coefficients.

        # C++ parity: sobolrsg.cpp:78557-78701 (the ``switch`` on the family).
        Returns C++'s ``maxTabulated``.
        """
        dim = self._dim
        if direction_integers is DirectionIntegers.Unit:
            # C++ parity: sobolrsg.cpp:78561-78568.
            max_tabulated = dim
            for k in range(1, max_tabulated):
                for ell in range(1, degree[k] + 1):
                    di[ell - 1][k] = (1 << (32 - ell)) & _MASK32
            return max_tabulated

        rows = direction_integer_initializers(_FAMILY_FILE[direction_integers])
        # C++ computes maxTabulated as sizeof(table)/sizeof(ptr) + 1: the
        # tables start at dimension 2, so the +1 converts a row count into a
        # dimension count.
        max_tabulated = len(rows) + 1
        for k in range(1, min(dim, max_tabulated)):
            row = rows[k - 1]
            for j, value in enumerate(row):
                di[j][k] = (value << (32 - j - 1)) & _MASK32
        return max_tabulated

    def _fill_random(
        self, di: list[list[int]], degree: list[int], max_tabulated: int, seed: int
    ) -> None:
        """Random free direction integers past the tabulated range.

        # C++ parity: sobolrsg.cpp:78704-78727.
        """
        if self._dim <= max_tabulated:
            return
        rng = MersenneTwisterUniformRng(seed)
        for k in range(max_tabulated, self._dim):
            for ell in range(1, degree[k] + 1):
                while True:
                    u = rng.next().value
                    value = int(u * (1 << ell))
                    if value & 1:
                        break
                # Only the l leftmost bits can be non-zero and the l-th is set.
                di[ell - 1][k] = (value << (32 - ell)) & _MASK32

    def _fill_recurrence(
        self, di: list[list[int]], degree: list[int], ppmt: list[int]
    ) -> None:
        """Extend each dimension to 32 direction integers by recurrence.

        # C++ parity: sobolrsg.cpp:78731-78753 — eq. 8.19 of "Monte Carlo
        # Methods in Finance" (Jaeckel).
        """
        for k in range(1, self._dim):
            gk = degree[k]
            pk = ppmt[k]
            for ell in range(gk, 32):
                n = di[ell - gk][k] >> gk
                # a[k][j] = ppmt[k] >> (gk - j - 1); the highest coefficient is
                # unused and the lowest is always set, which is why neither is
                # part of the encoding.
                for j in range(1, gk):
                    if (pk >> (gk - j - 1)) & 1:
                        n ^= di[ell - j][k]
                n ^= di[ell - gk][k]
                di[ell][k] = n & _MASK32

    # -- C++ public API ----------------------------------------------------

    def skip_to(self, n: int) -> list[int]:
        """Skip to the ``n``-th sample; returns its integer sequence.

        # C++ parity: ``SobolRsg::skipTo`` (sobolrsg.cpp:78775-78805).
        """
        big_n = (n + 1) & _MASK32
        di = self._direction_integers
        seq = self._integer_sequence
        if self._use_gray_code:
            # C++ uses log(N)/M_LN2 + 1 to bound the number of set bits.
            ops = int(math.log(float(big_n)) / _M_LN2) + 1
            gray = big_n ^ (big_n >> 1)
            for k in range(self._dim):
                acc = 0
                for index in range(ops):
                    if (gray >> index) & 1:
                        acc ^= di[index][k]
                seq[k] = acc
        else:
            for k in range(self._dim):
                seq[k] = 0
            mask = 1
            for index in range(32):
                if big_n & mask:
                    for k in range(self._dim):
                        seq[k] ^= di[index][k]
                mask <<= 1
        self._sequence_counter = n
        return list(seq)

    def next_int32_sequence(self) -> list[int]:
        """Next draw as raw 32-bit Sobol integers.

        # C++ parity: ``SobolRsg::nextInt32Sequence``
        # (sobolrsg.cpp:78807-78846).

        Note the C++ behaviour reproduced here for ``use_gray_code=False``:
        ``skipTo(sequenceCounter_)`` runs *before* the counter is advanced and
        the first draw does not advance it at all, so the first two calls
        return the same vector. ``Burley2020SobolRsg`` addresses the sequence
        by an explicit index and is unaffected.
        """
        if not self._use_gray_code:
            self.skip_to(self._sequence_counter)
            if self._first_draw:
                self._first_draw = False
            else:
                self._sequence_counter = (self._sequence_counter + 1) & _MASK32
                qassert.require(self._sequence_counter != 0, "period exceeded")
            return list(self._integer_sequence)

        if self._first_draw:
            # Precomputed in the constructor.
            self._first_draw = False
            return list(self._integer_sequence)

        self._sequence_counter = (self._sequence_counter + 1) & _MASK32
        qassert.require(self._sequence_counter != 0, "period exceeded")

        # Antonov-Saleev: use the Gray code G(n) rather than n itself, so a
        # single direction integer per dimension has to be XORed in.
        n = self._sequence_counter
        j = 0
        while n & 1:
            n >>= 1
            j += 1
        di = self._direction_integers
        seq = self._integer_sequence
        for k in range(self._dim):
            seq[k] ^= di[j][k]
        return list(seq)

    def next_sequence(self) -> Array:
        """Next draw, normalised into ``(0, 1)``.

        # C++ parity: ``SobolRsg::nextSequence`` (sobolrsg.hpp:135-142). The
        # C++ ``Sample::weight`` is always 1 for Sobol and is not exposed.
        """
        v = self.next_int32_sequence()
        seq = self._sequence
        for k in range(self._dim):
            seq[k] = v[k] * _SOBOL_NORM
        return seq.copy()

    def last_sequence(self) -> Array:
        """Most recent normalised draw.

        # C++ parity: ``SobolRsg::lastSequence``.
        """
        return self._sequence.copy()

    def dimension(self) -> int:
        """Output vector dimension.

        # C++ parity: ``SobolRsg::dimension``.
        """
        return self._dim


__all__ = ["PPMT_MAX_DIM", "DirectionIntegers", "SobolRsg"]
