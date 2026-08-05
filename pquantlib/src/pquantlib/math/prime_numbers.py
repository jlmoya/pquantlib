"""Prime-number generator.

# C++ parity: ql/math/primenumbers.{hpp,cpp} (v1.43), after Peter Jaeckel,
#             "Monte Carlo Methods in Finance".

A lazily extended static table, not a sieve: the first fifteen primes are
hard-coded (the first two are load-bearing for the bootstrap) and everything
past them comes from trial division against the primes already found. The
class is stateful in C++ — ``primeNumbers_`` is a file-scope
``std::vector<BigNatural>`` that grows across calls — and the Python port
keeps that shape, because ``HaltonRsg`` and the Faure sequence index into it
by absolute position.
"""

from __future__ import annotations

import math
from typing import ClassVar, Final

# C++ parity: primenumbers.cpp:32-39 — ``firstPrimes``.
_FIRST_PRIMES: Final[tuple[int, ...]] = (
    # the first two primes are mandatory for bootstrapping
    2,
    3,
    # optional additional precomputed primes
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
)


class PrimeNumbers:
    """Get and store one prime after another.

    # C++ parity: ``class PrimeNumbers`` — primenumbers.hpp:41-49,
    # primenumbers.cpp:43-67.
    """

    _prime_numbers: ClassVar[list[int]] = []

    @staticmethod
    def get(absolute_index: int) -> int:
        """The ``absolute_index``-th prime, zero-based (``get(0) == 2``).

        # C++ parity: ``PrimeNumbers::get`` — primenumbers.cpp:45-54.
        """
        if not PrimeNumbers._prime_numbers:
            PrimeNumbers._prime_numbers.extend(_FIRST_PRIMES)
        while len(PrimeNumbers._prime_numbers) <= absolute_index:
            PrimeNumbers._next_prime_number()
        return PrimeNumbers._prime_numbers[absolute_index]

    @staticmethod
    def _next_prime_number() -> int:
        # C++ parity: ``PrimeNumbers::nextPrimeNumber`` —
        # primenumbers.cpp:56-67. The trial division starts at index 1
        # because the even numbers are skipped by the outer ``m += 2``, and
        # the loop deliberately compares ``p <= n`` *after* fetching p, so
        # the last divisor tried can exceed sqrt(m) by one prime.
        primes = PrimeNumbers._prime_numbers
        m = primes[-1]
        while True:
            # skip the even numbers
            m += 2
            n = int(math.sqrt(float(m)))
            # i = 1 since the even numbers have already been skipped
            i = 1
            while True:
                p = primes[i]
                i += 1
                if not (m % p != 0 and p <= n):
                    break
            if not p <= n:
                break
        primes.append(m)
        return m


__all__ = ["PrimeNumbers"]
