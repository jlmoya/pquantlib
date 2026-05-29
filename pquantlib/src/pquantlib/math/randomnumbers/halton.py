"""Halton low-discrepancy sequence generator.

# C++ parity: ql/math/randomnumbers/haltonrsg.{hpp,cpp} (v1.42.1).

Halton sequences extend van der Corput sequences across multiple
dimensions: the i-th coordinate uses the i-th prime as its base. They
are deterministic and low-discrepancy, which makes them well-suited
for multi-start global-optimization seeds (the L10-A
``SabrInterpolation(max_guesses=...)`` Halton multi-start consumer).

PQuantLib divergences:

* **Prime numbers.** C++ uses ``PrimeNumbers::get(i)`` (a lazily-grown
  sieve). PQuantLib inlines the first 64 primes as a static tuple — the
  L10-A consumer caps ``max_guesses`` at ~50, so 64 dimensions are
  more than enough for current callers. Beyond 64 dimensions the
  constructor raises (extend the table if you need more).
* **Random start / random shift.** Ported by delegating to
  ``numpy.random.default_rng`` (per-seed deterministic). C++ pipes
  through a ``MersenneTwisterUniformRng``-backed
  ``RandomSequenceGenerator`` — the two RNGs differ but the Halton
  sequence shape is unaffected; the optional shifts only translate
  the deterministic sequence and the L10-A consumer doesn't probe
  bit-identical agreement against the C++ output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from pquantlib import qassert

# First 64 primes — enough dimensions for any realistic multi-start
# optimization. Beyond this the constructor errors with a clear message.
# # C++ parity: PrimeNumbers::get(i) for i = 0..63.
_PRIMES: Final[tuple[int, ...]] = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311,
)


@dataclass(frozen=True, slots=True)
class HaltonSample:
    """A Halton sample: ``value`` vector + uniform-weight 1.0.

    # C++ parity: ``HaltonRsg::sample_type``, which is
    # ``Sample<std::vector<Real>>``.
    """

    value: tuple[float, ...]
    weight: float = 1.0


class HaltonRsg:
    """Halton low-discrepancy sequence generator.

    Args:
        dimensionality: number of dimensions in each sample (>= 1).
        seed: RNG seed for the random-start / random-shift jitter
            (default 0 — deterministic given a seed).
        random_start: if True, draw a random per-dimension integer
            offset added to the Halton counter. Default True (matches
            the C++ default).
        random_shift: if True, draw a random per-dimension uniform
            offset added (mod 1) to each sample value. Default False.

    Use:

        rsg = HaltonRsg(dimensionality=4, seed=42)
        sample = rsg.next_sequence()
        # sample.value is a 4-tuple of floats in [0, 1).

    # C++ parity: ql/math/randomnumbers/haltonrsg.hpp ``class HaltonRsg``.
    """

    def __init__(
        self,
        dimensionality: int,
        seed: int = 0,
        random_start: bool = True,
        random_shift: bool = False,
    ) -> None:
        qassert.require(dimensionality > 0, "dimensionality must be greater than 0")
        qassert.require(
            dimensionality <= len(_PRIMES),
            f"HaltonRsg supports up to {len(_PRIMES)} dimensions; "
            f"got {dimensionality}",
        )
        self._dim: int = dimensionality
        self._counter: int = 0
        # Default offsets are zero.
        starts = np.zeros(dimensionality, dtype=np.int64)
        shifts = np.zeros(dimensionality, dtype=np.float64)
        if random_start or random_shift:
            rng = np.random.default_rng(seed)
            if random_start:
                # Use uint32 to mirror the C++ ``unsigned long`` start
                # offsets (32-bit on Windows MSVC; the C++ Mersenne
                # Twister output is 32-bit anyway).
                starts = rng.integers(0, np.iinfo(np.uint32).max, size=dimensionality).astype(
                    np.int64,
                )
            if random_shift:
                shifts = rng.random(dimensionality)
        self._random_start: np.ndarray = starts
        self._random_shift: np.ndarray = shifts
        # Cache the last-emitted vector for ``last_sequence``.
        self._last: tuple[float, ...] = tuple(0.0 for _ in range(dimensionality))

    # --- public API ----------------------------------------------------

    def dimension(self) -> int:
        return self._dim

    def next_sequence(self) -> HaltonSample:
        """Return the next Halton sample.

        # C++ parity: ``HaltonRsg::nextSequence`` (haltonrsg.cpp:56-72).
        """
        self._counter += 1
        result = [0.0] * self._dim
        for i in range(self._dim):
            b = _PRIMES[i]
            k = self._counter + int(self._random_start[i])
            h = 0.0
            f = 1.0
            while k != 0:
                f /= b
                h += (k % b) * f
                k //= b
            v = h + float(self._random_shift[i])
            # Wrap to [0, 1).
            v -= int(v)
            result[i] = v
        self._last = tuple(result)
        return HaltonSample(value=self._last)

    def last_sequence(self) -> HaltonSample:
        """Return the most recently emitted sample.

        # C++ parity: ``HaltonRsg::lastSequence``.
        """
        return HaltonSample(value=self._last)


__all__ = ["HaltonRsg", "HaltonSample"]
