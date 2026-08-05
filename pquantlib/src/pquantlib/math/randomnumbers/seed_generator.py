"""Random seed generator.

# C++ parity: ql/math/randomnumbers/seedgenerator.{hpp,cpp} (v1.43).

``SeedGenerator`` is the singleton every QuantLib RNG falls back to when it is
constructed with ``seed == 0``.

.. rubric:: It is clock-seeded, and that is deliberate

seedgenerator.cpp:34 reads ``auto firstSeed = (unsigned long)(std::time(nullptr));``
— the entry point of the chain is the wall clock, so ``SeedGenerator`` output
is **not** reproducible across runs and neither is any RNG constructed with
seed 0. That is the documented C++ behaviour ("a random seed will be chosen
based on clock()") and this port keeps it.

What *is* deterministic, and what a port has to get right, is the
transformation applied to that first seed:

1. ``first = MT19937(firstSeed)``; ``secondSeed = first.nextInt32()``;
2. ``second = MT19937(secondSeed)``; ``skip = second.nextInt32() % 1000``;
3. four more ``second.nextInt32()`` draws become the ``init_by_array`` key of
   the final generator;
4. the final generator discards ``skip`` draws.

``initialize`` therefore takes an optional ``first_seed`` — a seam that C++
does not have, because C++ can only test this chain by freezing the clock.
The cross-validation test drives it with fixed values and compares against a
C++ probe that replays the same four steps.
"""

from __future__ import annotations

import time
from typing import ClassVar

from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng


class SeedGenerator:
    """Singleton source of initialisation seeds.

    # C++ parity: ``SeedGenerator : public Singleton<SeedGenerator>``
    # (seedgenerator.hpp:38-47).
    """

    _instance: ClassVar[SeedGenerator | None] = None

    __slots__ = ("_rng",)

    def __init__(self) -> None:
        # C++ parity: seedgenerator.cpp:29-31 — "we need to prevent rng from
        # being default-initialized", hence the throw-away seed 42 before
        # initialize() replaces it.
        self._rng: MersenneTwisterUniformRng = MersenneTwisterUniformRng(42)
        self.initialize()

    @classmethod
    def instance(cls) -> SeedGenerator:
        """The process-wide instance.

        # C++ parity: ``Singleton<SeedGenerator>::instance()``.
        """
        existing = cls._instance
        if existing is not None:
            return existing
        # Publish before running __init__: initialize() builds Mersenne
        # Twisters, and a (2^-32-likely) zero intermediate seed would
        # otherwise re-enter instance() and recurse forever. C++ has the same
        # latent cycle; its Singleton simply hands back the object under
        # construction, which is what this ordering reproduces.
        obj = cls.__new__(cls)
        cls._instance = obj
        obj.__init__()
        return obj

    def initialize(self, first_seed: int | None = None) -> None:
        """(Re)build the underlying generator.

        # C++ parity: ``SeedGenerator::initialize`` (seedgenerator.cpp:33-57).

        Args:
            first_seed: entry point of the chain. ``None`` (the only value C++
                can produce) reads the wall clock, exactly as
                ``std::time(nullptr)`` does.
        """
        if first_seed is None:
            first_seed = int(time.time())

        # firstSeed is chosen based on the clock and used for the first rng.
        first = MersenneTwisterUniformRng(first_seed)
        # "secondSeed is as random as it could be" (C++ comment verbatim).
        second_seed = first.next_int32()
        second = MersenneTwisterUniformRng(second_seed)

        # Use the second rng to initialize the final one.
        skip = second.next_int32() % 1000
        init = [second.next_int32() for _ in range(4)]

        self._rng = MersenneTwisterUniformRng.from_seeds(init)
        for _ in range(skip):
            self._rng.next_int32()

    def get(self) -> int:
        """Next seed.

        # C++ parity: ``SeedGenerator::get`` (seedgenerator.cpp:59-61).
        """
        return self._rng.next_int32()


__all__ = ["SeedGenerator"]
