"""Randomised (random-shift) low-discrepancy sequence.

# C++ parity: ql/math/randomnumbers/randomizedlds.hpp (v1.43) —
# ``template <class LDS, class PRS> class RandomizedLDS``.

Random-shifts a uniform low-discrepancy sequence of dimension N by adding,
modulo 1 per coordinate, one pseudo-random uniform vector drawn once from
``PRS``. This is the estimator behind Randomised Quasi Monte Carlo: the shift
is what turns a deterministic QMC point set into an unbiased estimator whose
variance can be estimated by re-shifting.

The interaction of the two streams is the whole content, so note precisely
when each advances: the pseudo-random shift is drawn **once in the
constructor** and then held fixed; only ``next_randomizer`` draws a new one,
and it simultaneously rewinds the low-discrepancy generator to its pristine
state so the same QMC point set is re-used under the new shift.

The wrap is C++'s, verbatim: ``if (x > 1.0) x -= 1.0`` — a single conditional
subtraction, not ``fmod``, and the boundary is strict, so a sum of exactly
1.0 is left alone.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import numpy as np

from pquantlib import qassert
from pquantlib.math.array import Array
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.random_number_generator import (
    SequenceSample,
    UniformSequenceGenerator,
    sequence_parts,
)
from pquantlib.math.randomnumbers.random_sequence_generator import (
    RandomSequenceGenerator,
)
from pquantlib.math.randomnumbers.sobol_rsg import SobolRsg


def _default_lds(dimensionality: int, seed: int) -> UniformSequenceGenerator:
    """The LDS type every QuantLib instantiation of ``RandomizedLDS`` uses."""
    return SobolRsg(dimensionality, seed)


class RandomizedLDS[
    Lds: UniformSequenceGenerator,
    Prs: UniformSequenceGenerator = RandomSequenceGenerator[MersenneTwisterUniformRng],
]:
    """Random-shifted low-discrepancy sequence.

    # C++ parity: ``RandomizedLDS<LDS, PRS>`` (randomizedlds.hpp:59-85).

    Args:
        ldsg: the low-discrepancy generator.
        prsg: the pseudo-random generator supplying the shift. ``None``
            builds a ``RandomSequenceGenerator`` over a
            ``MersenneTwisterUniformRng`` of the same dimension — the C++
            default template argument, and the one-argument constructor.
    """

    __slots__ = (
        "_dimension",
        "_ldsg",
        "_pristine_ldsg",
        "_prsg",
        "_randomizer",
        "_randomizer_weight",
        "_x",
        "_x_weight",
    )

    def __init__(self, ldsg: Lds, prsg: Prs | None = None) -> None:
        # C++ parity: randomizedlds.hpp:87-113 (both reference constructors).
        self._ldsg: Lds = ldsg
        self._pristine_ldsg: Lds = copy.deepcopy(ldsg)
        self._dimension: int = ldsg.dimension()
        if prsg is None:
            self._prsg: UniformSequenceGenerator = RandomSequenceGenerator.from_seed(
                self._dimension
            )
        else:
            qassert.require(
                prsg.dimension() == self._dimension,
                f"generator mismatch: {self._dimension}-dim low discrepancy "
                f"and {prsg.dimension()}-dim pseudo random",
            )
            self._prsg = prsg
        self._x: Array = np.zeros(self._dimension, dtype=np.float64)
        self._x_weight: float = 1.0
        self._randomizer: Array = np.zeros(self._dimension, dtype=np.float64)
        self._randomizer_weight: float = 1.0
        self._draw_randomizer()

    @staticmethod
    def from_dimensionality(
        dimensionality: int,
        lds_seed: int = 0,
        prs_seed: int = 0,
        lds_factory: Callable[[int, int], UniformSequenceGenerator] | None = None,
    ) -> RandomizedLDS[
        UniformSequenceGenerator, RandomSequenceGenerator[MersenneTwisterUniformRng]
    ]:
        """Build both generators from a dimension and two seeds.

        # C++ parity: ``RandomizedLDS(Size, BigNatural, BigNatural)``
        # (randomizedlds.hpp:115-124).

        C++ names the LDS type as a template argument; Python takes it as
        ``lds_factory``, defaulting to ``SobolRsg`` — the type every QuantLib
        instantiation of this template uses.
        """
        factory = _default_lds if lds_factory is None else lds_factory
        return RandomizedLDS(
            factory(dimensionality, lds_seed),
            RandomSequenceGenerator.from_seed(dimensionality, prs_seed),
        )

    def _draw_randomizer(self) -> None:
        values, weight = sequence_parts(self._prsg.next_sequence())
        self._randomizer = np.array(values, dtype=np.float64, copy=True)
        self._randomizer_weight = weight

    def next_randomizer(self) -> None:
        """Draw a new shift and rewind the low-discrepancy generator.

        # C++ parity: ``nextRandomizer`` (randomizedlds.hpp:74-77).
        """
        self._draw_randomizer()
        self._ldsg = copy.deepcopy(self._pristine_ldsg)

    def next_sequence(self) -> SequenceSample:
        """Next shifted point.

        # C++ parity: ``nextSequence`` (randomizedlds.hpp:126-140).
        """
        values, weight = sequence_parts(self._ldsg.next_sequence())
        x = self._x
        rnd = self._randomizer
        for i in range(self._dimension):
            v = float(rnd[i]) + float(values[i])
            if v > 1.0:
                v -= 1.0
            x[i] = v
        self._x_weight = self._randomizer_weight * weight
        return SequenceSample(value=x.copy(), weight=self._x_weight)

    def last_sequence(self) -> SequenceSample:
        """Most recent shifted point.

        # C++ parity: ``lastSequence`` (randomizedlds.hpp:71-73).
        """
        return SequenceSample(value=self._x.copy(), weight=self._x_weight)

    def dimension(self) -> int:
        """Output vector dimension.

        # C++ parity: ``dimension`` (randomizedlds.hpp:78).
        """
        return self._dimension


__all__ = ["RandomizedLDS"]
