"""Gaussian sequence generators driving Monte Carlo paths.

# C++ parity: ql/math/randomnumbers/inversecumulativersg.hpp +
# ql/math/randomnumbers/randomsequencegenerator.hpp +
# ql/math/randomnumbers/rngtraits.hpp (v1.42.1).

C++ composes Gaussian sequences via a template stack:

    RandomSequenceGenerator<URNG> →
        InverseCumulativeRsg<URsg, InverseCumulativeNormal> →
            rsg_type used by PathGenerator

i.e. each call to ``next_sequence`` draws ``dimension`` uniforms
from the underlying URNG, then maps each through
``InverseCumulativeNormal`` to get ``dimension`` independent
standard-normal variates plus a multiplicative weight.

The Python port collapses this stack into two cooperating classes:

* :class:`UniformRandomSequenceGenerator` — wraps a scalar uniform
  RNG (MT19937 typically) and emits length-``dimension`` ndarray
  samples per ``next_sequence`` call.
* :class:`InverseCumulativeNormalRsg` — wraps a uniform-sequence
  generator and applies
  :func:`pquantlib.math.distributions.inverse_cumulative_normal.InverseCumulativeNormal.standard_value`
  element-wise.

The factory :func:`make_pseudo_random_rsg(dimension, seed)` mirrors
C++ ``PseudoRandom::make_sequence_generator(dim, seed)`` — the only
combination needed for ``MCEuropeanEngine`` and friends in this
cluster.  Sobol-driven low-discrepancy generators reuse the same
``InverseCumulativeNormalRsg`` adapter (already validated in L5-A);
a low-discrepancy factory lands in a future cluster.

Both generators expose ``dimension()`` (so
:class:`PathGenerator` can validate ``dim == time_steps`` /
``factors * time_steps``), ``next_sequence()`` returning a
``SequenceSample`` of ``(values: ndarray[float64], weight: float)``,
and ``last_sequence()`` returning the most recent sample (used by
``PathGenerator.antithetic`` to fetch the previous draw and negate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.math.distributions.inverse_cumulative_normal import InverseCumulativeNormal
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.random_number_generator import RandomNumberGenerator


@dataclass(frozen=True, slots=True)
class SequenceSample:
    """Multi-dimensional draw with a (multiplicative) weight.

    # C++ parity: ``Sample<std::vector<Real>>`` (sample.hpp).
    """

    value: npt.NDArray[np.float64]
    weight: float = 1.0


class UniformRandomSequenceGenerator:
    """Length-``dimension`` uniform sequence over a scalar URNG.

    # C++ parity: ``RandomSequenceGenerator<URNG>``
    # (randomsequencegenerator.hpp).
    """

    __slots__ = ("_dimension", "_last", "_urng", "_weight")

    def __init__(self, dimension: int, urng: RandomNumberGenerator) -> None:
        qassert.require(dimension >= 1, "dimension must be >= 1")
        self._dimension: int = dimension
        self._urng: RandomNumberGenerator = urng
        self._last: npt.NDArray[np.float64] = np.zeros(dimension, dtype=np.float64)
        self._weight: float = 1.0

    def dimension(self) -> int:
        return self._dimension

    def next_sequence(self) -> SequenceSample:
        out = np.empty(self._dimension, dtype=np.float64)
        w = 1.0
        for i in range(self._dimension):
            sample = self._urng.next()
            out[i] = sample.value
            w *= sample.weight
        self._last = out
        self._weight = w
        return SequenceSample(value=out, weight=w)

    def last_sequence(self) -> SequenceSample:
        return SequenceSample(value=self._last, weight=self._weight)


class InverseCumulativeNormalRsg:
    """Gaussian sequence generator over a uniform-sequence generator.

    # C++ parity: ``InverseCumulativeRsg<URsg, InverseCumulativeNormal>``
    # (inversecumulativersg.hpp).  Maps each uniform in the sequence
    # through the Acklam (Beasley-Springer) inverse-normal
    # approximation to obtain a unit-variance Gaussian variate.
    """

    __slots__ = ("_dimension", "_last", "_urs_gen", "_weight")

    def __init__(self, urs_gen: UniformRandomSequenceGenerator) -> None:
        self._urs_gen: UniformRandomSequenceGenerator = urs_gen
        self._dimension: int = urs_gen.dimension()
        self._last: npt.NDArray[np.float64] = np.zeros(self._dimension, dtype=np.float64)
        self._weight: float = 1.0

    def dimension(self) -> int:
        return self._dimension

    def next_sequence(self) -> SequenceSample:
        unif = self._urs_gen.next_sequence()
        gauss = np.empty(self._dimension, dtype=np.float64)
        for i in range(self._dimension):
            gauss[i] = InverseCumulativeNormal.standard_value(float(unif.value[i]))
        self._last = gauss
        self._weight = unif.weight
        return SequenceSample(value=gauss, weight=unif.weight)

    def last_sequence(self) -> SequenceSample:
        return SequenceSample(value=self._last, weight=self._weight)


def make_pseudo_random_rsg(dimension: int, seed: int) -> InverseCumulativeNormalRsg:
    """Factory mirroring C++ ``PseudoRandom::make_sequence_generator``.

    # C++ parity: ``GenericPseudoRandom<MT19937, InverseCumulativeNormal>
    #              ::make_sequence_generator(dim, seed)`` (rngtraits.hpp:51-57).

    Wires:
        Mersenne-Twister (seed) → UniformRandomSequenceGenerator(dim) →
        InverseCumulativeNormalRsg(dim) → caller-side PathGenerator.

    The seed default of 0 in C++ falls back to a clock-based seed via
    SeedGenerator; the Python MT rejects seed=0 (carve-out from L1).
    For deterministic tests pass an explicit nonzero seed.
    """
    qassert.require(seed != 0, "seed must be nonzero (SeedGenerator carve-out)")
    urng = MersenneTwisterUniformRng(seed)
    urs = UniformRandomSequenceGenerator(dimension, urng)
    return InverseCumulativeNormalRsg(urs)


__all__ = [
    "InverseCumulativeNormalRsg",
    "SequenceSample",
    "UniformRandomSequenceGenerator",
    "make_pseudo_random_rsg",
]
