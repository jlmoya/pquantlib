"""Sobol Brownian-bridge sequence generators.

# C++ parity: ql/math/randomnumbers/sobolbrownianbridgersg.{hpp,cpp} (v1.43)
#             ``SobolBrownianBridgeRsg`` / ``Burley2020SobolBrownianBridgeRsg``.

Adapters that present a ``SobolBrownianGenerator`` — a path-shaped object with
``nextPath`` / ``nextStep`` — through the flat sequence-generator interface.

The content is the *dimension ordering*. A ``factors x steps`` grid of Sobol
dimensions has to be assigned to (factor, step) pairs, and Sobol's leading
dimensions are the well-equidistributed ones, so the assignment decides which
parts of the path get the good variates:

* ``Factors`` — dimension 0..steps-1 walk the first factor's whole path;
* ``Steps`` — the first ``factors`` dimensions cover step 0 across all
  factors;
* ``Diagonal`` (the default) — a diagonal sweep that gives the best variates
  to the most important factors *and* the largest bridge steps at once.

Each factor's permuted slice is then Brownian-bridged, and the flat output is
laid out step-major: ``seq[i * factors + f]`` is factor ``f`` at step ``i``.
"""

from __future__ import annotations

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.randomnumbers.sobol_rsg import DirectionIntegers
from pquantlib.models.marketmodels.browniangenerators.sobol_brownian_generator import (
    Burley2020SobolBrownianGenerator,
    SobolBrownianGenerator,
    SobolBrownianGeneratorBase,
)

#: Re-exported so callers of this module do not have to reach into
#: ``models.marketmodels`` for the enum.
#: # C++ parity: ``SobolBrownianGenerator::Ordering``.
Ordering = SobolBrownianGeneratorBase.Ordering


def _fill(gen: SobolBrownianGeneratorBase, seq: Array) -> None:
    """Drive one path out of ``gen`` into the flat vector ``seq``.

    # C++ parity: the anonymous-namespace ``setNextSequence``
    # (sobolbrownianbridgersg.cpp:30-38).
    """
    gen.next_path()
    factors = gen.number_of_factors()
    output = [0.0] * factors
    for i in range(gen.number_of_steps()):
        gen.next_step(output)
        seq[i * factors : (i + 1) * factors] = output


class SobolBrownianBridgeRsg:
    """Sobol + Brownian bridge behind the sequence-generator interface.

    # C++ parity: ``SobolBrownianBridgeRsg``
    # (sobolbrownianbridgersg.hpp:32-49, .cpp:40-63).
    """

    __slots__ = ("_gen", "_seq")

    def __init__(
        self,
        factors: int,
        steps: int,
        ordering: SobolBrownianGeneratorBase.Ordering = Ordering.DIAGONAL,
        seed: int = 0,
        direction_integers: DirectionIntegers = DirectionIntegers.JoeKuoD7,
    ) -> None:
        # C++ parity: sobolbrownianbridgersg.cpp:40-46. Note the default
        # direction-integer family here is JoeKuoD7, not the Jaeckel default
        # of SobolRsg itself.
        self._seq: Array = np.zeros(factors * steps, dtype=np.float64)
        self._gen: SobolBrownianGenerator = SobolBrownianGenerator(
            factors, steps, ordering, seed, direction_integers
        )

    def next_sequence(self) -> Array:
        """Next bridged path, laid out step-major.

        # C++ parity: ``nextSequence`` (sobolbrownianbridgersg.cpp:48-53).
        The C++ ``Sample::weight`` is always 1 and is not exposed.
        """
        _fill(self._gen, self._seq)
        return self._seq.copy()

    def last_sequence(self) -> Array:
        """Most recent path.

        # C++ parity: ``lastSequence`` (sobolbrownianbridgersg.cpp:55-58).
        """
        return self._seq.copy()

    def dimension(self) -> int:
        """``factors * steps``.

        # C++ parity: ``dimension`` (sobolbrownianbridgersg.cpp:60-62).
        """
        return self._gen.number_of_factors() * self._gen.number_of_steps()


class Burley2020SobolBrownianBridgeRsg:
    """Owen-scrambled Sobol + Brownian bridge, sequence-generator interface.

    # C++ parity: ``Burley2020SobolBrownianBridgeRsg``
    # (sobolbrownianbridgersg.hpp:51-70, .cpp:65-88).
    """

    __slots__ = ("_gen", "_seq")

    def __init__(
        self,
        factors: int,
        steps: int,
        ordering: SobolBrownianGeneratorBase.Ordering = Ordering.DIAGONAL,
        seed: int = 42,
        direction_integers: DirectionIntegers = DirectionIntegers.JoeKuoD7,
        scramble_seed: int = 43,
    ) -> None:
        # C++ parity: sobolbrownianbridgersg.cpp:65-74.
        self._seq: Array = np.zeros(factors * steps, dtype=np.float64)
        self._gen: Burley2020SobolBrownianGenerator = Burley2020SobolBrownianGenerator(
            factors, steps, ordering, seed, direction_integers, scramble_seed
        )

    def next_sequence(self) -> Array:
        """Next bridged path, laid out step-major.

        # C++ parity: ``nextSequence`` (sobolbrownianbridgersg.cpp:76-80).
        """
        _fill(self._gen, self._seq)
        return self._seq.copy()

    def last_sequence(self) -> Array:
        """Most recent path.

        # C++ parity: ``lastSequence`` (sobolbrownianbridgersg.cpp:82-85).
        """
        return self._seq.copy()

    def dimension(self) -> int:
        """``factors * steps``.

        # C++ parity: ``dimension`` (sobolbrownianbridgersg.cpp:87-89).
        """
        return self._gen.number_of_factors() * self._gen.number_of_steps()


__all__ = ["Burley2020SobolBrownianBridgeRsg", "Ordering", "SobolBrownianBridgeRsg"]
