"""Inverse-cumulative random *sequence* generator.

# C++ parity: ql/math/randomnumbers/inversecumulativersg.hpp (v1.43) —
# ``template <class USG, class IC> class InverseCumulativeRsg``.

Maps every coordinate of a uniform sequence through an inverse cumulative
distribution, carrying the sequence's weight across unchanged. This is the
class that turns ``SobolRsg`` into the Gaussian sequence generator behind
``LowDiscrepancy`` and ``SobolBrownianGenerator``.
"""

from __future__ import annotations

import numpy as np

from pquantlib.math.array import Array
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.randomnumbers.random_number_generator import (
    InverseCumulative,
    SequenceSample,
    UniformSequenceGenerator,
    sequence_parts,
)


class InverseCumulativeRsg[
    Usg: UniformSequenceGenerator, Ic: InverseCumulative = InverseCumulativeNormal
]:
    """Uniform sequences mapped through an inverse cumulative distribution.

    # C++ parity: ``InverseCumulativeRsg<USG, IC>``
    # (inversecumulativersg.hpp:60-73).

    Both C++ constructors collapse into one optional argument: omit
    ``inverse_cumulative`` and you get ``InverseCumulativeNormal()``, which is
    what C++'s default-constructed ``IC ICD_;`` member gives.
    """

    __slots__ = ("_dimension", "_icd", "_usg", "_x", "_x_weight")

    def __init__(
        self,
        uniform_sequence_generator: Usg,
        inverse_cumulative: Ic | None = None,
    ) -> None:
        # C++ parity: inversecumulativersg.hpp:75-83.
        self._usg: Usg = uniform_sequence_generator
        self._dimension: int = uniform_sequence_generator.dimension()
        self._x: Array = np.zeros(self._dimension, dtype=np.float64)
        self._x_weight: float = 1.0
        self._icd: InverseCumulative = (
            InverseCumulativeNormal() if inverse_cumulative is None else inverse_cumulative
        )

    def next_sequence(self) -> SequenceSample:
        """Next sequence, coordinate-wise through the inverse cumulative.

        # C++ parity: ``nextSequence`` (inversecumulativersg.hpp:85-97).
        """
        values, weight = sequence_parts(self._usg.next_sequence())
        x = self._x
        icd = self._icd
        for i in range(self._dimension):
            x[i] = icd(float(values[i]))
        self._x_weight = weight
        return SequenceSample(value=x.copy(), weight=weight)

    def last_sequence(self) -> SequenceSample:
        """Most recent mapped sequence.

        # C++ parity: ``lastSequence`` (inversecumulativersg.hpp:66).
        """
        return SequenceSample(value=self._x.copy(), weight=self._x_weight)

    def dimension(self) -> int:
        """Output vector dimension.

        # C++ parity: ``dimension`` (inversecumulativersg.hpp:67).
        """
        return self._dimension


__all__ = ["InverseCumulativeRsg"]
