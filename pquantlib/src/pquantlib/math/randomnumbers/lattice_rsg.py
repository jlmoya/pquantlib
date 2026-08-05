"""Rank-1 lattice sequence generator.

# C++ parity: ql/math/randomnumbers/latticersg.{hpp,cpp} (v1.43) —
# ``class LatticeRsg``.

The ``i``-th point of a rank-1 lattice rule is ``frac(i * z / N)`` coordinate
by coordinate, with ``z`` a generating vector from :class:`LatticeRule` and
``N`` the number of points. Note that ``i`` starts at 0, so the first draw is
the origin.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib.math.array import Array


class LatticeRsg:
    """Rank-1 lattice low-discrepancy sequence.

    # C++ parity: ``LatticeRsg`` (latticersg.hpp:33-53).

    Args:
        dimensionality: number of coordinates per point.
        z: generating vector; only its first ``dimensionality`` entries are
            read (C++ indexes ``z_[j]`` for ``j < dimensionality_``).
        n: number of lattice points.
    """

    __slots__ = ("_dim", "_i", "_n", "_sequence", "_z")

    def __init__(self, dimensionality: int, z: Array, n: int) -> None:
        # C++ parity: latticersg.cpp:26-28.
        self._dim: int = dimensionality
        self._n: int = n
        self._z: Array = np.asarray(z, dtype=np.float64)
        self._i: int = 0
        self._sequence: Array = np.zeros(dimensionality, dtype=np.float64)

    def skip_to(self, n: int) -> None:
        """Skip ``n`` points.

        # C++ parity: ``skipTo`` (latticersg.cpp:30-33). The doc comment says
        # "skip to the n-th sample", but the body is ``i_ += n`` — a relative
        # advance, not an absolute seek. Behaviour, not comment, is ported.
        """
        self._i += n

    def next_sequence(self) -> Array:
        """Next lattice point.

        # C++ parity: ``nextSequence`` (latticersg.cpp:35-47). The C++
        # ``Sample::weight`` is always 1 and is not exposed.
        """
        seq = self._sequence
        z = self._z
        i = self._i
        n = self._n
        for j in range(self._dim):
            theta = i * float(z[j]) / n
            seq[j] = math.fmod(theta, 1.0)
        self._i = i + 1
        return seq.copy()

    def last_sequence(self) -> Array:
        """Most recent point.

        # C++ parity: ``lastSequence`` (latticersg.hpp:45).
        """
        return self._sequence.copy()

    def dimension(self) -> int:
        """Output vector dimension.

        # C++ parity: ``dimension`` (latticersg.hpp:44).
        """
        return self._dim


__all__ = ["LatticeRsg"]
