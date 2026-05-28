"""Uniform1dMesher — equally-spaced 1-D mesher.

# C++ parity: ql/methods/finitedifferences/meshers/uniform1dmesher.hpp
# (v1.42.1).

Lays ``size`` nodes uniformly on ``[start, end]``. Forward / backward
spacings are constant = ``(end - start) / (size - 1)`` everywhere
except at the boundaries (``dplus`` is NaN at the last node,
``dminus`` is NaN at the first node — matching C++'s ``Null<Real>``
sentinel).
"""

from __future__ import annotations

import math

from pquantlib import qassert
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher


class Uniform1dMesher(Fdm1dMesher):
    """Equispaced 1-D mesher on ``[start, end]`` with ``size`` nodes.

    # C++ parity: ``class Uniform1dMesher : public Fdm1dMesher``.
    """

    def __init__(self, start: float, end: float, size: int) -> None:
        super().__init__(size)
        qassert.require(end > start, "end must be larger than start")
        dx = (end - start) / (size - 1)
        # Vectorised: locations[i] = start + i * dx; final element exactly = end.
        for i in range(size - 1):
            self._locations[i] = start + i * dx
            self._dplus[i] = dx
            self._dminus[i + 1] = dx
        self._locations[-1] = end
        self._dplus[-1] = math.nan
        self._dminus[0] = math.nan


__all__ = ["Uniform1dMesher"]
