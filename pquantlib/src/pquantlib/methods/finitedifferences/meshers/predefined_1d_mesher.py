"""Predefined1dMesher — 1-D mesher built from an explicit list of points.

# C++ parity: ql/methods/finitedifferences/meshers/predefined1dmesher.hpp
# (v1.43) — header-only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final

from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher


@final
class Predefined1dMesher(Fdm1dMesher):
    """1-D mesher whose locations are given verbatim.

    # C++ parity: ``class Predefined1dMesher : public Fdm1dMesher``.
    """

    def __init__(self, x: Sequence[float]) -> None:
        super().__init__(len(x))
        for i, xi in enumerate(x):
            self._locations[i] = xi
        self._dplus[-1] = math.nan
        self._dminus[0] = math.nan
        for i in range(len(x) - 1):
            self._dplus[i] = self._dminus[i + 1] = x[i + 1] - x[i]


__all__ = ["Predefined1dMesher"]
