"""Glued1dMesher — composite 1-D mesher of two child meshers.

# C++ parity: ql/experimental/finitedifferences/glued1dmesher.{hpp,cpp}
# (v1.42.1).

Splices ``left_mesher`` (whose rightmost location must be
<= ``right_mesher``'s leftmost location) with ``right_mesher`` into a
single ``Fdm1dMesher``. If the two endpoints are *close* (within the
``close()`` floating-point tolerance), the shared point is collapsed.

Used to build piecewise-uniform meshes for energy + Kluge models.
"""

from __future__ import annotations

import math
from typing import final

from pquantlib import qassert
from pquantlib.math.closeness import close
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher


@final
class Glued1dMesher(Fdm1dMesher):
    """Composite 1-D mesher gluing two child meshers.

    # C++ parity: ``class Glued1dMesher : public Fdm1dMesher``.
    """

    def __init__(self, left_mesher: Fdm1dMesher, right_mesher: Fdm1dMesher) -> None:
        left_loc = left_mesher.locations()
        right_loc = right_mesher.locations()
        common_point = close(float(left_loc[-1]), float(right_loc[0]))
        size = left_loc.size + right_loc.size - (1 if common_point else 0)

        qassert.require(
            float(left_loc[-1]) <= float(right_loc[0]),
            f"left mesher rightmost ({left_loc[-1]}) > right mesher leftmost ({right_loc[0]})",
        )

        super().__init__(size)

        # Copy left locations as-is.
        self._locations[: left_loc.size] = left_loc
        # Skip the duplicated point on the right side if common.
        offset = 1 if common_point else 0
        self._locations[left_loc.size :] = right_loc[offset:]

        # Recompute dplus / dminus from the spliced locations.
        for i in range(self._locations.size - 1):
            d = float(self._locations[i + 1]) - float(self._locations[i])
            self._dplus[i] = d
            self._dminus[i + 1] = d
        # Boundary sentinels (NaN).
        self._dplus[-1] = math.nan
        self._dminus[0] = math.nan


__all__ = ["Glued1dMesher"]
