"""Fdm1dMesher — abstract 1-D mesher (locations + per-node spacings).

# C++ parity: ql/methods/finitedifferences/meshers/fdm1dmesher.hpp
# (v1.42.1).

A 1-D mesher stores three parallel arrays of length ``size``:

* ``locations[i]`` — physical position at node i.
* ``dplus[i]`` — forward spacing ``loc[i+1] - loc[i]`` (None at last node).
* ``dminus[i]`` — backward spacing ``loc[i] - loc[i-1]`` (None at first
  node).

Concrete 1-D meshers populate the three arrays during construction.
A 1-D mesher is *not* an ``FdmMesher`` directly — it has no layout —
but it is the building block of ``FdmBlackScholesMesher`` (1-D) and
``FdmMesherComposite`` (multi-D composition of 1-D meshers).

C++ uses ``Null<Real>`` as the boundary sentinel for ``dplus.back()``
and ``dminus.front()``. Python uses ``math.nan`` — never read at
the boundary because the operator code checks the coordinate first.
"""

from __future__ import annotations

import math

import numpy as np

from pquantlib.math.array import Array


class Fdm1dMesher:
    """Base 1-D mesher (locations + dplus + dminus arrays).

    # C++ parity: ``class Fdm1dMesher`` — abstract-by-convention.
    Concrete subclasses populate ``_locations``, ``_dplus``,
    ``_dminus`` in the constructor.
    """

    def __init__(self, size: int) -> None:
        self._locations: Array = np.zeros(size, dtype=np.float64)
        self._dplus: Array = np.full(size, math.nan, dtype=np.float64)
        self._dminus: Array = np.full(size, math.nan, dtype=np.float64)

    def size(self) -> int:
        return int(self._locations.shape[0])

    def location(self, index: int) -> float:
        return float(self._locations[index])

    def locations(self) -> Array:
        """Return the underlying locations array.

        # C++ parity: returns ``const std::vector<Real>&`` — Python
        # returns the numpy array view directly. Callers must not
        # mutate.
        """
        return self._locations

    def dplus(self, index: int) -> float:
        """Forward spacing at ``index``; NaN at the last node.

        # C++ parity: returns ``Null<Real>`` at ``size-1`` — Python
        # surfaces ``nan`` so callers can detect via ``math.isnan``.
        """
        return float(self._dplus[index])

    def dminus(self, index: int) -> float:
        """Backward spacing at ``index``; NaN at the first node."""
        return float(self._dminus[index])


__all__ = ["Fdm1dMesher"]
