"""FdmMesherIntegral — tensor-product integral of a grid function.

# C++ parity: ql/methods/finitedifferences/utilities/fdmmesherintegral.{hpp,cpp}
# @ v1.43 (6b57206e0).

Integrates a flat, layout-ordered ``Array`` over the whole composite mesh by
recursing on the *last* 1-D mesher: the values are sliced into
``len(last_mesher)`` contiguous blocks of ``sub_size`` entries, each block is
integrated over the remaining directions, and the resulting ``g`` is fed to
the supplied 1-D integrator against the last mesher's locations.

``integrator_1d`` has the C++ shape ``Real(const Array& x, const Array& f)``
— e.g. :class:`~pquantlib.math.integrals.discrete_integrals.DiscreteSimpsonIntegral`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import final

import numpy as np

from pquantlib.math.array import Array
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)

Integrator1d = Callable[[Array, Array], float]


@final
class FdmMesherIntegral:
    """Integral of a grid function over a composite mesh.

    # C++ parity: ``class FdmMesherIntegral``.
    """

    __slots__ = ("_integrator_1d", "_meshers")

    def __init__(self, mesher: FdmMesherComposite, integrator_1d: Integrator1d) -> None:
        self._meshers: tuple[Fdm1dMesher, ...] = tuple(mesher.get_fdm_1d_meshers())
        self._integrator_1d: Integrator1d = integrator_1d

    def integrate(self, f: Array) -> float:
        """Integrate ``f`` (flat, layout-ordered) over the whole mesh.

        # C++ parity: ``FdmMesherIntegral::integrate(const Array&)`` — the
        # recursion peels the last 1-D mesher off each round.
        """
        x = self._meshers[-1].locations()

        if len(self._meshers) == 1:
            return self._integrator_1d(x, f)

        sub_mesher = FdmMesherComposite(*self._meshers[:-1])
        sub_integral = FdmMesherIntegral(sub_mesher, self._integrator_1d)
        sub_size = sub_mesher.layout().size()

        g = np.empty(x.shape[0], dtype=np.float64)
        for i in range(x.shape[0]):
            g[i] = sub_integral.integrate(f[i * sub_size : (i + 1) * sub_size])

        return self._integrator_1d(x, g)


__all__ = ["FdmMesherIntegral"]
