"""FdmSimpleProcess1dMesher — process-driven 1-D mesher.

# C++ parity: ql/methods/finitedifferences/meshers/fdmsimpleprocess1dmesher.{hpp,cpp}
# (v1.42.1).

Builds a 1-D mesh anchored at the process's ``x0`` by averaging the
inverse-normal quantiles of the process's evolution at ``tAvgSteps``
time slices, then accumulating per-slice locations and dividing by
``tAvgSteps``. The construction matches C++ exactly so OU + ExtOU +
other 1-D processes get the same mesh from the same parameters.

For a Gaussian-distributed process (OU is one), the resulting mesh
is approximately equispaced on the (1 - 2*eps) confidence interval,
concentrated near ``x0``.
"""

from __future__ import annotations

import numpy as np

from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.methods.finitedifferences.meshers.fdm_1d_mesher import Fdm1dMesher
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D


class FdmSimpleProcess1dMesher(Fdm1dMesher):
    """Process-driven 1-D mesher.

    # C++ parity: ``class FdmSimpleProcess1dMesher : public Fdm1dMesher``.

    Parameters
    ----------
    size:
        Number of grid points.
    process:
        1-D stochastic process providing ``x0``, ``evolve``.
    maturity:
        Terminal time used as the right end of the averaging window.
    t_avg_steps:
        Number of intermediate time slices used for the per-slice
        quantile average. C++ default is 10.
    epsilon:
        Tail probability — the mesh covers ``(eps, 1 - eps)`` of
        the marginal distribution. C++ default is 1e-4.
    mandatory_point:
        Optional anchor point — overrides ``process.x0()`` in the
        per-slice ``qMin`` / ``qMax`` resolution. C++ default is
        ``Null<Real>``.
    """

    def __init__(
        self,
        size: int,
        process: StochasticProcess1D,
        maturity: float,
        t_avg_steps: int = 10,
        epsilon: float = 1e-4,
        mandatory_point: float | None = None,
    ) -> None:
        super().__init__(size)
        x0 = process.x0()
        inv_norm = InverseCumulativeNormal()

        # Accumulate per-slice contributions into self._locations.
        self._locations = np.zeros(size, dtype=np.float64)
        for slice_idx in range(1, t_avg_steps + 1):
            t = (maturity * slice_idx) / t_avg_steps
            mp = mandatory_point if mandatory_point is not None else x0
            evolve_low = process.evolve_1d(0.0, x0, t, inv_norm(epsilon))
            evolve_high = process.evolve_1d(0.0, x0, t, inv_norm(1.0 - epsilon))
            q_min = min(mp, x0, evolve_low)
            q_max = max(mp, x0, evolve_high)
            dp = (1.0 - 2.0 * epsilon) / (size - 1)
            self._locations[0] += q_min
            p = epsilon
            for i in range(1, size - 1):
                p += dp
                self._locations[i] += process.evolve_1d(0.0, x0, t, inv_norm(p))
            self._locations[-1] += q_max

        # Average over t_avg_steps slices.
        self._locations /= float(t_avg_steps)

        # Per-node spacings.
        for i in range(size - 1):
            spacing = self._locations[i + 1] - self._locations[i]
            self._dplus[i] = spacing
            self._dminus[i + 1] = spacing
        # Boundary cells get NaN (mirroring C++ Null<Real>).


__all__ = ["FdmSimpleProcess1dMesher"]
