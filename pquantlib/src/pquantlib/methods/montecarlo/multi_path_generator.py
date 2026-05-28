"""MultiPathGenerator — multi-asset MC path generation.

# C++ parity: ql/methods/montecarlo/multipathgenerator.hpp (v1.42.1) —
# ``template <class GSG> class MultiPathGenerator``.

Used for multi-D processes (Heston, G2) where each step needs
``factors`` independent Gaussians.  The Gaussian sequence generator
emits dim = ``factors * (time_grid.size() - 1)`` independent normal
variates per draw; the path generator slices them ``factors`` at a
time and feeds them into ``process.evolve(t, x, dt, dw)``.

Brownian bridge is *not* supported in the multi-D case in C++ —
``QL_FAIL("Brownian bridge not supported")``.  The Python port
preserves that behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.exceptions import LibraryException
from pquantlib.methods.montecarlo.multi_path import MultiPath
from pquantlib.methods.montecarlo.path_generator import GaussianSequenceGeneratorProtocol
from pquantlib.processes.stochastic_process import StochasticProcess
from pquantlib.time.time_grid import TimeGrid


@dataclass(frozen=True, slots=True)
class MultiPathSample:
    """One sampled (MultiPath, weight) pair.

    # C++ parity: ``Sample<MultiPath>`` (sample.hpp).
    """

    value: MultiPath
    weight: float = 1.0


class MultiPathGenerator:
    """Multi-asset MC path generator over an n-D process.

    # C++ parity: ``MultiPathGenerator<GSG>`` (multipathgenerator.hpp).
    """

    __slots__ = (
        "_brownian_bridge",
        "_generator",
        "_multi_path",
        "_n_assets",
        "_n_factors",
        "_process",
        "_time_grid",
    )

    def __init__(
        self,
        process: StochasticProcess,
        time_grid: TimeGrid,
        generator: GaussianSequenceGeneratorProtocol,
        brownian_bridge: bool = False,
    ) -> None:
        """Build a multi-D path generator.

        # C++ parity: ``MultiPathGenerator(process, times, gsg, bb=false)``
        # (multipathgenerator.hpp:72-88).
        """
        self._brownian_bridge: bool = brownian_bridge
        self._process: StochasticProcess = process
        self._generator: GaussianSequenceGeneratorProtocol = generator
        self._time_grid: TimeGrid = time_grid
        self._n_assets: int = process.size()
        self._n_factors: int = process.factors()
        qassert.require(
            generator.dimension() == self._n_factors * (len(time_grid) - 1),
            f"dimension ({generator.dimension()}) is not equal to "
            f"({self._n_factors} * {len(time_grid) - 1}) the number of "
            "factors times the number of time steps",
        )
        qassert.require(len(time_grid) > 1, "no times given")
        self._multi_path: MultiPath = MultiPath.from_assets_and_grid(
            self._n_assets, time_grid
        )

    # --- path generation ---------------------------------------------------

    def next(self) -> MultiPathSample:
        return self._next(antithetic=False)

    def antithetic(self) -> MultiPathSample:
        return self._next(antithetic=True)

    def _next(self, *, antithetic: bool) -> MultiPathSample:
        # C++ parity: ``MultiPathGenerator::next(bool)`` (multipathgenerator.hpp:103-150).
        if self._brownian_bridge:
            # C++ parity: ``QL_FAIL("Brownian bridge not supported")``.
            raise LibraryException("Brownian bridge not supported")

        seq = (
            self._generator.last_sequence()
            if antithetic
            else self._generator.next_sequence()
        )

        m = self._n_assets
        n = self._n_factors
        path = self._multi_path

        # Set initial values across all assets.
        asset: npt.NDArray[np.float64] = self._process.initial_values()
        for j in range(m):
            path[j].values[0] = float(asset[j])

        temp = np.empty(n, dtype=np.float64)
        for i in range(1, path.path_size()):
            offset = (i - 1) * n
            t = self._time_grid[i - 1]
            dt = self._time_grid.dt(i - 1)
            if antithetic:
                for k in range(n):
                    temp[k] = -float(seq.value[offset + k])
            else:
                for k in range(n):
                    temp[k] = float(seq.value[offset + k])
            asset = self._process.evolve(t, asset, dt, temp)
            for j in range(m):
                path[j].values[i] = float(asset[j])
        return MultiPathSample(value=path, weight=seq.weight)


__all__ = ["MultiPathGenerator", "MultiPathSample"]
