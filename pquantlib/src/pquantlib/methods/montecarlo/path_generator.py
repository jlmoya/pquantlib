"""PathGenerator — single-asset path generation from a 1-D process.

# C++ parity: ql/methods/montecarlo/pathgenerator.hpp (v1.42.1) —
# ``template <class GSG> class PathGenerator``.

The C++ class is templated on the Gaussian sequence generator type
(typically ``InverseCumulativeRsg<RandomSequenceGenerator<MT19937>,
InverseCumulativeNormal>``).  The Python port collapses the template
by taking any ``GaussianSequenceGeneratorProtocol``-compatible
instance (structural Protocol — duck-typed on
``next_sequence`` / ``last_sequence`` / ``dimension``).

Two construction modes mirror C++:

* ``PathGenerator(process, length, time_steps, gsg, brownian_bridge)``
  — internally builds a ``TimeGrid.regular(length, time_steps)``.
* ``PathGenerator.with_time_grid(process, time_grid, gsg, brownian_bridge)``
  — caller supplies an explicit grid (used by MC engines that have
  per-instrument fixings).

The ``next()`` and ``antithetic()`` methods return a
``PathSample`` (Path + weight).  C++ returns a *reference* to a
``mutable Sample<Path>`` member; the Python port returns a fresh
sample each call (cleaner; the Path's ``values`` ndarray is
re-populated in place but a new ``PathSample`` is wrapped around it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from pquantlib import qassert
from pquantlib.methods.montecarlo.brownian_bridge import BrownianBridge
from pquantlib.methods.montecarlo.gaussian_sequence_generator import SequenceSample
from pquantlib.methods.montecarlo.path import Path
from pquantlib.processes.stochastic_process_1d import StochasticProcess1D
from pquantlib.time.time_grid import TimeGrid


class GaussianSequenceGeneratorProtocol(Protocol):
    """Structural type for Gaussian sequence generators (length-dim ndarray)."""

    def next_sequence(self) -> SequenceSample: ...

    def last_sequence(self) -> SequenceSample: ...

    def dimension(self) -> int: ...


@dataclass(frozen=True, slots=True)
class PathSample:
    """One sampled (Path, weight) pair.

    # C++ parity: ``Sample<Path>`` (sample.hpp).
    """

    value: Path
    weight: float = 1.0


class PathGenerator:
    """Single-asset MC path generator over a 1-D process.

    # C++ parity: ``PathGenerator<GSG>`` (pathgenerator.hpp).
    """

    __slots__ = (
        "_bb",
        "_brownian_bridge",
        "_dimension",
        "_generator",
        "_path",
        "_process_1d",
        "_temp",
        "_time_grid",
    )

    def __init__(
        self,
        process: StochasticProcess1D,
        length: float,
        time_steps: int,
        generator: GaussianSequenceGeneratorProtocol,
        brownian_bridge: bool = False,
    ) -> None:
        """Build a path generator over a regular grid.

        # C++ parity: 4-arg ctor (pathgenerator.hpp:81-93).
        """
        self._time_grid: TimeGrid = TimeGrid.regular(length, time_steps)
        self._generator: GaussianSequenceGeneratorProtocol = generator
        self._dimension: int = generator.dimension()
        self._brownian_bridge: bool = brownian_bridge
        self._process_1d: StochasticProcess1D = process
        qassert.require(
            self._dimension == time_steps,
            f"sequence generator dimensionality ({self._dimension}) != "
            f"timeSteps ({time_steps})",
        )
        self._path: Path = Path(self._time_grid)
        self._temp: npt.NDArray[np.float64] = np.zeros(self._dimension, dtype=np.float64)
        self._bb: BrownianBridge = BrownianBridge.from_time_grid(self._time_grid)

    @classmethod
    def with_time_grid(
        cls,
        process: StochasticProcess1D,
        time_grid: TimeGrid,
        generator: GaussianSequenceGeneratorProtocol,
        brownian_bridge: bool = False,
    ) -> PathGenerator:
        """Build with an explicit time grid (rather than (length, steps)).

        # C++ parity: 5-arg ctor (pathgenerator.hpp:95-107).
        """
        pg = cls.__new__(cls)
        pg._time_grid = time_grid
        pg._generator = generator
        pg._dimension = generator.dimension()
        pg._brownian_bridge = brownian_bridge
        pg._process_1d = process
        qassert.require(
            pg._dimension == len(time_grid) - 1,
            f"sequence generator dimensionality ({pg._dimension}) != "
            f"timeSteps ({len(time_grid) - 1})",
        )
        pg._path = Path(time_grid)
        pg._temp = np.zeros(pg._dimension, dtype=np.float64)
        pg._bb = BrownianBridge.from_time_grid(time_grid)
        return pg

    # --- inspectors --------------------------------------------------------

    def size(self) -> int:
        """Sequence dimensionality (= time steps)."""
        return self._dimension

    def dimension(self) -> int:
        """Alias for :meth:`size` — required by ``PathGeneratorProtocol``."""
        return self._dimension

    @property
    def time_grid(self) -> TimeGrid:
        return self._time_grid

    # --- path generation ---------------------------------------------------

    def next(self) -> PathSample:
        """Generate one fresh path.

        # C++ parity: ``PathGenerator::next`` (pathgenerator.hpp:111-113).
        """
        return self._next(antithetic=False)

    def antithetic(self) -> PathSample:
        """Generate the antithetic of the last-drawn sequence.

        # C++ parity: ``PathGenerator::antithetic`` (pathgenerator.hpp:117-119).

        Re-uses the *last* Gaussian sequence (via ``last_sequence``)
        but negates each entry to obtain the negated Brownian
        increments.  Caller is responsible for invoking ``next()`` (or
        ``antithetic()``) on every other draw so the underlying
        sequence cursor stays in lock-step with the canonical-draw
        count.
        """
        return self._next(antithetic=True)

    def _next(self, *, antithetic: bool) -> PathSample:
        # C++ parity: ``PathGenerator::next(bool)`` (pathgenerator.hpp:122-154).
        seq = (
            self._generator.last_sequence()
            if antithetic
            else self._generator.next_sequence()
        )

        if self._brownian_bridge:
            self._temp[:] = self._bb.transform(seq.value)
        else:
            self._temp[:] = seq.value

        path = self._path
        path.values[0] = self._process_1d.x0()
        for i in range(1, path.length()):
            t = self._time_grid[i - 1]
            dt = self._time_grid.dt(i - 1)
            dw = -self._temp[i - 1] if antithetic else self._temp[i - 1]
            path.values[i] = self._process_1d.evolve_1d(
                t,
                float(path.values[i - 1]),
                dt,
                float(dw),
            )
        return PathSample(value=path, weight=seq.weight)


__all__ = ["GaussianSequenceGeneratorProtocol", "PathGenerator", "PathSample"]
