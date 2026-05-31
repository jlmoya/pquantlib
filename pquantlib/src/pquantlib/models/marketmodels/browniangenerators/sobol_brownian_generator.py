"""SobolBrownianGenerator — Sobol Brownian generator + base + factory.

# C++ parity: ql/models/marketmodels/browniangenerators/
# sobolbrowniangenerator.{hpp,cpp} (v1.42.1).

Incremental Brownian generator using a Sobol low-discrepancy generator, the
inverse-cumulative Gaussian method, and Brownian bridging. The
``SobolBrownianGeneratorBase`` holds the ordering schema + bridge; the
``SobolBrownianGenerator`` concrete wraps a ``SobolRsg`` mapped through
``InverseCumulativeNormal``.

Three orderings assign the best-quality Sobol dimensions to factors / steps /
a diagonal schema (see ``Ordering``).

Divergences from C++:

- The underlying Sobol stream is pquantlib's ``SobolRsg`` (scipy Joe-Kuo
  direction numbers), whereas C++ defaults to the Jaeckel direction-integer
  family. For ``factors * steps > 2`` the *stream* therefore differs; the
  deterministic parts of this class that are stream-independent — the
  ``ordered_indices`` schema and the ``transform`` Brownian-bridge algebra —
  match C++ exactly and are the cross-validated surfaces. The
  ``direction_integers`` argument is accepted for signature parity and
  forwarded to ``SobolRsg`` (which ignores it). The Burley2020 Sobol variant
  is deferred (a thin subclass once a downstream consumer requires it).
- C++ bridges via ``boost::make_permutation_iterator`` over
  ``sample.value``; the Python port gathers the permuted slice into a plain
  list and calls the L5 ``BrownianBridge.transform`` (which uses unit-time
  steps, so its ``/sqrt(dt)`` normalization is the identity and the output
  matches the C++ unit-time bridge).
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from pquantlib import qassert
from pquantlib.math.distributions.inverse_cumulative_normal import (
    InverseCumulativeNormal,
)
from pquantlib.math.randomnumbers.sobol_rsg import SobolRsg
from pquantlib.methods.montecarlo.brownian_bridge import BrownianBridge
from pquantlib.models.marketmodels.brownian_generator import (
    BrownianGenerator,
    BrownianGeneratorFactory,
)


def _fill_by_factor(m: list[list[int]], factors: int, steps: int) -> None:
    # C++ parity: sobolbrowniangenerator.cpp fillByFactor.
    counter = 0
    for i in range(factors):
        for j in range(steps):
            m[i][j] = counter
            counter += 1


def _fill_by_step(m: list[list[int]], factors: int, steps: int) -> None:
    # C++ parity: sobolbrowniangenerator.cpp fillByStep.
    counter = 0
    for j in range(steps):
        for i in range(factors):
            m[i][j] = counter
            counter += 1


def _fill_by_diagonal(m: list[list[int]], factors: int, steps: int) -> None:
    # C++ parity: sobolbrowniangenerator.cpp fillByDiagonal (variate 2 used
    # for the second factor's full path).
    i0 = 0
    j0 = 0
    i = 0
    j = 0
    counter = 0
    while counter < factors * steps:
        m[i][j] = counter
        counter += 1
        if i == 0 or j == steps - 1:
            # completed a diagonal; start a new one
            if i0 < factors - 1:
                i0 += 1
                j0 = 0
            else:
                i0 = factors - 1
                j0 += 1
            i = i0
            j = j0
        else:
            i -= 1
            j += 1


class SobolBrownianGeneratorBase(BrownianGenerator):
    """Base Sobol Brownian generator (ordering + bridge, abstract stream).

    # C++ parity: sobolbrowniangenerator.hpp SobolBrownianGeneratorBase.

    Subclasses provide the Sobol (already-Gaussian) sample via
    ``_next_sequence``.
    """

    class Ordering(IntEnum):
        """Sobol-dimension-to-(factor, step) assignment schema.

        # C++ parity: sobolbrowniangenerator.hpp
        # SobolBrownianGeneratorBase::Ordering.
        """

        FACTORS = 0  #: best-quality variates drive the first factor's path
        STEPS = 1  #: best-quality variates drive the largest steps
        DIAGONAL = 2  #: diagonal schema mixing factors + steps

    def __init__(self, factors: int, steps: int, ordering: Ordering) -> None:
        self._factors = factors
        self._steps = steps
        self._ordering = ordering
        self._bridge = BrownianBridge(steps)
        self._last_step = 0
        self._ordered_indices: list[list[int]] = [
            [0] * steps for _ in range(factors)
        ]
        self._bridged_variates: list[list[float]] = [
            [0.0] * steps for _ in range(factors)
        ]

        if ordering == self.Ordering.FACTORS:
            _fill_by_factor(self._ordered_indices, factors, steps)
        elif ordering == self.Ordering.STEPS:
            _fill_by_step(self._ordered_indices, factors, steps)
        elif ordering == self.Ordering.DIAGONAL:
            _fill_by_diagonal(self._ordered_indices, factors, steps)
        else:  # pragma: no cover - defensive
            qassert.fail("unknown ordering")

    def _next_sequence(self) -> np.ndarray:
        """Next Sobol (already-Gaussian) ``factors*steps`` sample.

        # C++ parity: SobolBrownianGeneratorBase::nextSequence (pure virtual).
        """
        raise NotImplementedError

    def next_path(self) -> float:
        """Draw a Sobol sample and Brownian-bridge it; return weight 1.0.

        # C++ parity: sobolbrowniangenerator.cpp
        # SobolBrownianGeneratorBase::nextPath.
        """
        sample = self._next_sequence()
        for i in range(self._factors):
            # gather the permutation sample[orderedIndices_[i][.]] in step order
            permuted = np.array(
                [sample[self._ordered_indices[i][j]] for j in range(self._steps)],
                dtype=np.float64,
            )
            bridged = self._bridge.transform(permuted)
            for j in range(self._steps):
                self._bridged_variates[i][j] = float(bridged[j])
        self._last_step = 0
        return 1.0

    def next_step(self, output: list[float]) -> float:
        """Fill ``output`` with the bridged variates for the current step.

        # C++ parity: sobolbrowniangenerator.cpp
        # SobolBrownianGeneratorBase::nextStep.
        """
        for i in range(self._factors):
            output[i] = self._bridged_variates[i][self._last_step]
        self._last_step += 1
        return 1.0

    def number_of_factors(self) -> int:
        return self._factors

    def number_of_steps(self) -> int:
        return self._steps

    # -- test interface (C++ parity) --------------------------------------

    def ordered_indices(self) -> list[list[int]]:
        """The factor x step Sobol-dimension assignment schema.

        # C++ parity: SobolBrownianGeneratorBase::orderedIndices.
        """
        return self._ordered_indices

    def transform(self, variates: list[list[float]]) -> list[list[float]]:
        """Brownian-bridge a ``factors*steps`` x ``nPaths`` variate block.

        # C++ parity: sobolbrowniangenerator.cpp
        # SobolBrownianGeneratorBase::transform.

        ``variates[k][path]`` (``k`` in ``[0, factors*steps)``); returns
        ``retVal[factor][path*steps + step]``.
        """
        dim = self._factors * self._steps
        qassert.require(len(variates) == dim, "inconsistent variate vector")
        n_paths = len(variates[0])
        ret_val: list[list[float]] = [
            [0.0] * (n_paths * self._steps) for _ in range(self._factors)
        ]
        for j in range(n_paths):
            sample = [variates[k][j] for k in range(dim)]
            for i in range(self._factors):
                permuted = np.array(
                    [sample[self._ordered_indices[i][s]] for s in range(self._steps)],
                    dtype=np.float64,
                )
                bridged = self._bridge.transform(permuted)
                base = j * self._steps
                for s in range(self._steps):
                    ret_val[i][base + s] = float(bridged[s])
        return ret_val


class SobolBrownianGenerator(SobolBrownianGeneratorBase):
    """Sobol Brownian generator (Sobol stream through inverse-cumulative normal).

    # C++ parity: sobolbrowniangenerator.hpp SobolBrownianGenerator.
    """

    Ordering = SobolBrownianGeneratorBase.Ordering

    def __init__(
        self,
        factors: int,
        steps: int,
        ordering: SobolBrownianGeneratorBase.Ordering,
        seed: int = 0,
        direction_integers: str | None = None,
    ) -> None:
        super().__init__(factors, steps, ordering)
        # C++ parity: InverseCumulativeRsg<SobolRsg, InverseCumulativeNormal>
        # (SobolRsg(factors*steps, seed, integers), InverseCumulativeNormal()).
        self._sobol = SobolRsg(factors * steps, seed, direction_integers)
        self._icn = InverseCumulativeNormal()

    def _next_sequence(self) -> np.ndarray:
        # C++ parity: SobolBrownianGenerator::nextSequence — Sobol uniforms
        # mapped one-by-one through the inverse-cumulative normal.
        uniforms = self._sobol.next_sequence()
        return np.array(
            [self._icn(float(u)) for u in uniforms], dtype=np.float64
        )


class SobolBrownianGeneratorFactory(BrownianGeneratorFactory):
    """Factory building ``SobolBrownianGenerator`` instances.

    # C++ parity: sobolbrowniangenerator.hpp SobolBrownianGeneratorFactory.
    """

    def __init__(
        self,
        ordering: SobolBrownianGeneratorBase.Ordering,
        seed: int = 0,
        direction_integers: str | None = None,
    ) -> None:
        self._ordering = ordering
        self._seed = seed
        self._direction_integers = direction_integers

    def create(self, factors: int, steps: int) -> BrownianGenerator:
        return SobolBrownianGenerator(
            factors, steps, self._ordering, self._seed, self._direction_integers
        )
