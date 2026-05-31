"""MarketModel — abstract base for market models (pseudo-root generator).

# C++ parity: ql/models/marketmodels/marketmodel.{hpp,cpp} (v1.42.1).

For each time step a market model produces the pseudo-square-root of the
covariance matrix for that step. ``MarketModel`` is the abstract base; the
``covariance`` / ``total_covariance`` / ``time_dependent_volatility``
methods are concrete (derived from the per-step ``pseudo_root`` via
``A @ A.T``), with lazily-cached results matching the C++ ``mutable``
buffers.

``MarketModelFactory`` is the abstract factory base.

Divergences from C++:

- C++ ``MarketModelFactory`` inherits ``Observable``; the pquantlib base is
  a plain ABC (the observer machinery is not needed by the W9-A consumers,
  and the concrete factories in W9-B/C can mix in the observable pattern if
  required).
- ``pseudo_root(i)`` returns a ``Matrix`` (numpy 2-D float64), matching the
  C++ ``const Matrix&``; the cached covariance buffers are numpy arrays.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert

if TYPE_CHECKING:
    from pquantlib.math.matrix import Matrix
    from pquantlib.models.marketmodels.evolution_description import EvolutionDescription


class MarketModel(ABC):
    """Abstract base for market models.

    # C++ parity: marketmodel.hpp MarketModel.

    Generates, per time step, the pseudo-square-root of that step's
    covariance matrix.
    """

    def __init__(self) -> None:
        # C++ parity: mutable std::vector<Matrix> covariance_, totalCovariance_.
        self._covariance: list[Matrix] = []
        self._total_covariance: list[Matrix] = []

    @abstractmethod
    def initial_rates(self) -> list[float]:
        """The initial (time-zero) forward rates."""

    @abstractmethod
    def displacements(self) -> list[float]:
        """The displacement (shift) of each rate."""

    @abstractmethod
    def evolution(self) -> EvolutionDescription:
        """The evolution description (rate/evolution time grid)."""

    @abstractmethod
    def number_of_rates(self) -> int:
        """Number of forward rates."""

    @abstractmethod
    def number_of_factors(self) -> int:
        """Number of driving factors."""

    @abstractmethod
    def number_of_steps(self) -> int:
        """Number of evolution steps."""

    @abstractmethod
    def pseudo_root(self, i: int) -> Matrix:
        """Pseudo-square-root of the covariance matrix for step ``i``."""

    def covariance(self, i: int) -> Matrix:
        """Covariance matrix for step ``i`` (``= pseudoRoot @ pseudoRoot.T``).

        # C++ parity: marketmodel.cpp MarketModel::covariance.
        """
        if not self._covariance:
            steps = self.number_of_steps()
            self._covariance = [
                self.pseudo_root(j) @ self.pseudo_root(j).T for j in range(steps)
            ]
        qassert.require(
            i < len(self._covariance),
            f"i ({i}) must be less than covariance_.size() ({len(self._covariance)})",
        )
        return self._covariance[i]

    def total_covariance(self, end_index: int) -> Matrix:
        """Cumulative covariance through step ``end_index``.

        # C++ parity: marketmodel.cpp MarketModel::totalCovariance.
        """
        if not self._total_covariance:
            steps = self.number_of_steps()
            self._total_covariance = [np.zeros((0, 0))] * steps
            # call covariance(0) to trigger calculation
            self._total_covariance[0] = self.covariance(0).copy()
            for j in range(1, steps):
                self._total_covariance[j] = self._total_covariance[j - 1] + self._covariance[j]
        qassert.require(
            end_index < len(self._covariance),
            f"endIndex ({end_index}) must be less than covariance_.size() "
            f"({len(self._covariance)})",
        )
        return self._total_covariance[end_index]

    def time_dependent_volatility(self, i: int) -> list[float]:
        """Per-step volatility of rate ``i`` from the covariance diagonal.

        # C++ parity: marketmodel.cpp MarketModel::timeDependentVolatility.
        """
        qassert.require(
            i < self.number_of_rates(),
            f"index ({i}) must less than number of rates ({self.number_of_rates()})",
        )
        steps = self.number_of_steps()
        result = [0.0] * steps
        evolution_time = self.evolution().evolution_times()
        last_time = 0.0
        for j in range(steps):
            tau = evolution_time[j] - last_time
            this_variance = float(self.covariance(j)[i, i])
            result[j] = math.sqrt(this_variance / tau)
            last_time = evolution_time[j]
        return result


class MarketModelFactory(ABC):
    """Abstract base for market-model factories.

    # C++ parity: marketmodel.hpp MarketModelFactory.
    """

    @abstractmethod
    def create(
        self, evolution: EvolutionDescription, number_of_factors: int
    ) -> MarketModel:
        """Build a ``MarketModel`` for the given evolution + factor count."""
