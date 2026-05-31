"""BrownianGenerator — abstract Gaussian-increment generator + factory.

# C++ parity: ql/models/marketmodels/browniangenerator.hpp (v1.42.1).

A ``BrownianGenerator`` produces, per step, a vector of independent standard
Gaussian variates driving the market-model evolution. ``next_step`` fills a
caller-supplied output vector and returns the step weight; ``next_path``
starts a new path and returns its weight.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BrownianGenerator(ABC):
    """Abstract base for Brownian (Gaussian-increment) generators.

    # C++ parity: browniangenerator.hpp BrownianGenerator.
    """

    @abstractmethod
    def next_step(self, output: list[float]) -> float:
        """Fill ``output`` with this step's Gaussian variates; return its weight.

        ``output`` is a caller-supplied buffer (C++ out-parameter) of length
        ``number_of_factors()``, filled in place.
        """

    @abstractmethod
    def next_path(self) -> float:
        """Start a new path; return its weight."""

    @abstractmethod
    def number_of_factors(self) -> int:
        """Number of driving factors per step."""

    @abstractmethod
    def number_of_steps(self) -> int:
        """Number of evolution steps per path."""


class BrownianGeneratorFactory(ABC):
    """Abstract base for Brownian-generator factories.

    # C++ parity: browniangenerator.hpp BrownianGeneratorFactory.
    """

    @abstractmethod
    def create(self, factors: int, steps: int) -> BrownianGenerator:
        """Build a ``BrownianGenerator`` for the given factor / step counts."""
