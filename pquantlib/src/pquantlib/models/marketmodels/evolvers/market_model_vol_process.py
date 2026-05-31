"""MarketModelVolProcess — abstract stochastic-vol process for SVD evolvers.

# C++ parity: ql/models/marketmodels/evolvers/marketmodelvolprocess.hpp
# (v1.42.1). (Header-only in C++; the .cpp is empty.)

The external, uncorrelated volatility process driving the displaced-diffusion
``SVDDFwdRatePc`` evolver ("Shifted BGM" with a stochastic vol process, after
Brace, *Engineering BGM*). The evolver draws ``variates_per_step`` extra
Gaussian variates per step, feeds them to ``next_step``, and reads the
resulting ``step_sd`` to scale that step's drift + diffusion.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MarketModelVolProcess(ABC):
    """Abstract base for market-model volatility processes.

    # C++ parity: marketmodelvolprocess.hpp MarketModelVolProcess.
    """

    @abstractmethod
    def variates_per_step(self) -> int:
        """Number of Gaussian variates consumed per evolution step."""

    @abstractmethod
    def number_steps(self) -> int:
        """Number of evolution steps."""

    @abstractmethod
    def next_path(self) -> None:
        """Reset the process to the start of a new path."""

    @abstractmethod
    def next_step(self, variates: list[float]) -> float:
        """Advance one step using ``variates``; return the step weight.

        # C++ parity: ``Real nextstep(const std::vector<Real>&)``.
        """

    @abstractmethod
    def step_sd(self) -> float:
        """Standard-deviation multiplier for the step just advanced.

        # C++ parity: ``Real stepSd() const``.
        """

    @abstractmethod
    def state_variables(self) -> list[float]:
        """The current state variables (e.g. the instantaneous variance)."""

    @abstractmethod
    def number_state_variables(self) -> int:
        """Number of state variables."""
