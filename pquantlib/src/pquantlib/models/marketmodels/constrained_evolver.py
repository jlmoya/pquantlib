"""ConstrainedEvolver — constrained market-model evolver (abstract).

# C++ parity: ql/models/marketmodels/constrainedevolver.hpp (v1.42.1).

Abstract ``MarketModelEvolver`` variant adding the two extra methods needed
to fix rates via importance sampling, for the Fries-Joshi proxy-simulation
approach to Greeks (driven by ``ProxyGreekEngine``).
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.models.marketmodels.evolver import MarketModelEvolver


class ConstrainedEvolver(MarketModelEvolver):
    """Abstract constrained market-model evolver.

    # C++ parity: constrainedevolver.hpp ConstrainedEvolver.
    """

    @abstractmethod
    def set_constraint_type(
        self,
        start_index_of_swap_rate: list[int],
        end_index_of_swap_rate: list[int],
    ) -> None:
        """Set the per-step swap-rate constraint index ranges (call once).

        # C++ parity: constrainedevolver.hpp setConstraintType.
        """

    @abstractmethod
    def set_this_constraint(
        self,
        rate_constraints: list[float],
        is_constraint_active: list[bool],
    ) -> None:
        """Set this path's constrained swap-rate values (call before each path).

        # C++ parity: constrainedevolver.hpp setThisConstraint.
        """
