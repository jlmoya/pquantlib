"""MarketModelBasisSystem — abstract Longstaff-Schwartz regression basis.

# C++ parity: ql/models/marketmodels/callability/marketmodelbasissystem.hpp
# (v1.42.1).

A ``MarketModelNodeDataProvider`` whose node data are the basis-function values
used to regress the continuation value in the Longstaff-Schwartz exercise
calibration. ``number_of_data() == number_of_functions()``.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.models.marketmodels.callability.node_data_provider import (
    MarketModelNodeDataProvider,
)


class MarketModelBasisSystem(MarketModelNodeDataProvider):
    """Abstract basis system for Longstaff-Schwartz exercise regression.

    # C++ parity: marketmodelbasissystem.hpp MarketModelBasisSystem.
    """

    @abstractmethod
    def number_of_functions(self) -> list[int]:
        """The number of basis functions per exercise."""

    def number_of_data(self) -> list[int]:
        return self.number_of_functions()

    @abstractmethod
    def clone(self) -> MarketModelBasisSystem:
        """A newly-allocated copy of this basis system."""
