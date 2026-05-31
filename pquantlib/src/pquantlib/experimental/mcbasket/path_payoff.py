"""PathPayoff — abstract payoff over a full multi-asset path.

# C++ parity: ql/experimental/mcbasket/pathpayoff.hpp (v1.42.1).

A ``PathPayoff`` computes, for a single realized multi-asset path, the
vector of cashflow payments and (optionally) the early-exercise values
and regression states at each fixing time. The path is presented as a
matrix with one row per asset and one column per fixing time; the
per-fixing forward yield term structures are supplied alongside.

This is the experimental-mcbasket analogue of the scalar ``Payoff``
hierarchy. The C++ Visitor ``accept`` machinery is omitted (PQuantLib
does not exercise visitor-based payoff comparison).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from pquantlib.termstructures.yield_term_structure import YieldTermStructure


class PathPayoff(ABC):
    """Abstract base for path-dependent multi-asset payoffs.

    # C++ parity: ``PathPayoff``.
    """

    @abstractmethod
    def name(self) -> str:
        """Short name (for output/comparison, not type-switching)."""

    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""

    @abstractmethod
    def value(
        self,
        path: npt.NDArray[np.float64],
        forward_term_structures: Sequence[YieldTermStructure],
        payments: npt.NDArray[np.float64],
        exercises: npt.NDArray[np.float64],
        states: list[npt.NDArray[np.float64]],
    ) -> None:
        """Fill ``payments`` (and optionally ``exercises``/``states``).

        # C++ parity: ``PathPayoff::value``.

        Args:
            path: ``(n_assets, n_times)`` matrix of asset values at each
                fixing time.
            forward_term_structures: per-fixing forward yield curves.
            payments: output array (length ``n_times``) of cashflows.
            exercises: output array (length ``n_times``) of exercise
                values; left empty to signal exercise is impossible.
            states: output list of per-fixing regression states; left
                empty to signal exercise is impossible.

        If cancelled at time ``i``, all payments on and before ``i`` are
        taken into account plus ``exercises[i]`` (cancellation at ``i``
        does NOT cancel ``payments[i]``).
        """

    @abstractmethod
    def basis_system_dimension(self) -> int:
        """Dimension of the regression basis (== len of each state)."""


__all__ = ["PathPayoff"]
