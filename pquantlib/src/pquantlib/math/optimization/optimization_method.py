"""Optimization method abstract class.

# C++ parity: ql/math/optimization/method.hpp (v1.42.1).

Empty abstract — concrete methods (LM, BFGS, Simplex, etc.) are
deferred to a later cluster.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria, Type
    from pquantlib.math.optimization.problem import Problem


class OptimizationMethod(ABC):
    """Abstract base for constrained optimization methods.

    # C++ parity: ``class OptimizationMethod`` in
    # ql/math/optimization/method.hpp:36-43 (v1.42.1).
    """

    @abstractmethod
    def minimize(self, problem: Problem, end_criteria: EndCriteria) -> Type:
        """Minimize ``problem`` subject to ``end_criteria``; return the outcome."""
        ...
