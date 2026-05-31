"""MarketModelParametricExercise — abstract parametric early-exercise rule.

# C++ parity:
# ql/models/marketmodels/callability/marketmodelparametricexercise.hpp +
# ql/methods/montecarlo/parametricexercise.hpp (v1.42.1).

A ``MarketModelNodeDataProvider`` (supplying the exercise *variables* per node)
that also exposes the ``ParametricExercise`` contract: the exercise decision is
a parametric function ``exercise(exerciseNumber, parameters, variables)`` of the
free parameters and the node variables. ``number_of_data() ==
number_of_variables()``.
"""

from __future__ import annotations

from abc import abstractmethod

from pquantlib.models.marketmodels.callability.node_data_provider import (
    MarketModelNodeDataProvider,
)


class MarketModelParametricExercise(MarketModelNodeDataProvider):
    """Abstract parametric exercise for the market model.

    # C++ parity: marketmodelparametricexercise.hpp + parametricexercise.hpp.
    """

    # --- ParametricExercise interface ---------------------------------------
    @abstractmethod
    def number_of_variables(self) -> list[int]:
        """The number of node variables per exercise."""

    @abstractmethod
    def number_of_parameters(self) -> list[int]:
        """The number of free parameters per exercise."""

    @abstractmethod
    def exercise(
        self,
        exercise_number: int,
        parameters: list[float],
        variables: list[float],
    ) -> bool:
        """Whether to exercise given parameters + node variables."""

    @abstractmethod
    def guess(self, exercise_number: int, parameters: list[float]) -> None:
        """Fill ``parameters`` with an initial guess for the exercise (in place)."""

    def number_of_data(self) -> list[int]:
        return self.number_of_variables()

    @abstractmethod
    def clone(self) -> MarketModelParametricExercise:
        """A newly-allocated copy of this parametric exercise."""
