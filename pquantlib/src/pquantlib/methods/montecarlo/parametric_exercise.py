"""ParametricExercise + genericEarlyExerciseOptimization.

# C++ parity: ql/methods/montecarlo/parametricexercise.{hpp,cpp} (v1.43).

A parametric exercise strategy is described by, per exercise date, a number of
state variables and a number of free parameters, plus a predicate
``exercise(i, parameters, variables)`` and an initial ``guess(i)``.

``generic_early_exercise_optimization`` walks the exercise dates backwards,
optimising each date's parameters against the biased (in-sample) estimate of
the option value, and returns that biased estimate. Backward induction folds
each date's decision into the previous date's cumulated cash flows, exactly
as C++ does — including the fact that the returned number is the *in-sample*
estimate and is therefore biased high.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.math.optimization.constraint import NoConstraint
from pquantlib.math.optimization.cost_function import CostFunction
from pquantlib.math.optimization.problem import Problem

if TYPE_CHECKING:
    from pquantlib.math.optimization.end_criteria import EndCriteria
    from pquantlib.math.optimization.optimization_method import OptimizationMethod
    from pquantlib.models.marketmodels.callability.collect_node_data import NodeData


class ParametricExercise(ABC):
    """Exercise strategy described by per-date free parameters.

    # C++ parity: ``class ParametricExercise``.
    """

    @abstractmethod
    def number_of_variables(self) -> list[int]:
        """State variables per exercise date. # C++: ``numberOfVariables``."""

    @abstractmethod
    def number_of_parameters(self) -> list[int]:
        """Free parameters per exercise date. # C++: ``numberOfParameters``."""

    @abstractmethod
    def exercise(
        self, exercise_number: int, parameters: list[float], variables: list[float]
    ) -> bool:
        """Exercise decision at ``exercise_number``. # C++: ``exercise``."""

    @abstractmethod
    def guess(self, exercise_number: int, parameters: list[float]) -> None:
        """Fill ``parameters`` in place with an initial guess. # C++: ``guess``."""


class _ValueEstimate(CostFunction):
    """# C++ parity: the anonymous-namespace ``ValueEstimate`` cost function."""

    def __init__(
        self,
        simulation_data: list[NodeData],
        exercise: ParametricExercise,
        exercise_index: int,
    ) -> None:
        self._simulation_data = simulation_data
        self._exercise = exercise
        self._exercise_index = exercise_index
        self._parameters: list[float] = [0.0] * exercise.number_of_parameters()[
            exercise_index
        ]
        if not any(d.is_valid for d in simulation_data):
            raise LibraryException("no valid paths")

    def value(self, x: Array) -> float:
        self._parameters = np.asarray(x, dtype=np.float64).tolist()
        total = 0.0
        n = 0
        for d in self._simulation_data:
            if d.is_valid:
                n += 1
                if self._exercise.exercise(
                    self._exercise_index, self._parameters, d.values
                ):
                    total += d.exercise_value
                else:
                    total += d.cumulated_cash_flows
        return -total / n

    def values(self, x: Array) -> Array:
        del x
        raise LibraryException("values method not implemented")


def generic_early_exercise_optimization(
    simulation_data: list[list[NodeData]],
    exercise: ParametricExercise,
    parameters: list[list[float]],
    end_criteria: EndCriteria,
    method: OptimizationMethod,
) -> float:
    """Optimise the exercise parameters backwards; return the biased estimate.

    # C++ parity: ``genericEarlyExerciseOptimization``. ``parameters`` is an
    out-parameter in C++ and is mutated in place here too (resized to
    ``steps-1`` rows).
    """
    steps = len(simulation_data)
    parameters.clear()
    parameters.extend([] for _ in range(steps - 1))

    for i in range(steps - 1, 0, -1):
        exercise_data = simulation_data[i]
        parameters[i - 1] = [0.0] * exercise.number_of_parameters()[i - 1]

        f = _ValueEstimate(exercise_data, exercise, i - 1)

        exercise.guess(i - 1, parameters[i - 1])
        guess = np.array(parameters[i - 1], dtype=np.float64)

        problem = Problem(f, NoConstraint(), guess)
        method.minimize(problem, end_criteria)

        parameters[i - 1] = np.asarray(problem.current_value, dtype=np.float64).tolist()

        previous_data = simulation_data[i - 1]
        for j in range(len(previous_data)):
            if exercise_data[j].is_valid:
                if exercise.exercise(i - 1, parameters[i - 1], exercise_data[j].values):
                    previous_data[j].cumulated_cash_flows += exercise_data[j].exercise_value
                else:
                    previous_data[j].cumulated_cash_flows += exercise_data[
                        j
                    ].cumulated_cash_flows

    initial_data = simulation_data[0]
    return sum(d.cumulated_cash_flows for d in initial_data) / len(initial_data)


__all__ = ["ParametricExercise", "generic_early_exercise_optimization"]
