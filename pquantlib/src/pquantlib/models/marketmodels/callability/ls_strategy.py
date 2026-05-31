"""LongstaffSchwartzExerciseStrategy — the calibrated LS exercise strategy.

# C++ parity: ql/models/marketmodels/callability/lsstrategy.{hpp,cpp} (v1.42.1).

Given a basis system, its per-exercise basis coefficients (from
``generic_longstaff_schwartz_regression``), the evolution + numeraires and the
rebate/control exercise values, this decides at each exercise time whether to
exercise by comparing the deflated exercise value against the
basis-regressed continuation value
``control_value + dot(alphas, basis_values)``.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from pquantlib.models.marketmodels.discounter import MarketModelDiscounter
from pquantlib.models.marketmodels.evolution_description import check_compatibility
from pquantlib.models.marketmodels.utilities import is_in_subset

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.callability.exercise_value import (
        MarketModelExerciseValue,
    )
    from pquantlib.models.marketmodels.callability.market_model_basis_system import (
        MarketModelBasisSystem,
    )
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.evolution_description import EvolutionDescription


class LongstaffSchwartzExerciseStrategy:
    """The calibrated Longstaff-Schwartz exercise strategy.

    # C++ parity: lsstrategy.hpp LongstaffSchwartzExerciseStrategy.

    Satisfies the W11-A ``ExerciseStrategy`` Protocol consumed by
    ``CallSpecifiedMultiProduct``.
    """

    def __init__(
        self,
        basis_system: MarketModelBasisSystem,
        basis_coefficients: list[list[float]],
        evolution: EvolutionDescription,
        numeraires: list[int],
        exercise: MarketModelExerciseValue,
        control: MarketModelExerciseValue,
    ) -> None:
        self._basis_system = basis_system.clone()
        self._basis_coefficients = [list(c) for c in basis_coefficients]
        self._exercise = exercise.clone()
        self._control = control.clone()
        self._numeraires = list(numeraires)

        check_compatibility(evolution, numeraires)
        self._relevant_times = evolution.evolution_times()

        self._is_basis_time = is_in_subset(
            self._relevant_times, self._basis_system.evolution().evolution_times()
        )
        self._is_rebate_time = is_in_subset(
            self._relevant_times, self._exercise.evolution().evolution_times()
        )
        self._is_control_time = is_in_subset(
            self._relevant_times, self._control.evolution().evolution_times()
        )

        self._exercise_index = [0] * len(self._relevant_times)
        self._is_exercise_time = [False] * len(self._relevant_times)
        self._exercise_times: list[float] = []
        v = self._exercise.is_exercise_time()
        exercises = 0
        idx = 0
        for i in range(len(self._relevant_times)):
            self._exercise_index[i] = exercises
            if self._is_rebate_time[i]:
                self._is_exercise_time[i] = v[idx]
                idx += 1
                if self._is_exercise_time[i]:
                    self._exercise_times.append(self._relevant_times[i])
                    exercises += 1

        rate_times = evolution.rate_times()
        rebate_times = self._exercise.possible_cash_flow_times()
        self._rebate_discounters = [
            MarketModelDiscounter(t, rate_times) for t in rebate_times
        ]
        control_times = self._control.possible_cash_flow_times()
        self._control_discounters = [
            MarketModelDiscounter(t, rate_times) for t in control_times
        ]

        basis_sizes = self._basis_system.number_of_functions()
        self._basis_values: list[list[float]] = [
            [0.0] * basis_sizes[i]
            for i in range(self._basis_system.number_of_exercises())
        ]

        self._current_index = 0
        self._principal_in_numeraire_portfolio = 1.0
        self._new_principal = 1.0

    def exercise_times(self) -> list[float]:
        return self._exercise_times

    def relevant_times(self) -> list[float]:
        return self._relevant_times

    def reset(self) -> None:
        self._exercise.reset()
        self._control.reset()
        self._basis_system.reset()
        self._current_index = 0
        self._principal_in_numeraire_portfolio = 1.0
        self._new_principal = 1.0

    def exercise(self, current_state: CurveState) -> bool:
        exercise_index = self._exercise_index[self._current_index - 1]

        exercise_cf = self._exercise.value(current_state)
        exercise_value = (
            exercise_cf.amount
            * self._rebate_discounters[exercise_cf.time_index].numeraire_bonds(
                current_state, self._numeraires[self._current_index - 1]
            )
            / self._principal_in_numeraire_portfolio
        )

        control_cf = self._control.value(current_state)
        control_value = (
            control_cf.amount
            * self._control_discounters[control_cf.time_index].numeraire_bonds(
                current_state, self._numeraires[self._current_index - 1]
            )
            / self._principal_in_numeraire_portfolio
        )

        self._basis_system.values(
            current_state, self._basis_values[exercise_index]
        )

        alphas = self._basis_coefficients[exercise_index]
        continuation_value = control_value + sum(
            a * b for a, b in zip(alphas, self._basis_values[exercise_index], strict=False)
        )

        return exercise_value >= continuation_value

    def next_step(self, current_state: CurveState) -> None:
        self._principal_in_numeraire_portfolio = self._new_principal

        if self._is_rebate_time[self._current_index]:
            self._exercise.next_step(current_state)
        if self._is_control_time[self._current_index]:
            self._control.next_step(current_state)
        if self._is_basis_time[self._current_index]:
            self._basis_system.next_step(current_state)

        if self._current_index < len(self._numeraires) - 1:
            numeraire = self._numeraires[self._current_index]
            next_numeraire = self._numeraires[self._current_index + 1]
            self._new_principal *= current_state.discount_ratio(
                numeraire, next_numeraire
            )

        self._current_index += 1

    def clone(self) -> LongstaffSchwartzExerciseStrategy:
        return copy.deepcopy(self)
