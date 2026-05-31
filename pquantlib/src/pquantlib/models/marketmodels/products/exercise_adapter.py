"""ExerciseAdapter — adapts a MarketModelExerciseValue to a product.

# C++ parity: ql/models/marketmodels/products/multistep/exerciseadapter.{hpp,cpp}
# (v1.42.1).

Turns a ``MarketModelExerciseValue`` (the callability "what do I get if I
exercise now" object) into a plain ``MarketModelMultiProduct`` that pays the
exercise value at the (single) exercise time and then terminates.

The ``MarketModelExerciseValue`` Protocol here is a forward-declaration of the
W11-C callability interface; W11-C provides the concretes
(``NothingExerciseValue``, ``SwapRateTrigger`` etc.). Its required shape:

    number_of_exercises() -> int
    evolution() -> EvolutionDescription
    possible_cash_flow_times() -> list[float]
    next_step(current_state: CurveState) -> None
    reset() -> None
    is_exercise_time() -> list[bool]
    value(current_state: CurveState) -> CashFlow
    clone() -> MarketModelExerciseValue
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.products.multiproduct_multistep import (
    MultiProductMultiStep,
)


@runtime_checkable
class MarketModelExerciseValue(Protocol):
    """The callability exercise-value interface (W11-C provides concretes).

    # C++ parity: ql/models/marketmodels/callability/exercisevalue.hpp
    # MarketModelExerciseValue.
    """

    def number_of_exercises(self) -> int: ...
    def evolution(self) -> EvolutionDescription: ...
    def possible_cash_flow_times(self) -> list[float]: ...
    def next_step(self, current_state: CurveState) -> None: ...
    def reset(self) -> None: ...
    def is_exercise_time(self) -> list[bool]: ...
    def value(self, current_state: CurveState) -> CashFlow: ...
    def clone(self) -> MarketModelExerciseValue: ...


class ExerciseAdapter(MultiProductMultiStep):
    """Adapts an exercise value to a market-model product.

    # C++ parity: exerciseadapter.hpp ExerciseAdapter.
    """

    def __init__(
        self, exercise: MarketModelExerciseValue, number_of_products: int = 1
    ) -> None:
        super().__init__(exercise.evolution().rate_times())
        self._exercise = exercise.clone()
        self._number_of_products = number_of_products
        self._is_exercise_time = list(self._exercise.is_exercise_time())
        self._current_index = 0

    # --- MarketModelMultiProduct interface (overrides) ----------------------
    def possible_cash_flow_times(self) -> list[float]:
        return self._exercise.possible_cash_flow_times()

    def number_of_products(self) -> int:
        return self._number_of_products

    def evolution(self) -> EvolutionDescription:
        return self._exercise.evolution()

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 1

    def reset(self) -> None:
        self._exercise.reset()
        self._current_index = 0

    def exercise_value(self) -> MarketModelExerciseValue:
        return self._exercise

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        for i in range(len(number_cash_flows_this_step)):
            number_cash_flows_this_step[i] = 0
        done = False

        self._exercise.next_step(current_state)
        if self._is_exercise_time[self._current_index]:
            cashflow = self._exercise.value(current_state)
            number_cash_flows_this_step[0] = 1
            dst = cash_flows_generated[0][0]
            dst.time_index = cashflow.time_index
            dst.amount = cashflow.amount
            done = True
        self._current_index += 1
        return done or self._current_index == len(self._is_exercise_time)

    def clone(self) -> MarketModelMultiProduct:
        return ExerciseAdapter(self._exercise, self._number_of_products)
