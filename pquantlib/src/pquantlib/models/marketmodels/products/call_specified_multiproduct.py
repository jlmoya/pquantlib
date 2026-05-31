"""CallSpecifiedMultiProduct — wrap a product with a callable exercise schedule.

# C++ parity:
# ql/models/marketmodels/products/multistep/callspecifiedmultiproduct.{hpp,cpp}
# (v1.42.1).

Wraps an underlying ``MarketModelMultiProduct`` with an ``ExerciseStrategy``
(when to call) and a rebate product (what is paid on call). Before the call the
underlying's cash flows flow through; once the strategy exercises, the rebate's
cash flows flow through instead (offset into the combined cash-flow-time
vector). The evolution merges four time series: the underlying's evolution
times, the strategy's exercise times, the rebate's evolution times, and the
strategy's relevant times.

The ``ExerciseStrategy`` Protocol here is a forward-declaration of the W11-C
callability interface; W11-C provides the concretes (e.g.
``SwapRateTrigger``-style strategies). Its required shape (mirrors the C++
``ExerciseStrategy<CurveState>``):

    exercise_times() -> list[float]
    relevant_times() -> list[float]
    reset() -> None
    exercise(current_state: CurveState) -> bool
    next_step(current_state: CurveState) -> None
    clone() -> ExerciseStrategy

Divergences from C++:

- C++ uses ``Clone<>`` for underlying / strategy / rebate; the Python port
  ``.clone()``-s the underlying and rebate on construction (single ownership).
  The strategy is held by reference and ``.clone()``-d only when this product is
  cloned, matching the C++ deep-copy on copy-construction.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from pquantlib import qassert
from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.products.cash_rebate import MarketModelCashRebate
from pquantlib.models.marketmodels.utilities import merge_times


@runtime_checkable
class ExerciseStrategy(Protocol):
    """The callability exercise-strategy interface (W11-C provides concretes).

    # C++ parity: ql/methods/montecarlo/exercisestrategy.hpp
    # ExerciseStrategy<CurveState>.
    """

    def exercise_times(self) -> list[float]: ...
    def relevant_times(self) -> list[float]: ...
    def reset(self) -> None: ...
    def exercise(self, current_state: CurveState) -> bool: ...
    def next_step(self, current_state: CurveState) -> None: ...
    def clone(self) -> ExerciseStrategy: ...


class CallSpecifiedMultiProduct(MarketModelMultiProduct):
    """An underlying product made callable by an exercise strategy + rebate.

    # C++ parity: callspecifiedmultiproduct.hpp CallSpecifiedMultiProduct.
    """

    def __init__(
        self,
        underlying: MarketModelMultiProduct,
        exercise_strategy: ExerciseStrategy,
        rebate: MarketModelMultiProduct | None = None,
    ) -> None:
        self._underlying = underlying.clone()
        self._strategy = exercise_strategy
        products = self._underlying.number_of_products()
        d1 = underlying.evolution()
        rate_times1 = d1.rate_times()
        evolution_times1 = d1.evolution_times()
        exercise_times = exercise_strategy.exercise_times()

        if rebate is not None:
            self._rebate: MarketModelMultiProduct = rebate.clone()
            rate_times2 = self._rebate.evolution().rate_times()
            qassert.require(
                len(rate_times1) == len(rate_times2)
                and all(a == b for a, b in zip(rate_times1, rate_times2, strict=False)),
                "incompatible rate times",
            )
        else:
            description = EvolutionDescription(rate_times1, exercise_times)
            amounts: Matrix = np.zeros(
                (products, len(exercise_times)), dtype=np.float64
            )
            self._rebate = MarketModelCashRebate(
                description, exercise_times, amounts, products
            )

        all_evolution_times = [
            list(evolution_times1),
            list(exercise_times),
            list(self._rebate.evolution().evolution_times()),
            list(exercise_strategy.relevant_times()),
        ]
        merged_evolution_times, self._is_present = merge_times(all_evolution_times)

        # TODO (C++): add relevant rates
        self._evolution = EvolutionDescription(rate_times1, merged_evolution_times)

        self._cash_flow_times = list(self._underlying.possible_cash_flow_times())
        self._rebate_offset = len(self._cash_flow_times)
        self._cash_flow_times.extend(self._rebate.possible_cash_flow_times())

        self._dummy_cash_flows_this_step = [0] * products
        n = self._rebate.max_number_of_cash_flows_per_product_per_step()
        self._dummy_cash_flows_generated: list[list[CashFlow]] = [
            [CashFlow() for _ in range(n)] for _ in range(products)
        ]

        self._was_called = False
        self._current_index = 0
        self._callable = True

    # --- MarketModelMultiProduct interface ----------------------------------
    def suggested_numeraires(self) -> list[int]:
        return self._underlying.suggested_numeraires()

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def possible_cash_flow_times(self) -> list[float]:
        return self._cash_flow_times

    def number_of_products(self) -> int:
        return self._underlying.number_of_products()

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return max(
            self._underlying.max_number_of_cash_flows_per_product_per_step(),
            self._rebate.max_number_of_cash_flows_per_product_per_step(),
        )

    def reset(self) -> None:
        self._underlying.reset()
        self._rebate.reset()
        self._strategy.reset()
        self._current_index = 0
        self._was_called = False

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        is_underlying_time = self._is_present[0][self._current_index]
        is_exercise_time = self._is_present[1][self._current_index]
        is_rebate_time = self._is_present[2][self._current_index]
        is_strategy_relevant_time = self._is_present[3][self._current_index]

        done = False

        if not self._was_called and is_strategy_relevant_time:
            self._strategy.next_step(current_state)

        if not self._was_called and is_exercise_time and self._callable:
            self._was_called = self._strategy.exercise(current_state)

        if self._was_called:
            if is_rebate_time:
                done = self._rebate.next_time_step(
                    current_state, number_cash_flows_this_step, cash_flows_generated
                )
                for i in range(len(number_cash_flows_this_step)):
                    for j in range(number_cash_flows_this_step[i]):
                        cash_flows_generated[i][j].time_index += self._rebate_offset
        else:
            if is_rebate_time:
                self._rebate.next_time_step(
                    current_state,
                    self._dummy_cash_flows_this_step,
                    self._dummy_cash_flows_generated,
                )
            if is_underlying_time:
                done = self._underlying.next_time_step(
                    current_state, number_cash_flows_this_step, cash_flows_generated
                )

        self._current_index += 1
        return done or self._current_index == len(self._evolution.evolution_times())

    def clone(self) -> MarketModelMultiProduct:
        return CallSpecifiedMultiProduct(
            self._underlying, self._strategy.clone(), self._rebate
        )

    # --- inspectors ---------------------------------------------------------
    def underlying(self) -> MarketModelMultiProduct:
        return self._underlying

    def strategy(self) -> ExerciseStrategy:
        return self._strategy

    def rebate(self) -> MarketModelMultiProduct:
        return self._rebate

    def enable_callability(self) -> None:
        self._callable = True

    def disable_callability(self) -> None:
        self._callable = False
