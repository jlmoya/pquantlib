"""UpperBoundEngine — Andersen-Broadie dual upper bound for Bermudan LMM.

# C++ parity: ql/models/marketmodels/callability/upperboundengine.{hpp,cpp}
# (v1.42.1).

The Andersen-Broadie primal-dual upper-bound estimator for a callable
LIBOR-market-model product. The engine evolves the underlying, its rebate, a
hedge, the hedge rebate and a *decorated* callable hedge together; at each
exercise date it sub-simulates the unexercised hedge value (via a fresh inner
``AccountingEngine``) and forms the dual martingale increment, accumulating the
running maximum of the hedged-portfolio value. The expected maximum is the
upper bound; ``initial_numeraire_value`` rescales it to cash.

# C++ parity note: ``DecoratedHedge`` records the ``CurveState`` at each step
# while *not* exercising, so that on ``reset()`` it can replay those states and
# bring the inner callable hedge to the current point of the outer path rather
# than to the path start. The Python port stores deep ``clone()``-d curve
# states; everything else mirrors the C++ control flow verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pquantlib.models.marketmodels.accounting_engine import AccountingEngine
from pquantlib.models.marketmodels.discounter import MarketModelDiscounter
from pquantlib.models.marketmodels.products.call_specified_multiproduct import (
    CallSpecifiedMultiProduct,
    ExerciseStrategy,
)
from pquantlib.models.marketmodels.products.exercise_adapter import ExerciseAdapter
from pquantlib.models.marketmodels.products.multiproduct_composite import (
    MultiProductComposite,
)
from pquantlib.models.marketmodels.utilities import is_in_subset

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pquantlib.models.marketmodels.callability.exercise_value import (
        MarketModelExerciseValue,
    )
    from pquantlib.models.marketmodels.curve_state import CurveState
    from pquantlib.models.marketmodels.evolver import MarketModelEvolver
    from pquantlib.models.marketmodels.multi_product import (
        CashFlow,
        MarketModelMultiProduct,
    )

_QL_MIN_REAL = -1.7976931348623157e308


class _DecoratedHedge(CallSpecifiedMultiProduct):
    """A callable hedge that records / replays curve states for re-hedging.

    # C++ parity: upperboundengine.cpp (anonymous) DecoratedHedge.
    """

    def __init__(self, product: CallSpecifiedMultiProduct) -> None:
        # Re-build a CallSpecifiedMultiProduct from the same parts (C++ copy-ctor).
        super().__init__(
            product.underlying(), product.strategy().clone(), product.rebate()
        )
        n = self.number_of_products()
        self._dh_number_cash_flows: list[int] = [0] * n
        self._dh_cash_flows: list[list[CashFlow]] = [
            [self.CashFlow() for _ in range(self.max_number_of_cash_flows_per_product_per_step())]
            for _ in range(n)
        ]
        # C++ parity: the copy-ctor copies savedStates_ / lastSavedStep_ /
        # recording_; the AccountingEngine clones its product, so this state must
        # survive cloning (otherwise the inner replay is lost).
        if isinstance(product, _DecoratedHedge):
            self._saved_states: list[CurveState] = [
                s.clone() for s in product._saved_states
            ]
            self._last_saved_step = product._last_saved_step
            self._recording = product._recording
        else:
            self._saved_states = []
            self._last_saved_step = 0
            self._recording = True

    def reset(self) -> None:
        super().reset()
        self.disable_callability()
        for i in range(self._last_saved_step):
            super().next_time_step(
                self._saved_states[i],
                self._dh_number_cash_flows,
                self._dh_cash_flows,
            )
        self.enable_callability()

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        if self._recording:
            self._saved_states.append(current_state.clone())
        return super().next_time_step(
            current_state, number_cash_flows_this_step, cash_flows_generated
        )

    def clone(self) -> MarketModelMultiProduct:
        # Re-derive from parts (mirrors C++ DecoratedHedge copy-ctor).
        return _DecoratedHedge(self)

    def save(self) -> None:
        self._last_saved_step = len(self._saved_states)

    def clear(self) -> None:
        self._last_saved_step = 0
        self._saved_states = []
        self._recording = True

    def start_recording(self) -> None:
        self._recording = True

    def stop_recording(self) -> None:
        self._recording = False


class UpperBoundEngine:
    """Andersen-Broadie dual upper-bound engine for Bermudan LMM products.

    # C++ parity: upperboundengine.hpp UpperBoundEngine.

    Pre-condition: ``underlying`` and ``hedge`` must share the same rate times
    and exercise times.
    """

    def __init__(
        self,
        evolver: MarketModelEvolver,
        inner_evolvers: Sequence[MarketModelEvolver],
        underlying: MarketModelMultiProduct,
        rebate: MarketModelExerciseValue,
        hedge: MarketModelMultiProduct,
        hedge_rebate: MarketModelExerciseValue,
        hedge_strategy: ExerciseStrategy,
        initial_numeraire_value: float,
    ) -> None:
        self._evolver = evolver
        self._inner_evolvers = list(inner_evolvers)
        self._initial_numeraire_value = initial_numeraire_value

        self._composite = MultiProductComposite()
        self._composite.add(underlying)
        self._composite.add(ExerciseAdapter(rebate))
        self._composite.add(hedge)
        self._composite.add(ExerciseAdapter(hedge_rebate))
        self._composite.add(
            _DecoratedHedge(
                CallSpecifiedMultiProduct(
                    hedge, hedge_strategy, ExerciseAdapter(hedge_rebate)
                )
            )
        )
        self._composite.finalize()

        self._underlying_offset = 0
        self._underlying_size = underlying.number_of_products()
        self._rebate_offset = self._underlying_size
        self._rebate_size = 1
        self._hedge_offset = self._underlying_size + self._rebate_size
        self._hedge_size = hedge.number_of_products()
        self._hedge_rebate_offset = (
            self._underlying_size + self._rebate_size + self._hedge_size
        )
        self._hedge_rebate_size = 1

        self._number_of_products = self._composite.number_of_products()
        evolution_times = self._composite.evolution().evolution_times()
        self._number_of_steps = len(evolution_times)
        self._is_exercise_time = is_in_subset(
            evolution_times, hedge_strategy.exercise_times()
        )

        self._number_cash_flows_this_step = [0] * self._number_of_products
        max_flows = self._composite.max_number_of_cash_flows_per_product_per_step()
        self._cash_flows_generated: list[list[CashFlow]] = [
            [self._composite.CashFlow() for _ in range(max_flows)]
            for _ in range(self._number_of_products)
        ]

        cash_flow_times = self._composite.possible_cash_flow_times()
        rate_times = self._composite.evolution().rate_times()
        self._discounters = [
            MarketModelDiscounter(t, rate_times) for t in cash_flow_times
        ]

    def multiple_path_values(
        self, stats: object, outer_paths: int, inner_paths: int
    ) -> None:
        """Run ``outer_paths`` outer paths (each with ``inner_paths`` sub-sims).

        ``stats`` must expose ``add(value, weight)``.
        """
        for _ in range(outer_paths):
            value, weight = self.single_path_value(inner_paths)
            stats.add(value, weight)  # type: ignore[attr-defined]

    def single_path_value(self, inner_paths: int) -> tuple[float, float]:
        """One outer-path upper-bound estimate + its weight.

        # C++ parity: upperboundengine.cpp UpperBoundEngine::singlePathValue.
        """
        callable_hedge = self._composite.item(4)
        assert isinstance(callable_hedge, _DecoratedHedge)
        strategy = callable_hedge.strategy()

        maximum_value = _QL_MIN_REAL
        numeraires_held = 0.0
        weight = self._evolver.start_new_path()
        callable_hedge.clear()
        self._composite.reset()
        callable_hedge.disable_callability()
        principal_in_numeraire_portfolio = 1.0
        exercise = 0

        for k in range(self._number_of_steps):
            weight *= self._evolver.advance_step()
            self._composite.next_time_step(
                self._evolver.current_state(),
                self._number_cash_flows_this_step,
                self._cash_flows_generated,
            )

            underlying_cash_flows = self._collect_cash_flows(
                k,
                principal_in_numeraire_portfolio,
                self._underlying_offset,
                self._underlying_offset + self._underlying_size,
            )
            hedge_cash_flows = self._collect_cash_flows(
                k,
                principal_in_numeraire_portfolio,
                self._hedge_offset,
                self._hedge_offset + self._hedge_size,
            )
            rebate_cash_flow = self._collect_cash_flows(
                k,
                principal_in_numeraire_portfolio,
                self._rebate_offset,
                self._rebate_offset + self._rebate_size,
            )
            hedge_rebate_cash_flow = self._collect_cash_flows(
                k,
                principal_in_numeraire_portfolio,
                self._hedge_rebate_offset,
                self._hedge_rebate_offset + self._hedge_rebate_size,
            )

            numeraires_held += underlying_cash_flows - hedge_cash_flows

            if self._is_exercise_time[k]:
                unexercised_hedge_value = 0.0

                if k != self._number_of_steps - 1:
                    current_evolver = self._inner_evolvers[exercise]
                    exercise += 1
                    current_evolver.set_initial_state(self._evolver.current_state())

                    callable_hedge.stop_recording()
                    callable_hedge.enable_callability()
                    callable_hedge.save()

                    inner_engine = AccountingEngine(current_evolver, callable_hedge, 1.0)
                    inner_stats = _MeanCollector(callable_hedge.number_of_products())
                    inner_engine.multiple_path_values(inner_stats, inner_paths)

                    values = inner_stats.mean()
                    unexercised_hedge_value = (
                        sum(values) / principal_in_numeraire_portfolio
                    )

                    callable_hedge.disable_callability()
                    callable_hedge.start_recording()

                portfolio_value = numeraires_held
                if strategy.exercise(self._evolver.current_state()):
                    portfolio_value += rebate_cash_flow - hedge_rebate_cash_flow
                    numeraires_held += unexercised_hedge_value - hedge_rebate_cash_flow
                else:
                    portfolio_value += rebate_cash_flow - unexercised_hedge_value

                maximum_value = max(maximum_value, portfolio_value)

            if k < self._number_of_steps - 1:
                numeraire = self._evolver.numeraires()[k]
                next_numeraire = self._evolver.numeraires()[k + 1]
                principal_in_numeraire_portfolio *= self._evolver.current_state().discount_ratio(
                    numeraire, next_numeraire
                )

        maximum_value = max(maximum_value, numeraires_held)
        maximum_value *= self._initial_numeraire_value
        return maximum_value, weight

    def _collect_cash_flows(
        self,
        current_step: int,
        principal_in_numeraire_portfolio: float,
        begin_product: int,
        end_product: int,
    ) -> float:
        # C++ parity: upperboundengine.cpp UpperBoundEngine::collectCashFlows.
        numeraire = self._evolver.numeraires()[current_step]
        numeraire_units = 0.0
        for i in range(begin_product, end_product):
            cashflows = self._cash_flows_generated[i]
            for j in range(self._number_cash_flows_this_step[i]):
                discounter = self._discounters[cashflows[j].time_index]
                numeraire_units += cashflows[j].amount * discounter.numeraire_bonds(
                    self._evolver.current_state(), numeraire
                )
        return numeraire_units / principal_in_numeraire_portfolio


class _MeanCollector:
    """Minimal SequenceStatisticsInc shim collecting per-dimension means.

    # C++ parity: the SequenceStatisticsInc used by the inner AccountingEngine
    # (only its ``mean()`` is consumed).
    """

    def __init__(self, dimension: int) -> None:
        self._sums = [0.0] * dimension
        self._weights = 0.0

    def add(self, values: list[float], weight: float = 1.0) -> None:
        for i, v in enumerate(values):
            self._sums[i] += v * weight
        self._weights += weight

    def mean(self) -> list[float]:
        if self._weights == 0.0:
            return list(self._sums)
        return [s / self._weights for s in self._sums]
