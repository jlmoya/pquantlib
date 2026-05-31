"""AccountingEngine — BGM Monte-Carlo cash-flow accounting engine.

# C++ parity: ql/models/marketmodels/accountingengine.{hpp,cpp} (v1.42.1).

The core BGM Monte-Carlo pricing loop: drives a ``MarketModelEvolver`` over a
``MarketModelMultiProduct``, converting each generated cash flow into units of
the per-step numeraire (via ``MarketModelDiscounter``) and accumulating the
numeraire-rebased path value, then handing the per-product path values to a
sequence-statistics accumulator.

Divergences from C++:

- C++ takes ``SequenceStatisticsInc&`` (boost-accumulator sequence stats);
  pquantlib has no ``SequenceStatistics`` port yet, so ``multiple_path_values``
  accepts any object exposing ``add(values, weight)`` (the exact C++ call
  site). A minimal in-test shim supplies one; a full ``SequenceStatistics``
  port lands when a consumer needs it.
- C++ ``Clone<MarketModelMultiProduct>`` deep-copies the product; the Python
  port calls ``product.clone()`` for the same single-ownership semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pquantlib.models.marketmodels.discounter import MarketModelDiscounter

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.evolver import MarketModelEvolver
    from pquantlib.models.marketmodels.multi_product import (
        CashFlow,
        MarketModelMultiProduct,
    )


class SequenceStatsLike(Protocol):
    """Minimal sequence-statistics protocol (C++ ``SequenceStatisticsInc``)."""

    def add(self, values: list[float], weight: float) -> None: ...


class AccountingEngine:
    """Engine collecting cash flows along a market-model simulation.

    # C++ parity: accountingengine.hpp AccountingEngine.
    """

    def __init__(
        self,
        evolver: MarketModelEvolver,
        product: MarketModelMultiProduct,
        initial_numeraire_value: float,
    ) -> None:
        self._evolver = evolver
        self._product = product.clone()
        self._initial_numeraire_value = initial_numeraire_value
        self._number_products = self._product.number_of_products()

        # workspace (C++ parity: preallocated member buffers)
        self._numeraires_held = [0.0] * self._number_products
        self._number_cash_flows_this_step = [0] * self._number_products
        max_flows = self._product.max_number_of_cash_flows_per_product_per_step()
        self._cash_flows_generated: list[list[CashFlow]] = [
            [self._product.CashFlow() for _ in range(max_flows)]
            for _ in range(self._number_products)
        ]

        cash_flow_times = self._product.possible_cash_flow_times()
        rate_times = self._product.evolution().rate_times()
        self._discounters = [
            MarketModelDiscounter(t, rate_times) for t in cash_flow_times
        ]

    def _single_path_values(self, values: list[float]) -> float:
        # C++ parity: accountingengine.cpp AccountingEngine::singlePathValues.
        for i in range(self._number_products):
            self._numeraires_held[i] = 0.0
        weight = self._evolver.start_new_path()
        self._product.reset()
        principal_in_numeraire_portfolio = 1.0

        done = False
        while not done:
            this_step = self._evolver.current_step()
            weight *= self._evolver.advance_step()
            done = self._product.next_time_step(
                self._evolver.current_state(),
                self._number_cash_flows_this_step,
                self._cash_flows_generated,
            )
            numeraire = self._evolver.numeraires()[this_step]

            for i in range(self._number_products):
                cashflows = self._cash_flows_generated[i]
                for j in range(self._number_cash_flows_this_step[i]):
                    discounter = self._discounters[cashflows[j].time_index]
                    bonds = cashflows[j].amount * discounter.numeraire_bonds(
                        self._evolver.current_state(), numeraire
                    )
                    self._numeraires_held[i] += bonds / principal_in_numeraire_portfolio

            if not done:
                next_numeraire = self._evolver.numeraires()[this_step + 1]
                principal_in_numeraire_portfolio *= self._evolver.current_state().discount_ratio(
                    numeraire, next_numeraire
                )

        for i in range(len(self._numeraires_held)):
            values[i] = self._numeraires_held[i] * self._initial_numeraire_value
        return weight

    def multiple_path_values(
        self, stats: SequenceStatsLike, number_of_paths: int
    ) -> None:
        """Run ``number_of_paths`` paths, feeding each value vector to ``stats``.

        # C++ parity: accountingengine.cpp AccountingEngine::multiplePathValues.
        """
        values = [0.0] * self._product.number_of_products()
        for _ in range(number_of_paths):
            weight = self._single_path_values(values)
            stats.add(values, weight)
