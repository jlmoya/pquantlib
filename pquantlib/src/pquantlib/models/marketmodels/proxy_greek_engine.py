"""ProxyGreekEngine — Fries-Joshi proxy-simulation Greek engine.

# C++ parity: ql/models/marketmodels/proxygreekengine.{hpp,cpp} (v1.42.1).

Runs the original evolver plus a family of ``ConstrainedEvolver`` variants
(each constrained on a per-step swap rate seen along the original path), then
forms Greeks as diff-weighted linear combinations of the original + modified
path values. The original-evolver leg also records, per step, the swap rate
used to constrain the modified evolvers (importance sampling).

Divergences from C++:

- ``multiple_path_values`` takes any sequence-statistics-like object (with
  ``add(values)``); see ``AccountingEngine`` for the same rationale. The
  ``modified_stats`` argument is a matching nested list of such accumulators.
- ``Clone<MarketModelMultiProduct>`` -> ``product.clone()``.
- C++ ``std::valarray<bool> constraintsActive_`` -> a Python ``list[bool]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pquantlib.models.marketmodels.discounter import MarketModelDiscounter

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.constrained_evolver import ConstrainedEvolver
    from pquantlib.models.marketmodels.evolver import MarketModelEvolver
    from pquantlib.models.marketmodels.multi_product import (
        CashFlow,
        MarketModelMultiProduct,
    )


class StatsAddLike(Protocol):
    """Minimal sequence-statistics protocol exposing ``add(values)``."""

    def add(self, values: list[float], weight: float = ...) -> None: ...


class ProxyGreekEngine:
    """Proxy-simulation Greek engine.

    # C++ parity: proxygreekengine.hpp ProxyGreekEngine.
    """

    def __init__(
        self,
        evolver: MarketModelEvolver,
        constrained_evolvers: list[list[ConstrainedEvolver]],
        diff_weights: list[list[list[float]]],
        start_index_of_constraint: list[int],
        end_index_of_constraint: list[int],
        product: MarketModelMultiProduct,
        initial_numeraire_value: float,
    ) -> None:
        self._original_evolver = evolver
        self._constrained_evolvers = constrained_evolvers
        self._diff_weights = diff_weights
        self._start_index_of_constraint = start_index_of_constraint
        self._end_index_of_constraint = end_index_of_constraint
        self._product = product.clone()
        self._initial_numeraire_value = initial_numeraire_value
        self._number_products = self._product.number_of_products()

        # workspace
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
        evolution_times = self._product.evolution().evolution_times()
        self._constraints = [0.0] * len(evolution_times)
        self._constraints_active = [False] * len(evolution_times)

    def _single_evolver_values(
        self,
        evolver: MarketModelEvolver,
        values: list[float],
        store_rates: bool = False,
    ) -> None:
        # C++ parity: proxygreekengine.cpp ProxyGreekEngine::singleEvolverValues.
        for i in range(self._number_products):
            self._numeraires_held[i] = 0.0
        weight = evolver.start_new_path()
        self._product.reset()
        principal_in_numeraire_portfolio = 1.0

        if store_rates:
            for k in range(len(self._constraints_active)):
                self._constraints_active[k] = False

        done = False
        while not done:
            this_step = evolver.current_step()
            weight *= evolver.advance_step()
            done = self._product.next_time_step(
                evolver.current_state(),
                self._number_cash_flows_this_step,
                self._cash_flows_generated,
            )
            if store_rates:
                self._constraints[this_step] = evolver.current_state().swap_rate(
                    self._start_index_of_constraint[this_step],
                    self._end_index_of_constraint[this_step],
                )
                self._constraints_active[this_step] = True

            numeraire = evolver.numeraires()[this_step]

            for i in range(self._number_products):
                cashflows = self._cash_flows_generated[i]
                for j in range(self._number_cash_flows_this_step[i]):
                    discounter = self._discounters[cashflows[j].time_index]
                    bonds = cashflows[j].amount * discounter.numeraire_bonds(
                        evolver.current_state(), numeraire
                    )
                    self._numeraires_held[i] += (
                        weight * bonds / principal_in_numeraire_portfolio
                    )

            if not done:
                next_numeraire = evolver.numeraires()[this_step + 1]
                principal_in_numeraire_portfolio *= evolver.current_state().discount_ratio(
                    numeraire, next_numeraire
                )

        for i in range(len(self._numeraires_held)):
            values[i] = self._numeraires_held[i] * self._initial_numeraire_value

    def single_path_values(
        self,
        values: list[float],
        modified_values: list[list[list[float]]],
    ) -> None:
        """Run the original + all constrained evolvers for one path.

        # C++ parity: proxygreekengine.cpp ProxyGreekEngine::singlePathValues.
        """
        self._single_evolver_values(self._original_evolver, values, True)
        for i in range(len(self._constrained_evolvers)):
            for j in range(len(self._constrained_evolvers[i])):
                self._constrained_evolvers[i][j].set_this_constraint(
                    self._constraints, self._constraints_active
                )
                self._single_evolver_values(
                    self._constrained_evolvers[i][j], modified_values[i][j]
                )

    def multiple_path_values(
        self,
        stats: StatsAddLike,
        modified_stats: list[list[StatsAddLike]],
        number_of_paths: int,
    ) -> None:
        """Run ``number_of_paths`` paths; accumulate base + diff-weighted Greeks.

        # C++ parity: proxygreekengine.cpp ProxyGreekEngine::multiplePathValues.
        """
        n = self._product.number_of_products()
        values = [0.0] * n
        modified_values: list[list[list[float]]] = []
        for i in range(len(self._constrained_evolvers)):
            row: list[list[float]] = []
            for _ in range(len(self._constrained_evolvers[i])):
                row.append([0.0] * n)
            modified_values.append(row)

        results = [0.0] * n

        for _ in range(number_of_paths):
            self.single_path_values(values, modified_values)
            stats.add(values)

            for j in range(len(self._diff_weights)):
                for k in range(len(self._diff_weights[j])):
                    weights = self._diff_weights[j][k]
                    for ell in range(n):
                        results[ell] = weights[0] * values[ell]
                        for m in range(1, len(weights)):
                            results[ell] += weights[m] * modified_values[j][m - 1][ell]
                    modified_stats[j][k].add(results)
