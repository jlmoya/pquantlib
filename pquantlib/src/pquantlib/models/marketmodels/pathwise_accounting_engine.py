"""PathwiseAccountingEngine — pathwise (Delta) accounting engine.

# C++ parity: ql/models/marketmodels/pathwiseaccountingengine.{hpp,cpp}
# (v1.42.1) — the ``PathwiseAccountingEngine`` class (Deltas only).

Collects cash flows along a market-model simulation for the pathwise
(Giles-Glasserman smoking-adjoints) computation of Deltas. Works with a
displaced LMM and needs the model's pseudo-roots + displacements; the
backwards sweep accumulates the per-rate sensitivity matrix ``V`` from the
generated cash flows.

Divergences from C++:

- C++ takes ``ext::shared_ptr<LogNormalFwdRateEuler>`` because the Delta
  recursion is intimately tied to log-normal Euler evolution; the simple
  Delta engine ported here only calls the generic ``MarketModelEvolver``
  interface (``start_new_path`` / ``current_step`` / ``advance_step`` /
  ``current_state``), so it is typed against that abstract base. A concrete
  ``LogNormalFwdRateEuler`` lands in W10; the Vegas variants
  (``PathwiseVegasAccountingEngine`` / ``...Outer...``) additionally need the
  evolver's ``browniansThisStep`` + ``RatePseudoRootJacobian``. As of W11-D
  both blockers are satisfied (``LogNormalFwdRateEuler.browniansThisStep`` from
  W10-B and ``RatePseudoRootJacobian`` from W11-D), so the Vegas engines are
  *unblocked*; the ~1200-line GG smoking-adjoints Vegas backward sweep itself
  is left as a follow-up beyond W11-D's 4-class pathwise-greeks scope.
- ``SequenceStatisticsInc&`` -> any ``add(values, weight)`` accumulator.
- C++ ``Matrix`` workspace -> numpy float64 2-D arrays.
- ``Clone<MarketModelPathwiseMultiProduct>`` -> ``product.clone()``.
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Protocol

import numpy as np

from pquantlib.models.marketmodels.pathwise_discounter import (
    MarketModelPathwiseDiscounter,
)

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.evolver import MarketModelEvolver
    from pquantlib.models.marketmodels.market_model import MarketModel
    from pquantlib.models.marketmodels.pathwise_multi_product import (
        MarketModelPathwiseMultiProduct,
        PathwiseCashFlow,
    )


class SequenceStatsLike(Protocol):
    """Minimal sequence-statistics protocol (C++ ``SequenceStatisticsInc``)."""

    def add(self, values: list[float], weight: float) -> None: ...


class PathwiseAccountingEngine:
    """Engine collecting cash flows for pathwise Delta computation.

    # C++ parity: pathwiseaccountingengine.hpp PathwiseAccountingEngine.
    """

    def __init__(
        self,
        evolver: MarketModelEvolver,
        product: MarketModelPathwiseMultiProduct,
        pseudo_root_structure: MarketModel,
        initial_numeraire_value: float,
    ) -> None:
        self._evolver = evolver
        self._product = product.clone()
        self._pseudo_root_structure = pseudo_root_structure
        self._initial_numeraire_value = initial_numeraire_value
        self._number_products = self._product.number_of_products()
        self._do_deflation = not self._product.already_deflated()

        self._number_rates = pseudo_root_structure.number_of_rates()
        self._number_steps = pseudo_root_structure.number_of_steps()
        n_rates = self._number_rates
        n_steps = self._number_steps

        # workspace
        self._numeraires_held = [0.0] * self._number_products
        self._number_cash_flows_this_step = [0] * self._number_products
        self._deflator_and_derivatives = [0.0] * (n_rates + 1)

        # Discounts_: (steps+1, rates+1); column 0 == 1.0 (P(t_0,t_0))
        self._discounts = np.zeros((n_steps + 1, n_rates + 1), dtype=np.float64)
        self._discounts[:, 0] = 1.0

        cash_flow_times = self._product.possible_cash_flow_times()
        self._number_cash_flow_times = len(cash_flow_times)

        # V_: one (steps+1, rates) matrix per product
        self._v = [
            np.zeros((n_steps + 1, n_rates), dtype=np.float64)
            for _ in range(self._number_products)
        ]
        # numberCashFlowsThisIndex_[product][cashFlowTimeIndex]
        self._number_cash_flows_this_index = [
            [0] * self._number_cash_flow_times for _ in range(self._number_products)
        ]
        # totalCashFlowsThisIndex_[product] = (cashFlowTimes, rates+1)
        self._total_cash_flows_this_index = [
            np.zeros((self._number_cash_flow_times, n_rates + 1), dtype=np.float64)
            for _ in range(self._number_products)
        ]

        self._libor_ratios = np.zeros((n_steps + 1, n_rates), dtype=np.float64)
        self._steps_discounts_squared = np.zeros((n_steps + 1, n_rates), dtype=np.float64)
        self._libor_rates = np.zeros((n_steps + 1, n_rates), dtype=np.float64)

        max_flows = self._product.max_number_of_cash_flows_per_product_per_step()
        self._cash_flows_generated: list[list[PathwiseCashFlow]] = []
        for _ in range(self._number_products):
            row: list[PathwiseCashFlow] = []
            for _ in range(max_flows):
                cf = self._product.CashFlow()
                cf.amount = [0.0] * (n_rates + 1)
                row.append(cf)
            self._cash_flows_generated.append(row)

        rate_times = self._product.evolution().rate_times()
        evolution_times = self._product.evolution().evolution_times()
        self._discounters = [
            MarketModelPathwiseDiscounter(t, rate_times) for t in cash_flow_times
        ]

        # cashFlowIndicesThisStep_[step] = list of cash-flow-time indices whose
        # flow happens after the last-completed step.
        self._cash_flow_indices_this_step: list[list[int]] = [
            [] for _ in range(n_steps)
        ]
        for i in range(self._number_cash_flow_times):
            # C++ parity: it = upper_bound(evolutionTimes, cashFlowTimes[i]);
            # if (it != begin) --it; index = it - begin.
            ub = bisect.bisect_right(evolution_times, cash_flow_times[i])
            index = ub - 1 if ub != 0 else 0
            self._cash_flow_indices_this_step[index].append(i)

        self._partials = np.zeros(
            (pseudo_root_structure.number_of_factors(), n_rates), dtype=np.float64
        )

    def _single_path_values(self, values: list[float]) -> float:  # noqa: PLR0915
        # C++ parity: pathwiseaccountingengine.cpp
        # PathwiseAccountingEngine::singlePathValues.
        n_rates = self._number_rates
        n_steps = self._number_steps
        initial_forwards = list(self._pseudo_root_structure.initial_rates())
        current_forwards = list(initial_forwards)

        # clear accumulation variables
        for i in range(self._number_products):
            self._numeraires_held[i] = 0.0
            self._number_cash_flows_this_index[i] = [0] * self._number_cash_flow_times
            self._total_cash_flows_this_index[i].fill(0.0)
            self._v[i].fill(0.0)

        weight = self._evolver.start_new_path()
        self._product.reset()

        this_step = 0
        done = False
        while not done:
            this_step = self._evolver.current_step()
            store_step = this_step + 1
            weight *= self._evolver.advance_step()

            done = self._product.next_time_step(
                self._evolver.current_state(),
                self._number_cash_flows_this_step,
                self._cash_flows_generated,
            )

            last_forwards = current_forwards
            current_forwards = list(self._evolver.current_state().forward_rates())

            state = self._evolver.current_state()
            for i in range(n_rates):
                x = state.discount_ratio(i + 1, i)
                self._steps_discounts_squared[store_step, i] = x * x
                self._libor_ratios[store_step, i] = current_forwards[i] / last_forwards[i]
                self._libor_rates[store_step, i] = current_forwards[i]
                self._discounts[store_step, i + 1] = state.discount_ratio(i + 1, 0)

            for i in range(self._number_products):
                for j in range(self._number_cash_flows_this_step[i]):
                    k = self._cash_flows_generated[i][j].time_index
                    self._number_cash_flows_this_index[i][k] += 1
                    amount = self._cash_flows_generated[i][j].amount
                    total_row = self._total_cash_flows_this_index[i]
                    for ell in range(n_rates + 1):
                        total_row[k, ell] += amount[ell] * weight

        # backwards computation
        factors = self._pseudo_root_structure.number_of_factors()
        taus = self._pseudo_root_structure.evolution().rate_taus()
        flows_found = False
        final_step_done = this_step

        for current_step in range(n_steps - 1, -1, -1):
            step_to_use = min(current_step, final_step_done) + 1

            for cash_flow_index in self._cash_flow_indices_this_step[current_step]:
                no_flows = True
                for ell in range(self._number_products):
                    if self._number_cash_flows_this_index[ell][cash_flow_index] != 0:
                        no_flows = False
                        break
                flows_found = flows_found or (not no_flows)

                if not no_flows:
                    if self._do_deflation:
                        self._discounters[cash_flow_index].get_factors(
                            self._libor_rates,
                            self._discounts,
                            step_to_use,
                            self._deflator_and_derivatives,
                        )
                    for j in range(self._number_products):
                        if self._number_cash_flows_this_index[j][cash_flow_index] > 0:
                            total_row = self._total_cash_flows_this_index[j]
                            deflated_cash_flow = float(total_row[cash_flow_index, 0])
                            if self._do_deflation:
                                deflated_cash_flow *= self._deflator_and_derivatives[0]
                            self._numeraires_held[j] += deflated_cash_flow

                            for i in range(1, n_rates + 1):
                                this_derivative = float(total_row[cash_flow_index, i])
                                if self._do_deflation:
                                    this_derivative *= self._deflator_and_derivatives[0]
                                    this_derivative += (
                                        float(total_row[cash_flow_index, 0])
                                        * self._deflator_and_derivatives[i]
                                    )
                                self._v[j][step_to_use, i - 1] += this_derivative

            if flows_found:
                next_step_to_use = min(current_step - 1, final_step_done)
                next_step_index = next_step_to_use + 1
                if next_step_index != step_to_use:
                    this_pseudo_root = self._pseudo_root_structure.pseudo_root(current_step)
                    for i in range(self._number_products):
                        v_i = self._v[i]
                        # compute partials
                        for f in range(factors):
                            libor = self._libor_rates[step_to_use, n_rates - 1]
                            v_last = v_i[step_to_use, n_rates - 1]
                            pseudo = this_pseudo_root[n_rates - 1, f]
                            self._partials[f, n_rates - 1] = libor * v_last * pseudo
                            for r in range(n_rates - 2, -1, -1):
                                term = (
                                    self._libor_rates[step_to_use, r]
                                    * v_i[step_to_use, r]
                                    * this_pseudo_root[r, f]
                                )
                                self._partials[f, r] = self._partials[f, r + 1] + term
                        for j in range(n_rates):
                            next_v = v_i[step_to_use, j] * self._libor_ratios[step_to_use, j]
                            v_i[next_step_index, j] = next_v
                            summand = 0.0
                            for f in range(factors):
                                summand += this_pseudo_root[j, f] * self._partials[f, j]
                            summand *= taus[j] * self._steps_discounts_squared[step_to_use, j]
                            v_i[next_step_index, j] += summand

        # write answer into values
        for i in range(self._number_products):
            values[i] = self._numeraires_held[i] * self._initial_numeraire_value
            for j in range(n_rates):
                values[(i + 1) * self._number_products + j] = (
                    float(self._v[i][0, j]) * self._initial_numeraire_value
                )

        # weight already folded in (lower-variance form); return 1.0
        return 1.0

    def multiple_path_values(
        self, stats: SequenceStatsLike, number_of_paths: int
    ) -> None:
        """Run ``number_of_paths`` paths, feeding each value vector to ``stats``.

        # C++ parity: pathwiseaccountingengine.cpp
        # PathwiseAccountingEngine::multiplePathValues.
        """
        values = [0.0] * (self._product.number_of_products() * (self._number_rates + 1))
        for _ in range(number_of_paths):
            weight = self._single_path_values(values)
            stats.add(values, weight)
