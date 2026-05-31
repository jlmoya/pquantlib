"""collect_node_data + the generic Longstaff-Schwartz regression.

# C++ parity:
# ql/models/marketmodels/callability/collectnodedata.{hpp,cpp} +
# ql/methods/montecarlo/nodedata.hpp +
# ql/methods/montecarlo/genericlsregression.{hpp,cpp} (v1.42.1).

``collect_node_data`` runs the (single-product) market-model simulation over
``number_of_paths`` paths, recording per-exercise ``NodeData`` (the deflated
exercise value, the basis-function values, the deflated control value and the
running cumulated cash flows). ``generic_longstaff_schwartz_regression`` then
back-propagates a least-squares regression of the continuation value onto the
basis functions, producing the per-exercise basis coefficients consumed by
``LongstaffSchwartzExerciseStrategy``.

The regression mirrors the C++ ``genericLongstaffSchwartzRegression`` exactly:
the normal-equations matrix ``C`` and target are built from the basis/cash-flow
covariance + means, and solved via SVD least squares (here ``numpy.linalg.lstsq``
— the closest direct analogue of the C++ ``SVD(C).solveFor(target)``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pquantlib import qassert
from pquantlib.math.statistics.sequence_statistics import SequenceStatistics
from pquantlib.models.marketmodels.discounter import MarketModelDiscounter
from pquantlib.models.marketmodels.utilities import is_in_subset

if TYPE_CHECKING:
    from pquantlib.models.marketmodels.callability.exercise_value import (
        MarketModelExerciseValue,
    )
    from pquantlib.models.marketmodels.callability.node_data_provider import (
        MarketModelNodeDataProvider,
    )
    from pquantlib.models.marketmodels.evolver import MarketModelEvolver
    from pquantlib.models.marketmodels.multi_product import MarketModelMultiProduct


@dataclass(slots=True)
class NodeData:
    """Per-exercise simulation node data for Longstaff-Schwartz.

    # C++ parity: ql/methods/montecarlo/nodedata.hpp NodeData.
    """

    exercise_value: float = 0.0
    cumulated_cash_flows: float = 0.0
    values: list[float] = field(default_factory=list[float])
    control_value: float = 0.0
    is_valid: bool = False


def collect_node_data(  # noqa: PLR0915
    evolver: MarketModelEvolver,
    product: MarketModelMultiProduct,
    data_provider: MarketModelNodeDataProvider,
    rebate: MarketModelExerciseValue,
    control: MarketModelExerciseValue,
    number_of_paths: int,
) -> list[list[NodeData]]:
    """Collect per-exercise LS node data over ``number_of_paths`` MC paths.

    # C++ parity: collectnodedata.cpp collectNodeData. Returns the
    ``collectedData`` out-parameter (``exercises+1`` rows by ``number_of_paths``).
    """
    qassert.require(
        product.number_of_products() == 1, "a single product is required"
    )

    number_cash_flows_this_step = [0]
    cash_flows_generated = [
        [
            product.CashFlow()
            for _ in range(product.max_number_of_cash_flows_per_product_per_step())
        ]
    ]

    rate_times = product.evolution().rate_times()
    cash_flow_times = product.possible_cash_flow_times()
    rebate_times = rebate.possible_cash_flow_times()
    control_times = control.possible_cash_flow_times()

    product_discounters = [
        MarketModelDiscounter(t, rate_times) for t in cash_flow_times
    ]
    rebate_discounters = [MarketModelDiscounter(t, rate_times) for t in rebate_times]
    control_discounters = [MarketModelDiscounter(t, rate_times) for t in control_times]

    evolution = product.evolution()
    numeraires = evolver.numeraires()
    evolution_times = evolution.evolution_times()

    is_product_time = is_in_subset(
        evolution_times, product.evolution().evolution_times()
    )
    is_rebate_time = is_in_subset(evolution_times, rebate.evolution().evolution_times())
    is_control_time = is_in_subset(
        evolution_times, control.evolution().evolution_times()
    )
    is_basis_time = is_in_subset(
        evolution_times, data_provider.evolution().evolution_times()
    )
    is_exercise_time = [False] * len(evolution_times)
    v = rebate.is_exercise_time()
    exercises = 0
    idx = 0
    for i in range(len(evolution_times)):
        if is_rebate_time[i]:
            if v[idx]:
                is_exercise_time[i] = True
                exercises += 1
            idx += 1

    collected_data: list[list[NodeData]] = [
        [NodeData() for _ in range(number_of_paths)] for _ in range(exercises + 1)
    ]

    for path in range(number_of_paths):
        evolver.start_new_path()
        product.reset()
        rebate.reset()
        control.reset()
        data_provider.reset()
        principal_in_numeraire_portfolio = 1.0

        done = False
        next_exercise = 0
        collected_data[0][path].cumulated_cash_flows = 0.0
        while not done:
            current_step = evolver.current_step()
            evolver.advance_step()
            current_state = evolver.current_state()
            numeraire = numeraires[current_step]

            if is_rebate_time[current_step]:
                rebate.next_step(current_state)
            if is_control_time[current_step]:
                control.next_step(current_state)
            if is_basis_time[current_step]:
                data_provider.next_step(current_state)

            if is_exercise_time[current_step]:
                data = collected_data[next_exercise + 1][path]

                exercise_value = rebate.value(current_state)
                data.exercise_value = (
                    exercise_value.amount
                    * rebate_discounters[exercise_value.time_index].numeraire_bonds(
                        current_state, numeraire
                    )
                    / principal_in_numeraire_portfolio
                )

                values: list[float] = []
                data_provider.values(current_state, values)
                data.values = values

                control_value = control.value(current_state)
                data.control_value = (
                    control_value.amount
                    * control_discounters[control_value.time_index].numeraire_bonds(
                        current_state, numeraire
                    )
                    / principal_in_numeraire_portfolio
                )

                data.cumulated_cash_flows = 0.0
                data.is_valid = True
                next_exercise += 1

            if is_product_time[current_step]:
                done = product.next_time_step(
                    current_state, number_cash_flows_this_step, cash_flows_generated
                )
                for j in range(number_cash_flows_this_step[0]):
                    cf = cash_flows_generated[0][j]
                    collected_data[next_exercise][path].cumulated_cash_flows += (
                        cf.amount
                        * product_discounters[cf.time_index].numeraire_bonds(
                            current_state, numeraire
                        )
                        / principal_in_numeraire_portfolio
                    )

            if not done:
                next_numeraire = numeraires[current_step + 1]
                principal_in_numeraire_portfolio *= current_state.discount_ratio(
                    numeraire, next_numeraire
                )

        # fill the remaining (un)collected data with nulls
        for j in range(next_exercise, exercises):
            data = collected_data[j + 1][path]
            data.exercise_value = 0.0
            data.control_value = 0.0
            data.cumulated_cash_flows = 0.0
            data.is_valid = False

    return collected_data


def generic_longstaff_schwartz_regression(
    simulation_data: list[list[NodeData]],
) -> tuple[list[list[float]], float]:
    """Back-propagating LS regression of the continuation value.

    # C++ parity: genericlsregression.cpp genericLongstaffSchwartzRegression.

    Returns ``(basis_coefficients, biased_estimate)``: ``basis_coefficients`` has
    ``len(simulation_data) - 1`` rows (one per exercise), and the biased estimate
    is the path-average of the back-propagated cumulated cash flows (the
    in-sample lower-bound price).
    """
    steps = len(simulation_data)
    basis_coefficients: list[list[float]] = [[] for _ in range(steps - 1)]

    for i in range(steps - 1, 0, -1):
        exercise_data = simulation_data[i]

        # 1) covariance matrix of basis-function values + deflated cash flows
        n = len(exercise_data[0].values)
        stats = SequenceStatistics(n + 1)
        for d in exercise_data:
            if d.is_valid:
                temp = [*d.values, d.cumulated_cash_flows - d.control_value]
                stats.add(temp)

        means = stats.mean()
        covariance = np.asarray(stats.covariance(), dtype=np.float64)

        c_mat = np.empty((n, n), dtype=np.float64)
        target = np.empty(n, dtype=np.float64)
        for k in range(n):
            target[k] = covariance[k][n] + means[k] * means[n]
            for ell in range(k + 1):
                value = covariance[k][ell] + means[k] * means[ell]
                c_mat[k][ell] = value
                c_mat[ell][k] = value

        # 2) least-squares regression (SVD solve, as in C++ SVD(C).solveFor)
        alphas = np.linalg.lstsq(c_mat, target, rcond=None)[0]
        basis_coefficients[i - 1] = alphas.tolist()

        # 3) divide paths into exercise / non-exercise domains
        for j, d in enumerate(exercise_data):
            if d.is_valid:
                exercise_value = d.exercise_value
                continuation_value = d.cumulated_cash_flows
                estimated_continuation_value = (
                    float(np.dot(np.asarray(d.values), alphas)) + d.control_value
                )
                chosen = (
                    exercise_value
                    if estimated_continuation_value <= exercise_value
                    else continuation_value
                )
                simulation_data[i - 1][j].cumulated_cash_flows += chosen

    estimate = [d.cumulated_cash_flows for d in simulation_data[0]]
    biased_estimate = sum(estimate) / len(estimate) if estimate else 0.0
    return basis_coefficients, biased_estimate
