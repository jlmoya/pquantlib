"""Tests for the W9-C accounting / greek engines + constrained evolver.

Cross-validates ``AccountingEngine`` + ``ProxyGreekEngine`` against
``migration-harness/references/cluster/w9c.json`` using Python stub evolvers /
products that mirror the C++ probe's deterministic flat 1-step LMM world.

``PathwiseAccountingEngine`` is validated against a self-consistent
degenerate-path analytic reference (its C++ ctor needs a concrete
LogNormalFwdRateEuler from W10, so it cannot be probed yet).

C++ parity:
  ql/models/marketmodels/accountingengine.{hpp,cpp}
  ql/models/marketmodels/proxygreekengine.{hpp,cpp}
  ql/models/marketmodels/pathwiseaccountingengine.{hpp,cpp}
  ql/models/marketmodels/constrainedevolver.hpp
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.statistics.incremental_statistics import IncrementalStatistics
from pquantlib.models.marketmodels.accounting_engine import AccountingEngine
from pquantlib.models.marketmodels.constrained_evolver import ConstrainedEvolver
from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.curvestates.lmm_curve_state import LMMCurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.evolver import MarketModelEvolver
from pquantlib.models.marketmodels.market_model import MarketModel
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.pathwise_accounting_engine import (
    PathwiseAccountingEngine,
)
from pquantlib.models.marketmodels.pathwise_discounter import (
    MarketModelPathwiseDiscounter,
)
from pquantlib.models.marketmodels.pathwise_multi_product import (
    MarketModelPathwiseMultiProduct,
    PathwiseCashFlow,
)
from pquantlib.models.marketmodels.proxy_greek_engine import ProxyGreekEngine
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight

# --- constants matching the C++ probe world ---------------------------------
_FORWARD = 0.05
_NUMERAIRE = 3  # terminal bond (rateTimes index 3)
_INIT_NUMERAIRE = 100.0
_CASH_AMOUNT = 1.0


def _rate_times4() -> list[float]:
    return [0.5, 1.0, 1.5, 2.0]


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w9c")


# --- sequence-statistics shim (stands in for C++ SequenceStatisticsInc) ------
class _SeqStats:
    def __init__(self, dimension: int) -> None:
        self._stats = [IncrementalStatistics() for _ in range(dimension)]

    def add(self, values: list[float], weight: float = 1.0) -> None:
        for i, v in enumerate(values):
            self._stats[i].add(v, weight)

    def mean(self) -> list[float]:
        return [s.mean() for s in self._stats]

    def samples(self) -> int:
        return self._stats[0].samples()


# --- Python stubs mirroring the C++ probe -----------------------------------
class _StubEvolver(MarketModelEvolver):
    def __init__(self) -> None:
        self._numeraires = [_NUMERAIRE]
        self._state = LMMCurveState(_rate_times4())
        self._state.set_on_forward_rates([_FORWARD, _FORWARD, _FORWARD])
        self._step = 0

    def numeraires(self) -> list[int]:
        return self._numeraires

    def start_new_path(self) -> float:
        self._step = 0
        return 1.0

    def advance_step(self) -> float:
        self._step += 1
        return 1.0

    def current_step(self) -> int:
        return self._step

    def current_state(self) -> CurveState:
        return self._state

    def set_initial_state(self, curve_state: CurveState) -> None:
        pass


class _StubProduct(MarketModelMultiProduct):
    def __init__(self) -> None:
        self._evolution = EvolutionDescription(_rate_times4())
        self._done = False

    def suggested_numeraires(self) -> list[int]:
        return [_NUMERAIRE]

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def possible_cash_flow_times(self) -> list[float]:
        return [1.5]

    def number_of_products(self) -> int:
        return 1

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 1

    def reset(self) -> None:
        self._done = False

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[CashFlow]],
    ) -> bool:
        if not self._done:
            number_cash_flows_this_step[0] = 1
            cash_flows_generated[0][0].time_index = 0
            cash_flows_generated[0][0].amount = _CASH_AMOUNT
            self._done = True
            return True
        number_cash_flows_this_step[0] = 0
        return True

    def clone(self) -> MarketModelMultiProduct:
        return _StubProduct()


def test_accounting_engine(ref: dict[str, Any]) -> None:
    evolver = _StubEvolver()
    product = _StubProduct()
    engine = AccountingEngine(evolver, product, _INIT_NUMERAIRE)
    stats = _SeqStats(1)
    engine.multiple_path_values(stats, 4)
    # Deterministic degenerate path -> TIGHT vs the C++ accumulation.
    tight(stats.mean()[0], ref["acc_mean"])
    assert stats.samples() == int(ref["acc_samples"])


def test_accounting_engine_value_breakdown() -> None:
    # The single cash flow of 1.0 at t=1.5 is rebased to the terminal
    # numeraire (index 3): numeraireBonds = P(1.5)/P(2.0) = 1 + f*tau =
    # 1 + 0.05*0.5 = 1.025, scaled by initialNumeraire 100 -> 102.5.
    evolver = _StubEvolver()
    product = _StubProduct()
    engine = AccountingEngine(evolver, product, _INIT_NUMERAIRE)
    stats = _SeqStats(1)
    engine.multiple_path_values(stats, 1)
    tight(stats.mean()[0], 102.5)


# --- ProxyGreekEngine: unconstrained leg ------------------------------------
def test_proxy_greek_engine_unconstrained(ref: dict[str, Any]) -> None:
    evolver = _StubEvolver()
    product = _StubProduct()
    # No constrained evolvers / diff weights -> the original-evolver leg only.
    engine = ProxyGreekEngine(
        evolver,
        [],  # constrained_evolvers
        [],  # diff_weights
        [0, 0, 0],  # start_index_of_constraint (per evolution step)
        [3, 3, 3],  # end_index_of_constraint
        product,
        _INIT_NUMERAIRE,
    )
    values = [0.0]
    modified: list[list[list[float]]] = []
    engine.single_path_values(values, modified)
    tight(values[0], ref["proxy_value"])


def test_proxy_greek_engine_multiple_path_values() -> None:
    # With no diff weights, multiple_path_values just accumulates the base leg.
    evolver = _StubEvolver()
    product = _StubProduct()
    engine = ProxyGreekEngine(
        evolver, [], [], [0, 0, 0], [3, 3, 3], product, _INIT_NUMERAIRE
    )
    stats = _SeqStats(1)
    engine.multiple_path_values(stats, [], 3)
    tight(stats.mean()[0], 102.5)


# --- ConstrainedEvolver is abstract -----------------------------------------
def test_constrained_evolver_is_abstract() -> None:
    assert issubclass(ConstrainedEvolver, MarketModelEvolver)
    with pytest.raises(TypeError):
        ConstrainedEvolver()  # type: ignore[abstract]


# --- PathwiseAccountingEngine: self-consistent degenerate-path reference -----
class _StubPathwiseProduct(MarketModelPathwiseMultiProduct):
    """1-rate world, single unit cash flow at t=1.0 with zero derivatives."""

    def __init__(self) -> None:
        self._evolution = EvolutionDescription([0.5, 1.0])  # 1 rate
        self._done = False

    def suggested_numeraires(self) -> list[int]:
        return [1]

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def possible_cash_flow_times(self) -> list[float]:
        return [1.0]

    def number_of_products(self) -> int:
        return 1

    def max_number_of_cash_flows_per_product_per_step(self) -> int:
        return 1

    def already_deflated(self) -> bool:
        return False

    def reset(self) -> None:
        self._done = False

    def next_time_step(
        self,
        current_state: CurveState,
        number_cash_flows_this_step: list[int],
        cash_flows_generated: list[list[PathwiseCashFlow]],
    ) -> bool:
        if not self._done:
            number_cash_flows_this_step[0] = 1
            cash_flows_generated[0][0].time_index = 0
            cash_flows_generated[0][0].amount[0] = _CASH_AMOUNT
            cash_flows_generated[0][0].amount[1] = 0.0
            self._done = True
            return True
        number_cash_flows_this_step[0] = 0
        return True

    def clone(self) -> MarketModelPathwiseMultiProduct:
        return _StubPathwiseProduct()


class _StubEvolver1Rate(MarketModelEvolver):
    def __init__(self) -> None:
        self._state = LMMCurveState([0.5, 1.0])
        self._state.set_on_forward_rates([_FORWARD])
        self._step = 0

    def numeraires(self) -> list[int]:
        return [1]

    def start_new_path(self) -> float:
        self._step = 0
        return 1.0

    def advance_step(self) -> float:
        self._step += 1
        return 1.0

    def current_step(self) -> int:
        return self._step

    def current_state(self) -> CurveState:
        return self._state

    def set_initial_state(self, curve_state: CurveState) -> None:
        pass


class _StubMarketModel1Rate(MarketModel):
    def __init__(self) -> None:
        super().__init__()
        self._evolution = EvolutionDescription([0.5, 1.0])
        self._pseudo = np.array([[0.10]], dtype=np.float64)

    def initial_rates(self) -> list[float]:
        return [_FORWARD]

    def displacements(self) -> list[float]:
        return [0.0]

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def number_of_rates(self) -> int:
        return 1

    def number_of_factors(self) -> int:
        return 1

    def number_of_steps(self) -> int:
        return 1

    def pseudo_root(self, i: int) -> np.ndarray:
        return self._pseudo


def test_pathwise_accounting_engine_degenerate() -> None:
    # 1-rate world, single unit cash flow at the rate time t=1.0, money-market
    # (numeraire bond) measure, flat forward, zero pay-off derivative. We
    # independently reconstruct the engine's full Giles-Glasserman recursion
    # for this 1-step path and check the value + the V[0] sensitivity row
    # match (the C++ engine cannot be probed yet — it needs a concrete
    # LogNormalFwdRateEuler from W10 — so this is a self-consistent analytic
    # cross-check of the ported recursion).
    engine = PathwiseAccountingEngine(
        _StubEvolver1Rate(), _StubPathwiseProduct(), _StubMarketModel1Rate(), 1.0
    )
    stats = _SeqStats(2)
    engine.multiple_path_values(stats, 1)
    means = stats.mean()

    tau = 0.5
    p1 = 1.0 / (1.0 + _FORWARD * tau)  # P(t_0, t_1) on the simulation account
    pseudo = 0.10

    # 1) deflator + its forward-derivative from the pathwise discounter.
    disc = MarketModelPathwiseDiscounter(1.0, [0.5, 1.0])
    discounts = np.array([[1.0, 1.0], [1.0, p1]], dtype=np.float64)
    libor = np.array([[0.0], [_FORWARD]], dtype=np.float64)
    factors = [0.0, 0.0]
    disc.get_factors(libor, discounts, 1, factors)
    deflator, deflator_deriv = factors[0], factors[1]

    # value = amount * deflator * initialNumeraire (initialNumeraire = 1).
    tight(means[0], _CASH_AMOUNT * deflator)

    # 2) replicate the engine's backward sweep for the single step.
    #    V[1][0] = amount*deflator_deriv (pay-off deriv 0); step_to_use=1.
    v_step1 = _CASH_AMOUNT * deflator_deriv
    #    propagate V[1] -> V[0]: liborRatio[1][0] = f/f = 1.
    libor_ratio = 1.0
    steps_disc_sq = p1 * p1  # discountRatio(1,0)^2
    #    partials[0][0] = libor * V[1][0] * pseudo
    partial = _FORWARD * v_step1 * pseudo
    summand = pseudo * partial * tau * steps_disc_sq
    v_step0 = v_step1 * libor_ratio + summand

    tight(means[1], v_step0)


def test_pathwise_accounting_engine_value_positive() -> None:
    engine = PathwiseAccountingEngine(
        _StubEvolver1Rate(), _StubPathwiseProduct(), _StubMarketModel1Rate(), 1.0
    )
    stats = _SeqStats(2)
    engine.multiple_path_values(stats, 2)
    means = stats.mean()
    # discounted unit cash flow < 1, derivative negative (df decreasing in f).
    assert 0.9 < means[0] < 1.0
    assert means[1] < 0.0
