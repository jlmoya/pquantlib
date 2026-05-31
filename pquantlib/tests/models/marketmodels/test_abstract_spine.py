"""Tests for the W9-A market-model abstract spine.

Covers the abstract bases (cannot instantiate; minimal concrete subclasses
satisfy the interface) and the concrete ``MarketModel`` covariance /
total-covariance / time-dependent-volatility algebra.

C++ parity:
  ql/models/marketmodels/marketmodel.{hpp,cpp}
  ql/models/marketmodels/evolver.hpp
  ql/models/marketmodels/multiproduct.hpp
  ql/models/marketmodels/pathwisemultiproduct.hpp
  ql/models/marketmodels/browniangenerator.hpp
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.brownian_generator import (
    BrownianGenerator,
    BrownianGeneratorFactory,
)
from pquantlib.models.marketmodels.curve_state import CurveState
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.evolver import MarketModelEvolver
from pquantlib.models.marketmodels.market_model import MarketModel, MarketModelFactory
from pquantlib.models.marketmodels.multi_product import (
    CashFlow,
    MarketModelMultiProduct,
)
from pquantlib.models.marketmodels.pathwise_multi_product import (
    MarketModelPathwiseMultiProduct,
    PathwiseCashFlow,
)
from pquantlib.testing.tolerance import tight


def test_abstract_bases_cannot_instantiate() -> None:
    for cls in (
        MarketModel,
        MarketModelFactory,
        MarketModelEvolver,
        MarketModelMultiProduct,
        MarketModelPathwiseMultiProduct,
        BrownianGenerator,
        BrownianGeneratorFactory,
        CurveState,
    ):
        with pytest.raises(TypeError):
            cls()  # type: ignore[abstract]


# A minimal concrete MarketModel with constant per-step pseudo-roots, so the
# covariance algebra is hand-checkable.
class _ConstPseudoRootModel(MarketModel):
    def __init__(self, evolution: EvolutionDescription, pseudo_roots: list[Matrix]) -> None:
        super().__init__()
        self._evolution = evolution
        self._pseudo_roots = pseudo_roots
        self._n_rates = evolution.number_of_rates()
        self._n_factors = pseudo_roots[0].shape[1]

    def initial_rates(self) -> list[float]:
        return [0.05] * self._n_rates

    def displacements(self) -> list[float]:
        return [0.0] * self._n_rates

    def evolution(self) -> EvolutionDescription:
        return self._evolution

    def number_of_rates(self) -> int:
        return self._n_rates

    def number_of_factors(self) -> int:
        return self._n_factors

    def number_of_steps(self) -> int:
        return self._evolution.number_of_steps()

    def pseudo_root(self, i: int) -> Matrix:
        return self._pseudo_roots[i]


def _model() -> _ConstPseudoRootModel:
    rate_times = [0.5 * (i + 1) for i in range(4)]  # 4 times -> 3 rates, 3 steps
    evolution = EvolutionDescription(rate_times)
    # Each step: a 3x2 pseudo-root with a known A@A.T.
    a = np.array([[0.10, 0.02], [0.03, 0.09], [0.01, 0.07]], dtype=np.float64)
    pseudo_roots = [a.copy(), (0.5 * a).copy(), (0.25 * a).copy()]
    return _ConstPseudoRootModel(evolution, pseudo_roots)


def test_market_model_covariance() -> None:
    m = _model()
    a = m.pseudo_root(0)
    expected = a @ a.T
    cov0 = m.covariance(0)
    for i in range(3):
        for j in range(3):
            tight(float(cov0[i, j]), float(expected[i, j]))


def test_market_model_total_covariance() -> None:
    m = _model()
    # total covariance through step 2 = sum of all three per-step covariances
    expected = m.covariance(0) + m.covariance(1) + m.covariance(2)
    total = m.total_covariance(2)
    for i in range(3):
        for j in range(3):
            tight(float(total[i, j]), float(expected[i, j]))


def test_market_model_time_dependent_volatility() -> None:
    m = _model()
    evolution_times = m.evolution().evolution_times()  # {0.5, 1.0, 1.5}
    # rate 0 variance per step / tau, then sqrt
    last_time = 0.0
    for j in range(m.number_of_steps()):
        tau = evolution_times[j] - last_time
        var = float(m.covariance(j)[0, 0])
        expected = math.sqrt(var / tau)
        tight(m.time_dependent_volatility(0)[j], expected)
        last_time = evolution_times[j]


def test_multi_product_cashflow_dataclass() -> None:
    cf = CashFlow()
    assert cf.time_index == 0
    assert cf.amount == 0.0
    cf.time_index = 2
    cf.amount = 1.5
    assert (cf.time_index, cf.amount) == (2, 1.5)
    # nested alias is the same type
    assert MarketModelMultiProduct.CashFlow is CashFlow


def test_pathwise_cashflow_dataclass() -> None:
    cf = PathwiseCashFlow()
    assert cf.time_index == 0
    assert cf.amount == []
    cf.amount = [1.0, 0.5, 0.25]
    assert cf.amount == [1.0, 0.5, 0.25]
    assert MarketModelPathwiseMultiProduct.CashFlow is PathwiseCashFlow
