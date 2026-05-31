"""Shared fixtures for the W11-D pathwise-greeks tests.

Reproduces the deterministic FlatVol market model built by
``migration-harness/cpp/probes/cluster_w11d/probe.cpp``: 6 rates, 3 factors,
semiannual rate times, exponential forward correlation (longTermCorr=0.5,
beta=0.2), forwards 0.03+0.002*i, vols 0.12+0.005*i, displacements 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.matrix import Matrix
from pquantlib.models.marketmodels.correlations.exp_correlations import (
    ExponentialForwardCorrelation,
)
from pquantlib.models.marketmodels.evolution_description import EvolutionDescription
from pquantlib.models.marketmodels.market_model import MarketModel
from pquantlib.models.marketmodels.models.flat_vol import FlatVol
from pquantlib.models.marketmodels.models.pseudo_root_facade import PseudoRootFacade
from pquantlib.testing.reference_reader import load as load_reference

N_RATES = 6
N_FACTORS = 3


def _setup_data() -> tuple[list[float], list[float], list[float], list[float]]:
    n = N_RATES
    rate_times = [0.5 * (i + 1) for i in range(n + 1)]
    forwards = [0.03 + 0.002 * i for i in range(n)]
    displacements = [0.0] * n
    volatilities = [0.12 + 0.005 * i for i in range(n)]
    return rate_times, forwards, displacements, volatilities


def make_setup() -> tuple[MarketModel, EvolutionDescription]:
    """Build the deterministic FlatVol model matching the probe."""
    rate_times, forwards, displacements, volatilities = _setup_data()
    evolution = EvolutionDescription(rate_times)
    corr = ExponentialForwardCorrelation(rate_times, 0.5, 0.2)
    model = FlatVol(volatilities, corr, evolution, N_FACTORS, forwards, displacements)
    return model, evolution


def make_facade_setup(ref_pseudo_roots: list[float]) -> MarketModel:
    """Build a ``PseudoRootFacade`` from the C++-emitted pseudo-roots.

    The BGM pseudo-root is unique only up to a per-column sign (spectral
    eigenvector ambiguity): C++'s Jacobi ``SymmetricSchurDecomposition`` and
    numpy's LAPACK ``eigh`` pin different signs for the eigenvectors of the
    dead-rate-truncated covariance matrices, even though ``B @ B.T`` is
    identical (see ``align(pseudo_sqrt)`` W10-B). Feeding the *exact* C++
    pseudo-roots through a facade pins the signs to C++'s, so the
    sign-sensitive (linear-in-pseudo-root) volatility/variance derivatives
    cross-validate TIGHT with sign — isolating the test to the derivative
    algorithm itself rather than the arbitrary spectral sign.
    """
    rate_times, forwards, displacements, _ = _setup_data()
    prs = reshape_stack(ref_pseudo_roots, N_RATES, N_RATES, N_FACTORS)
    return PseudoRootFacade(prs, rate_times, forwards, displacements)


def reshape(flat: list[float], rows: int, cols: int) -> Matrix:
    """Reshape a row-major flattened list into a (rows, cols) Matrix."""
    return np.asarray(flat, dtype=np.float64).reshape(rows, cols)


def reshape_stack(flat: list[float], count: int, rows: int, cols: int) -> list[Matrix]:
    """Reshape a concatenation of ``count`` row-major (rows,cols) Matrices."""
    arr = np.asarray(flat, dtype=np.float64).reshape(count, rows, cols)
    return [arr[i] for i in range(count)]


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w11d")
