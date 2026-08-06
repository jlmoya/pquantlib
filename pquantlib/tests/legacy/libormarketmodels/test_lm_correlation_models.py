"""Cross-validation of the LIBOR-market-model correlation models against C++ v1.43.

Covers ``LmCorrelationModel``, ``LmExponentialCorrelationModel``,
``LmLinearExponentialCorrelationModel`` and ``LmConstWrapperCorrelationModel``.

Expected values come from ``migration-harness/references/v143/legacy/lmm.json``
(``migration-harness/cpp/probes/v143_legacy_lmm/probe.cpp``). None of these
classes reads a date, so this module needs no ``Settings`` pinning.

``pseudo_sqrt`` is checked ENTRY BY ENTRY, not through ``B @ B.T``: the
eigenvector sign convention (first component non-negative) is what makes the
Python spectral root match C++'s Jacobi one, and ``B @ B.T`` is blind to a
sign flip while ``LfmCovarianceProxy.diffusion`` is not.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pytest

from pquantlib.legacy.libormarketmodels.lm_const_wrapper_corr_model import (
    LmConstWrapperCorrelationModel,
)
from pquantlib.legacy.libormarketmodels.lm_exp_corr_model import (
    LmExponentialCorrelationModel,
)
from pquantlib.legacy.libormarketmodels.lm_lin_exp_corr_model import (
    LmLinearExponentialCorrelationModel,
)
from pquantlib.math.optimization.constraint import (
    BoundaryConstraint,
    PositiveConstraint,
)
from pquantlib.models.parameter import ConstantParameter
from pquantlib.testing import reference_reader
from pquantlib.testing.tolerance import tight

EXP_SIZE: Final[int] = 7
EXP_RHO: Final[float] = 0.13
LINEXP_SIZE: Final[int] = 8
LINEXP_RHO: Final[float] = 0.42
LINEXP_BETA: Final[float] = 0.73
LINEXP_FACTORS: Final[int] = 3


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/legacy/lmm")


def _assert_close(actual: Any, expected: Any, label: str) -> None:
    act = np.asarray(actual, dtype=np.float64)
    exp = np.asarray(expected, dtype=np.float64)
    assert act.shape == exp.shape, f"{label}: shape {act.shape} != {exp.shape}"
    for idx, (a, e) in enumerate(zip(act.ravel(), exp.ravel(), strict=True)):
        tight(float(a), float(e), reason=f"{label}[{idx}]")


# --- LmExponentialCorrelationModel -------------------------------------------


def test_exponential_correlation_inspectors_and_matrices(cpp: dict[str, Any]) -> None:
    ref = cpp["lm_exponential_correlation"]
    model = LmExponentialCorrelationModel(EXP_SIZE, EXP_RHO)

    assert model.size() == ref["size"] == EXP_SIZE
    # base-class factors(): full rank
    assert model.factors() == ref["factors"] == EXP_SIZE
    assert model.is_time_independent() is ref["is_time_independent"] is True
    _assert_close([p(0.0) for p in model.params()], ref["params"], "expcorr.params")

    _assert_close(model.correlation(0.0), ref["correlation_t0"], "expcorr.correlation")
    # time independence is a claim about the VALUES, not just the flag
    _assert_close(model.correlation(3.7), ref["correlation_t3_7"], "expcorr.correlation_t")
    _assert_close(model.pseudo_sqrt(0.0), ref["pseudo_sqrt_t0"], "expcorr.pseudo_sqrt")


def test_exponential_correlation_scalar_matches_the_matrix(cpp: dict[str, Any]) -> None:
    ref = cpp["lm_exponential_correlation"]
    model = LmExponentialCorrelationModel(EXP_SIZE, EXP_RHO)
    got = [
        [model.correlation_scalar(i, j, 1.9) for j in range(EXP_SIZE)]
        for i in range(EXP_SIZE)
    ]
    _assert_close(got, ref["correlation_scalar_t1_9"], "expcorr.correlation_scalar")


def test_exponential_correlation_set_params_regenerates_both_caches(
    cpp: dict[str, Any],
) -> None:
    """A port whose set_params forgot generate_arguments would keep the old cache."""
    ref = cpp["lm_exponential_correlation"]
    model = LmExponentialCorrelationModel(EXP_SIZE, EXP_RHO)
    model.set_params([ConstantParameter(0.47, PositiveConstraint())])

    _assert_close(
        [p(0.0) for p in model.params()], ref["params_after_set"], "expcorr.params_after"
    )
    _assert_close(
        model.correlation(0.0), ref["correlation_after_set"], "expcorr.correlation_after"
    )
    _assert_close(
        model.pseudo_sqrt(0.0), ref["pseudo_sqrt_after_set"], "expcorr.pseudo_sqrt_after"
    )


def test_exponential_correlation_pseudo_sqrt_reconstructs_the_matrix() -> None:
    """C++ test-suite testSimpleCovarianceModels, tolerance 1e-14."""
    model = LmExponentialCorrelationModel(10, 0.1)
    b = model.pseudo_sqrt(0.0)
    recon = model.correlation(0.0) - b @ b.T
    assert np.max(np.abs(recon)) <= 1e-14


# --- LmLinearExponentialCorrelationModel -------------------------------------


def test_linexp_correlation_honours_an_explicit_factor_count(cpp: dict[str, Any]) -> None:
    """``factors`` is not just an inspector: it reshapes pseudoSqrt AND, through
    ``corrMatrix_ = B B^T``, the correlation matrix itself.
    """
    ref = cpp["lm_linexp_correlation_3factors"]
    model = LmLinearExponentialCorrelationModel(
        LINEXP_SIZE, LINEXP_RHO, LINEXP_BETA, LINEXP_FACTORS
    )

    assert model.size() == ref["size"] == LINEXP_SIZE
    assert model.factors() == ref["factors"] == LINEXP_FACTORS
    assert model.is_time_independent() is ref["is_time_independent"] is True
    _assert_close([p(0.0) for p in model.params()], ref["params"], "linexpcorr.params")

    assert model.pseudo_sqrt(0.0).shape == (LINEXP_SIZE, LINEXP_FACTORS)
    _assert_close(model.correlation(0.0), ref["correlation_t0"], "linexpcorr.correlation")
    _assert_close(model.correlation(2.2), ref["correlation_t2_2"], "linexpcorr.correlation_t")
    _assert_close(model.pseudo_sqrt(0.0), ref["pseudo_sqrt_t0"], "linexpcorr.pseudo_sqrt")
    got = [
        [model.correlation_scalar(i, j, 0.4) for j in range(LINEXP_SIZE)]
        for i in range(LINEXP_SIZE)
    ]
    _assert_close(got, ref["correlation_scalar_t0_4"], "linexpcorr.correlation_scalar")


def test_linexp_correlation_rank_reduced_matrix_is_not_the_analytic_formula(
    cpp: dict[str, Any],
) -> None:
    """Guards the ``corrMatrix_ = pseudoSqrt * transpose(pseudoSqrt)`` step.

    With three factors out of eight, the stored matrix must DIFFER from
    ``rho + (1 - rho) exp(-beta |i - j|)`` — otherwise the rank reduction was
    dropped.
    """
    model = LmLinearExponentialCorrelationModel(
        LINEXP_SIZE, LINEXP_RHO, LINEXP_BETA, LINEXP_FACTORS
    )
    analytic = np.array(
        [
            [
                LINEXP_RHO + (1 - LINEXP_RHO) * np.exp(-LINEXP_BETA * abs(i - j))
                for j in range(LINEXP_SIZE)
            ]
            for i in range(LINEXP_SIZE)
        ]
    )
    assert not np.allclose(model.correlation(0.0), analytic, atol=1e-6)
    # ... but the diagonal is pinned exactly by normalizePseudoRoot
    assert np.allclose(np.diag(model.correlation(0.0)), 1.0, atol=1e-14)


def test_linexp_correlation_default_factors_is_full_rank(cpp: dict[str, Any]) -> None:
    ref = cpp["lm_linexp_correlation_default_factors"]
    model = LmLinearExponentialCorrelationModel(LINEXP_SIZE, LINEXP_RHO, LINEXP_BETA)

    assert model.factors() == ref["factors"] == LINEXP_SIZE
    _assert_close(model.correlation(0.0), ref["correlation_t0"], "linexpcorrD.correlation")
    _assert_close(model.pseudo_sqrt(0.0), ref["pseudo_sqrt_t0"], "linexpcorrD.pseudo_sqrt")


def test_linexp_correlation_set_params_regenerates_both_caches(
    cpp: dict[str, Any],
) -> None:
    ref = cpp["lm_linexp_correlation_3factors"]
    model = LmLinearExponentialCorrelationModel(
        LINEXP_SIZE, LINEXP_RHO, LINEXP_BETA, LINEXP_FACTORS
    )
    # rho is NEGATIVE here — only reachable through the BoundaryConstraint(-1, 1)
    model.set_params(
        [
            ConstantParameter(-0.15, BoundaryConstraint(-1.0, 1.0)),
            ConstantParameter(1.35, PositiveConstraint()),
        ]
    )
    _assert_close(
        [p(0.0) for p in model.params()], ref["params_after_set"], "linexpcorr.params_after"
    )
    _assert_close(
        model.correlation(0.0),
        ref["correlation_after_set"],
        "linexpcorr.correlation_after",
    )
    _assert_close(
        model.pseudo_sqrt(0.0),
        ref["pseudo_sqrt_after_set"],
        "linexpcorr.pseudo_sqrt_after",
    )


# --- LmConstWrapperCorrelationModel ------------------------------------------


def test_const_wrapper_correlation_forwards_everything_but_the_params(
    cpp: dict[str, Any],
) -> None:
    ref = cpp["lm_const_wrapper_correlation"]
    inner = LmExponentialCorrelationModel(EXP_SIZE, EXP_RHO)
    wrapper = LmConstWrapperCorrelationModel(inner)

    assert wrapper.size() == ref["size"] == EXP_SIZE
    assert wrapper.factors() == ref["factors"] == inner.factors()
    assert len(wrapper.params()) == ref["n_params"] == 0
    assert wrapper.is_time_independent() is ref["is_time_independent"] is True
    assert wrapper.correlation_model() is inner

    _assert_close(wrapper.correlation(0.0), ref["correlation_t0"], "wrapcorr.correlation")
    _assert_close(wrapper.pseudo_sqrt(0.0), ref["pseudo_sqrt_t0"], "wrapcorr.pseudo_sqrt")
    tight(
        wrapper.correlation_scalar(2, 5, 1.1),
        ref["correlation_scalar_2_5_t1_1"],
        reason="wrapcorr.correlation_scalar",
    )


def test_const_wrapper_correlation_tracks_a_factor_reduced_model() -> None:
    """``factors()`` must come from the wrapped model, not the base default."""
    inner = LmLinearExponentialCorrelationModel(
        LINEXP_SIZE, LINEXP_RHO, LINEXP_BETA, LINEXP_FACTORS
    )
    wrapper = LmConstWrapperCorrelationModel(inner)
    assert wrapper.size() == LINEXP_SIZE
    assert wrapper.factors() == LINEXP_FACTORS
    assert wrapper.pseudo_sqrt(0.0).shape == (LINEXP_SIZE, LINEXP_FACTORS)


def test_const_wrapper_correlation_set_params_cannot_move_the_wrapped_model() -> None:
    inner = LmExponentialCorrelationModel(EXP_SIZE, EXP_RHO)
    wrapper = LmConstWrapperCorrelationModel(inner)
    before = wrapper.correlation(0.0)
    wrapper.set_params([])
    assert np.array_equal(wrapper.correlation(0.0), before)
    assert [p(0.0) for p in inner.params()] == [EXP_RHO]
