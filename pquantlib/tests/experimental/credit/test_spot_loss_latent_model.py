"""Tests for SpotLossLatentModel.

The C++ class implements the Bennani-Maetz / Li (2009) spot recovery
model; the Python port simplifies to the single-factor reduction. Tests
exercise structural invariants because there is no closed-form C++
probe path that exposes the raw arithmetic (the C++ helper is a
template-bound private impl that requires a Basket).

# C++ parity: ql/experimental/credit/spotlosslatentmodel.hpp.
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.one_factor_copula import (
    OneFactorGaussianCopula,
)
from pquantlib.experimental.credit.spot_loss_latent_model import (
    SpotLossLatentModel,
)
from pquantlib.testing import tolerance


def test_spot_loss_latent_model_round_trips() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = SpotLossLatentModel(
        cop,
        rr_mean=[0.40, 0.45],
        rr_loading=[0.30, 0.30],
        model_a=0.10,
    )
    assert m.pool_size() == 2
    assert m.rr_mean() == [0.40, 0.45]
    assert m.rr_loading() == [0.30, 0.30]
    tolerance.tight(m.model_a(), 0.10)
    assert m.copula() is cop


def test_spot_loss_latent_model_rejects_mismatched_sizes() -> None:
    cop = OneFactorGaussianCopula(0.20)
    with pytest.raises(LibraryException, match="sizes differ"):
        SpotLossLatentModel(
            cop, rr_mean=[0.40, 0.45], rr_loading=[0.30], model_a=0.10
        )


def test_spot_loss_latent_model_rejects_invalid_rr_mean() -> None:
    cop = OneFactorGaussianCopula(0.20)
    with pytest.raises(LibraryException, match="rr_mean"):
        SpotLossLatentModel(
            cop, rr_mean=[0.40, 1.20], rr_loading=[0.30, 0.30], model_a=0.10
        )


def test_spot_loss_latent_model_rejects_negative_model_a() -> None:
    cop = OneFactorGaussianCopula(0.20)
    with pytest.raises(LibraryException, match="model_a"):
        SpotLossLatentModel(
            cop, rr_mean=[0.40], rr_loading=[0.30], model_a=-0.10
        )


def test_spot_loss_latent_model_conditional_default_matches_copula() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = SpotLossLatentModel(
        cop, rr_mean=[0.40], rr_loading=[0.30], model_a=0.10
    )
    tolerance.tight(
        m.conditional_default_probability(0.10, 0.5),
        cop.conditional_probability(0.10, 0.5),
    )


def test_spot_loss_latent_model_zero_prob_returns_rr_mean() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = SpotLossLatentModel(
        cop, rr_mean=[0.40], rr_loading=[0.30], model_a=0.10
    )
    tolerance.tight(m.exp_conditional_recovery(0, 1e-15, 0.5), 0.40)


def test_spot_loss_latent_model_conditional_recovery_in_range() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = SpotLossLatentModel(
        cop, rr_mean=[0.40], rr_loading=[0.30], model_a=0.10
    )
    # Sample across a few factor values — should always be a probability.
    for mm in (-2.0, -1.0, 0.0, 1.0, 2.0):
        rr = m.exp_conditional_recovery(0, 0.10, mm)
        assert 0.0 <= rr <= 1.0


def test_spot_loss_latent_model_expected_loss_zero_prob() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = SpotLossLatentModel(
        cop, rr_mean=[0.40], rr_loading=[0.30], model_a=0.10
    )
    tolerance.tight(m.expected_loss(0, 1e-15), 0.0)


def test_spot_loss_latent_model_expected_loss_in_range() -> None:
    """Expected loss is between 0 and probability of default."""
    cop = OneFactorGaussianCopula(0.20)
    m = SpotLossLatentModel(
        cop, rr_mean=[0.40], rr_loading=[0.30], model_a=0.10
    )
    expected = m.expected_loss(0, 0.05)
    assert 0.0 <= expected <= 0.05  # less than PD * 1


def test_spot_loss_latent_model_rejects_out_of_range_name() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = SpotLossLatentModel(
        cop, rr_mean=[0.40], rr_loading=[0.30], model_a=0.10
    )
    with pytest.raises(LibraryException, match="i_name"):
        m.exp_conditional_recovery(5, 0.10, 0.5)
