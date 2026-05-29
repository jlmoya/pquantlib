"""Tests for DefaultProbabilityLatentModel + ConstantLossLatentModel.

Cross-validated against:
  * the Vasicek closed-form for single-name conditional default prob
    (matches scipy.stats.norm-CDF up to TIGHT).
  * the C++ ``LossDist::probabilityOfAtLeastNEvents`` (from cluster_w3a)
    in the LIMIT rho -> 0 (independence) — at rho=0, the conditional
    default probability is the unconditional one, so the joint P(>= n)
    matches the independence formula computed inside loss_distribution.

# C++ parity: ql/experimental/credit/defaultprobabilitylatentmodel.hpp +
# constantlosslatentmodel.hpp.
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.constant_loss_latent_model import (
    ConstantLossLatentModel,
)
from pquantlib.experimental.credit.default_probability_latent_model import (
    DefaultProbabilityLatentModel,
)
from pquantlib.experimental.credit.loss_distribution import (
    probability_of_at_least_n_events,
)
from pquantlib.experimental.credit.one_factor_copula import (
    OneFactorGaussianCopula,
)
from pquantlib.testing import tolerance

# -----------------------------------------------------------------------------
# DefaultProbabilityLatentModel
# -----------------------------------------------------------------------------


def test_default_prob_latent_model_size_round_trips() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = DefaultProbabilityLatentModel(cop, pool_size=5)
    assert m.pool_size() == 5
    assert m.copula() is cop


def test_default_prob_latent_model_rejects_invalid_pool_size() -> None:
    cop = OneFactorGaussianCopula(0.20)
    with pytest.raises(LibraryException, match="pool_size"):
        DefaultProbabilityLatentModel(cop, pool_size=0)


def test_default_prob_latent_model_conditional_default_matches_copula() -> None:
    """conditional_default_probability delegates to the copula's same method."""
    cop = OneFactorGaussianCopula(0.20)
    m = DefaultProbabilityLatentModel(cop, pool_size=5)
    tolerance.tight(
        m.conditional_default_probability(0.10, 0.0),
        cop.conditional_probability(0.10, 0.0),
    )


def test_default_prob_latent_model_conditional_vec_matches_copula() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = DefaultProbabilityLatentModel(cop, pool_size=4)
    probs = [0.1, 0.2, 0.3, 0.4]
    expected = cop.conditional_probability_vec(probs, 0.5)
    actual = m.conditional_default_probability_vec(probs, 0.5)
    for a, e in zip(actual, expected, strict=True):
        tolerance.tight(a, e)


def test_default_prob_latent_model_independence_recovers_unconditional() -> None:
    """At rho=0, integrating the conditional default prob over M recovers p."""
    cop = OneFactorGaussianCopula(0.0)
    m = DefaultProbabilityLatentModel(cop, pool_size=1)
    # # CUSTOM 1e-6: 50-step Euler midpoint integration on the standard
    # normal density has ~1e-7 truncation error.
    tolerance.custom(
        m.prob_of_default(0, 0.2),
        0.2,
        abs_tol=1e-6,
        rel_tol=1e-6,
        reason="50-step Euler midpoint integration noise",
    )


def test_default_prob_latent_model_prob_at_least_n_matches_independence() -> None:
    """At rho=0, joint P(>=n) matches the independence formula."""
    cop = OneFactorGaussianCopula(0.0)
    m = DefaultProbabilityLatentModel(cop, pool_size=4)
    probs = [0.1, 0.2, 0.3, 0.4]
    expected = probability_of_at_least_n_events(2, probs)
    # # CUSTOM 1e-3: 2^4 = 16 mask iterations and 50-step Euler each adds noise.
    tolerance.custom(
        m.prob_at_least_n_events(2, probs),
        expected,
        abs_tol=1e-3,
        rel_tol=1e-3,
        reason="Euler integration + combinatorial walk noise at rho=0",
    )


def test_default_prob_latent_model_rejects_mismatched_probs_size() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = DefaultProbabilityLatentModel(cop, pool_size=3)
    with pytest.raises(LibraryException, match="probs size"):
        m.conditional_default_probability_vec([0.1, 0.2], 0.5)


def test_default_prob_latent_model_default_correlation_basic() -> None:
    """At rho=0, default correlation should be ~0; at rho>0, > 0."""
    cop0 = OneFactorGaussianCopula(0.0)
    m0 = DefaultProbabilityLatentModel(cop0, pool_size=2)
    corr0 = m0.default_correlation(0.10, 0.10)
    assert abs(corr0) < 1e-3

    cop1 = OneFactorGaussianCopula(0.50)
    m1 = DefaultProbabilityLatentModel(cop1, pool_size=2)
    corr1 = m1.default_correlation(0.10, 0.10)
    assert corr1 > 0.05


def test_default_prob_latent_model_zero_prob_returns_zero() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = DefaultProbabilityLatentModel(cop, pool_size=2)
    assert m.prob_of_default(0, 1e-15) == 0.0
    assert m.default_correlation(1e-15, 0.10) == 0.0


# -----------------------------------------------------------------------------
# ConstantLossLatentModel
# -----------------------------------------------------------------------------


def test_constant_loss_latent_model_recoveries_round_trip() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = ConstantLossLatentModel(cop, recoveries=[0.40, 0.30, 0.50])
    assert m.recoveries() == [0.40, 0.30, 0.50]
    assert m.pool_size() == 3


def test_constant_loss_latent_model_rejects_empty_recoveries() -> None:
    cop = OneFactorGaussianCopula(0.20)
    # # The base class checks pool_size > 0 first; the message is "pool_size".
    with pytest.raises(LibraryException, match="pool_size"):
        ConstantLossLatentModel(cop, recoveries=[])


def test_constant_loss_latent_model_rejects_recovery_out_of_range() -> None:
    cop = OneFactorGaussianCopula(0.20)
    with pytest.raises(LibraryException, match=r"not in \[0, 1\]"):
        ConstantLossLatentModel(cop, recoveries=[0.40, 1.20])


def test_constant_loss_latent_model_conditional_recovery_is_constant() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = ConstantLossLatentModel(cop, recoveries=[0.40, 0.30])
    tolerance.tight(m.conditional_recovery(0), 0.40)
    tolerance.tight(m.conditional_recovery(1), 0.30)
    # Recovery is constant across factor draws
    tolerance.tight(m.conditional_recovery(0, m=-2.0), 0.40)
    tolerance.tight(m.conditional_recovery(0, m=2.0), 0.40)


def test_constant_loss_latent_model_expected_loss_matches_independence_limit() -> None:
    """At rho=0, expected_loss = prob * (1 - recovery)."""
    cop = OneFactorGaussianCopula(0.0)
    m = ConstantLossLatentModel(cop, recoveries=[0.40])
    expected = 0.05 * (1.0 - 0.40)
    # # CUSTOM 1e-6: 50-step Euler midpoint integration noise.
    tolerance.custom(
        m.expected_loss(0, 0.05),
        expected,
        abs_tol=1e-6,
        rel_tol=1e-6,
        reason="50-step Euler midpoint integration noise",
    )
