"""Tests for RandomDefaultLatentModel + RandomLossLatentModel.

Cross-validation: at rho=0 (independence) the MC default-count
distribution should approach the binomial(prob_uniform, n) distribution;
at high rho the default count distribution concentrates at 0 and n. We
exercise both regimes with enough MC paths to keep the test stable.

# C++ parity: ql/experimental/credit/randomdefaultlatentmodel.hpp +
# randomlosslatentmodel.hpp.
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.one_factor_copula import (
    OneFactorGaussianCopula,
)
from pquantlib.experimental.credit.random_default_latent_model import (
    RandomDefaultLatentModel,
)
from pquantlib.experimental.credit.random_loss_latent_model import (
    RandomLossLatentModel,
    constant_recovery_draw,
)
from pquantlib.testing import tolerance

# -----------------------------------------------------------------------------
# RandomDefaultLatentModel
# -----------------------------------------------------------------------------


def test_random_default_latent_model_round_trips() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = RandomDefaultLatentModel(cop, pool_size=5, seed=123)
    assert m.pool_size() == 5
    assert m.copula() is cop


def test_random_default_latent_model_rejects_invalid_pool_size() -> None:
    cop = OneFactorGaussianCopula(0.20)
    with pytest.raises(LibraryException, match="pool_size"):
        RandomDefaultLatentModel(cop, pool_size=0)


def test_random_default_latent_model_rejects_mismatched_probs() -> None:
    cop = OneFactorGaussianCopula(0.20)
    m = RandomDefaultLatentModel(cop, pool_size=3)
    with pytest.raises(LibraryException, match="probs size"):
        m.simulate_default_counts(probs=[0.1, 0.2], n_paths=10)


def test_random_default_latent_model_independence_matches_binomial_mean() -> None:
    """At rho=0 with uniform pd, expected default count = N * p."""
    cop = OneFactorGaussianCopula(0.0)
    n = 20
    m = RandomDefaultLatentModel(cop, pool_size=n, seed=42)
    p = 0.10
    counts = m.simulate_default_counts([p] * n, n_paths=2000)
    avg = sum(counts) / len(counts)
    # # CUSTOM 0.5: 2000-path MC stderr at p=0.1 across n=20 is ~sqrt(20*0.1*0.9 / 2000) ~ 0.03 per draw; * 20 = 0.6
    tolerance.custom(
        avg,
        n * p,
        abs_tol=0.5,
        rel_tol=0.1,
        reason="MC noise at 2000 paths for n=20 binomial mean",
    )


def test_random_default_latent_model_high_rho_concentrates() -> None:
    """At rho close to 1, default counts concentrate at 0 and n."""
    cop = OneFactorGaussianCopula(0.95)
    n = 10
    m = RandomDefaultLatentModel(cop, pool_size=n, seed=42)
    p = 0.20
    counts = m.simulate_default_counts([p] * n, n_paths=500)
    # Fraction of paths with all-or-none defaults
    extremes = sum(1 for c in counts if c in (0, n))
    # At rho=0.95 expect > 60% extremes
    assert extremes / float(len(counts)) > 0.50


def test_random_default_latent_model_prob_at_least_one_independence() -> None:
    """At rho=0, P(>=1) = 1 - (1-p)^n."""
    cop = OneFactorGaussianCopula(0.0)
    n = 5
    m = RandomDefaultLatentModel(cop, pool_size=n, seed=42)
    p = 0.20
    actual = m.prob_at_least_n_events_mc(1, [p] * n, n_paths=2000)
    expected = 1.0 - (1.0 - p) ** n
    tolerance.custom(
        actual,
        expected,
        abs_tol=0.05,
        rel_tol=0.05,
        reason="MC noise at 2000 paths for P(>=1) at rho=0",
    )


# -----------------------------------------------------------------------------
# RandomLossLatentModel
# -----------------------------------------------------------------------------


def test_random_loss_latent_model_round_trips() -> None:
    cop = OneFactorGaussianCopula(0.20)
    draw = constant_recovery_draw([0.40, 0.30])
    m = RandomLossLatentModel(
        cop, notionals=[1.0, 1.0], recovery_draw=draw, seed=123
    )
    assert m.pool_size() == 2
    assert m.notionals() == [1.0, 1.0]
    assert m.copula() is cop


def test_random_loss_latent_model_rejects_empty_notionals() -> None:
    cop = OneFactorGaussianCopula(0.20)
    with pytest.raises(LibraryException, match="non-empty"):
        RandomLossLatentModel(
            cop,
            notionals=[],
            recovery_draw=constant_recovery_draw([]),
        )


def test_random_loss_latent_model_expected_loss_independence() -> None:
    """At rho=0 with uniform pd + uniform rr, expected loss ~ N * p * (1 - rr)."""
    cop = OneFactorGaussianCopula(0.0)
    n = 20
    draw = constant_recovery_draw([0.40] * n)
    m = RandomLossLatentModel(
        cop,
        notionals=[1.0] * n,
        recovery_draw=draw,
        seed=42,
    )
    p = 0.10
    actual = m.expected_loss_mc([p] * n, n_paths=2000)
    expected = n * p * (1.0 - 0.40)
    tolerance.custom(
        actual,
        expected,
        abs_tol=0.5,
        rel_tol=0.3,
        reason="MC noise at 2000 paths for expected loss at rho=0",
    )


def test_random_loss_latent_model_zero_pd_zero_loss() -> None:
    """With near-zero pd, expected loss is 0 (no defaults observed)."""
    cop = OneFactorGaussianCopula(0.20)
    n = 10
    draw = constant_recovery_draw([0.40] * n)
    m = RandomLossLatentModel(
        cop,
        notionals=[1.0] * n,
        recovery_draw=draw,
        seed=42,
    )
    actual = m.expected_loss_mc([1e-15] * n, n_paths=200)
    tolerance.tight(actual, 0.0)


def test_random_loss_latent_model_rejects_mismatched_probs() -> None:
    cop = OneFactorGaussianCopula(0.20)
    draw = constant_recovery_draw([0.40, 0.30])
    m = RandomLossLatentModel(
        cop, notionals=[1.0, 1.0], recovery_draw=draw
    )
    with pytest.raises(LibraryException, match="probs size"):
        m.simulate_loss_distribution(probs=[0.1], n_paths=10)


def test_random_loss_latent_model_deterministic_seed() -> None:
    """Two runs with the same seed produce identical loss distributions."""
    cop = OneFactorGaussianCopula(0.20)
    n = 5
    draw = constant_recovery_draw([0.40] * n)
    m1 = RandomLossLatentModel(
        cop, notionals=[1.0] * n, recovery_draw=draw, seed=42
    )
    m2 = RandomLossLatentModel(
        cop, notionals=[1.0] * n, recovery_draw=draw, seed=42
    )
    p = [0.10] * n
    losses1 = m1.simulate_loss_distribution(p, n_paths=100)
    losses2 = m2.simulate_loss_distribution(p, n_paths=100)
    assert losses1 == losses2
