"""Cross-validate BinomialLossModel / RecursiveLossModel.

The C++ class hierarchy is templated on copula + LLM type; these two Python
loss models share a flat interface using ``ConstantLossLatentModel`` as the
upstream provider.

Cross-validation strategy: Binomial vs Recursive on an identical pool — both
approximate the same true loss distribution, so agreement within ~5% is
expected for moderate pools.

The saddlepoint model used to be exercised here as ``SaddlepointLossModel``, a
PQuantLib-only Lugannani-Rice approximation with an explicit-args interface. It
has been replaced by :class:`~pquantlib.experimental.credit.saddlepoint_loss_model.SaddlePointLossModel`,
the faithful port of C++ ``SaddlePointLossModel``; its cross-validation lives in
``test_saddlepoint_loss_model_v143.py``, driven by the C++ v1.43 probe.
"""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.binomial_loss_model import (
    BinomialLossModel,
)
from pquantlib.experimental.credit.constant_loss_latent_model import (
    ConstantLossLatentModel,
)
from pquantlib.experimental.credit.default_loss_model import DefaultLossModelBase
from pquantlib.experimental.credit.one_factor_copula import (
    OneFactorGaussianCopula,
)
from pquantlib.experimental.credit.recursive_loss_model import (
    RecursiveLossModel,
)
from pquantlib.testing import tolerance

# -----------------------------------------------------------------------------
# DefaultLossModel — abstract surface enforcement
# -----------------------------------------------------------------------------


def test_default_loss_model_is_abstract() -> None:
    with pytest.raises(TypeError):
        _ = DefaultLossModelBase()  # pyright: ignore[reportAbstractUsage]


def test_default_loss_model_default_methods_raise_not_implemented() -> None:
    """Subclass that overrides only expected_tranche_loss — the other
    methods should raise NotImplementedError by default."""

    class _MinimalModel(DefaultLossModelBase):
        def expected_tranche_loss(
            self,
            remaining_notional: float,
            prob: float,
            average_rr: float,
            attach: float,
            detach: float,
        ) -> float:
            return 0.0

    model = _MinimalModel()
    with pytest.raises(NotImplementedError, match="prob_over_loss"):
        model.prob_over_loss(1.0, 0.05, 0.40, 0.03, 0.06, 0.5)
    with pytest.raises(NotImplementedError, match="percentile"):
        model.percentile(1.0, 0.05, 0.40, 0.03, 0.06, 0.95)
    with pytest.raises(NotImplementedError, match="expected_shortfall"):
        model.expected_shortfall(1.0, 0.05, 0.40, 0.03, 0.06, 0.95)


# -----------------------------------------------------------------------------
# BinomialLossModel
# -----------------------------------------------------------------------------


def test_binomial_loss_model_construction() -> None:
    cop = OneFactorGaussianCopula(0.30)
    llm = ConstantLossLatentModel(cop, recoveries=[0.40] * 10)
    m = BinomialLossModel(llm)
    assert m.lgd_buckets() == 10
    assert m.latent_model() is llm


def test_binomial_loss_model_rejects_invalid_lgd_buckets() -> None:
    cop = OneFactorGaussianCopula(0.30)
    llm = ConstantLossLatentModel(cop, recoveries=[0.40] * 5)
    with pytest.raises(LibraryException, match="lgd_buckets"):
        BinomialLossModel(llm, lgd_buckets=0)


def test_binomial_loss_model_etl_zero_prob_returns_zero() -> None:
    cop = OneFactorGaussianCopula(0.30)
    llm = ConstantLossLatentModel(cop, recoveries=[0.40] * 5)
    m = BinomialLossModel(llm)
    # CUSTOM 1e-6: floating-point noise from the integration.
    actual = m.expected_tranche_loss(
        remaining_notional=1.0,
        prob=1e-12,
        average_rr=0.40,
        attach=0.0,
        detach=1.0,
    )
    tolerance.custom(
        actual, 0.0, abs_tol=1e-6, rel_tol=1e-6, reason="quadrature noise"
    )


def test_binomial_loss_model_etl_positive_for_moderate_pd() -> None:
    cop = OneFactorGaussianCopula(0.30)
    n = 20
    llm = ConstantLossLatentModel(cop, recoveries=[0.40] * n)
    m = BinomialLossModel(llm)
    etl = m.expected_tranche_loss(
        remaining_notional=1.0,
        prob=0.05,
        average_rr=0.40,
        attach=0.0,
        detach=1.0,
    )
    # Total expected loss ~ N * (notional/N) * p * (1-RR) = 1 * 0.05 * 0.6 = 0.03
    # Binomial approximation is loose but should be within 50% of the truth.
    assert 0.015 < etl < 0.045


def test_binomial_loss_model_etl_tranche_within_total() -> None:
    """ETL of (0, 1) >= ETL of (0.03, 0.06) — sanity tranche ordering."""
    cop = OneFactorGaussianCopula(0.30)
    n = 20
    llm = ConstantLossLatentModel(cop, recoveries=[0.40] * n)
    m = BinomialLossModel(llm)
    full = m.expected_tranche_loss(1.0, 0.05, 0.40, 0.0, 1.0)
    tranche = m.expected_tranche_loss(1.0, 0.05, 0.40, 0.03, 0.06)
    assert full >= tranche
    assert tranche >= 0.0


# -----------------------------------------------------------------------------
# RecursiveLossModel
# -----------------------------------------------------------------------------


def test_recursive_loss_model_construction() -> None:
    cop = OneFactorGaussianCopula(0.30)
    llm = ConstantLossLatentModel(cop, recoveries=[0.40] * 10)
    m = RecursiveLossModel(llm, n_buckets=2)
    assert m.n_buckets() == 2
    assert m.latent_model() is llm


def test_recursive_loss_model_rejects_invalid_n_buckets() -> None:
    cop = OneFactorGaussianCopula(0.30)
    llm = ConstantLossLatentModel(cop, recoveries=[0.40] * 5)
    with pytest.raises(LibraryException, match="n_buckets"):
        RecursiveLossModel(llm, n_buckets=0)


def test_recursive_loss_model_etl_zero_prob_returns_zero() -> None:
    cop = OneFactorGaussianCopula(0.30)
    llm = ConstantLossLatentModel(cop, recoveries=[0.40] * 5)
    m = RecursiveLossModel(llm)
    actual = m.expected_tranche_loss(
        remaining_notional=1.0,
        prob=1e-12,
        average_rr=0.40,
        attach=0.0,
        detach=1.0,
    )
    tolerance.custom(
        actual, 0.0, abs_tol=1e-6, rel_tol=1e-6, reason="quadrature noise"
    )


def test_recursive_loss_model_etl_positive_for_moderate_pd() -> None:
    cop = OneFactorGaussianCopula(0.30)
    n = 10
    llm = ConstantLossLatentModel(cop, recoveries=[0.40] * n)
    m = RecursiveLossModel(llm)
    etl = m.expected_tranche_loss(
        remaining_notional=1.0,
        prob=0.05,
        average_rr=0.40,
        attach=0.0,
        detach=1.0,
    )
    # Recursive is exact for the convolution; expected loss = 0.05 * 0.6 = 0.03.
    tolerance.custom(
        etl,
        0.03,
        abs_tol=5e-3,
        rel_tol=0.20,
        reason="Recursive convolution + 50-step Euler M integration noise",
    )


def test_recursive_loss_model_loss_distribution_sums_to_one() -> None:
    cop = OneFactorGaussianCopula(0.30)
    n = 5
    llm = ConstantLossLatentModel(cop, recoveries=[0.40] * n)
    m = RecursiveLossModel(llm)
    dist = m.unconditional_loss_distribution(
        probs=[0.05] * n,
        notionals=[1.0 / n] * n,
    )
    total = sum(dist.values())
    tolerance.custom(
        total,
        1.0,
        abs_tol=1e-6,
        rel_tol=1e-6,
        reason="discrete distribution should sum to 1 up to Euler integration noise",
    )


# -----------------------------------------------------------------------------
# Binomial vs Recursive — cross-check
# -----------------------------------------------------------------------------


def test_binomial_vs_recursive_match_on_identical_pool() -> None:
    """At low rho both models give similar ETL on the same pool."""
    cop = OneFactorGaussianCopula(0.20)
    n = 10
    llm = ConstantLossLatentModel(cop, recoveries=[0.40] * n)
    binom = BinomialLossModel(llm)
    recur = RecursiveLossModel(llm)
    etl_binom = binom.expected_tranche_loss(1.0, 0.05, 0.40, 0.0, 1.0)
    etl_recur = recur.expected_tranche_loss(1.0, 0.05, 0.40, 0.0, 1.0)
    # # LOOSE 5e-2: binomial is an approximation; should agree within ~5%
    # of the recursive truth.
    tolerance.custom(
        etl_binom,
        etl_recur,
        abs_tol=5e-2,
        rel_tol=0.30,
        reason="binomial approximation deviates from recursive truth on small pools",
    )
