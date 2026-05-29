"""Cross-validate GaussianLHPLossModel against the C++ probe.

Probe source: migration-harness/cpp/probes/cluster_w3b/probe.cpp
Reference:    migration-harness/references/cluster/w3b.json

Tests cover:
  * expected_tranche_loss at four (attach, detach) levels for the
    LHP closed form @ (corr=0.30, prob=0.05, RR=0.40).
  * expected_tranche_loss in a stressed-default scenario
    (corr=0.20, prob=0.15, RR=0.30).
  * percentile_portfolio_loss_fraction at p=0.995, 0.990, 0.950.
  * Edge cases: attach >= detach -> 0, prob == 0 -> 0, ETL across
    [0, 1] reproduces prob * (1 - RR) exactly (no roundoff because the
    LHP formula degenerates).
  * Validation errors: out-of-range correlation / recovery.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.gaussian_lhp_loss_model import (
    GaussianLHPLossModel,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w3b")


# -----------------------------------------------------------------------------
# Construction + validation
# -----------------------------------------------------------------------------


def test_lhp_loss_model_round_trips() -> None:
    model = GaussianLHPLossModel(0.30, 0.40)
    tolerance.tight(model.correlation(), 0.30)
    tolerance.tight(model.recovery_rate(), 0.40)


def test_lhp_loss_model_rejects_out_of_range_correlation() -> None:
    with pytest.raises(LibraryException, match=r"correlation"):
        GaussianLHPLossModel(1.5, 0.40)
    with pytest.raises(LibraryException, match=r"correlation"):
        GaussianLHPLossModel(0.0, 0.40)
    with pytest.raises(LibraryException, match=r"correlation"):
        GaussianLHPLossModel(1.0, 0.40)


def test_lhp_loss_model_rejects_out_of_range_recovery() -> None:
    with pytest.raises(LibraryException, match=r"recovery"):
        GaussianLHPLossModel(0.30, 1.5)
    with pytest.raises(LibraryException, match=r"recovery"):
        GaussianLHPLossModel(0.30, -0.10)


# -----------------------------------------------------------------------------
# C++ probe round-trip: ETL @ (corr=0.30, prob=0.05, RR=0.40)
# -----------------------------------------------------------------------------


def test_lhp_etl_matches_cpp_at_each_tranche(cpp_ref: dict[str, Any]) -> None:
    ref = cpp_ref["lhp_etl"]
    model = GaussianLHPLossModel(ref["corr"], ref["avgRR"])
    prob = ref["prob"]
    avg_rr = ref["avgRR"]
    # # TIGHT: BivariateCumulativeNormalDistribution uses scipy multi-variate
    # CDF which matches the C++ Drezner-Wesolowsky formula at < 1e-14.
    tolerance.tight(
        model.expected_tranche_loss(1.0, prob, avg_rr, 0.0, 0.03),
        ref["etl_0_3"],
    )
    tolerance.tight(
        model.expected_tranche_loss(1.0, prob, avg_rr, 0.03, 0.06),
        ref["etl_3_6"],
    )
    tolerance.tight(
        model.expected_tranche_loss(1.0, prob, avg_rr, 0.06, 0.09),
        ref["etl_6_9"],
    )
    tolerance.tight(
        model.expected_tranche_loss(1.0, prob, avg_rr, 0.09, 0.12),
        ref["etl_9_12"],
    )
    # # The full 0-100% tranche ETL equals prob * (1 - RR) up to the
    # K=1 clip-to-(1-1e-12) epsilon — the LHP formula reduces to that.
    tolerance.tight(
        model.expected_tranche_loss(1.0, prob, avg_rr, 0.0, 1.0),
        ref["etl_0_100"],
    )


def test_lhp_etl_stress_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    ref = cpp_ref["lhp_etl_stress"]
    model = GaussianLHPLossModel(0.20, 0.30)
    tolerance.tight(
        model.expected_tranche_loss(1.0, 0.15, 0.30, 0.0, 0.05),
        ref["etl_0_5"],
    )
    tolerance.tight(
        model.expected_tranche_loss(1.0, 0.15, 0.30, 0.05, 0.15),
        ref["etl_5_15"],
    )
    tolerance.tight(
        model.expected_tranche_loss(1.0, 0.15, 0.30, 0.15, 0.30),
        ref["etl_15_30"],
    )


def test_lhp_percentile_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    ref = cpp_ref["lhp_percentile"]
    model = GaussianLHPLossModel(0.30, 0.40)
    tolerance.tight(
        model.percentile_portfolio_loss_fraction(0.05, 0.40, 0.995),
        ref["p995"],
    )
    tolerance.tight(
        model.percentile_portfolio_loss_fraction(0.05, 0.40, 0.99),
        ref["p990"],
    )
    tolerance.tight(
        model.percentile_portfolio_loss_fraction(0.05, 0.40, 0.95),
        ref["p950"],
    )


# -----------------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------------


def test_lhp_etl_attach_geq_detach_returns_zero() -> None:
    model = GaussianLHPLossModel(0.30, 0.40)
    tolerance.tight(
        model.expected_tranche_loss(1.0, 0.05, 0.40, 0.06, 0.03), 0.0
    )
    tolerance.tight(
        model.expected_tranche_loss(1.0, 0.05, 0.40, 0.05, 0.05), 0.0
    )


def test_lhp_etl_zero_prob_returns_zero() -> None:
    model = GaussianLHPLossModel(0.30, 0.40)
    tolerance.tight(
        model.expected_tranche_loss(1.0, 0.0, 0.40, 0.03, 0.06), 0.0
    )


def test_lhp_etl_zero_notional_returns_zero() -> None:
    model = GaussianLHPLossModel(0.30, 0.40)
    tolerance.tight(
        model.expected_tranche_loss(0.0, 0.05, 0.40, 0.03, 0.06), 0.0
    )


def test_lhp_etl_scales_linearly_with_notional() -> None:
    model = GaussianLHPLossModel(0.30, 0.40)
    etl1 = model.expected_tranche_loss(1.0, 0.05, 0.40, 0.03, 0.06)
    etl10 = model.expected_tranche_loss(10.0, 0.05, 0.40, 0.03, 0.06)
    tolerance.tight(etl10, 10.0 * etl1)


# -----------------------------------------------------------------------------
# Percentile + prob over loss + ESF
# -----------------------------------------------------------------------------


def test_lhp_percentile_zero_returns_zero() -> None:
    model = GaussianLHPLossModel(0.30, 0.40)
    tolerance.tight(
        model.percentile_portfolio_loss_fraction(0.05, 0.40, 0.0), 0.0
    )


def test_lhp_percentile_one_returns_one_minus_rr() -> None:
    """At perctl=1, the percentile loss fraction tends to (1-RR)."""
    model = GaussianLHPLossModel(0.30, 0.40)
    # # CUSTOM 1e-3: at perctl=1 the formula saturates at (1-RR) * Phi(+inf)
    # which is (1-RR). The clip-to-(1-eps) introduces a tiny shortfall.
    actual = model.percentile_portfolio_loss_fraction(0.05, 0.40, 1.0)
    tolerance.custom(
        actual,
        1.0 - 0.40,
        abs_tol=1e-3,
        rel_tol=1e-3,
        reason="clip-to-(1-eps) tail truncation in percentile formula",
    )


def test_lhp_percentile_in_tranche_clips_at_detach() -> None:
    model = GaussianLHPLossModel(0.30, 0.40)
    # Wide tranche, very high perctl -> capped at detach - attach.
    expected_cap = 0.06 - 0.03
    actual = model.percentile(1.0, 0.05, 0.40, 0.03, 0.06, 0.9999)
    tolerance.tight(actual, expected_cap)


def test_lhp_prob_over_loss_zero_attach_zero_loss_returns_one() -> None:
    """When attach == 0 and remaining_loss_fraction == 0, ptfl_fract == 0
    is below QL_EPSILON so the model short-circuits to 1.0.

    # C++ parity: gaussianlhplossmodel.cpp:144 if(portfFract <= QL_EPSILON) return 1.;
    """
    model = GaussianLHPLossModel(0.30, 0.40)
    tolerance.tight(
        model.prob_over_loss(1.0, 0.05, 0.40, 0.0, 0.06, 0.0), 1.0
    )


def test_lhp_prob_over_loss_above_max_returns_zero() -> None:
    """Above the unrecoverable loss the prob is 0."""
    model = GaussianLHPLossModel(0.30, 0.40)
    # ptfl_fract = 0.03 + 1.0 * (0.99 - 0.03) = 0.99 > 1 - 0.40 = 0.60.
    tolerance.tight(
        model.prob_over_loss(1.0, 0.05, 0.40, 0.03, 0.99, 1.0), 0.0
    )


def test_lhp_prob_over_loss_in_range() -> None:
    model = GaussianLHPLossModel(0.30, 0.40)
    p = model.prob_over_loss(1.0, 0.05, 0.40, 0.03, 0.06, 0.5)
    assert 0.0 <= p <= 1.0


def test_lhp_expected_shortfall_is_finite() -> None:
    model = GaussianLHPLossModel(0.30, 0.40)
    es = model.expected_shortfall(1.0, 0.05, 0.40, 0.03, 0.06, 0.95)
    # ESF in [0, detach - attach] * remaining_notional = [0, 0.03].
    assert 0.0 <= es <= 0.03 + 1e-9


def test_lhp_expected_shortfall_rejects_out_of_range_percentile() -> None:
    model = GaussianLHPLossModel(0.30, 0.40)
    with pytest.raises(LibraryException, match=r"perctl"):
        model.percentile_portfolio_loss_fraction(0.05, 0.40, 1.5)
    with pytest.raises(LibraryException, match=r"perctl"):
        model.percentile_portfolio_loss_fraction(0.05, 0.40, -0.10)
