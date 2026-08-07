"""Cross-validate the finite-pool loss models against C++ QuantLib v1.43.

Covers ``HomogeneousPoolLossModel`` (homogeneouspooldef.hpp) and
``InhomogeneousPoolLossModel`` (inhomogeneouspooldef.hpp).

Probe source: migration-harness/cpp/probes/v143_experimental_creditloss/probe.cpp
Reference:    migration-harness/references/v143/experimental/creditloss.json

Both models are pinned array-by-array on the whole loss ``Distribution`` before
any summary statistic, so a mismatch localises to a bucket rather than being
hidden inside an integral. Both are also run on a second integration grid
(40 buckets over [-4, 4] in 20 steps) to pin the (min, max, n_steps) plumbing.

Everything here is TIGHT: neither model performs a root search, so there is no
solver-path amplification of floating-point noise.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.credit.distribution import Distribution
from pquantlib.experimental.credit.homogeneous_pool import (
    HomogeneousPoolLossModel,
)
from pquantlib.experimental.credit.inhomogeneous_pool import (
    InhomogeneousPoolLossModel,
)
from pquantlib.testing import tolerance
from tests.experimental.credit import _creditloss_fixture as fx

#: probe.cpp:864.
N_BUCKETS = 25
#: probe.cpp:873, 887 — the quantiles the probe walks.
QUANTILES = (0.10, 0.50, 0.90, 0.95, 0.99)


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return fx.load_reference()


def _assert_distribution(
    dist: Distribution, prefix: str, cpp_ref: dict[str, Any]
) -> None:
    """Pin every observable array of a ``Distribution`` under ``prefix``."""
    n = dist.size()
    assert n == cpp_ref[f"{prefix}_size"]
    for suffix, getter in (
        ("_x", dist.x),
        ("_dx", dist.dx),
        ("_density", dist.density),
        ("_cumulative", dist.cumulative),
        ("_excess", dist.excess),
        ("_cumulative_excess", dist.cumulative_excess),
        ("_average", dist.average),
    ):
        expected = cpp_ref[prefix + suffix]
        for i in range(n):
            tolerance.tight(getter(i), expected[i])


# -----------------------------------------------------------------------------
# HomogeneousPoolLossModel
# -----------------------------------------------------------------------------


def test_homogeneous_loss_distribution_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    f = fx.build_fixture()
    model = HomogeneousPoolLossModel(f.latent_model, N_BUCKETS)
    f.basket.set_loss_model(model)
    _assert_distribution(
        model.loss_distrib(f.calc_date), "homog_loss_distrib_5y", cpp_ref
    )


def test_homogeneous_expected_tranche_loss_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    f = fx.build_fixture()
    model = HomogeneousPoolLossModel(f.latent_model, N_BUCKETS)
    f.basket.set_loss_model(model)
    tolerance.tight(
        model.expected_tranche_loss(f.calc_date),
        cpp_ref["homog_expected_tranche_loss_5y"],
    )
    tolerance.tight(
        model.expected_tranche_loss(f.calc_date_2y),
        cpp_ref["homog_expected_tranche_loss_2y"],
    )


def test_homogeneous_percentile_and_shortfall_match_cpp(
    cpp_ref: dict[str, Any],
) -> None:
    f = fx.build_fixture()
    model = HomogeneousPoolLossModel(f.latent_model, N_BUCKETS)
    f.basket.set_loss_model(model)
    for q, expected in zip(QUANTILES, cpp_ref["homog_percentile_5y"], strict=True):
        tolerance.tight(model.percentile(f.calc_date, q), expected)
    for q, expected in zip(
        QUANTILES, cpp_ref["homog_expected_shortfall_5y"], strict=True
    ):
        tolerance.tight(model.expected_shortfall(f.calc_date, q), expected)


def test_homogeneous_alternate_integration_grid_matches_cpp(
    cpp_ref: dict[str, Any],
) -> None:
    """Non-default (max, min, n_steps) — pins the integration plumbing."""
    f = fx.build_fixture()
    model = HomogeneousPoolLossModel(f.latent_model, 40, 4.0, -4.0, 20)
    f.basket.set_loss_model(model)
    _assert_distribution(
        model.loss_distrib(f.calc_date), "homog_alt_grid_loss_distrib_5y", cpp_ref
    )
    tolerance.tight(
        model.expected_tranche_loss(f.calc_date),
        cpp_ref["homog_alt_grid_expected_tranche_loss_5y"],
    )


# -----------------------------------------------------------------------------
# InhomogeneousPoolLossModel
# -----------------------------------------------------------------------------


def test_inhomogeneous_loss_distribution_matches_cpp(cpp_ref: dict[str, Any]) -> None:
    f = fx.build_fixture()
    model = InhomogeneousPoolLossModel(f.latent_model, N_BUCKETS)
    f.basket.set_loss_model(model)
    _assert_distribution(
        model.loss_distrib(f.calc_date), "inhomog_loss_distrib_5y", cpp_ref
    )


def test_inhomogeneous_expected_tranche_loss_matches_cpp(
    cpp_ref: dict[str, Any],
) -> None:
    f = fx.build_fixture()
    model = InhomogeneousPoolLossModel(f.latent_model, N_BUCKETS)
    f.basket.set_loss_model(model)
    tolerance.tight(
        model.expected_tranche_loss(f.calc_date),
        cpp_ref["inhomog_expected_tranche_loss_5y"],
    )
    tolerance.tight(
        model.expected_tranche_loss(f.calc_date_2y),
        cpp_ref["inhomog_expected_tranche_loss_2y"],
    )


def test_inhomogeneous_percentile_and_shortfall_match_cpp(
    cpp_ref: dict[str, Any],
) -> None:
    f = fx.build_fixture()
    model = InhomogeneousPoolLossModel(f.latent_model, N_BUCKETS)
    f.basket.set_loss_model(model)
    for q, expected in zip(QUANTILES, cpp_ref["inhomog_percentile_5y"], strict=True):
        tolerance.tight(model.percentile(f.calc_date, q), expected)
    for q, expected in zip(
        QUANTILES, cpp_ref["inhomog_expected_shortfall_5y"], strict=True
    ):
        tolerance.tight(model.expected_shortfall(f.calc_date, q), expected)


def test_inhomogeneous_alternate_integration_grid_matches_cpp(
    cpp_ref: dict[str, Any],
) -> None:
    f = fx.build_fixture()
    model = InhomogeneousPoolLossModel(f.latent_model, 40, 4.0, -4.0, 20)
    f.basket.set_loss_model(model)
    _assert_distribution(
        model.loss_distrib(f.calc_date), "inhomog_alt_grid_loss_distrib_5y", cpp_ref
    )
    tolerance.tight(
        model.expected_tranche_loss(f.calc_date),
        cpp_ref["inhomog_alt_grid_expected_tranche_loss_5y"],
    )


# -----------------------------------------------------------------------------
# Structure
# -----------------------------------------------------------------------------


def test_both_models_agree_on_uniform_notionals(cpp_ref: dict[str, Any]) -> None:
    """The probe fixture is uniform-notional, so the two algorithms coincide.

    ``LossDistHomogeneous`` (exact convolution off one shared volume) and
    ``LossDistBucketing`` (Hull-White bucketing of heterogeneous volumes) must
    return the same distribution when every LGD is equal — which is what the
    C++ reference itself shows, so this is a structural check on top of the
    per-model pins above rather than an independent one.
    """
    for key in ("expected_tranche_loss_5y", "expected_tranche_loss_2y"):
        tolerance.loose(
            cpp_ref["homog_" + key],
            cpp_ref["inhomog_" + key],
            reason=(
                "the two convolution algorithms accumulate the same buckets in "
                "different orders; C++ itself differs in the last 2 ulp"
            ),
        )


@pytest.mark.parametrize(
    "cls", [HomogeneousPoolLossModel, InhomogeneousPoolLossModel]
)
def test_multifactor_copula_is_rejected(cls: type) -> None:
    """# C++ parity: homogeneouspooldef.hpp:59-60 / inhomogeneouspooldef.hpp:66-67."""
    from pquantlib.experimental.credit.constant_loss_latent_model import (  # noqa: PLC0415
        ConstantLossLatentmodel,
    )
    from pquantlib.experimental.credit.default_probability_latent_model import (  # noqa: PLC0415
        LatentModelIntegrationType,
    )
    from pquantlib.experimental.math.gaussian_copula_policy import (  # noqa: PLC0415
        GaussianCopulaPolicy,
    )

    weights = [[0.4, 0.3] for _ in range(5)]
    lm = ConstantLossLatentmodel(
        weights,
        fx.RECOVERIES,
        GaussianCopulaPolicy(weights),
        LatentModelIntegrationType.GaussianQuadrature,
    )
    with pytest.raises(LibraryException, match="not implemented for multifactor"):
        cls(lm, N_BUCKETS)
