"""Cross-validate the generic LatentModel against C++ inspectors + integration.

Probe source: migration-harness/cpp/probes/cluster_w6c/probe.cpp
Reference:    migration-harness/references/cluster/w6c.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.experimental.math.gaussian_copula_policy import GaussianCopulaPolicy
from pquantlib.experimental.math.latent_model import LatentModel
from pquantlib.experimental.math.multidim_integrator import MultidimIntegral
from pquantlib.experimental.math.t_copula_policy import TCopulaPolicy
from pquantlib.math.integrals.trapezoid import TrapezoidIntegral
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w6c")


def _gaussian_model() -> LatentModel:
    w = [[0.5], [0.4], [0.3]]
    return LatentModel(w, GaussianCopulaPolicy(w))


def test_latent_inspectors(cpp_ref: dict[str, Any]) -> None:
    lm = _gaussian_model()
    assert lm.size() == int(cpp_ref["latent_size"])
    assert lm.num_factors() == int(cpp_ref["latent_numFactors"])
    assert lm.num_total_factors() == int(cpp_ref["latent_numTotalFactors"])
    tolerance.tight(lm.idiosync_fctrs()[0], cpp_ref["latent_idiosync_0"])


def test_latent_correlation(cpp_ref: dict[str, Any]) -> None:
    lm = _gaussian_model()
    tolerance.tight(lm.latent_variable_correl(0, 1), cpp_ref["latent_correl_0_1"])
    tolerance.tight(lm.latent_variable_correl(0, 0), cpp_ref["latent_correl_0_0"])


def test_latent_copula_passthrough(cpp_ref: dict[str, Any]) -> None:
    lm = _gaussian_model()
    cy = lm.cumulative_y(0.7, 0)
    tolerance.tight(cy, cpp_ref["latent_cumY_0"])
    tolerance.tight(lm.inverse_cumulative_y(cy, 0), cpp_ref["latent_invY_0"])


def test_latent_var_value(cpp_ref: dict[str, Any]) -> None:
    lm = _gaussian_model()
    # all_factors = [M, Z_0, Z_1, Z_2]; Y_0 = 0.5*M + idio_0 * Z_0
    all_factors = [1.0, 0.5, -0.5, 0.25]
    tolerance.tight(lm.latent_var_value(all_factors, 0), cpp_ref["latent_varValue_0"])


def test_latent_factor_weights_roundtrip() -> None:
    w = [[0.5], [0.4], [0.3]]
    lm = LatentModel(w, GaussianCopulaPolicy(w))
    assert lm.factor_weights() == w


def test_latent_single_factor_factory() -> None:
    lm = LatentModel.single_factor([0.5, 0.4, 0.3], GaussianCopulaPolicy)
    assert lm.size() == 3
    assert lm.num_factors() == 1
    tolerance.tight(lm.latent_variable_correl(0, 1), 0.2)


def test_latent_from_correlation_factory() -> None:
    lm = LatentModel.from_correlation(0.5, 4, GaussianCopulaPolicy)
    assert lm.size() == 4
    # all names share the same loading; pairwise correl = 0.5^2 = 0.25.
    tolerance.tight(lm.latent_variable_correl(0, 1), 0.25)


def test_latent_inconsistent_factor_rows() -> None:
    with pytest.raises(ValueError, match="different number of factors"):
        LatentModel([[0.5, 0.1], [0.4]], GaussianCopulaPolicy([[0.5, 0.1]]))


class _TrapLMIntegration:
    """Single-axis trapezoid integration facility over [-8, 8]."""

    def __init__(self) -> None:
        t = TrapezoidIntegral(1e-9, 1_000_000)
        self._mdi = MultidimIntegral([t])

    def integrate(self, f: Any) -> float:
        return self._mdi(f, [-8.0], [8.0])


def test_latent_integrated_expected_value_density_normalises() -> None:
    # E[1] over the systemic factor density integrates the density to ~1.
    lm = LatentModel([[0.5]], GaussianCopulaPolicy([[0.5]]), _TrapLMIntegration())
    tolerance.loose(
        lm.integrated_expected_value(lambda x: 1.0),
        1.0,
        reason="density integrates to 1 over a wide trapezoid grid.",
    )


def test_latent_integrated_expected_value_unit_variance() -> None:
    # The systemic factor has unit variance: E[M^2] ~ 1.
    lm = LatentModel([[0.5]], GaussianCopulaPolicy([[0.5]]), _TrapLMIntegration())
    tolerance.loose(
        lm.integrated_expected_value(lambda x: x[0] ** 2),
        1.0,
        reason="standard-normal systemic factor has unit variance.",
    )


def test_latent_integrated_without_facility_raises() -> None:
    lm = _gaussian_model()
    with pytest.raises(ValueError, match="no integration facility"):
        lm.integrated_expected_value(lambda x: 1.0)


def test_latent_with_t_copula() -> None:
    # The same template works with the Student-t copula policy.
    w = [[0.5], [0.4]]
    lm = LatentModel(w, TCopulaPolicy(w, [3, 3]))
    assert lm.size() == 2
    assert lm.num_factors() == 1
    # correlation is structural (independent of the copula law).
    tolerance.tight(lm.latent_variable_correl(0, 1), 0.2)
