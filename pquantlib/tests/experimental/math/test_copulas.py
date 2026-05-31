"""Cross-validate copula policies + copula RNGs.

Probe source: migration-harness/cpp/probes/cluster_w6c/probe.cpp
Reference:    migration-harness/references/cluster/w6c.json
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.experimental.math.clayton_copula_rng import ClaytonCopulaRng
from pquantlib.experimental.math.farlie_gumbel_morgenstern_copula_rng import (
    FarlieGumbelMorgensternCopulaRng,
)
from pquantlib.experimental.math.frank_copula_rng import FrankCopulaRng
from pquantlib.experimental.math.gaussian_copula_policy import GaussianCopulaPolicy
from pquantlib.experimental.math.polar_student_t_rng import PolarStudentTRng
from pquantlib.experimental.math.t_copula_policy import TCopulaPolicy
from pquantlib.math.randomnumbers.mersenne_twister import (
    MersenneTwisterUniformRng,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp_ref() -> dict[str, Any]:
    return reference_reader.load("cluster/w6c")


# ---- Gaussian copula policy ----


def test_gaussian_copula_policy(cpp_ref: dict[str, Any]) -> None:
    gcp = GaussianCopulaPolicy([[0.5], [0.4]])
    assert gcp.num_factors() == int(cpp_ref["gauss_copula_numFactors"])
    cy = gcp.cumulative_y(0.7, 0)
    tolerance.tight(cy, cpp_ref["gauss_copula_cumY_0_7"])
    tolerance.tight(
        gcp.inverse_cumulative_y(cy, 0), cpp_ref["gauss_copula_invY_of_cumY"]
    )
    tolerance.tight(gcp.cumulative_z(0.3), cpp_ref["gauss_copula_cumZ_0_3"])
    tolerance.tight(gcp.density([0.1, -0.2]), cpp_ref["gauss_copula_density"])


def test_gaussian_copula_inverse_round_trip() -> None:
    gcp = GaussianCopulaPolicy([[0.5], [0.4]])
    for p in (0.1, 0.5, 0.9):
        # The inverse-normal rational approximation (Beasley-Springer-Moro)
        # is accurate to ~1e-9, so the round trip is LOOSE, not bit-exact.
        tolerance.loose(
            gcp.cumulative_y(gcp.inverse_cumulative_y(p, 0), 0),
            p,
            reason="inverse-normal rational approx is ~1e-9 accurate.",
        )


def test_gaussian_copula_non_normal_rejected() -> None:
    with pytest.raises(LibraryException, match="Non normal"):
        GaussianCopulaPolicy([[1.5]])


# ---- Student-t copula policy ----


def test_t_copula_policy(cpp_ref: dict[str, Any]) -> None:
    tcp = TCopulaPolicy([[0.5], [0.4]], [3, 3])
    assert tcp.num_factors() == int(cpp_ref["t_copula_numFactors"])
    cy = tcp.cumulative_y(0.7, 0)
    tolerance.tight(cy, cpp_ref["t_copula_cumY_0_7"])
    tolerance.loose(
        tcp.inverse_cumulative_y(cy, 0),
        cpp_ref["t_copula_invY_of_cumY"],
        reason="inverse is a 1e-6-accuracy Brent root solve.",
    )
    tolerance.tight(tcp.cumulative_z(0.3), cpp_ref["t_copula_cumZ_0_3"])
    tolerance.tight(tcp.variance_factors()[0], cpp_ref["t_copula_varfac_0"])


def test_t_copula_finite_variance_required() -> None:
    with pytest.raises(LibraryException, match="finite variance"):
        TCopulaPolicy([[0.5]], [2, 3])


def test_t_copula_factor_count_mismatch() -> None:
    with pytest.raises(LibraryException, match="Incompatible number of T"):
        TCopulaPolicy([[0.5]], [3, 3, 3])


# ---- copula RNGs ----


def test_clayton_copula_rng_structure() -> None:
    mt = MersenneTwisterUniformRng(42)
    rng = ClaytonCopulaRng(mt, theta=2.0)
    s = rng.next()
    assert len(s.value) == 2
    # u1 is the first uniform; both outputs are valid probabilities.
    assert 0.0 <= s.value[0] <= 1.0
    assert 0.0 <= s.value[1] <= 1.0
    assert s.weight == 1.0


def test_clayton_copula_rng_u1_is_first_uniform() -> None:
    mt1 = MersenneTwisterUniformRng(7)
    first = mt1.next().value
    mt2 = MersenneTwisterUniformRng(7)
    rng = ClaytonCopulaRng(mt2, theta=2.0)
    tolerance.exact(rng.next().value[0], first)


def test_clayton_invalid_theta() -> None:
    mt = MersenneTwisterUniformRng(1)
    with pytest.raises(LibraryException, match="must be different from 0"):
        ClaytonCopulaRng(mt, theta=0.0)
    with pytest.raises(LibraryException, match="greater or equal to -1"):
        ClaytonCopulaRng(mt, theta=-2.0)


def test_frank_copula_rng_structure() -> None:
    mt = MersenneTwisterUniformRng(42)
    rng = FrankCopulaRng(mt, theta=2.0)
    s = rng.next()
    assert len(s.value) == 2
    assert 0.0 <= s.value[1] <= 1.0


def test_frank_invalid_theta() -> None:
    mt = MersenneTwisterUniformRng(1)
    with pytest.raises(LibraryException, match="must be different from 0"):
        FrankCopulaRng(mt, theta=0.0)


def test_fgm_copula_rng_structure() -> None:
    mt = MersenneTwisterUniformRng(42)
    rng = FarlieGumbelMorgensternCopulaRng(mt, theta=0.5)
    s = rng.next()
    assert len(s.value) == 2


def test_fgm_invalid_theta() -> None:
    mt = MersenneTwisterUniformRng(1)
    with pytest.raises(LibraryException, match=r"must be in \[-1,1\]"):
        FarlieGumbelMorgensternCopulaRng(mt, theta=1.5)


# ---- polar Student-t RNG ----


def test_polar_student_t_rng_stream(cpp_ref: dict[str, Any]) -> None:
    mt = MersenneTwisterUniformRng(42)
    rng = PolarStudentTRng(5.0, mt)
    tolerance.tight(rng.next().value, cpp_ref["polar_t5_seed42_sample0"])
    tolerance.tight(rng.next().value, cpp_ref["polar_t5_seed42_sample1"])


def test_polar_student_t_invalid_dof() -> None:
    mt = MersenneTwisterUniformRng(1)
    with pytest.raises(LibraryException, match="degrees of freedom"):
        PolarStudentTRng(0.0, mt)
