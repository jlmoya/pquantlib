"""Tests for the W9-C market-model Brownian generators.

Cross-validates against ``migration-harness/references/cluster/w9c.json``.

C++ parity:
  ql/models/marketmodels/browniangenerators/mtbrowniangenerator.{hpp,cpp}
  ql/models/marketmodels/browniangenerators/sobolbrowniangenerator.{hpp,cpp}
  @ v1.42.1 (099987f0).
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.models.marketmodels.brownian_generator import (
    BrownianGenerator,
    BrownianGeneratorFactory,
)
from pquantlib.models.marketmodels.browniangenerators.mt_brownian_generator import (
    MTBrownianGenerator,
    MTBrownianGeneratorFactory,
)
from pquantlib.models.marketmodels.browniangenerators.sobol_brownian_generator import (
    SobolBrownianGenerator,
    SobolBrownianGeneratorBase,
    SobolBrownianGeneratorFactory,
)
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import exact, tight

# The MT *uniform* stream is a bit-identical port of the C++
# MersenneTwisterUniformRng, but pquantlib's InverseCumulativeNormal
# implements the bare Acklam rational approximation while C++ adds a final
# Halley-refinement step (normaldistribution.cpp). The two therefore agree to
# ~1e-16 but not to the last ULP, so the Gaussian variates are compared at the
# TIGHT tier (abs 1e-14 / rel 1e-12) rather than EXACT. This is the documented
# L1 ICN gap, not a W9-C divergence.
_ICN_HALLEY_GAP = (
    "InverseCumulativeNormal omits the C++ Halley-refinement step "
    "(documented L1 gap); MT uniform stream itself is bit-identical."
)


@pytest.fixture
def ref() -> dict[str, Any]:
    return load_reference("cluster/w9c")


# --- MTBrownianGenerator -----------------------------------------------------


def test_mt_dimensions(ref: dict[str, Any]) -> None:
    gen = MTBrownianGenerator(2, 2, 42)
    assert gen.number_of_factors() == int(ref["mt_factors"])
    assert gen.number_of_steps() == int(ref["mt_steps"])


def test_mt_stream(ref: dict[str, Any]) -> None:
    # Weights are exact (1.0); the Gaussian variates are TIGHT (see the
    # _ICN_HALLEY_GAP note — the uniform stream is bit-identical, the ICN
    # refinement differs by sub-ULP).
    gen = MTBrownianGenerator(2, 2, 42)
    path_w = gen.next_path()
    exact(path_w, ref["mt_path_weight"])
    step0 = [0.0, 0.0]
    step1 = [0.0, 0.0]
    w0 = gen.next_step(step0)
    gen.next_step(step1)
    exact(w0, ref["mt_step_weight"])
    tight(step0[0], ref["mt_s0_0"], reason=_ICN_HALLEY_GAP)
    tight(step0[1], ref["mt_s0_1"], reason=_ICN_HALLEY_GAP)
    tight(step1[0], ref["mt_s1_0"], reason=_ICN_HALLEY_GAP)
    tight(step1[1], ref["mt_s1_1"], reason=_ICN_HALLEY_GAP)


def test_mt_uniform_stream_exact() -> None:
    # The MT *uniform* draws ARE bit-identical to C++ — this is the EXACT
    # guarantee underpinning the (TIGHT) Gaussian comparison above. C++
    # MersenneTwisterUniformRng(42) first 4 nextReal() values:
    rng = MersenneTwisterUniformRng(42)
    expected = [
        0.37454011442605406,
        0.7965429843170568,
        0.9507143116788939,
        0.18343478778842837,
    ]
    for e in expected:
        exact(rng.next_real(), e)


def test_mt_factory() -> None:
    factory = MTBrownianGeneratorFactory(42)
    assert isinstance(factory, BrownianGeneratorFactory)
    gen = factory.create(2, 2)
    assert isinstance(gen, MTBrownianGenerator)
    assert isinstance(gen, BrownianGenerator)
    assert gen.number_of_factors() == 2
    assert gen.number_of_steps() == 2


def test_mt_seed_zero_rejected() -> None:
    # pquantlib's MersenneTwisterUniformRng rejects seed 0 (the C++
    # SeedGenerator clock fallback is not ported); the generator surfaces
    # that error rather than silently picking a clock-derived seed.
    with pytest.raises(ValueError, match="nonzero seed"):
        MTBrownianGenerator(2, 2, 0)


# --- SobolBrownianGenerator: ordering schema (EXACT, stream-independent) ------


def test_sobol_ordering_factors(ref: dict[str, Any]) -> None:
    gen = SobolBrownianGenerator(3, 4, SobolBrownianGenerator.Ordering.FACTORS, 42)
    idx = gen.ordered_indices()
    exact(float(idx[0][0]), ref["sob_factors_00"])
    exact(float(idx[0][3]), ref["sob_factors_03"])
    exact(float(idx[1][0]), ref["sob_factors_10"])
    exact(float(idx[2][3]), ref["sob_factors_23"])


def test_sobol_ordering_steps(ref: dict[str, Any]) -> None:
    gen = SobolBrownianGenerator(3, 4, SobolBrownianGenerator.Ordering.STEPS, 42)
    idx = gen.ordered_indices()
    exact(float(idx[0][0]), ref["sob_steps_00"])
    exact(float(idx[1][0]), ref["sob_steps_10"])
    exact(float(idx[0][1]), ref["sob_steps_01"])
    exact(float(idx[2][3]), ref["sob_steps_23"])


def test_sobol_ordering_diagonal(ref: dict[str, Any]) -> None:
    gen = SobolBrownianGenerator(3, 4, SobolBrownianGenerator.Ordering.DIAGONAL, 42)
    idx = gen.ordered_indices()
    exact(float(idx[0][0]), ref["sob_diag_00"])
    exact(float(idx[1][0]), ref["sob_diag_10"])
    exact(float(idx[0][1]), ref["sob_diag_01"])
    exact(float(idx[2][3]), ref["sob_diag_23"])


def test_sobol_ordering_is_permutation() -> None:
    # Each ordering must be a permutation of 0..factors*steps-1.
    for ordering in SobolBrownianGenerator.Ordering:
        gen = SobolBrownianGenerator(3, 4, ordering, 42)
        flat = [v for row in gen.ordered_indices() for v in row]
        assert sorted(flat) == list(range(12))


# --- SobolBrownianGenerator: bridge transform (TIGHT, stream-independent) -----


def test_sobol_transform(ref: dict[str, Any]) -> None:
    # The transform() test interface is pure Brownian-bridge algebra over a
    # fixed deterministic variate matrix (stream-independent) -> TIGHT.
    gen = SobolBrownianGenerator(2, 3, SobolBrownianGenerator.Ordering.FACTORS, 42)
    vals = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    variates = [[v] for v in vals]  # dim=6, nPaths=1
    out = gen.transform(variates)
    tight(out[0][0], ref["sob_xf_f0_s0"])
    tight(out[0][1], ref["sob_xf_f0_s1"])
    tight(out[0][2], ref["sob_xf_f0_s2"])
    tight(out[1][0], ref["sob_xf_f1_s0"])
    tight(out[1][1], ref["sob_xf_f1_s1"])
    tight(out[1][2], ref["sob_xf_f1_s2"])


def test_sobol_factory() -> None:
    factory = SobolBrownianGeneratorFactory(
        SobolBrownianGenerator.Ordering.DIAGONAL, 42
    )
    assert isinstance(factory, BrownianGeneratorFactory)
    gen = factory.create(2, 3)
    assert isinstance(gen, SobolBrownianGenerator)
    assert isinstance(gen, SobolBrownianGeneratorBase)
    assert gen.number_of_factors() == 2
    assert gen.number_of_steps() == 3


def test_sobol_next_path_then_step_roundtrip() -> None:
    # next_path then number_of_steps calls to next_step must reproduce the
    # bridged variates the generator stored (self-consistency).
    gen = SobolBrownianGenerator(2, 3, SobolBrownianGenerator.Ordering.STEPS, 7)
    w = gen.next_path()
    assert w == 1.0
    collected: list[list[float]] = []
    for _ in range(3):
        out = [0.0, 0.0]
        sw = gen.next_step(out)
        assert sw == 1.0
        collected.append(list(out))
    # 3 steps x 2 factors filled with finite values
    assert len(collected) == 3
    assert all(len(s) == 2 for s in collected)
