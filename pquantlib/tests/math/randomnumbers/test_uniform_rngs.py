"""Cross-validate L1-D uniform RNGs against the C++ probe.

Probe source: migration-harness/cpp/probes/cluster_d/probe.cpp
Reference:    migration-harness/references/cluster/d.json

Each RNG is seeded with 42 (per the probe) and the first 5 ``next().value``
outputs are asserted bit-identical to the C++ reference via ``tolerance.exact``.
RNG sequences are bit-deterministic — agreement at the EXACT tier is the
binding correctness criterion for this cluster.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.math.randomnumbers.knuth import KnuthUniformRng
from pquantlib.math.randomnumbers.lecuyer import LecuyerUniformRng
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.random_number_generator import RandomNumberGenerator, Sample
from pquantlib.math.randomnumbers.ranlux import Ranlux3UniformRng
from pquantlib.math.randomnumbers.xoshiro256_starstar import Xoshiro256StarStarUniformRng
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/d")


# --- protocol / sample sanity ------------------------------------------


def test_sample_defaults_weight_to_one() -> None:
    sample = Sample(value=0.5)
    assert sample.value == 0.5
    assert sample.weight == 1.0


def test_sample_is_immutable() -> None:
    sample = Sample(value=0.5, weight=2.0)
    with pytest.raises(AttributeError):
        sample.value = 1.0  # type: ignore[misc]


def test_all_rngs_satisfy_protocol() -> None:
    assert isinstance(MersenneTwisterUniformRng(seed=42), RandomNumberGenerator)
    assert isinstance(KnuthUniformRng(seed=42), RandomNumberGenerator)
    assert isinstance(LecuyerUniformRng(seed=42), RandomNumberGenerator)
    assert isinstance(Ranlux3UniformRng(seed=42), RandomNumberGenerator)
    assert isinstance(Xoshiro256StarStarUniformRng(seed=42), RandomNumberGenerator)


def test_scalar_rngs_report_dimension_one() -> None:
    assert MersenneTwisterUniformRng(seed=42).dimension() == 1
    assert KnuthUniformRng(seed=42).dimension() == 1
    assert LecuyerUniformRng(seed=42).dimension() == 1
    assert Ranlux3UniformRng(seed=42).dimension() == 1
    assert Xoshiro256StarStarUniformRng(seed=42).dimension() == 1


# --- bit-exact sequence cross-validation -------------------------------


def test_mersenne_twister_matches_cpp(cpp: dict[str, Any]) -> None:
    rng = MersenneTwisterUniformRng(seed=42)
    expected = cpp["mt19937"]
    for exp in expected:
        tolerance.exact(rng.next().value, float(exp))


def test_knuth_matches_cpp(cpp: dict[str, Any]) -> None:
    rng = KnuthUniformRng(seed=42)
    expected = cpp["knuth"]
    for exp in expected:
        tolerance.exact(rng.next().value, float(exp))


def test_lecuyer_matches_cpp(cpp: dict[str, Any]) -> None:
    rng = LecuyerUniformRng(seed=42)
    expected = cpp["lecuyer"]
    for exp in expected:
        tolerance.exact(rng.next().value, float(exp))


def test_ranlux3_matches_cpp(cpp: dict[str, Any]) -> None:
    rng = Ranlux3UniformRng(seed=42)
    expected = cpp["ranlux3"]
    for exp in expected:
        tolerance.exact(rng.next().value, float(exp))


def test_xoshiro256_starstar_matches_cpp(cpp: dict[str, Any]) -> None:
    rng = Xoshiro256StarStarUniformRng(seed=42)
    expected = cpp["xoshiro256ss"]
    for exp in expected:
        tolerance.exact(rng.next().value, float(exp))


# --- seed-0 rejection (pquantlib-specific divergence) ------------------


def test_mersenne_twister_rejects_seed_zero() -> None:
    with pytest.raises(ValueError, match="nonzero seed"):
        MersenneTwisterUniformRng(seed=0)


def test_knuth_rejects_seed_zero() -> None:
    with pytest.raises(ValueError, match="nonzero seed"):
        KnuthUniformRng(seed=0)


def test_lecuyer_rejects_seed_zero() -> None:
    with pytest.raises(ValueError, match="nonzero seed"):
        LecuyerUniformRng(seed=0)


def test_xoshiro256_starstar_rejects_seed_zero() -> None:
    with pytest.raises(ValueError, match="nonzero seed"):
        Xoshiro256StarStarUniformRng(seed=0)


# --- xoshiro from_state -------------------------------------------------


def test_xoshiro256_starstar_from_state_rejects_all_zero() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        Xoshiro256StarStarUniformRng.from_state(0, 0, 0, 0)


def test_xoshiro256_starstar_from_state_is_deterministic() -> None:
    # Two instances built from the same explicit 256-bit state must
    # produce identical sequences (state is the only RNG observable).
    state = (
        0x1234567890ABCDEF,
        0xDEADBEEFCAFEBABE,
        0x0F1E2D3C4B5A6978,
        0xFEDCBA9876543210,
    )
    a = Xoshiro256StarStarUniformRng.from_state(*state)
    b = Xoshiro256StarStarUniformRng.from_state(*state)
    for _ in range(10):
        tolerance.exact(a.next().value, b.next().value)


# --- determinism (re-seeding reproduces the same sequence) -------------


def test_mersenne_twister_re_seeding_is_deterministic() -> None:
    first = [MersenneTwisterUniformRng(seed=42).next().value for _ in range(1)]
    second = [MersenneTwisterUniformRng(seed=42).next().value for _ in range(1)]
    tolerance.exact(first[0], second[0])


def test_lecuyer_buffer_index_does_not_overflow() -> None:
    # Stress: pull 200 samples. The Bays-Durham shuffle index is
    # ``y // bufferNormalizer`` in [0, 31]; if normalizer or m1 are
    # off-by-one we'd index out of the 32-element buffer.
    rng = LecuyerUniformRng(seed=42)
    for _ in range(200):
        sample = rng.next()
        assert 0.0 <= sample.value < 1.0


def test_ranlux3_discard_block_cycle_completes() -> None:
    # Pull 250 samples — more than P = 223, so we cross the
    # block-discard boundary at least once.
    rng = Ranlux3UniformRng(seed=42)
    for _ in range(250):
        sample = rng.next()
        assert 0.0 <= sample.value < 1.0
