"""Tests for the Gaussian-sequence generator stack.

# C++ parity: ql/math/randomnumbers/randomsequencegenerator.hpp +
# inversecumulativersg.hpp + rngtraits.hpp (v1.42.1).
"""

from __future__ import annotations

import math

import pytest

from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.methods.montecarlo.gaussian_sequence_generator import (
    InverseCumulativeNormalRsg,
    UniformRandomSequenceGenerator,
    make_pseudo_random_rsg,
)


def test_uniform_rsg_dimension_and_draws() -> None:
    urng = MersenneTwisterUniformRng(42)
    rsg = UniformRandomSequenceGenerator(5, urng)
    assert rsg.dimension() == 5
    sample = rsg.next_sequence()
    assert sample.value.shape == (5,)
    # Uniforms must be in (0, 1).
    for v in sample.value:
        assert 0.0 < v < 1.0
    assert sample.weight == 1.0


def test_uniform_rsg_last_sequence_caches() -> None:
    urng = MersenneTwisterUniformRng(42)
    rsg = UniformRandomSequenceGenerator(3, urng)
    s1 = rsg.next_sequence()
    s2 = rsg.last_sequence()
    # Same values for the same draw.
    for a, b in zip(s1.value, s2.value, strict=True):
        assert a == b


def test_inverse_cumulative_normal_rsg_gaussianizes() -> None:
    urng = MersenneTwisterUniformRng(42)
    ursg = UniformRandomSequenceGenerator(10000, urng)
    gsg = InverseCumulativeNormalRsg(ursg)
    seq = gsg.next_sequence().value
    # Empirical mean/std of 10000 standard normals should be close to (0, 1).
    mean = sum(seq) / len(seq)
    var = sum((x - mean) ** 2 for x in seq) / len(seq)
    # ~3-sigma band, sigma_mean ~= 1/sqrt(10000) = 0.01
    assert abs(mean) < 0.05
    assert abs(math.sqrt(var) - 1.0) < 0.05


def test_make_pseudo_random_rsg_factory() -> None:
    gsg = make_pseudo_random_rsg(5, 42)
    assert gsg.dimension() == 5
    seq = gsg.next_sequence().value
    assert seq.shape == (5,)


def test_make_pseudo_random_rsg_seed0_rejected() -> None:
    with pytest.raises(Exception, match="seed"):
        make_pseudo_random_rsg(5, 0)
