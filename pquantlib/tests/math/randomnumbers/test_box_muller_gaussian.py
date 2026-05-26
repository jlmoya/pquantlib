"""Cross-validate BoxMullerGaussianRng over MT19937 against the C++ probe.

Probe source: migration-harness/cpp/probes/cluster_d/probe.cpp
Reference:    migration-harness/references/cluster/d.json key ``box_muller_mt``.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.math.randomnumbers.box_muller_gaussian import BoxMullerGaussianRng
from pquantlib.math.randomnumbers.mersenne_twister import MersenneTwisterUniformRng
from pquantlib.math.randomnumbers.random_number_generator import (
    RandomNumberGenerator,
    Sample,
)
from pquantlib.testing import reference_reader, tolerance


class _CountingUniformRng:
    """Wrap any uniform RNG and count ``next()`` calls — test-only."""

    def __init__(self, base: RandomNumberGenerator) -> None:
        self._base = base
        self.calls = 0

    def next(self) -> Sample:
        self.calls += 1
        return self._base.next()

    def dimension(self) -> int:
        return 1


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("cluster/d")


def test_box_muller_over_mt_matches_cpp(cpp: dict[str, Any]) -> None:
    uniform = MersenneTwisterUniformRng(seed=42)
    rng = BoxMullerGaussianRng(uniform)
    expected = cpp["box_muller_mt"]
    for exp in expected:
        tolerance.exact(rng.next().value, float(exp))


def test_box_muller_alternates_pair_caching() -> None:
    # First call consumes a pair of uniforms (or more, if Marsaglia's
    # polar rejection rolls a sample with r >= 1) and returns the
    # first half. Second call must not consume any uniforms — it
    # emits the cached second half.
    counting = _CountingUniformRng(MersenneTwisterUniformRng(seed=42))
    rng = BoxMullerGaussianRng(counting)
    rng.next()  # may consume 2, 4, 6... uniforms
    pre_calls = counting.calls
    rng.next()  # zero additional uniform calls
    assert counting.calls == pre_calls


def test_box_muller_satisfies_rng_protocol() -> None:
    uniform = MersenneTwisterUniformRng(seed=42)
    rng = BoxMullerGaussianRng(uniform)
    assert isinstance(rng, RandomNumberGenerator)


def test_box_muller_weight_is_product_of_uniform_weights() -> None:
    # MT19937 emits weight 1.0; the product is 1.0.
    uniform = MersenneTwisterUniformRng(seed=42)
    rng = BoxMullerGaussianRng(uniform)
    sample = rng.next()
    assert sample.weight == 1.0
