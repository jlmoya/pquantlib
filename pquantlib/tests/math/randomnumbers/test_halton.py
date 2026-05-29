"""Tests for HaltonRsg (low-discrepancy multi-start seed for L10-A)."""

from __future__ import annotations

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.randomnumbers.halton import HaltonRsg
from pquantlib.testing import tolerance


def test_basic_construction_works() -> None:
    rsg = HaltonRsg(dimensionality=2, seed=42, random_start=False, random_shift=False)
    assert rsg.dimension() == 2


def test_deterministic_when_no_random_offsets() -> None:
    """Halton sequence is purely deterministic when no jitter is applied."""
    rsg = HaltonRsg(dimensionality=2, seed=42, random_start=False, random_shift=False)
    s1 = rsg.next_sequence().value
    # First Halton point at counter=1 in 2D: (1/2, 1/3).
    tolerance.tight(s1[0], 0.5)
    tolerance.tight(s1[1], 1.0 / 3.0)


def test_first_few_2d_values_match_known() -> None:
    """First 4 Halton 2D vectors at counter 1-4 are well known."""
    rsg = HaltonRsg(dimensionality=2, seed=42, random_start=False, random_shift=False)
    expected = [
        (1 / 2, 1 / 3),
        (1 / 4, 2 / 3),
        (3 / 4, 1 / 9),
        (1 / 8, 4 / 9),
    ]
    for exp in expected:
        v = rsg.next_sequence().value
        tolerance.tight(v[0], exp[0])
        tolerance.tight(v[1], exp[1])


def test_values_in_unit_interval() -> None:
    """All sequence values are in [0, 1)."""
    rsg = HaltonRsg(dimensionality=4, seed=42, random_start=True, random_shift=True)
    for _ in range(20):
        s = rsg.next_sequence().value
        for v in s:
            assert 0.0 <= v < 1.0


def test_zero_dimensionality_rejected() -> None:
    with pytest.raises(LibraryException, match="dimensionality"):
        HaltonRsg(dimensionality=0)


def test_too_many_dimensions_rejected() -> None:
    """We only ship the first 64 primes."""
    with pytest.raises(LibraryException, match="dimensions"):
        HaltonRsg(dimensionality=65)


def test_last_sequence_matches_next() -> None:
    rsg = HaltonRsg(dimensionality=2, seed=42, random_start=False, random_shift=False)
    s = rsg.next_sequence()
    last = rsg.last_sequence()
    assert s.value == last.value


def test_random_start_changes_sequence() -> None:
    """A non-zero random_start shifts the deterministic prefix."""
    rsg_no = HaltonRsg(dimensionality=2, seed=42, random_start=False, random_shift=False)
    rsg_yes = HaltonRsg(dimensionality=2, seed=42, random_start=True, random_shift=False)
    s_no = rsg_no.next_sequence().value
    s_yes = rsg_yes.next_sequence().value
    # With a non-zero random offset the two sequences will differ.
    assert s_no != s_yes


def test_same_seed_reproducible() -> None:
    """Same seed → same random offsets → same sequence."""
    rsg1 = HaltonRsg(dimensionality=3, seed=42, random_start=True, random_shift=True)
    rsg2 = HaltonRsg(dimensionality=3, seed=42, random_start=True, random_shift=True)
    for _ in range(5):
        v1 = rsg1.next_sequence().value
        v2 = rsg2.next_sequence().value
        assert v1 == v2


def test_sample_has_unit_weight() -> None:
    rsg = HaltonRsg(dimensionality=2, random_start=False)
    sample = rsg.next_sequence()
    assert sample.weight == 1.0


def test_3d_third_axis_uses_base_5() -> None:
    """Third dimension uses prime 5 — first value at counter=1 should be 1/5."""
    rsg = HaltonRsg(dimensionality=3, seed=0, random_start=False, random_shift=False)
    v = rsg.next_sequence().value
    tolerance.tight(v[0], 0.5)  # base 2
    tolerance.tight(v[1], 1.0 / 3.0)  # base 3
    tolerance.tight(v[2], 0.2)  # base 5
