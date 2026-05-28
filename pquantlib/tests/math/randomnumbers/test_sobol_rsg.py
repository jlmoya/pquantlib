"""Cross-validate SobolRsg + Burley2020SobolRsg.

Reference: ``migration-harness/references/l5a/foundations.json`` —
``sobol_joekuod5_d2_s42`` (matches scipy's Joe-Kuo direction numbers
for dim<=2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.randomnumbers.burley_2020_sobol_rsg import Burley2020SobolRsg
from pquantlib.math.randomnumbers.sobol_rsg import PPMT_MAX_DIM, SobolRsg
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("l5a/foundations")


# -- core SobolRsg ----------------------------------------------------


def test_first_five_match_cpp_dim2(cpp: dict[str, Any]) -> None:
    # C++ first 5 vectors at (dim=2, seed=42) — for d <= 2 every C++
    # direction-integer family produces the same prefix, and scipy's
    # Joe-Kuo agrees on the same prefix. Bit-exact match.
    rsg = SobolRsg(dimensionality=2, seed=42)
    expected = cpp["sobol_joekuod5_d2_s42"]
    for ev in expected:
        v = rsg.next_sequence()
        assert v.shape == (2,)
        for actual, exp in zip(v.tolist(), ev, strict=True):
            tolerance.exact(actual, float(exp))


def test_dimension_accessor() -> None:
    rsg = SobolRsg(dimensionality=5)
    assert rsg.dimension() == 5


def test_last_sequence_returns_last_draw() -> None:
    rsg = SobolRsg(dimensionality=2, seed=42)
    v = rsg.next_sequence()
    last = rsg.last_sequence()
    # Identity is not guaranteed (impl may return cached copy) — value
    # equality is.
    assert np.array_equal(v, last)


def test_last_sequence_before_draw_raises() -> None:
    rsg = SobolRsg(dimensionality=2)
    with pytest.raises(LibraryException, match="before next_sequence"):
        rsg.last_sequence()


def test_skip_to_advances_counter(cpp: dict[str, Any]) -> None:
    # skip_to(3) should make next_sequence return the 4th vector (1-indexed
    # in C++ Gray-code terms: position 3 in the post-origin sequence).
    # Validate by comparing skip_to(2) + next() to the explicit 3rd draw.
    expected = cpp["sobol_joekuod5_d2_s42"]
    rsg = SobolRsg(dimensionality=2, seed=42)
    rsg.skip_to(2)
    v = rsg.next_sequence()
    for actual, exp in zip(v.tolist(), expected[2], strict=True):
        tolerance.exact(actual, float(exp))


def test_reset_restarts_sequence(cpp: dict[str, Any]) -> None:
    rsg = SobolRsg(dimensionality=2, seed=42)
    rsg.next_sequence()
    rsg.next_sequence()
    rsg.reset()
    v = rsg.next_sequence()
    for actual, exp in zip(v.tolist(), cpp["sobol_joekuod5_d2_s42"][0], strict=True):
        tolerance.exact(actual, float(exp))


def test_dimension_validation() -> None:
    with pytest.raises(LibraryException, match=">= 1"):
        SobolRsg(dimensionality=0)
    with pytest.raises(LibraryException, match=">= 1"):
        SobolRsg(dimensionality=-1)
    with pytest.raises(LibraryException, match=f"<= {PPMT_MAX_DIM}"):
        SobolRsg(dimensionality=PPMT_MAX_DIM + 1)


def test_high_dimension_runs() -> None:
    # scipy supports up to ~21201 dimensions via Joe-Kuo; check that
    # the construction succeeds at a non-trivial dim and the output
    # has the right shape.
    rsg = SobolRsg(dimensionality=64, seed=1)
    v = rsg.next_sequence()
    assert v.shape == (64,)
    assert np.all((v >= 0.0) & (v < 1.0))


def test_skip_to_zero_resets() -> None:
    rsg = SobolRsg(dimensionality=2)
    rsg.next_sequence()
    rsg.skip_to(0)
    v = rsg.next_sequence()
    # First post-origin draw should be (0.5, 0.5) for dim=2.
    tolerance.exact(float(v[0]), 0.5)
    tolerance.exact(float(v[1]), 0.5)


def test_skip_to_negative_raises() -> None:
    rsg = SobolRsg(dimensionality=2)
    with pytest.raises(LibraryException, match="n >= 0"):
        rsg.skip_to(-1)


# -- Burley2020 (scrambled) -------------------------------------------


def test_burley_returns_unit_interval() -> None:
    rsg = Burley2020SobolRsg(dimensionality=3, seed=7)
    for _ in range(5):
        v = rsg.next_sequence()
        assert v.shape == (3,)
        assert np.all((v >= 0.0) & (v < 1.0))


def test_burley_is_deterministic_given_seed() -> None:
    rsg_a = Burley2020SobolRsg(dimensionality=3, seed=12345)
    rsg_b = Burley2020SobolRsg(dimensionality=3, seed=12345)
    for _ in range(4):
        va = rsg_a.next_sequence()
        vb = rsg_b.next_sequence()
        # scipy with same seed -> identical scrambled draws.
        for actual, exp in zip(va.tolist(), vb.tolist(), strict=True):
            tolerance.exact(actual, exp)


def test_burley_differs_from_unscrambled() -> None:
    plain = SobolRsg(dimensionality=2, seed=42)
    scrambled = Burley2020SobolRsg(dimensionality=2, seed=42)
    # Burley's first draw will not be (0.5, 0.5) — scrambling shifts
    # the entire sequence by a random offset.
    vp = plain.next_sequence()
    vs = scrambled.next_sequence()
    assert not np.allclose(vp, vs), (
        "Burley2020 must scramble — found identical first draw to plain SobolRsg"
    )


def test_burley_dimension_accessor() -> None:
    rsg = Burley2020SobolRsg(dimensionality=8, seed=1)
    assert rsg.dimension() == 8
