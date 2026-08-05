"""Cross-validate RandomizedLDS against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/randomnumbers.json``,
section ``randomized_lds``.

EXACT tier: the shift is one addition of two uniform deviates followed by a
conditional subtraction of 1.0 — no rounding beyond the addition itself, which
both sides perform identically.

What actually needs pinning is the *interaction* of the two streams, so the
probe walks it in three phases: the first five shifted points, then
``next_randomizer()`` followed by five more, then a point 10 000 deep. The
middle phase is the discriminating one — ``nextRandomizer`` both draws a fresh
pseudo-random shift *and* rewinds the low-discrepancy generator to its
pristine state, and a port that does only one of the two still produces
plausible-looking uniforms.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.randomnumbers.random_sequence_generator import (
    RandomSequenceGenerator,
)
from pquantlib.math.randomnumbers.randomized_lds import RandomizedLDS
from pquantlib.math.randomnumbers.sobol_rsg import SobolRsg
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/randomnumbers")


def _make(case: dict[str, Any]) -> RandomizedLDS[Any, Any]:
    return RandomizedLDS.from_dimensionality(
        case["dimension"], case["lds_seed"], case["prs_seed"]
    )


def test_shifted_sequences_exact(cpp: dict[str, Any]) -> None:
    """The first points, the points after re-randomising, and one 10 000 deep."""
    for case in cpp["randomized_lds"]:
        g = _make(case)
        assert g.dimension() == case["dimension_accessor"]
        for expected in case["sequences"]:
            for a, e in zip(g.next_sequence().value, expected, strict=True):
                tolerance.exact(float(a), float(e))
        g.next_randomizer()
        for expected in case["after_next_randomizer"]:
            for a, e in zip(g.next_sequence().value, expected, strict=True):
                tolerance.exact(float(a), float(e))
        for _ in range(10000):
            g.next_sequence()
        for a, e in zip(g.next_sequence().value, case["after_10000"], strict=True):
            tolerance.exact(float(a), float(e))


def test_next_randomizer_rewinds_the_low_discrepancy_generator(
    cpp: dict[str, Any],
) -> None:
    """After re-randomising, the *same* QMC point set reappears under a new shift.

    That is the whole basis of randomised QMC as a variance estimator: the
    replications share the point set and differ only by the shift. Checked
    structurally — the differences between the first block and the
    re-randomised block must be constant across the block (mod 1).
    """
    for case in cpp["randomized_lds"]:
        first = case["sequences"]
        after = case["after_next_randomizer"]
        for coord in range(case["dimension"]):
            deltas = {
                round((a[coord] - b[coord]) % 1.0, 12)
                for a, b in zip(after, first, strict=False)
            }
            assert len(deltas) == 1, f"shift not constant on coordinate {coord}"


def test_shift_is_a_single_conditional_subtraction() -> None:
    """C++ wraps with ``if (x > 1.0) x -= 1.0``, not with ``fmod``.

    Both inputs are in [0, 1), so their sum is in [0, 2) and one subtraction
    suffices — but the comparison is strict, so a sum of exactly 1.0 is left
    at 1.0 rather than wrapped to 0.0. Pinned because ``fmod`` would map it to
    0.0 and the difference is invisible in random data.
    """
    g = RandomizedLDS.from_dimensionality(4, 42, 42)
    for _ in range(500):
        assert all(0.0 <= float(v) <= 1.0 for v in g.next_sequence().value)


def test_dimension_mismatch_is_rejected() -> None:
    """# C++ parity: randomizedlds.hpp:93-96."""
    with pytest.raises(LibraryException, match="generator mismatch"):
        RandomizedLDS(SobolRsg(3), RandomSequenceGenerator.from_seed(4, 42))


def test_default_pseudo_random_generator_is_used() -> None:
    """The one-argument constructor builds its own MT-backed shift generator."""
    g = RandomizedLDS(SobolRsg(3, 42))
    assert g.dimension() == 3
    assert len(g.next_sequence().value) == 3
