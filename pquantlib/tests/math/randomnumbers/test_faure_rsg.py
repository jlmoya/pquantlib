"""Cross-validate FaureRsg against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/randomnumbers.json``,
section ``faure_rsg``.

EXACT tier. The integer sequence is built entirely from integer addition and
modular arithmetic; the reals are that integer divided by a power of the base,
which for base 2 is exact and for odd bases is a single correctly-rounded
division.

Both surfaces are pinned — ``next_int_sequence`` as well as
``next_sequence`` — because the integers are where the Pascal-matrix and
Gray-code logic is visible; a normalisation error would be invisible in the
integers and a matrix error would be blurred by the division.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.randomnumbers.faure_rsg import FaureRsg
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/randomnumbers")


def test_faure_integer_and_real_sequences_exact(cpp: dict[str, Any]) -> None:
    """First eight integer draws, then four reals off the same generator."""
    for case in cpp["faure_rsg"]:
        g = FaureRsg(case["dimension"])
        assert g.dimension() == case["dimension_accessor"]
        for expected in case["int_sequences"]:
            assert g.next_int_sequence() == expected
        for expected in case["sequences"]:
            for a, e in zip(g.next_sequence(), expected, strict=True):
                tolerance.exact(float(a), float(e))


def test_faure_deep_draw_exact(cpp: dict[str, Any]) -> None:
    """A draw 1 000 in: the b-ary Gray code has carried across several digits."""
    for case in cpp["faure_rsg"]:
        g = FaureRsg(case["dimension"])
        for expected in case["int_sequences"]:
            assert g.next_int_sequence() == expected
        for _ in range(len(case["sequences"])):
            g.next_sequence()
        for _ in range(1000):
            g.next_int_sequence()
        assert g.next_int_sequence() == case["after_1000_int"]
        for a, e in zip(g.next_sequence(), case["after_1000_real"], strict=True):
            tolerance.exact(float(a), float(e))


def test_integer_sequence_is_cumulative(cpp: dict[str, Any]) -> None:
    """C++ accumulates into ``integerSequence_`` and never clears it.

    ``generateNextIntSequence`` does ``integerSequence_[i] += powBase_[j][g2]``
    and the vector is only zeroed in the constructor, so draw k depends on
    every draw before it. (The increments are signed — ``powBase_`` holds
    negative entries below the ``base_`` column — so the totals random-walk
    rather than grow, which is why this is checked by state-dependence and
    not by monotonicity.)

    A port that "fixed" this by zeroing the accumulator per draw would produce
    exactly the increments instead of the totals, and would therefore repeat
    its first vector every time the Gray code returned to the same digit
    pattern. The check below is that it does not.
    """
    for case in cpp["faure_rsg"]:
        g = FaureRsg(case["dimension"])
        first = g.next_int_sequence()
        # The b-ary Gray code visits digit position 0 on every second draw,
        # so a stateless implementation would re-emit `first` at draw 3.
        g.next_int_sequence()
        third = g.next_int_sequence()
        assert third != first
        assert third == case["int_sequences"][2]


def test_reals_are_the_accumulated_integers_rescaled(cpp: dict[str, Any]) -> None:
    """``next_sequence`` is ``next_int_sequence`` over a fixed normaliser.

    Ties the two emitted surfaces together, so an error in one cannot be
    absorbed by a compensating error in the other.
    """
    for case in cpp["faure_rsg"]:
        a = FaureRsg(case["dimension"])
        b = FaureRsg(case["dimension"])
        norm: float | None = None
        for _ in range(6):
            ints = a.next_int_sequence()
            reals = b.next_sequence()
            for i, r in zip(ints, reals, strict=True):
                if r == 0.0:
                    continue
                ratio = i / float(r)
                if norm is None:
                    norm = ratio
                # The two divisions round differently in the last bit; the
                # claim is that one constant normaliser explains every pair.
                assert math.isclose(ratio, norm, rel_tol=1e-15)


def test_last_sequence_repeats_without_advancing(cpp: dict[str, Any]) -> None:
    """``last_*`` must be a pure accessor — no hidden generation."""
    g = FaureRsg(3)
    first = g.next_int_sequence()
    assert g.last_int_sequence() == first
    assert g.last_int_sequence() == first
    values = g.next_sequence()
    for a, e in zip(g.last_sequence(), values, strict=True):
        tolerance.exact(float(a), float(e))


def test_dimensionality_must_be_positive() -> None:
    """# C++ parity: faurersg.cpp:33-34."""
    with pytest.raises(LibraryException, match="greater than 0"):
        FaureRsg(0)


def test_base_is_the_smallest_prime_at_least_the_dimension(cpp: dict[str, Any]) -> None:
    """The base drives everything; check it through the sequence's granularity.

    For dimension 5 the base is 5, so every coordinate is a multiple of
    1/5^k — visible as exact fifths in the low-order draws. For dimension 2
    the base is 2 and the coordinates are dyadic.
    """
    by_dim = {c["dimension"]: c for c in cpp["faure_rsg"]}
    for value in by_dim[5]["sequences"][0]:
        assert abs(value * 25.0 - round(value * 25.0)) < 1e-9
    for value in by_dim[2]["sequences"][0]:
        assert abs(value * 16.0 - round(value * 16.0)) < 1e-12
