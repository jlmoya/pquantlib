"""Cross-validate SobolRsg + Burley2020SobolRsg against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/randomnumbers.json``,
sections ``sobol_rsg`` and ``burley2020_sobol_rsg``.

EXACT tier throughout, and deliberately so. A Sobol generator that agrees to
1e-12 is a different generator: the raw output is a 32-bit integer, the
normalisation is an exact power of two, and every operation in between is
XOR and shift. There is no floating-point rounding anywhere in the pipeline
that could licence a tolerance.

Each case is pinned three ways, because the three failure modes surface at
different points in the sequence:

* the first draws catch a wrong direction-integer table or an off-by-one in
  the constructor's precomputed first vector;
* ``skip_to(10000)`` catches the direct Gray-code evaluation;
* 10 000 *sequential* draws catch the incremental recurrence, which
  ``skip_to`` bypasses entirely — an error in "which bit of the counter do I
  XOR" is invisible for the first few hundred draws.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.randomnumbers.burley_2020_sobol_rsg import Burley2020SobolRsg
from pquantlib.math.randomnumbers.sobol_rsg import (
    PPMT_MAX_DIM,
    DirectionIntegers,
    SobolRsg,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/randomnumbers")


def test_sobol_cases_cover_every_direction_integer_family(cpp: dict[str, Any]) -> None:
    """The probe must exercise all ten C++ families, else the tables go unchecked."""
    families = {c["direction_integers"] for c in cpp["sobol_rsg"]}
    assert families == {d.name for d in DirectionIntegers}


def test_sobol_int_and_real_sequences_exact(cpp: dict[str, Any]) -> None:
    """First draws, as integers and as normalised reals, are bit-identical."""
    for case in cpp["sobol_rsg"]:
        di = DirectionIntegers[case["direction_integers"]]
        g = SobolRsg(case["dimension"], case["seed"], di, case["use_gray_code"])
        for expected in case["int_sequences"]:
            assert g.next_int32_sequence() == expected
        for expected in case["sequences"]:
            actual = g.next_sequence()
            for a, e in zip(actual, expected, strict=True):
                tolerance.exact(float(a), float(e))
        assert g.dimension() == case["dimension_accessor"]


def test_sobol_skip_to_exact(cpp: dict[str, Any]) -> None:
    """``skip_to`` lands on the same point the sequential walk would reach."""
    for case in cpp["sobol_rsg"]:
        di = DirectionIntegers[case["direction_integers"]]
        g = SobolRsg(case["dimension"], case["seed"], di, case["use_gray_code"])
        assert g.skip_to(10000) == case["skip_to_10000"]
        assert g.next_int32_sequence() == case["after_skip_to_10000"]
        actual = g.next_sequence()
        for a, e in zip(actual, case["after_skip_real"], strict=True):
            tolerance.exact(float(a), float(e))


def test_sobol_deep_sequential_draw_exact(cpp: dict[str, Any]) -> None:
    """Draw 10 001 taken one at a time — the recurrence, not the Gray-code jump."""
    for case in cpp["sobol_rsg"]:
        di = DirectionIntegers[case["direction_integers"]]
        g = SobolRsg(case["dimension"], case["seed"], di, case["use_gray_code"])
        for _ in range(10000):
            g.next_int32_sequence()
        assert g.next_int32_sequence() == case["sequential_10000"]


def test_sobol_plain_counter_repeats_its_first_draw(cpp: dict[str, Any]) -> None:
    """With ``use_gray_code=False`` the first two draws coincide — as in C++.

    ``nextInt32Sequence`` calls ``skipTo(sequenceCounter_)`` before advancing
    the counter and skips the advance entirely on the first draw, so draws 1
    and 2 are the same vector. Burley2020 is unaffected because it addresses
    the sequence by an explicit index. Pinned so a "tidy-up" cannot silently
    change the Burley stream.
    """
    plain = [c for c in cpp["sobol_rsg"] if not c["use_gray_code"]]
    assert plain, "probe must contain a use_gray_code=False case"
    for case in plain:
        assert case["int_sequences"][0] == case["int_sequences"][1]
        g = SobolRsg(
            case["dimension"],
            case["seed"],
            DirectionIntegers[case["direction_integers"]],
            False,
        )
        assert g.next_int32_sequence() == g.next_int32_sequence()


def test_burley_sequences_exact(cpp: dict[str, Any]) -> None:
    """Owen-scrambled draws are bit-identical, integers and reals alike."""
    for case in cpp["burley2020_sobol_rsg"]:
        di = DirectionIntegers[case["direction_integers"]]
        g = Burley2020SobolRsg(case["dimension"], case["seed"], di, case["scramble_seed"])
        for expected in case["int_sequences"]:
            assert g.next_int32_sequence() == expected
        for expected in case["sequences"]:
            actual = g.next_sequence()
            for a, e in zip(actual, expected, strict=True):
                tolerance.exact(float(a), float(e))
        assert g.dimension() == case["dimension_accessor"]


def test_burley_skip_to_and_deep_draw_exact(cpp: dict[str, Any]) -> None:
    """The scrambled index path and the sequential path agree with C++."""
    for case in cpp["burley2020_sobol_rsg"]:
        di = DirectionIntegers[case["direction_integers"]]
        g = Burley2020SobolRsg(case["dimension"], case["seed"], di, case["scramble_seed"])
        assert g.skip_to(10000) == case["skip_to_10000"]
        assert g.next_int32_sequence() == case["after_skip_to_10000"]

        h = Burley2020SobolRsg(case["dimension"], case["seed"], di, case["scramble_seed"])
        for _ in range(10000):
            h.next_int32_sequence()
        assert h.next_int32_sequence() == case["sequential_10000"]


def test_burley_actually_scrambles(cpp: dict[str, Any]) -> None:
    """Scrambling must move the point set, not just relabel it."""
    plain = SobolRsg(4, 42, DirectionIntegers.Jaeckel)
    scrambled = Burley2020SobolRsg(4, 42, DirectionIntegers.Jaeckel, 43)
    assert list(plain.next_sequence()) != list(scrambled.next_sequence())


def test_dimensionality_bounds() -> None:
    """# C++ parity: sobolrsg.cpp:78486-78491 — both QL_REQUIREs."""
    with pytest.raises(LibraryException, match="greater than 0"):
        SobolRsg(0)
    with pytest.raises(LibraryException, match="greater than 0"):
        SobolRsg(-1)
    with pytest.raises(LibraryException, match="exceeds the number of available"):
        SobolRsg(PPMT_MAX_DIM + 1)


def test_draws_lie_in_the_unit_interval(cpp: dict[str, Any]) -> None:
    """Structural: every coordinate of every family stays inside (0, 1)."""
    for case in cpp["sobol_rsg"]:
        g = SobolRsg(
            case["dimension"],
            case["seed"],
            DirectionIntegers[case["direction_integers"]],
            case["use_gray_code"],
        )
        for _ in range(20):
            assert all(0.0 <= float(v) < 1.0 for v in g.next_sequence())
