"""Cross-validate LatticeRule + LatticeRsg against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/randomnumbers.json``,
sections ``lattice_rule`` and ``lattice_rsg``.

EXACT tier. The generating vectors are integers; the points are
``fmod(i * z / N, 1)`` with ``N`` a power of two, so the division is exact and
so is the ``fmod``.

The probe dumps all 3 600 entries of each of the four families rather than a
checksum. The tables were transcribed mechanically out of a 14 495-line C++
file, and a checksum would happily accept two transposed entries.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.randomnumbers.lattice_rsg import LatticeRsg
from pquantlib.math.randomnumbers.lattice_rules import RULE_LENGTH, LatticeRule
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/randomnumbers")


def test_all_four_rules_are_pinned(cpp: dict[str, Any]) -> None:
    assert {c["label"] for c in cpp["lattice_rule"]} == {"A", "B", "C", "D"}


def test_generating_vectors_exact(cpp: dict[str, Any]) -> None:
    """Every one of the 4 x 3600 entries, not a checksum."""
    for case in cpp["lattice_rule"]:
        z = LatticeRule.get_rule(LatticeRule.Type[case["label"]], 1024)
        assert len(z) == case["size"] == RULE_LENGTH
        for a, e in zip(z, case["z"], strict=True):
            tolerance.exact(float(a), float(e))


def test_rules_are_distinct(cpp: dict[str, Any]) -> None:
    """Four families that all returned the same vector would pass a checksum."""
    vectors = {c["label"]: tuple(c["z"]) for c in cpp["lattice_rule"]}
    assert len(set(vectors.values())) == 4


def test_n_range_is_checked() -> None:
    """# C++ parity: latticerules.cpp:14464 — ``N >= 1024 && N <= pow(2.9,20)``."""
    with pytest.raises(LibraryException, match="N must be between"):
        LatticeRule.get_rule(LatticeRule.Type.A, 1023)
    with pytest.raises(LibraryException, match="N must be between"):
        LatticeRule.get_rule(LatticeRule.Type.A, int(2.9**20) + 1)
    LatticeRule.get_rule(LatticeRule.Type.A, 1024)


def test_lattice_points_exact(cpp: dict[str, Any]) -> None:
    """Sequence points, plus the point after ``skip_to(10000)``."""
    for case in cpp["lattice_rsg"]:
        z = LatticeRule.get_rule(LatticeRule.Type[case["label"]], case["N"])
        g = LatticeRsg(case["dimension"], z, case["N"])
        assert g.dimension() == case["dimension_accessor"]
        for expected in case["sequences"]:
            for a, e in zip(g.next_sequence(), expected, strict=True):
                tolerance.exact(float(a), float(e))
        g.skip_to(10000)
        for a, e in zip(g.next_sequence(), case["after_skip_to_10000"], strict=True):
            tolerance.exact(float(a), float(e))


def test_first_point_is_the_origin(cpp: dict[str, Any]) -> None:
    """The counter starts at 0, so the lattice's first point is (0, ..., 0).

    Worth pinning: a port that "helpfully" started at i = 1 (as Sobol's
    Gray-code counter does) would shift the entire point set by one, and
    every subsequent value would still look plausible.
    """
    for case in cpp["lattice_rsg"]:
        assert all(v == 0.0 for v in case["sequences"][0])
        z = LatticeRule.get_rule(LatticeRule.Type[case["label"]], case["N"])
        g = LatticeRsg(case["dimension"], z, case["N"])
        assert all(float(v) == 0.0 for v in g.next_sequence())


def test_skip_to_is_relative() -> None:
    """``skipTo(n)`` is ``i_ += n`` in C++, despite the "skip to" name.

    So skipping 10 then 10 lands on point 20, not on point 10. Pinned
    because the doc comment says the opposite and a port that follows the
    comment would be wrong.
    """
    z = LatticeRule.get_rule(LatticeRule.Type.A, 1024)
    stepwise = LatticeRsg(3, z, 1024)
    stepwise.skip_to(10)
    stepwise.skip_to(10)
    direct = LatticeRsg(3, z, 1024)
    direct.skip_to(20)
    for a, e in zip(stepwise.next_sequence(), direct.next_sequence(), strict=True):
        tolerance.exact(float(a), float(e))


def test_points_lie_in_the_unit_cube() -> None:
    z = LatticeRule.get_rule(LatticeRule.Type.B, 2048)
    g = LatticeRsg(4, z, 2048)
    for _ in range(200):
        assert all(0.0 <= float(v) < 1.0 for v in g.next_sequence())
