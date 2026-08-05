"""Cross-validate BivariateCumulativeStudentDistribution against the v1.43 probe.

Reference: ``migration-harness/references/v143/math/distributions.json`` —
``bivariate_student``. Dunnett & Sobel give two structurally different closed
forms, one for even ``n`` and one for odd ``n``, so both parities are sampled
(n = 1..15), together with rho at both signs and within 1e-6 of +/-1 where the
``f_x`` epsilon guard collapses to the degenerate limit.
"""

from __future__ import annotations

from typing import Any

from pquantlib.math.distributions.bivariate_student_t_distribution import (
    BivariateCumulativeStudentDistribution,
)
from pquantlib.testing import tolerance


def test_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for block in v143["bivariate_student"]:
        f = BivariateCumulativeStudentDistribution(block["n"], block["rho"])
        for x, y, expected in block["cases"]:
            tolerance.tight(f(x, y), expected, reason=f"n={block['n']}, rho={block['rho']}, x={x}, y={y}")


def test_both_parities_are_covered(v143: dict[str, Any]) -> None:
    """The even/odd split is the whole structure of the algorithm; assert the probe keeps it."""
    ns = {block["n"] for block in v143["bivariate_student"]}
    assert any(n % 2 == 0 for n in ns)
    assert any(n % 2 == 1 for n in ns)
    assert 1 in ns, "n == 1 skips the series entirely and must stay covered"


def test_symmetry_in_its_arguments(v143: dict[str, Any]) -> None:
    """P(X<=x, Y<=y) is symmetric under swapping (x, y) — a property the C++
    formula satisfies only because the second and third lines mirror each
    other. A transcription slip in one of them breaks this without breaking
    any single pinned value badly enough to notice.
    """
    for block in v143["bivariate_student"]:
        f = BivariateCumulativeStudentDistribution(block["n"], block["rho"])
        for x, y, _ in block["cases"]:
            tolerance.tight(f(x, y), f(y, x), reason=f"n={block['n']}, rho={block['rho']}")
