"""Cross-validate the Student-t family against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/distributions.json`` —
``student`` and ``inverse_student``. Degrees of freedom run from 1 (Cauchy,
where the tails are heaviest and the CDF hardest to invert) to 200 (nearly
normal), and ``x`` reaches +/-50 so the heavy tail is pinned, not just the body.
"""

from __future__ import annotations

from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.distributions.student_t_distribution import (
    CumulativeStudentDistribution,
    InverseCumulativeStudent,
    StudentDistribution,
)
from pquantlib.testing import tolerance


def test_pdf_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for block in v143["student"]:
        pdf = StudentDistribution(block["n"])
        for x, expected in block["pdf"]:
            tolerance.tight(pdf(x), expected, reason=f"n={block['n']}, x={x}")


def test_cdf_matches_cpp_tight(v143: dict[str, Any]) -> None:
    for block in v143["student"]:
        cdf = CumulativeStudentDistribution(block["n"])
        for x, expected in block["cdf"]:
            tolerance.tight(cdf(x), expected, reason=f"n={block['n']}, x={x}")


def test_inverse_matches_cpp_tight(v143: dict[str, Any]) -> None:
    """TIGHT even though the iteration only targets 1e-6.

    The accuracy parameter is a *stopping* criterion, not the accuracy of the
    answer: Newton's last step overshoots the target by orders of magnitude,
    and both sides take the same steps from the same start. What is being
    compared is two runs of one iteration, so the agreement is at the level of
    the arithmetic, not of the stopping rule.
    """
    block = v143["inverse_student"]
    for case in block["cases"]:
        inv = InverseCumulativeStudent(case["n"], block["accuracy"], block["max_iterations"])
        tolerance.tight(inv(case["y"]), case["v"], reason=f"n={case['n']}, y={case['y']}")


def test_inverse_actually_inverts_the_cdf(v143: dict[str, Any]) -> None:
    """Independent check that the pinned roots are roots."""
    block = v143["inverse_student"]
    for case in block["cases"]:
        cdf = CumulativeStudentDistribution(case["n"])
        tolerance.custom(
            cdf(case["v"]),
            case["y"],
            abs_tol=block["accuracy"],
            rel_tol=0.0,
            reason="the C++ iteration stops at |F(x) - y| <= accuracy, so that bounds the residual",
        )


def test_non_positive_degrees_of_freedom_raise() -> None:
    with pytest.raises(LibraryException, match="invalid parameter for t-distribution"):
        StudentDistribution(0)
    with pytest.raises(LibraryException, match="invalid parameter for t-distribution"):
        CumulativeStudentDistribution(-1)


def test_inverse_rejects_arguments_outside_the_unit_interval() -> None:
    inv = InverseCumulativeStudent(5)
    with pytest.raises(LibraryException, match="argument out of range"):
        inv(-0.1)
    with pytest.raises(LibraryException, match="argument out of range"):
        inv(1.1)
