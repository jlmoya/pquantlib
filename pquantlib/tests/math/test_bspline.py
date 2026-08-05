"""Cross-validate BSpline against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/tail.json`` — ``bspline``.
Three knot configurations: uniform linear, a clamped cubic with repeated end
knots, and a non-uniform quadratic. Sampled on the knots themselves (where the
half-open degree-0 support test decides), at midpoints, and outside the span.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.bspline import BSpline
from pquantlib.testing import tolerance


def test_matches_cpp_exactly(v143_tail: dict[str, Any]) -> None:
    """EXACT: the recursion is a fixed sequence of multiply-adds on both sides."""
    for block in v143_tail["bspline"]:
        spline = BSpline(block["p"], block["n"], block["knots"])
        for i, x, expected in block["values"]:
            actual = spline(i, x)
            if expected == "nan":
                assert math.isnan(actual), f"p={block['p']}, i={i}, x={x}"
            else:
                tolerance.exact(actual, expected, reason=f"p={block['p']}, i={i}, x={x}")


def test_repeated_knots_produce_nan_as_in_cpp(v143_tail: dict[str, Any]) -> None:
    """A clamped knot vector divides by a zero knot difference.

    C++ gets IEEE NaN; Python's ``/`` would raise. The port routes the
    recursion's divisions through an IEEE-semantics helper so it agrees. This
    is pinned explicitly because "raises" and "returns NaN" are both plausible
    and only one is C++.
    """
    clamped = [b for b in v143_tail["bspline"] if len(set(b["knots"])) < len(b["knots"])]
    assert clamped, "probe no longer covers a repeated-knot configuration"
    for block in clamped:
        spline = BSpline(block["p"], block["n"], block["knots"])
        nans = [(i, x) for i, x, v in block["values"] if v == "nan"]
        assert nans
        for i, x in nans:
            assert math.isnan(spline(i, x))


def test_degree_zero_support_is_half_open(v143_tail: dict[str, Any]) -> None:
    """N_{i,0} is 1 on [x_i, x_{i+1}) — the right end point belongs to the next
    interval, so the basis sums to 0, not 1, at the last knot.
    """
    block = next(b for b in v143_tail["bspline"] if len(set(b["knots"])) == len(b["knots"]))
    spline = BSpline(block["p"], block["n"], block["knots"])
    last = block["knots"][-1]
    assert sum(spline(i, last) for i in range(block["n"] + 1)) == 0.0


@pytest.mark.parametrize(
    ("p", "n", "knots", "message"),
    [
        (0, 2, [0.0, 1.0, 2.0, 3.0], "lowest degree B-spline has p = 1"),
        (1, 0, [0.0, 1.0, 2.0], "number of control points"),
        (3, 2, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "must have p <= n"),
        (1, 2, [0.0, 1.0, 2.0], "number of knots must equal"),
        (1, 2, [0.0, 2.0, 1.0, 3.0, 4.0], "nondecreasing"),
    ],
)
def test_constructor_validation(p: int, n: int, knots: list[float], message: str) -> None:
    with pytest.raises(LibraryException, match=message):
        BSpline(p, n, knots)


def test_index_beyond_n_raises() -> None:
    spline = BSpline(1, 2, [0.0, 1.0, 2.0, 3.0, 4.0])
    with pytest.raises(LibraryException, match="i must not be greater than n"):
        spline(3, 1.5)
