"""Cross-validate the ``Linear`` / ``BackwardFlat`` / ``ForwardFlat`` factories.

Reference: ``migration-harness/references/v143/math/interp/cubic.json``,
``flat`` section, plus the ``traits`` block.

These are the C++ template-argument objects — ``InterpolatedDiscountCurve<
LogLinear>``, ``PiecewiseYieldCurve<Discount, LogLinear>`` and friends name
them, and the bootstrap branches on their ``global`` / ``requiredPoints``
constants. Getting ``BackwardFlat::requiredPoints == 1`` wrong, for example,
would reject a legal single-pillar curve.

The interpolations themselves are also re-pinned here over a non-uniform,
non-monotone grid: the older ``cluster/e`` reference only exercises a
uniform integer grid, on which several off-by-one errors in ``locate`` are
invisible.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.array import Array
from pquantlib.math.interpolations.backward_flat import (
    BackwardFlat,
    BackwardFlatInterpolation,
)
from pquantlib.math.interpolations.forward_flat import (
    ForwardFlat,
    ForwardFlatInterpolation,
)
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import Linear, LinearInterpolation
from pquantlib.testing import tolerance
from tests.math.interpolations import _interp_reference as ref

_CASES: list[dict[str, Any]] = ref.load()["flat"]


def _build(name: str, xs: Array, ys: Array) -> Interpolation:
    if name.startswith("Linear"):
        return Linear().interpolate(xs, ys)
    if name.startswith("BackwardFlat"):
        return BackwardFlat().interpolate(xs, ys)
    if name.startswith("ForwardFlat"):
        return ForwardFlat().interpolate(xs, ys)
    raise AssertionError(f"probe case {name!r} has no builder")


@pytest.mark.parametrize("case", _CASES, ids=ref.names(_CASES))
def test_flat_case_matches_cpp(case: dict[str, Any]) -> None:
    """Value, both derivatives and the primitive match C++ at TIGHT."""
    xs, ys = ref.curve(case)
    f = _build(str(case["name"]), xs, ys)
    for key, evaluate in ref.evaluators(f):
        for raw_x, raw_expected in zip(case["eval_x"], case[key], strict=True):
            tolerance.tight(evaluate(ref.num(raw_x)), ref.num(raw_expected))


@pytest.mark.parametrize(
    ("factory", "key"),
    [
        (Linear, "Linear"),
        (BackwardFlat, "BackwardFlat"),
        (ForwardFlat, "ForwardFlat"),
    ],
    ids=["Linear", "BackwardFlat", "ForwardFlat"],
)
def test_traits_match_cpp(factory: type[Linear], key: str) -> None:
    """``global`` and ``requiredPoints`` — the bootstrap reads both."""
    traits = ref.load()["traits"][key]
    assert factory.global_ is bool(traits["global"])
    assert factory.required_points == int(traits["required_points"])


def test_factories_return_the_matching_interpolation() -> None:
    xs = np.array([0.0, 1.0, 2.5, 3.0], dtype=np.float64)
    ys = np.array([5.0, 3.0, 4.0, 2.0], dtype=np.float64)
    assert isinstance(Linear().interpolate(xs, ys), LinearInterpolation)
    assert isinstance(BackwardFlat().interpolate(xs, ys), BackwardFlatInterpolation)
    assert isinstance(ForwardFlat().interpolate(xs, ys), ForwardFlatInterpolation)


def test_backward_flat_accepts_a_single_pillar() -> None:
    """``BackwardFlat::requiredPoints == 1`` — the others need two."""
    single_x = np.array([2.0], dtype=np.float64)
    single_y = np.array([0.75], dtype=np.float64)
    f = BackwardFlat().interpolate(single_x, single_y)
    tolerance.exact(f(1.0, allow_extrapolation=True), 0.75)
    tolerance.exact(f(3.5, allow_extrapolation=True), 0.75)
    # ...and the primitive is the constant times the offset from the pillar.
    tolerance.exact(f.primitive(3.5, allow_extrapolation=True), 1.5 * 0.75)
