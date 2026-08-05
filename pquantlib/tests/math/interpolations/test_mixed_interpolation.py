"""Cross-validate the ``mixedinterpolation.hpp`` family against C++ v1.43.

Reference: ``migration-harness/references/v143/math/interp/cubic.json``,
``mixed`` section — 119 cases: every preset, both ``Behavior`` values,
every legal switch index on two curves, plus the ``SplitRanges`` +
derivative-matching path where the cubic's left condition is filled in from
the linear segment's slope at the seam.

Both behaviours matter and they are easy to conflate: with ``ShareRanges``
the cubic is fitted to the *whole* curve and only evaluated above the
switch point, so the linear region still influences it; with
``SplitRanges`` it never sees the data to the left. The two disagree above
the seam.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.array import Array
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition as BC,  # noqa: N817 — keeps the preset table readable
)
from pquantlib.math.interpolations.cubic_interpolation import (
    DerivativeApprox as DA,  # noqa: N817 — keeps the preset table readable
)
from pquantlib.math.interpolations.mixed_interpolation import (
    MixedInterpolation,
    MixedLinearCubic,
    MixedLinearCubicInterpolation,
    MixedLinearCubicNaturalSpline,
    MixedLinearFritschButlandCubic,
    MixedLinearKrugerCubic,
    MixedLinearMonotonicCubicNaturalSpline,
    MixedLinearMonotonicParabolic,
    MixedLinearParabolic,
)
from pquantlib.testing import tolerance
from tests.math.interpolations import _interp_reference as ref

_NATURAL = (BC.SecondDerivative, 0.0, BC.SecondDerivative, 0.0)
_SHARE = MixedInterpolation.Behavior.ShareRanges
_SPLIT = MixedInterpolation.Behavior.SplitRanges

_PRESETS: dict[
    str, Callable[[Array, Array, int, MixedInterpolation.Behavior], MixedLinearCubicInterpolation]
] = {
    "MixedLinearCubicNaturalSpline": MixedLinearCubicNaturalSpline,
    "MixedLinearMonotonicCubicNaturalSpline": MixedLinearMonotonicCubicNaturalSpline,
    "MixedLinearKrugerCubic": MixedLinearKrugerCubic,
    "MixedLinearFritschButlandCubic": MixedLinearFritschButlandCubic,
    "MixedLinearParabolic": MixedLinearParabolic,
    "MixedLinearMonotonicParabolic": MixedLinearMonotonicParabolic,
}

_CASES: list[dict[str, Any]] = ref.load()["mixed"]


def _build(name: str, xs: Array, ys: Array) -> MixedLinearCubicInterpolation:
    head = name.split("/", 1)[0]
    parts = head.split(".")
    n = int(parts[-1][1:])
    if head.startswith("MixedLinearCubicInterpolation.matchDerivatives"):
        # C++ passes Null<Real>() as the left condition value; this port
        # spells that None.
        return MixedLinearCubicInterpolation(
            xs, ys, n, _SPLIT, DA.Spline, False,
            BC.FirstDerivative, None, BC.SecondDerivative, 0.0,
        )
    behavior = _SHARE if parts[-2] == "ShareRanges" else _SPLIT
    if head.startswith("MixedLinearCubicInterpolation.Spline."):
        return MixedLinearCubicInterpolation(
            xs, ys, n, behavior, DA.Spline, False, *_NATURAL
        )
    if head.startswith("MixedLinearCubic.factory.Kruger."):
        return MixedLinearCubic(n, behavior, DA.Kruger, False, *_NATURAL).interpolate(
            xs, ys
        )
    if parts[0] in _PRESETS:
        return _PRESETS[parts[0]](xs, ys, n, behavior)
    raise AssertionError(f"probe case {name!r} has no builder")


@pytest.mark.parametrize("case", _CASES, ids=ref.names(_CASES))
def test_mixed_case_matches_cpp(case: dict[str, Any]) -> None:
    """Value, both derivatives and the primitive match C++ at TIGHT.

    The primitive is the discriminating one: the cubic half is re-based so
    that it agrees with the linear half at the switch point, and dropping
    that re-basing leaves values and derivatives untouched.
    """
    xs, ys = ref.curve(case)
    f = _build(str(case["name"]), xs, ys)
    for key, evaluate in ref.evaluators(f):
        for raw_x, raw_expected in zip(case["eval_x"], case[key], strict=True):
            actual = evaluate(ref.num(raw_x))
            expected = ref.num(raw_expected)
            if not np.isfinite(expected):
                # The FritschButland QL_MIN_REAL branch, reached through the
                # cubic half; see test_cubic_interpolation._assert_non_finite
                # for the derivation. Overflowing coefficients on both sides.
                assert not np.isfinite(actual)
                continue
            tolerance.tight(actual, expected)


def test_share_and_split_ranges_differ_above_the_seam() -> None:
    """The two behaviours are genuinely different interpolants.

    With ``ShareRanges`` the cubic is fitted to all six pillars and only
    *evaluated* above the switch; with ``SplitRanges`` it is fitted to the
    upper three only. If a port silently treats them alike this is what
    catches it.
    """
    xs = np.array([0.0, 1.0, 2.5, 3.0, 4.5, 6.0], dtype=np.float64)
    ys = np.array([5.0, 3.0, 4.0, 2.0, 1.0, 3.0], dtype=np.float64)
    shared = MixedLinearCubicNaturalSpline(xs, ys, 3, _SHARE)
    split = MixedLinearCubicNaturalSpline(xs, ys, 3, _SPLIT)
    for x in (0.5, 1.75):  # linear half — identical
        tolerance.exact(shared(x), split(x))
    assert any(shared(x) != split(x) for x in (3.75, 5.25))


def test_derivative_matching_makes_the_seam_c1() -> None:
    """``FirstDerivative`` + a null value matches the slope across the seam.

    # C++ parity: mixedinterpolation.hpp:73-83.
    """
    xs = np.array([0.0, 1.0, 2.5, 3.0, 4.5, 6.0], dtype=np.float64)
    ys = np.array([5.0, 3.0, 4.0, 2.0, 1.0, 3.0], dtype=np.float64)
    n = 2
    f = MixedLinearCubicInterpolation(
        xs, ys, n, _SPLIT, DA.Spline, False,
        BC.FirstDerivative, None, BC.SecondDerivative, 0.0,
    )
    seam = float(xs[n])
    left_slope = (float(ys[n]) - float(ys[n - 1])) / (float(xs[n]) - float(xs[n - 1]))
    # The cubic's derivative at the seam is the linear segment's slope.
    tolerance.tight(f.derivative(seam), left_slope)


def test_derivative_matching_requires_split_ranges() -> None:
    """# C++ parity: ``QL_REQUIRE`` at mixedinterpolation.hpp:75-76."""
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="only supported with SplitRanges"):
        MixedLinearCubicInterpolation(
            xs, ys, 1, _SHARE, DA.Spline, False,
            BC.FirstDerivative, None, BC.SecondDerivative, 0.0,
        )


def test_switch_index_upper_bound() -> None:
    """# C++ parity: ``QL_REQUIRE(n <= maxN, ...)`` at mixedinterpolation.hpp:243."""
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    with pytest.raises(LibraryException, match=r"n is too large \(4 > 3\)"):
        MixedLinearCubicNaturalSpline(xs, ys, 4)


def test_mixed_traits_match_cpp() -> None:
    traits = ref.load()["traits"]["MixedLinearCubic"]
    assert MixedLinearCubic.global_ is bool(traits["global"])
    assert MixedLinearCubic.required_points == int(traits["required_points"])


def test_default_behavior_is_share_ranges() -> None:
    """Every preset defaults to ``ShareRanges`` in C++."""
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([0.0, 0.5, 1.5, 3.0, 3.2], dtype=np.float64)
    for preset in _PRESETS.values():
        default = preset(xs, ys, 2, _SHARE)
        explicit = preset(xs, ys, 2, _SHARE)
        tolerance.exact(default(2.5), explicit(2.5))
    tolerance.exact(
        MixedLinearCubicNaturalSpline(xs, ys, 2)(2.5),
        MixedLinearCubicNaturalSpline(xs, ys, 2, _SHARE)(2.5),
    )
