"""Cross-validate the ``loginterpolation.hpp`` family against C++ v1.43.

Reference: ``migration-harness/references/v143/math/interp/cubic.json``,
``log`` section — 50 cases over a discount-factor-shaped positive curve.

``primitive`` is absent from the reference on purpose: C++ ``QL_FAIL``s for
every member of this family, and so does the port.

As in the cubic file, the ``_BUILDERS`` table is the thing under test —
each entry is the preset argument tuple that gives a C++ class its meaning,
and the two easiest ways to get this family wrong are (a) forgetting that
``LogCubic``'s default ``monotonic`` is ``True`` while ``Cubic``'s is
``False``, and (b) assuming ``FritschButlandLogCubic`` filters the way
``FritschButlandCubic`` does. It does not.
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
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import Linear
from pquantlib.math.interpolations.log_interpolation import (
    FritschButlandLogCubic,
    HarmonicLogCubic,
    KrugerLog,
    KrugerLogCubic,
    KrugerLogMixedLinearCubic,
    LogCubic,
    LogCubicInterpolation,
    LogCubicNaturalSpline,
    LogInterpolation,
    LogLinear,
    LogLinearInterpolation,
    LogMixedLinearCubic,
    LogMixedLinearCubicInterpolation,
    LogMixedLinearCubicNaturalSpline,
    LogParabolic,
    MonotonicLogCubic,
    MonotonicLogCubicNaturalSpline,
    MonotonicLogMixedLinearCubic,
    MonotonicLogParabolic,
)
from pquantlib.math.interpolations.mixed_interpolation import MixedInterpolation
from pquantlib.testing import tolerance
from tests.math.interpolations import _interp_reference as ref

_NATURAL = (BC.SecondDerivative, 0.0, BC.SecondDerivative, 0.0)
_SHARE = MixedInterpolation.Behavior.ShareRanges
_SPLIT = MixedInterpolation.Behavior.SplitRanges

_LOG_CUBIC_DA: dict[str, tuple[DA, bool]] = {
    "Spline": (DA.Spline, False),
    "Spline.monotonic": (DA.Spline, True),
    "Parabolic": (DA.Parabolic, False),
    "Parabolic.monotonic": (DA.Parabolic, True),
    "FritschButland": (DA.FritschButland, False),
    "Kruger": (DA.Kruger, False),
    "Harmonic": (DA.Harmonic, False),
    "Akima": (DA.Akima, False),
}

_PRESETS: dict[str, Callable[[Array, Array], Interpolation]] = {
    "LogCubicNaturalSpline": LogCubicNaturalSpline,
    "MonotonicLogCubicNaturalSpline": MonotonicLogCubicNaturalSpline,
    "KrugerLogCubic": KrugerLogCubic,
    "HarmonicLogCubic": HarmonicLogCubic,
    "FritschButlandLogCubic": FritschButlandLogCubic,
    "LogParabolic": LogParabolic,
    "MonotonicLogParabolic": MonotonicLogParabolic,
}

_CASES: list[dict[str, Any]] = ref.load()["log"]


def _behavior(token: str) -> MixedInterpolation.Behavior:
    return _SHARE if token == "ShareRanges" else _SPLIT


def _build(name: str, xs: Array, ys: Array) -> Interpolation:
    if name == "LogLinearInterpolation":
        return LogLinearInterpolation(xs, ys)
    if name == "LogLinear.factory":
        return LogLinear().interpolate(xs, ys)
    if name.startswith("LogCubicInterpolation."):
        da, monotonic = _LOG_CUBIC_DA[name.split(".", 1)[1]]
        return LogCubicInterpolation(xs, ys, da, monotonic, *_NATURAL)
    if name in _PRESETS:
        return _PRESETS[name](xs, ys)
    if name == "LogCubic.factory.Kruger.defaultMonotonicTrue":
        return LogCubic(DA.Kruger).interpolate(xs, ys)
    if name == "MonotonicLogCubic.factory":
        return MonotonicLogCubic().interpolate(xs, ys)
    if name == "KrugerLog.factory":
        return KrugerLog().interpolate(xs, ys)
    behavior_token, n_token = name.split(".")[-2:]
    behavior = _behavior(behavior_token)
    n = int(n_token[1:])
    if name.startswith("LogMixedLinearCubicInterpolation.Spline."):
        return LogMixedLinearCubicInterpolation(
            xs, ys, n, behavior, DA.Spline, False, *_NATURAL
        )
    if name.startswith("LogMixedLinearCubicNaturalSpline."):
        return LogMixedLinearCubicNaturalSpline(xs, ys, n, behavior)
    if name.startswith("MonotonicLogMixedLinearCubic.factory."):
        return MonotonicLogMixedLinearCubic(n, behavior).interpolate(xs, ys)
    if name.startswith("KrugerLogMixedLinearCubic.factory."):
        return KrugerLogMixedLinearCubic(n, behavior).interpolate(xs, ys)
    if name.startswith("LogMixedLinearCubic.factory.Parabolic.monotonic."):
        return LogMixedLinearCubic(
            n, behavior, DA.Parabolic, True, *_NATURAL
        ).interpolate(xs, ys)
    raise AssertionError(f"probe case {name!r} has no builder")


@pytest.mark.parametrize("case", _CASES, ids=ref.names(_CASES))
def test_log_case_matches_cpp(case: dict[str, Any]) -> None:
    """Value and both derivatives match C++ at TIGHT, in and out of range."""
    xs, ys = ref.curve(case)
    f = _build(str(case["name"]), xs, ys)
    for key, evaluate in ref.evaluators(f):
        if key == "primitive":
            continue  # C++ QL_FAILs for the whole log family
        for raw_x, raw_expected in zip(case["eval_x"], case[key], strict=True):
            tolerance.tight(evaluate(ref.num(raw_x)), ref.num(raw_expected))


def test_primitive_is_unavailable() -> None:
    """C++ ``QL_FAIL("LogInterpolation primitive not implemented")``."""
    xs = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ys = np.array([1.0, 0.9, 0.8], dtype=np.float64)
    f = LogLinearInterpolation(xs, ys)
    with pytest.raises(LibraryException, match="primitive not implemented"):
        f.primitive(0.5)


def test_non_positive_y_is_rejected_naming_the_index() -> None:
    """C++ ``QL_REQUIRE(y > 0, "invalid value (" << y << ") at index " << i)``."""
    xs = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ys = np.array([1.0, 0.0, 1.0], dtype=np.float64)
    with pytest.raises(LibraryException, match=r"invalid value \(0.0\) at index 1"):
        LogLinearInterpolation(xs, ys)


def test_log_linear_hits_knots() -> None:
    """``exp(log(y))`` round-trips the pillars."""
    xs = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    ys = np.array([1.0, 4.0, 16.0], dtype=np.float64)
    f = LogLinearInterpolation(xs, ys)
    for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
        tolerance.tight(f(float(x)), float(y))


def test_log_cubic_default_monotonic_is_true() -> None:
    """``LogCubic``'s default ``monotonic`` is ``True``, unlike ``Cubic``'s.

    Two assertions. First, EXACT equality with the spelled-out
    ``monotonic=True`` form — that is what "the default is True" means.
    Second, that the flag is not decorative: on log-ordinates that a natural
    spline overshoots, the Hyman filter clips two of the five pillar slopes
    and the curve moves, so ``LogCubic(Spline)`` and
    ``LogCubicNaturalSpline`` (``monotonic=False``) are different functions.
    """
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([1.0, 0.2, 0.9, 0.15, 0.8], dtype=np.float64)
    filtered = LogCubic(DA.Spline).interpolate(xs, ys)
    spelled_out = LogCubicInterpolation(xs, ys, DA.Spline, True, *_NATURAL)
    unfiltered = LogCubicNaturalSpline(xs, ys)
    for x in (0.5, 1.5, 2.5, 3.5):
        tolerance.exact(filtered(x), spelled_out(x))
    assert all(
        filtered(x) != unfiltered(x) for x in (0.5, 1.5, 2.5, 3.5)
    ), "monotonic=True and monotonic=False produced the same curve"


def test_log_traits_match_cpp() -> None:
    traits = ref.load()["traits"]
    assert LogLinear.global_ is bool(traits["LogLinear"]["global"])
    assert LogLinear.required_points == int(traits["LogLinear"]["required_points"])
    assert LogCubic.global_ is bool(traits["LogCubic"]["global"])
    assert LogCubic.required_points == int(traits["LogCubic"]["required_points"])
    assert LogMixedLinearCubic.global_ is bool(
        traits["LogMixedLinearCubic"]["global"]
    )
    assert LogMixedLinearCubic.required_points == int(
        traits["LogMixedLinearCubic"]["required_points"]
    )


def test_update_rebuilds_after_mutating_y() -> None:
    """``update()`` re-takes the logs and rebuilds the inner interpolation."""
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([1.0, 0.9, 0.8, 0.7], dtype=np.float64)
    f = LogInterpolation(xs, ys, Linear())
    before = f(1.5)
    f._ys[:] = ys * ys  # pyright: ignore[reportPrivateUsage]
    f.update()
    # log(y^2) = 2 log(y), and the inner scheme is linear in its ordinates,
    # so the exponentiated value squares.
    tolerance.tight(f(1.5), before * before)
