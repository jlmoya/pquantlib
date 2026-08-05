"""Cross-validate ``ConvexMonotoneInterpolation`` against C++ v1.43.

Reference: ``migration-harness/references/v143/math/interp/cubic.json``,
``convex_monotone`` section — 86 cases: five curve shapes chosen to reach
every arm of the ``SectionHelper`` dispatch, crossed with eight
``(quadraticity, monotonicity, force_positive)`` settings and both values
of the flat-final-period flag, plus the ``ConvexMonotone`` factory defaults
and the five-step ``localInterpolate`` chain that ``LocalBootstrap`` drives.

The curve shapes and what they are for:

``rates``      increasing discrete forwards — the ordinary case.
``humped``     alternating, so ``g_prev`` and ``g_next`` change sign and the
               2/3-helper branches and the ``eta`` clamps are reached.
``near_zero``  averages that dive towards zero, so the force-positive clamp
               zeroes ``f[0]`` and the ``*MinHelper`` split regions engage.
``flat``       constant averages, i.e. the ``|g| < 1e-14`` ``ConstantGradHelper``.
``two_point``  the ``length == 2`` short-circuit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.convex_monotone_interpolation import (
    ComboHelper,
    ConstantGradHelper,
    ConvexMonotone,
    ConvexMonotone2Helper,
    ConvexMonotone3Helper,
    ConvexMonotone4Helper,
    ConvexMonotone4MinHelper,
    ConvexMonotoneInterpolation,
    EverywhereConstantHelper,
    QuadraticHelper,
    QuadraticMinHelper,
    SectionHelper,
)
from pquantlib.testing import tolerance
from tests.math.interpolations import _interp_reference as ref

_CASES: list[dict[str, Any]] = ref.load()["convex_monotone"]

# The `near_zero` curve with force_positive drives f[0] to exactly 0.0.
# QuadraticMinHelper then computes
#     dAv = bAv^2 - 4*aAv*cAv  with  bAv = -24*(fPrev+fNext),
#                                    cAv = 4*(fPrev^2 + fPrev*fNext + fNext^2)
# whose exact value is 576*fPrev*fNext = 0 — but it is evaluated as the
# difference of two quantities of size 0.1384, so what survives is pure
# rounding noise of order eps*0.1384 ~ 3e-17. `sqrt` maps that to ~7e-9, and
# any difference in how the two runtimes round the subtraction (the reference
# build contracts it into an fma; CPython does not) shows up there at full
# size. Propagating: |d avRoot| <= 7.5e-9 / (2*36) = 1.0e-10, i.e. 2.0e-8
# relative on avRoot ~ 5.2e-3, hence on xRatio and on xVal. The value is
# c + b*xVal + a*xVal^2, so the relative error is at most
# (|b*xVal| + 2|a*xVal^2|)/|value| * 2.0e-8, which at the left-extrapolated
# point (xVal ~ -2.6) is ~5.8 * 2.0e-8 = 1.2e-7.
#
# It only bites outside the range: at and between the pillars the split
# region either returns exactly 0.0 or evaluates at xVal = 1, both of which
# are independent of xRatio.
_CANCELLED_DISCRIMINANT_REL_TOL = 2.0e-7
_CANCELLED_DISCRIMINANT_REASON = (
    "QuadraticMinHelper's auxiliary discriminant is an exact zero computed as "
    "a difference of two ~0.1384 terms (force_positive clamps f[0] to 0.0), so "
    "sqrt() amplifies rounding noise of order eps*0.1384 to ~7e-9; that is "
    "2.0e-8 relative on avRoot/xRatio and, through the quadratic in xVal at "
    "the left-extrapolated point, ~1.2e-7 relative on the value"
)


def _has_cancelled_discriminant(case: dict[str, Any]) -> bool:
    return (
        str(case["curve"]) == "near_zero"
        and bool(case["force_positive"])
        and float(case["quadraticity"]) > 0.0
    )


def _build(case: dict[str, Any]) -> ConvexMonotoneInterpolation:
    xs, ys = ref.curve(case)
    return ConvexMonotoneInterpolation(
        xs,
        ys,
        quadraticity=float(case["quadraticity"]),
        monotonicity=float(case["monotonicity"]),
        force_positive=bool(case["force_positive"]),
        flat_final_period=bool(case["flat_final_period"]),
    )


_PARAM_CASES = [c for c in _CASES if "quadraticity" in c]


@pytest.mark.parametrize("case", _PARAM_CASES, ids=ref.names(_PARAM_CASES))
def test_convex_monotone_case_matches_cpp(case: dict[str, Any]) -> None:
    """Value and primitive match C++ at TIGHT, in and out of range."""
    f = _build(case)
    xs, _ = ref.curve(case)
    x_min = float(xs[0])
    relaxed = _has_cancelled_discriminant(case)
    for key, evaluate in ref.evaluators(f):
        if key in ("derivative", "second_derivative"):
            continue  # C++ QL_FAILs for both
        for raw_x, raw_expected in zip(case["eval_x"], case[key], strict=True):
            x = ref.num(raw_x)
            expected = ref.num(raw_expected)
            actual = evaluate(x)
            if relaxed and x < x_min:
                tolerance.custom(
                    actual,
                    expected,
                    abs_tol=1e-14,
                    rel_tol=_CANCELLED_DISCRIMINANT_REL_TOL,
                    reason=_CANCELLED_DISCRIMINANT_REASON,
                )
            else:
                tolerance.tight(actual, expected)


def test_factory_defaults_match_cpp() -> None:
    """``ConvexMonotone()`` is ``quadraticity=0.3, monotonicity=0.7, force_positive``."""
    case = next(
        c for c in _CASES if c["name"] == "ConvexMonotone.factory.defaults"
    )
    xs, ys = ref.curve(case)
    f = ConvexMonotone().interpolate(xs, ys)
    for raw_x, raw_expected in zip(case["eval_x"], case["value"], strict=True):
        tolerance.tight(f(ref.num(raw_x), allow_extrapolation=True), ref.num(raw_expected))
    for raw_x, raw_expected in zip(case["eval_x"], case["primitive"], strict=True):
        tolerance.tight(
            f.primitive(ref.num(raw_x), allow_extrapolation=True), ref.num(raw_expected)
        )


def test_local_interpolate_chain_matches_cpp() -> None:
    """The ``LocalBootstrap`` growth path, step for step.

    Each step reuses the previous interpolation's settled section helpers and
    holds the final period flat until the curve reaches ``final_size``. A port
    that rebuilt every helper from scratch would still agree on the finished
    curve and disagree here — which is the point.
    """
    chain = [c for c in _CASES if str(c["name"]).startswith("ConvexMonotone.localInterpolate")]
    assert chain, "probe emitted no localInterpolate chain"
    factory = ConvexMonotone(0.3, 0.7, True)
    previous: ConvexMonotoneInterpolation | None = None
    for case in chain:
        xs, ys = ref.curve(case)
        current = factory.local_interpolate(
            xs, ys, int(case["localisation"]), previous, int(case["final_size"])
        )
        for raw_x, raw_expected in zip(case["eval_x"], case["value"], strict=True):
            tolerance.tight(
                current(ref.num(raw_x), allow_extrapolation=True), ref.num(raw_expected)
            )
        for raw_x, raw_expected in zip(case["eval_x"], case["primitive"], strict=True):
            tolerance.tight(
                current.primitive(ref.num(raw_x), allow_extrapolation=True),
                ref.num(raw_expected),
            )
        previous = current


# ---------------------------------------------------------------------------
# the dispatch actually reaches every helper
# ---------------------------------------------------------------------------


def _helper_types_reached() -> set[type]:
    found: set[type] = set()
    for case in _PARAM_CASES:
        f = _build(case)
        for helper in f.section_helpers:
            if helper is not None:
                found.add(type(helper))
        f_extrapolation = f._extrapolation_helper  # pyright: ignore[reportPrivateUsage]
        if f_extrapolation is not None:
            found.add(type(f_extrapolation))
    return found


def test_every_section_helper_is_exercised() -> None:
    """The probe's curve/parameter matrix reaches all nine helper shapes.

    Without this the ``*MinHelper`` classes could be present, untested and
    wrong — which is how they came to be missing in the first place: the
    plain quadratic agrees with ``QuadraticMinHelper`` bit for bit whenever
    the split region is inactive, so an ordinary forward curve cannot tell
    them apart.
    """
    reached = _helper_types_reached()
    for helper_type in (
        EverywhereConstantHelper,
        ConstantGradHelper,
        QuadraticHelper,
        QuadraticMinHelper,
        ConvexMonotone2Helper,
        ConvexMonotone3Helper,
        ConvexMonotone4Helper,
        ConvexMonotone4MinHelper,
        ComboHelper,
    ):
        assert helper_type in reached, f"{helper_type.__name__} never selected"


def test_quadratic_min_helper_splits_where_the_quadratic_would_go_negative() -> None:
    """``QuadraticMinHelper`` differs from ``QuadraticHelper`` only in the split.

    Same coefficients, same value, unless the discriminant conditions hold —
    then the min variant returns exactly 0 over the middle band.
    """
    plain = QuadraticHelper(0.0, 1.0, 0.0, 0.0155, 0.001, 0.0)
    minimal = QuadraticMinHelper(0.0, 1.0, 0.0, 0.0155, 0.001, 0.0)
    # x = 0.5 lands inside the flattened band.
    assert minimal.value(0.5) == 0.0
    assert plain.value(0.5) != 0.0
    # Both still hand back the same right-edge forward.
    tolerance.exact(minimal.f_next(), plain.f_next())


def test_convex_monotone4_min_helper_is_the_plain_helper_when_positive() -> None:
    """No split when ``A + f_average > 0`` — the two agree bit for bit."""
    args = (0.0, 1.0, -0.002, 0.003, 0.02, 0.6, 0.0)
    plain = ConvexMonotone4Helper(*args)
    minimal = ConvexMonotone4MinHelper(*args)
    for x in (0.0, 0.25, 0.6, 0.9, 1.0):
        tolerance.exact(minimal.value(x), plain.value(x))
        tolerance.exact(minimal.primitive(x), plain.primitive(x))


# ---------------------------------------------------------------------------
# construction, invariants and the C++ failure modes
# ---------------------------------------------------------------------------


def _xs_ys() -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float64)
    ys = np.array([0.0, 0.018, 0.020, 0.025, 0.028, 0.030], dtype=np.float64)
    return xs, ys


def test_construction_rejects_bad_quadraticity() -> None:
    xs, ys = _xs_ys()
    with pytest.raises(LibraryException, match="Quadraticity"):
        ConvexMonotoneInterpolation(xs, ys, quadraticity=1.5)


def test_construction_rejects_bad_monotonicity() -> None:
    xs, ys = _xs_ys()
    with pytest.raises(LibraryException, match="Monotonicity"):
        ConvexMonotoneInterpolation(xs, ys, monotonicity=-0.1)


def test_construction_rejects_single_pillar() -> None:
    with pytest.raises(LibraryException):
        ConvexMonotoneInterpolation(np.array([0.0]), np.array([1.0]))


def test_too_many_pre_existing_helpers_is_rejected() -> None:
    """# C++ parity: ``QL_REQUIRE`` at convexmonotoneinterpolation.hpp:208-209."""
    xs, ys = _xs_ys()
    helpers: dict[float, SectionHelper] = {
        float(x): EverywhereConstantHelper(0.02, 0.0, 0.0) for x in xs.tolist()
    }
    with pytest.raises(LibraryException, match="Too many existing helpers"):
        ConvexMonotoneInterpolation(xs, ys, pre_existing_helpers=helpers)


def test_derivative_not_implemented() -> None:
    """C++ ``QL_FAIL`` — the derivative is not computed for this interpolation."""
    xs, ys = _xs_ys()
    interp = ConvexMonotoneInterpolation(xs, ys, quadratic_constraint=False)
    with pytest.raises(NotImplementedError, match="derivative"):
        interp.derivative(1.5)


def test_second_derivative_not_implemented() -> None:
    xs, ys = _xs_ys()
    interp = ConvexMonotoneInterpolation(xs, ys, quadratic_constraint=False)
    with pytest.raises(NotImplementedError, match="second derivative"):
        interp.second_derivative(1.5)


def test_returns_non_negative_when_force_positive() -> None:
    """``force_positive`` keeps the curve off the negative side."""
    xs, ys = _xs_ys()
    interp = ConvexMonotoneInterpolation(
        xs, ys, quadratic_constraint=False, force_positive=True
    )
    for x_val in np.linspace(0.1, float(xs[-1]), 50):
        x = float(x_val)
        assert interp(x) >= -1e-12, f"f({x}) = {interp(x)} < 0"


def test_extrapolation_returns_constant_beyond_last_pillar() -> None:
    """# C++ parity: the extrapolation helper is an ``EverywhereConstantHelper``."""
    xs, ys = _xs_ys()
    interp = ConvexMonotoneInterpolation(xs, ys, quadratic_constraint=False)
    interp.enable_extrapolation(True)
    tolerance.exact(interp(float(xs[-1]) + 5.0), interp(float(xs[-1])))


def test_constant_input_yields_constant_output() -> None:
    """Constant averages collapse to ``ConstantGradHelper`` everywhere."""
    xs = np.array([0.0, 1.0, 2.0, 5.0, 10.0], dtype=np.float64)
    ys = np.array([0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float64)
    interp = ConvexMonotoneInterpolation(
        xs, ys, quadratic_constraint=False, force_positive=False
    )
    for x in (0.5, 1.5, 3.0, 7.0, 9.5):
        assert math.isclose(interp(x), 0.03, abs_tol=1e-12)


def test_quadratic_constraint_flag_selects_the_cpp_factory_defaults() -> None:
    xs, ys = _xs_ys()
    interp = ConvexMonotoneInterpolation(xs, ys, quadratic_constraint=True)
    assert interp.quadraticity == 0.3
    assert interp.monotonicity == 0.7
    other = ConvexMonotoneInterpolation(xs, ys, quadratic_constraint=False)
    assert other.quadraticity == 0.0
    assert other.monotonicity == 1.0


def test_two_point_construction_constant_helper() -> None:
    """# C++ parity: the single-period ``EverywhereConstantHelper`` short-circuit."""
    xs = np.array([0.0, 1.0], dtype=np.float64)
    ys = np.array([0.0, 0.02], dtype=np.float64)
    interp = ConvexMonotoneInterpolation(
        xs, ys, quadratic_constraint=False, force_positive=True
    )
    # y[0] is ignored; the single period takes y[1] as its constant.
    tolerance.exact(interp(0.5), 0.02)
    tolerance.exact(interp(1.0), 0.02)


def test_convex_monotone_traits_match_cpp() -> None:
    doc = ref.load()
    traits = doc["traits"]["ConvexMonotone"]
    assert ConvexMonotone.global_ is bool(traits["global"])
    assert ConvexMonotone.required_points == int(traits["required_points"])
    assert ConvexMonotone.data_size_adjustment == int(
        doc["convex_monotone_data_size_adjustment"]
    )
