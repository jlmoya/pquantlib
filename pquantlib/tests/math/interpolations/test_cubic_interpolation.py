"""Cross-validate the ``CubicInterpolation`` family against C++ v1.43.

Reference: ``migration-harness/references/v143/math/interp/cubic.json``,
``cubic`` section — 76 cases covering every ``DerivativeApprox`` crossed
with every usable ``BoundaryCondition``, every convenience preset, and the
2-, 3-, 5- and 6-point corners of the derivative stencils.

The ``_SPEC`` table below is not scaffolding: each entry is the
``(DerivativeApprox, monotonic, leftCondition, leftValue, rightCondition,
rightValue)`` tuple that a C++ class or call site actually uses, and getting
one of them wrong is exactly the failure this file exists to catch. It is
transcribed from cubicinterpolation.hpp and from the probe's own
constructor calls.

The older ``cluster/l9a`` assertions are kept at the end: they pin the same
two splines against an independently generated reference, so a regression
would have to fool both.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition as BC,  # noqa: N817 — keeps the preset table readable
)
from pquantlib.math.interpolations.cubic_interpolation import (
    Cubic,
    CubicInterpolation,
    CubicNaturalSpline,
    CubicSplineOvershootingMinimization1,
    CubicSplineOvershootingMinimization2,
    FritschButlandCubic,
    HarmonicCubic,
    KrugerCubic,
    MonotonicCubicNaturalSpline,
    MonotonicParabolic,
    Parabolic,
)
from pquantlib.math.interpolations.cubic_interpolation import (
    DerivativeApprox as DA,  # noqa: N817 — keeps the preset table readable
)
from pquantlib.testing import reference_reader, tolerance
from tests.math.interpolations import _interp_reference as ref

_NATURAL = (BC.SecondDerivative, 0.0, BC.SecondDerivative, 0.0)

#: Probe case name (before the ``/curve`` suffix) -> constructor arguments.
_SPEC: dict[str, tuple[DA, bool, BC, float, BC, float]] = {
    # --- CubicInterpolation, spelled out --------------------------------
    "Spline.Natural": (DA.Spline, False, *_NATURAL),
    "Spline.Natural.monotonic": (DA.Spline, True, *_NATURAL),
    "Spline.SecondDerivative.nonzero": (
        DA.Spline, False, BC.SecondDerivative, 0.3, BC.SecondDerivative, -0.2,
    ),
    "Spline.FirstDerivative": (
        DA.Spline, False, BC.FirstDerivative, 0.5, BC.FirstDerivative, -0.25,
    ),
    # C++ ignores the end-condition value for NotAKnot; the probe passes
    # deliberate junk (123.0 / -7.0) to prove it.
    "Spline.NotAKnot": (DA.Spline, False, BC.NotAKnot, 123.0, BC.NotAKnot, -7.0),
    "Spline.Lagrange": (DA.Spline, False, BC.Lagrange, 0.0, BC.Lagrange, 0.0),
    "Spline.NotAKnot.left.FirstDerivative.right": (
        DA.Spline, False, BC.NotAKnot, 0.0, BC.FirstDerivative, -0.4,
    ),
    "Spline.Lagrange.left.NotAKnot.right": (
        DA.Spline, False, BC.Lagrange, 0.0, BC.NotAKnot, 0.0,
    ),
    "SplineOM1": (DA.SplineOM1, False, *_NATURAL),
    "SplineOM2": (DA.SplineOM2, False, *_NATURAL),
    "Parabolic": (DA.Parabolic, False, *_NATURAL),
    "Parabolic.monotonic": (DA.Parabolic, True, *_NATURAL),
    "FritschButland": (DA.FritschButland, False, *_NATURAL),
    "FritschButland.monotonic": (DA.FritschButland, True, *_NATURAL),
    "Akima": (DA.Akima, False, *_NATURAL),
    "Akima.monotonic": (DA.Akima, True, *_NATURAL),
    "Kruger": (DA.Kruger, False, *_NATURAL),
    "Kruger.monotonic": (DA.Kruger, True, *_NATURAL),
    "Harmonic": (DA.Harmonic, False, *_NATURAL),
    "Harmonic.monotonic": (DA.Harmonic, True, *_NATURAL),
    "Spline.FirstDerivative/pair": (
        DA.Spline, False, BC.FirstDerivative, 1.0, BC.FirstDerivative, 2.0,
    ),
    # --- the convenience presets: each tuple IS the class ----------------
    "CubicNaturalSpline": (DA.Spline, False, *_NATURAL),
    "MonotonicCubicNaturalSpline": (DA.Spline, True, *_NATURAL),
    "CubicSplineOvershootingMinimization1": (DA.SplineOM1, False, *_NATURAL),
    "CubicSplineOvershootingMinimization2": (DA.SplineOM2, False, *_NATURAL),
    "AkimaCubicInterpolation": (DA.Akima, False, *_NATURAL),
    "KrugerCubic": (DA.Kruger, False, *_NATURAL),
    "HarmonicCubic": (DA.Harmonic, False, *_NATURAL),
    "FritschButlandCubic": (DA.FritschButland, True, *_NATURAL),
    "MonotonicParabolic": (DA.Parabolic, True, *_NATURAL),
    # --- the Cubic factory's defaults ------------------------------------
    "Cubic.factory.Kruger": (DA.Kruger, False, *_NATURAL),
    "Cubic.factory.defaults": (DA.Kruger, False, *_NATURAL),
}

_CASES: list[dict[str, Any]] = ref.load()["cubic"]


def _spec_for(name: str) -> tuple[DA, bool, BC, float, BC, float]:
    if name in _SPEC:
        return _SPEC[name]
    head = name.split("/", 1)[0]
    if head in _SPEC:
        return _SPEC[head]
    raise AssertionError(f"probe case {name!r} has no constructor spec")


def _build(case: dict[str, Any]) -> CubicInterpolation:
    xs, ys = ref.curve(case)
    return CubicInterpolation(xs, ys, *_spec_for(str(case["name"])))


def _assert_non_finite(actual: float, expected: float, coeffs: list[float]) -> None:
    """Compare a non-finite reference value.

    C++ produces non-finite values here only through the FritschButland
    ``QL_MIN_REAL`` / ``QL_MAX_REAL`` branch (cubicinterpolation.hpp:598-605),
    which pushes ``|b|`` past ``DBL_MAX / 2`` and ``|c|`` past ``DBL_MAX / 6``.

    Where no intermediate overflows, the port must agree exactly — nan for
    nan, ``+-inf`` for ``+-inf``.

    Where one does, the two sides can pick different non-finite values from
    the *same* arithmetic. ``derivative`` is ``a + (2*b + 3*c*h)*h``: the
    reference build (clang on arm64, ``-ffp-contract=fast`` by default)
    folds the sum into an ``fmadd``, so ``2*b`` is never rounded and the
    result is ``-inf``; CPython does not contract, so ``2*b`` rounds to
    ``+inf`` first and ``inf + -inf`` is ``nan``. Neither is more correct;
    the coefficients are garbage either way. So the requirement there is
    only that the port also refuses to return a number.
    """
    overflow = any(
        not np.isfinite(2.0 * c) or not np.isfinite(6.0 * c) for c in coeffs
    )
    if not overflow:
        assert actual == expected or (np.isnan(actual) and np.isnan(expected)), (
            f"non-finite mismatch with no overflowing intermediate: "
            f"{actual} vs {expected}"
        )
        return
    assert not np.isfinite(actual), (
        f"C++ overflowed to {expected} but the port returned the finite {actual}"
    )


@pytest.mark.parametrize("case", _CASES, ids=ref.names(_CASES))
def test_cubic_case_matches_cpp(case: dict[str, Any]) -> None:
    """Value, both derivatives and the primitive match C++ at TIGHT.

    Evaluated below the range, at every node, at every midpoint and above
    the range — extrapolation enabled, as the probe does.
    """
    f = _build(case)
    coeff_rows = list(
        zip(
            case.get("a", []), case.get("b", []), case.get("c", []), strict=True
        )
    )
    for key, evaluate in ref.evaluators(f):
        for raw_x, raw_expected in zip(case["eval_x"], case[key], strict=True):
            x = ref.num(raw_x)
            expected = ref.num(raw_expected)
            actual = evaluate(x)
            if not np.isfinite(expected):
                interval = min(max(f._locate(x), 0), len(coeff_rows) - 1)  # pyright: ignore[reportPrivateUsage]
                _assert_non_finite(
                    actual,
                    expected,
                    [ref.num(v) for v in coeff_rows[interval]] if coeff_rows else [],
                )
                continue
            tolerance.tight(actual, expected)


@pytest.mark.parametrize(
    "case", [c for c in _CASES if "a" in c], ids=ref.names([c for c in _CASES if "a" in c])
)
def test_cubic_coefficients_match_cpp(case: dict[str, Any]) -> None:
    """The per-interval coefficients and the monotonicity flags match C++.

    This is the discriminating check: ``a``/``b``/``c`` *are* the output of
    the derivative-approximation + boundary-condition + Hyman-filter
    pipeline, so a wrong preset shows up here directly instead of being
    diluted into a value.
    """
    f = _build(case)
    for key, got in (
        ("a", f.a_coefficients()),
        ("b", f.b_coefficients()),
        ("c", f.c_coefficients()),
        ("primitive_const", f.primitive_constants()),
    ):
        for actual, raw_expected in zip(got, case[key], strict=True):
            expected = ref.num(raw_expected)
            if not np.isfinite(expected):
                assert actual == expected or (
                    np.isnan(actual) and np.isnan(expected)
                ), f"{key}: {actual} vs {expected}"
                continue
            tolerance.tight(actual, expected)
    assert f.monotonicity_adjustments() == list(case["monotonicity_adjustments"])


# ---------------------------------------------------------------------------
# the convenience presets must be exactly their spelled-out equivalent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("preset", "spec"),
    [
        (CubicNaturalSpline, _SPEC["CubicNaturalSpline"]),
        (MonotonicCubicNaturalSpline, _SPEC["MonotonicCubicNaturalSpline"]),
        (
            CubicSplineOvershootingMinimization1,
            _SPEC["CubicSplineOvershootingMinimization1"],
        ),
        (
            CubicSplineOvershootingMinimization2,
            _SPEC["CubicSplineOvershootingMinimization2"],
        ),
        (KrugerCubic, _SPEC["KrugerCubic"]),
        (HarmonicCubic, _SPEC["HarmonicCubic"]),
        (FritschButlandCubic, _SPEC["FritschButlandCubic"]),
        (Parabolic, _SPEC["Parabolic"]),
        (MonotonicParabolic, _SPEC["MonotonicParabolic"]),
    ],
    ids=lambda v: getattr(v, "__name__", ""),
)
def test_preset_equals_its_spelled_out_form(
    preset: type[CubicInterpolation], spec: tuple[DA, bool, BC, float, BC, float]
) -> None:
    """EXACT: a preset is a fixed argument tuple, not an approximation of one."""
    xs = np.array([0.0, 1.0, 2.5, 3.0, 4.5, 6.0], dtype=np.float64)
    ys = np.array([5.0, 3.0, 4.0, 2.0, 1.0, 3.0], dtype=np.float64)
    a = preset(xs, ys)
    b = CubicInterpolation(xs, ys, *spec)
    for x in (-0.5, 0.5, 2.0, 3.75, 6.5):
        tolerance.exact(a(x, allow_extrapolation=True), b(x, allow_extrapolation=True))


def test_cubic_factory_defaults_are_kruger_natural() -> None:
    """C++ ``Cubic``'s defaults are Kruger + non-monotonic + natural."""
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([0.0, 0.5, 1.5, 3.0, 3.2], dtype=np.float64)
    a = Cubic().interpolate(xs, ys)
    b = KrugerCubic(xs, ys)
    for x in (0.25, 1.5, 3.7):
        tolerance.exact(a(x), b(x))


def test_cubic_traits_match_cpp() -> None:
    """``global`` and ``requiredPoints`` are read by the bootstrap machinery."""
    traits = ref.load()["traits"]["Cubic"]
    assert Cubic.global_ is bool(traits["global"])
    assert Cubic.required_points == int(traits["required_points"])


# ---------------------------------------------------------------------------
# failure modes that C++ has too
# ---------------------------------------------------------------------------


def test_periodic_boundary_condition_is_unimplemented() -> None:
    """C++ ``QL_FAIL("this end condition is not implemented yet")``."""
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="not implemented yet"):
        CubicInterpolation(
            xs, ys, DA.Spline, False, BC.Periodic, 0.0, BC.SecondDerivative, 0.0
        )
    with pytest.raises(LibraryException, match="not implemented yet"):
        CubicInterpolation(
            xs, ys, DA.Spline, False, BC.SecondDerivative, 0.0, BC.Periodic, 0.0
        )


def test_fourth_order_is_unimplemented() -> None:
    """C++ ``QL_FAIL("FourthOrder not implemented yet")``."""
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="FourthOrder not implemented yet"):
        CubicInterpolation(xs, ys, DA.FourthOrder, False, *_NATURAL)


def test_lagrange_needs_four_points() -> None:
    """C++ ``QL_REQUIRE`` at cubicinterpolation.hpp:397-402."""
    xs = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="Lagrange boundary condition requires"):
        CubicInterpolation(xs, ys, DA.Spline, False, BC.Lagrange, 0.0, BC.Lagrange, 0.0)


def test_akima_needs_four_points() -> None:
    """C++ ``QL_REQUIRE`` at cubicinterpolation.hpp:403-407."""
    xs = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="Akima approximation requires"):
        CubicInterpolation(xs, ys, DA.Akima, False, *_NATURAL)


def test_not_a_knot_both_ends_with_three_points_is_singular() -> None:
    """C++ throws here too, from inside the tridiagonal elimination.

    With three points and a not-a-knot condition at both ends the system has
    no unique solution: the elimination reaches ``bet == 0`` on the last row
    and ``TridiagonalOperator::solveFor`` raises "division by zero". The
    probe deliberately does not pin this case; the port must fail the same
    way rather than return a number.
    """
    xs = np.array([0.0, 1.5, 4.0], dtype=np.float64)
    ys = np.array([1.0, 2.5, 2.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="division by zero"):
        CubicInterpolation(xs, ys, DA.Spline, False, BC.NotAKnot, 0.0, BC.NotAKnot, 0.0)


def test_not_a_knot_with_two_points_is_refused() -> None:
    """Documented divergence: C++ reads out of bounds, this port refuses.

    ``dx_[1]`` (left arm) and ``dx_[n-3]`` (right arm) index past the end of
    a length-``n-1`` vector when ``n == 2``. C++'s ``std::vector::operator[]``
    reads whatever is there; Python's negative indexing would wrap around
    and silently produce a plausible wrong answer, which is worse.
    """
    xs = np.array([0.0, 2.0], dtype=np.float64)
    ys = np.array([1.0, 4.0], dtype=np.float64)
    with pytest.raises(LibraryException, match="NotAKnot boundary condition requires"):
        CubicInterpolation(xs, ys, DA.Spline, False, BC.NotAKnot, 0.0, BC.NotAKnot, 0.0)


def test_cubic_interpolation_extrapolation_guard() -> None:
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    interp = CubicNaturalSpline(xs, ys)
    with pytest.raises(LibraryException, match="extrapolation"):
        interp(5.0)
    _ = interp(5.0, allow_extrapolation=True)


def test_update_refreshes_after_mutating_y() -> None:
    """Mutating the pillar data and calling ``update()`` rebuilds the cubic."""
    xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    ys = np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    interp = CubicNaturalSpline(xs, ys)
    before = interp(1.5)
    interp._ys[:] = 2.0 * ys  # pyright: ignore[reportPrivateUsage]
    interp.update()
    # The spline is linear in y, so doubling y doubles the interpolant.
    tolerance.tight(interp(1.5), 2.0 * before)


# ---------------------------------------------------------------------------
# the pre-existing cluster/l9a reference — a second, independent pin
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def l9a() -> dict[str, Any]:
    return reference_reader.load("cluster/l9a")


def test_natural_spline_matches_l9a(l9a: dict[str, Any]) -> None:
    block = l9a["cubic_natural_spline"]
    xs = np.asarray(block["xs"], dtype=np.float64)
    ys = np.asarray(block["ys"], dtype=np.float64)
    interp = CubicNaturalSpline(xs, ys)
    for x, y in zip(block["xs"], block["pillars"], strict=True):
        tolerance.tight(interp(float(x)), float(y))
    for x, y in zip(block["mids_x"], block["mids_y"], strict=True):
        tolerance.tight(interp(float(x)), float(y))
    tolerance.tight(interp.derivative(1.5), float(block["derivative_at_1_5"]))
    tolerance.tight(
        interp.second_derivative(1.5), float(block["second_derivative_at_1_5"])
    )
    tolerance.tight(interp.primitive(2.5), float(block["primitive_at_2_5"]))


def test_monotonic_cubic_matches_l9a(l9a: dict[str, Any]) -> None:
    block = l9a["monotonic_cubic_natural_spline"]
    xs = np.asarray(block["xs"], dtype=np.float64)
    ys = np.asarray(block["ys"], dtype=np.float64)
    interp = MonotonicCubicNaturalSpline(xs, ys)
    for x, y in zip(block["xs"], block["pillars"], strict=True):
        tolerance.tight(interp(float(x)), float(y))
    for x, y in zip(block["mids_x"], block["mids_y"], strict=True):
        tolerance.tight(interp(float(x)), float(y))


def test_monotonic_cubic_preserves_monotonicity() -> None:
    """The filter's contract: a monotone input stays monotone on a fine grid."""
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    ys = np.array([0.0, 0.5, 1.5, 3.0, 3.2], dtype=np.float64)
    interp = MonotonicCubicNaturalSpline(xs, ys)
    values = [interp(float(x)) for x in np.linspace(0.0, 4.0, 401)]
    diffs = np.diff(values)
    assert (diffs >= -1e-15).all(), f"non-monotonic: min diff {diffs.min()}"
