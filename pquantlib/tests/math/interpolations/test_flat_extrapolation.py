"""Cross-validate FlatExtrapolator against the ``v143/flatextrapolation`` probe.

FlatExtrapolator is new in C++ QuantLib v1.43. The decorator is small, so what
the probe pins is behaviour at the seams rather than arithmetic in bulk: the
strict (rather than inclusive) boundary test in ``derivative`` /
``second_derivative``, the linear — not flat — extension of ``primitive``, and
the fact that the decorator owns its extrapolation flag independently of the
interpolation it wraps. Each of those is a plausible port bug that a
value-only check would not catch.

Three underlyings are decorated — natural cubic spline, not-a-knot cubic
spline and linear — plus raw-underlying reference blocks, so a failure can be
localised to the decorator versus the interpolator underneath it.

Tolerance: TIGHT. Pure interpolation arithmetic, no iteration and no
transcendental accumulation.

Not asserted: the probe also records ``empty()`` for each decorator. C++
``Interpolation::empty()`` is the PIMPL null-check (``!impl_``), which the
Python port deliberately has no analogue for — an ``Interpolation`` instance
here is always constructed. Documented carve-out of the base class, not of
this decorator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pquantlib.exceptions import LibraryException
from pquantlib.math.interpolations.cubic_interpolation import (
    BoundaryCondition,
    CubicInterpolation,
    DerivativeApprox,
)
from pquantlib.math.interpolations.flat_extrapolation import FlatExtrapolator
from pquantlib.math.interpolations.interpolation import Interpolation
from pquantlib.math.interpolations.linear import LinearInterpolation
from pquantlib.testing import reference_reader, tolerance

# The probe's data, verbatim.
X = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
Y_CUBIC = np.array([5.0, 3.0, 4.0, 2.0, 1.0], dtype=np.float64)
Y_LINEAR = np.array([1.0, 2.5, 2.0, 4.0, 3.5], dtype=np.float64)


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/flatextrapolation")


# --- underlyings — mirror the probe's definitions exactly -------------------


def _natural_cubic() -> CubicInterpolation:
    return CubicInterpolation(
        X,
        Y_CUBIC,
        derivative_approx=DerivativeApprox.Spline,
        monotonic=False,
        left_condition=BoundaryCondition.SecondDerivative,
        left_value=0.0,
        right_condition=BoundaryCondition.SecondDerivative,
        right_value=0.0,
    )


def _not_a_knot_cubic() -> CubicInterpolation:
    return CubicInterpolation(
        X,
        Y_CUBIC,
        derivative_approx=DerivativeApprox.Spline,
        monotonic=False,
        left_condition=BoundaryCondition.NotAKnot,
        left_value=0.0,
        right_condition=BoundaryCondition.NotAKnot,
        right_value=0.0,
    )


def _linear() -> LinearInterpolation:
    return LinearInterpolation(X, Y_LINEAR)


def _extrapolating(decorated: Interpolation) -> FlatExtrapolator:
    """The decorator with extrapolation enabled, as every sampled case needs."""
    decorated.enable_extrapolation()
    flat = FlatExtrapolator(decorated)
    flat.enable_extrapolation()
    return flat


# --- helpers ----------------------------------------------------------------


def _check_sampled(
    cpp: dict[str, Any], case: str, f: Callable[[float], float], suffix: str = "values"
) -> None:
    """Evaluate ``f`` at the probe's abscissae and compare with its recording."""
    xs = cpp[f"{case}_x"]
    expected = cpp[f"{case}_{suffix}"]
    assert len(xs) == len(expected)
    for x, want in zip(xs, expected, strict=True):
        tolerance.tight(f(float(x)), float(want), reason=f"{case} at x={x}")


# --- natural cubic spline ---------------------------------------------------


def test_cubic_values(cpp: dict[str, Any]) -> None:
    flat = _extrapolating(_natural_cubic())
    for case in (
        "cubic_value_at_nodes",
        "cubic_value_in_range_midpoints",
        "cubic_value_below_range",
        "cubic_value_above_range",
        "cubic_value_at_endpoints",
    ):
        _check_sampled(cpp, case, flat)


def test_cubic_derivatives(cpp: dict[str, Any]) -> None:
    flat = _extrapolating(_natural_cubic())
    for case in (
        "cubic_derivative_in_range",
        "cubic_derivative_at_endpoints",
        "cubic_derivative_outside_range",
    ):
        _check_sampled(cpp, case, flat.derivative)
    for case in (
        "cubic_second_derivative_in_range",
        "cubic_second_derivative_at_endpoints",
        "cubic_second_derivative_outside_range",
    ):
        _check_sampled(cpp, case, flat.second_derivative)


def test_cubic_primitive_extends_linearly(cpp: dict[str, Any]) -> None:
    """Outside the range the integrand is constant, so the primitive is affine.

    Not flat and not clamped: below ``x_min`` the slope is ``y[0]``, above
    ``x_max`` it is ``y[-1]``.
    """
    flat = _extrapolating(_natural_cubic())
    for case in (
        "cubic_primitive_in_range",
        "cubic_primitive_at_endpoints",
        "cubic_primitive_below_range",
        "cubic_primitive_above_range",
    ):
        _check_sampled(cpp, case, flat.primitive)


def test_cubic_accessors_delegate(cpp: dict[str, Any]) -> None:
    flat = _extrapolating(_natural_cubic())
    tolerance.exact(flat.x_min, float(cpp["cubic_accessors_x_min"]))
    tolerance.exact(flat.x_max, float(cpp["cubic_accessors_x_max"]))
    assert flat.x_values.tolist() == cpp["cubic_accessors_x_values"]
    assert flat.y_values.tolist() == cpp["cubic_accessors_y_values"]
    assert flat.x_values.size == cpp["cubic_accessors_size"]
    assert flat.allows_extrapolation is cpp["cubic_accessors_allows_extrapolation"]
    assert flat.is_in_range(flat.x_min) is cpp["cubic_accessors_is_in_range_x_min"]
    assert flat.is_in_range(flat.x_max) is cpp["cubic_accessors_is_in_range_x_max"]
    assert flat.is_in_range(2.0) is cpp["cubic_accessors_is_in_range_2"]
    assert flat.is_in_range(-1.0) is cpp["cubic_accessors_is_in_range_minus1"]
    assert flat.is_in_range(5.0) is cpp["cubic_accessors_is_in_range_5"]


def test_cubic_update_delegates_to_the_decorated_interpolation(cpp: dict[str, Any]) -> None:
    flat = _extrapolating(_natural_cubic())
    tolerance.tight(flat(2.0), float(cpp["cubic_value_at_2_before_update"]))
    flat.update()
    tolerance.tight(flat(2.0), float(cpp["cubic_value_at_2_after_update"]))


def test_decorator_owns_its_extrapolation_flag(cpp: dict[str, Any]) -> None:
    """Enabling extrapolation on the wrapped interpolation must NOT propagate.

    Getting this wrong turns a loud error into a silently wrong number, so it
    is asserted directly rather than inferred. Internally the decorator always
    passes ``allow_extrapolation=True`` downward, so the decorated object's own
    flag never matters either way.
    """
    decorated = _natural_cubic()
    decorated.enable_extrapolation()
    flat = FlatExtrapolator(decorated)

    assert decorated.allows_extrapolation is cpp["cubic_default_decorated_allows_extrapolation"]
    assert flat.allows_extrapolation is cpp["cubic_default_decorator_allows_extrapolation"]

    checks: list[tuple[str, Callable[[], float]]] = [
        ("cubic_default_value_below_throws", lambda: flat(-1.0)),
        ("cubic_default_value_above_throws", lambda: flat(5.0)),
        ("cubic_default_derivative_below_throws", lambda: flat.derivative(-1.0)),
        ("cubic_default_second_derivative_above_throws", lambda: flat.second_derivative(5.0)),
        ("cubic_default_primitive_below_throws", lambda: flat.primitive(-1.0)),
        ("cubic_default_value_in_range_throws", lambda: flat(2.0)),
    ]
    for key, call in checks:
        if cpp[key]:
            with pytest.raises(LibraryException, match="extrapolation"):
                call()
        else:
            call()  # must not raise

    # Once enabled on the decorator itself, the same calls succeed and clamp.
    flat.enable_extrapolation()
    tolerance.tight(flat(-1.0), 5.0)
    tolerance.tight(flat(5.0), 1.0)


# --- not-a-knot cubic spline: the discriminating boundary case -------------


def test_not_a_knot_strict_boundary(cpp: dict[str, Any]) -> None:
    """A not-a-knot spline has a NONZERO endpoint second derivative.

    That is exactly what separates the strict ``x < x_min || x > x_max`` test
    from an inclusive one: with ``<=`` / ``>=`` the endpoint would wrongly
    return zero, and a natural spline (second derivative 0 at both ends by
    construction) could never reveal it.
    """
    flat = _extrapolating(_not_a_knot_cubic())
    _check_sampled(cpp, "notaknot_value_outside_range", flat)
    _check_sampled(cpp, "notaknot_derivative_at_endpoints", flat.derivative)
    _check_sampled(cpp, "notaknot_second_derivative_at_endpoints", flat.second_derivative)
    _check_sampled(cpp, "notaknot_second_derivative_outside_range", flat.second_derivative)
    _check_sampled(cpp, "notaknot_primitive_outside_range", flat.primitive)

    at_endpoints = cpp["notaknot_second_derivative_at_endpoints_values"]
    assert abs(float(at_endpoints[0])) > 1.0, (
        "the not-a-knot endpoint second derivative must be nonzero for this test to discriminate"
    )
    assert flat.second_derivative(4.01) == 0.0


# --- linear underlying ------------------------------------------------------


def test_linear_underlying(cpp: dict[str, Any]) -> None:
    """Linear is independent of the spline solver, so the algebra is hand-checkable.

    Below the range the value is ``y[0] = 1.0`` and the primitive is
    ``0 + 1.0 * x``; above it the value is ``y[-1] = 3.5``.
    """
    flat = _extrapolating(_linear())
    _check_sampled(cpp, "linear_value_below_in_above_range", flat)
    _check_sampled(cpp, "linear_derivative", flat.derivative)
    _check_sampled(cpp, "linear_second_derivative", flat.second_derivative)
    _check_sampled(cpp, "linear_primitive", flat.primitive)


def test_linear_accessors(cpp: dict[str, Any]) -> None:
    flat = _extrapolating(_linear())
    tolerance.exact(flat.x_min, float(cpp["linear_accessors_x_min"]))
    tolerance.exact(flat.x_max, float(cpp["linear_accessors_x_max"]))
    assert flat.x_values.tolist() == cpp["linear_accessors_x_values"]
    assert flat.y_values.tolist() == cpp["linear_accessors_y_values"]
    assert flat.x_values.size == cpp["linear_accessors_size"]
    assert flat.is_in_range(2.0) is cpp["linear_accessors_is_in_range_2"]
    assert flat.is_in_range(-1.0) is cpp["linear_accessors_is_in_range_minus1"]
    assert flat.is_in_range(5.0) is cpp["linear_accessors_is_in_range_5"]


def test_decorated_interpolation_is_exposed() -> None:
    decorated = _linear()
    assert FlatExtrapolator(decorated).decorated_interpolation is decorated


# --- the undecorated underlyings -------------------------------------------


@pytest.mark.parametrize(
    ("case", "build"),
    [
        ("cubic_underlying_spline_reference", _natural_cubic),
        ("notaknot_underlying_spline_reference", _not_a_knot_cubic),
        ("linear_underlying_reference", _linear),
    ],
)
def test_underlying_interpolations_are_unchanged(
    cpp: dict[str, Any], case: str, build: Callable[[], Interpolation]
) -> None:
    """A divergence here would silently shift every decorated result."""
    raw = build()
    raw.enable_extrapolation()
    _check_sampled(cpp, case, raw, suffix="value")
    _check_sampled(cpp, case, raw.derivative, suffix="derivative")
    _check_sampled(cpp, case, raw.second_derivative, suffix="second_derivative")
    _check_sampled(cpp, case, raw.primitive, suffix="primitive")
