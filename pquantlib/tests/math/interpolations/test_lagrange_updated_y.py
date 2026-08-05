"""Cross-validate the UpdatedYInterpolation path against the v1.43 C++ probe.

Reference: ``migration-harness/references/v143/math/interp/kernel.json`` —
``lagrange_updated_y`` section. Six non-uniform nodes (so the barycentric
weights are asymmetric), evaluated at the nodes, between them and outside the
range on both sides.

The class under test is ``detail::UpdatedYInterpolation``, whose one method
``updatedValue(y, x)`` re-uses the cached barycentric weights against a fresh
ordinate vector. C++ reaches it by ``static_cast``-ing ``LagrangeInterpolation``'s
type-erased impl; the port makes it an ABC that ``LagrangeInterpolation``
implements.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pquantlib.math.interpolations.lagrange_interpolation import (
    LagrangeInterpolation,
    UpdatedYInterpolation,
)
from pquantlib.testing import reference_reader, tolerance


@pytest.fixture(scope="module")
def cpp() -> dict[str, Any]:
    return reference_reader.load("v143/math/interp/kernel")


def _make(block: dict[str, Any]) -> LagrangeInterpolation:
    interp = LagrangeInterpolation(
        np.asarray(block["xs"], dtype=np.float64),
        np.asarray(block["ys"], dtype=np.float64),
    )
    interp.enable_extrapolation()
    return interp


def test_plain_values_match_cpp_tight(cpp: dict[str, Any]) -> None:
    """Baseline: ``value(x)`` matches C++ to TIGHT on the same nodes."""
    block = cpp["lagrange_updated_y"]
    interp = _make(block)
    for x, expected in zip(block["eval_xs"], block["values"], strict=True):
        tolerance.tight(interp(float(x)), float(expected))


def test_updated_values_match_cpp_tight(cpp: dict[str, Any]) -> None:
    """``updated_value(y2, x)`` matches C++ ``value(Array, Real)`` to TIGHT.

    Same barycentric weights, different ordinates: the whole point is that
    the ``lambda`` vector depends only on the abscissae, so no refit happens.
    """
    block = cpp["lagrange_updated_y"]
    interp = _make(block)
    y2 = np.asarray(block["y2"], dtype=np.float64)
    for x, expected in zip(block["eval_xs"], block["updated_values"], strict=True):
        tolerance.tight(interp.updated_value(y2, float(x)), float(expected))


def test_updating_with_the_original_y_reproduces_value(cpp: dict[str, Any]) -> None:
    """Feeding the construction ordinates back must give ``value(x)`` again.

    C++ pins this separately, and it is the check that catches a port which
    accidentally reads the cached ``yBegin_`` instead of the argument.
    """
    block = cpp["lagrange_updated_y"]
    interp = _make(block)
    y0 = np.asarray(block["ys"], dtype=np.float64)
    for x, expected in zip(block["eval_xs"], block["values_with_own_y"], strict=True):
        tolerance.tight(interp.updated_value(y0, float(x)), float(expected))
        tolerance.tight(interp.updated_value(y0, float(x)), interp(float(x)))


def test_derivatives_match_cpp_tight(cpp: dict[str, Any]) -> None:
    """The barycentric derivative, including its on-node special case."""
    block = cpp["lagrange_updated_y"]
    interp = _make(block)
    for x, expected in zip(block["eval_xs"], block["derivatives"], strict=True):
        tolerance.tight(interp.derivative(float(x)), float(expected))


def test_value_with_is_the_same_method(cpp: dict[str, Any]) -> None:
    """The pre-existing ``value_with`` spelling still works (CLV models use it)."""
    block = cpp["lagrange_updated_y"]
    interp = _make(block)
    y2 = np.asarray(block["y2"], dtype=np.float64)
    for x in block["eval_xs"]:
        tolerance.exact(interp.value_with(y2, float(x)), interp.updated_value(y2, float(x)))


def test_implements_the_updated_y_interface(cpp: dict[str, Any]) -> None:
    """C++ ``LagrangeInterpolationImpl`` derives from ``UpdatedYInterpolation``."""
    assert isinstance(_make(cpp["lagrange_updated_y"]), UpdatedYInterpolation)


def test_wrong_length_y_is_rejected(cpp: dict[str, Any]) -> None:
    """C++ indexes ``y`` by node count without checking; we check first."""
    block = cpp["lagrange_updated_y"]
    interp = _make(block)
    with pytest.raises(Exception, match="node count"):
        interp.updated_value(np.array([1.0, 2.0], dtype=np.float64), 0.0)
