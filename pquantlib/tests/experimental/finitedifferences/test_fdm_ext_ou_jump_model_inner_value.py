"""Tests for FdmExtOUJumpModelInnerValue.

# C++ parity: ql/experimental/finitedifferences/fdmextoujumpmodelinnervalue.hpp
# @ v1.42.1.

Cross-validates against ``fdm_ext_ou_jump_model_inner_value`` section
of ``migration-harness/references/cluster/w5c.json``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from pquantlib.experimental.finitedifferences.fdm_ext_ou_jump_model_inner_value import (
    FdmExtOUJumpModelInnerValue,
)
from pquantlib.methods.finitedifferences.meshers.uniform_grid_mesher import (
    UniformGridMesher,
)
from pquantlib.methods.finitedifferences.operators.fdm_linear_op_layout import (
    FdmLinearOpLayout,
)
from pquantlib.payoffs import OptionType, PlainVanillaPayoff
from pquantlib.testing.reference_reader import load as load_reference
from pquantlib.testing.tolerance import tight


@pytest.fixture
def reference_data() -> dict[str, Any]:
    return load_reference("cluster/w5c")["fdm_ext_ou_jump_model_inner_value"]


def test_inner_value_matches_cpp(reference_data: dict[str, Any]) -> None:
    """Per-node inner values match the C++ probe to TIGHT precision.

    The lookup + exp + payoff are all closed-form transcendentals;
    the only divergence sources would be math library differences
    in ``exp`` between the platform libm in C++ and Python's math.exp
    (both delegate to libm — bit-identical on the same machine).
    """
    nx = reference_data["nx"]
    ny = reference_data["ny"]
    layout = FdmLinearOpLayout((nx, ny))
    mesher = UniformGridMesher(layout, [(-1.0, 1.0), (-0.5, 0.5)])
    payoff = PlainVanillaPayoff(OptionType.Call, reference_data["strike"])
    shape: list[tuple[float, float]] = [
        (entry[0], entry[1]) for entry in reference_data["shape"]
    ]
    calc = FdmExtOUJumpModelInnerValue(payoff, mesher, shape)

    actual: list[float] = []
    for iter_ in layout.iter():
        actual.append(calc.inner_value(iter_, reference_data["t_test"]))

    expected = reference_data["inner_values"]
    for actual_v, expected_v in zip(actual, expected, strict=True):
        tight(float(actual_v), float(expected_v))


def test_avg_inner_value_equals_inner_value() -> None:
    """``avg_inner_value`` is an alias of ``inner_value`` per C++."""
    layout = FdmLinearOpLayout((3, 3))
    mesher = UniformGridMesher(layout, [(-1.0, 1.0), (-1.0, 1.0)])
    payoff = PlainVanillaPayoff(OptionType.Call, 5.0)
    shape: list[tuple[float, float]] = [(0.0, 0.0), (1.0, 1.0)]
    calc = FdmExtOUJumpModelInnerValue(payoff, mesher, shape)
    iter_ = next(iter(layout.iter()))
    tight(
        calc.avg_inner_value(iter_, 0.5),
        calc.inner_value(iter_, 0.5),
    )


def test_inner_value_no_shape_means_zero_offset() -> None:
    """``shape=None`` ⇔ ``f(t) = 0``, so inner value = payoff(exp(x + y))."""
    layout = FdmLinearOpLayout((2, 2))
    mesher = UniformGridMesher(layout, [(0.0, 1.0), (0.0, 1.0)])
    payoff = PlainVanillaPayoff(OptionType.Call, 1.0)
    calc = FdmExtOUJumpModelInnerValue(payoff, mesher, shape=None)
    # First iter is (0, 0) → x=0, y=0 → exp(0+0+0)=1 → payoff(1) = max(1-1,0) = 0.
    iter0 = next(iter(layout.iter()))
    tight(calc.inner_value(iter0, 0.5), 0.0)
    # Last iter is (1, 1) → x=1, y=1 → exp(0+1+1) = e^2 ≈ 7.389 →
    # payoff(7.389) = max(7.389 - 1, 0) = 6.389.
    *_, iter_last = layout.iter()
    expected = math.exp(2.0) - 1.0
    tight(calc.inner_value(iter_last, 0.5), expected)
