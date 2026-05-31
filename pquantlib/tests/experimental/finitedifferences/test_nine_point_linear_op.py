"""Tests for NinePointLinearOp + SecondOrderMixedDerivativeOp.

# C++ parity: ql/methods/finitedifferences/operators/ninepointlinearop.hpp +
# ql/methods/finitedifferences/operators/secondordermixedderivativeop.hpp.

These ops are exercised end-to-end via FdmKlugeExtOUOp; here we
add unit-level smoke tests for the basic apply / mult contracts.
"""

from __future__ import annotations

import numpy as np

from pquantlib.experimental.finitedifferences.nine_point_linear_op import (
    NinePointLinearOp,
)
from pquantlib.experimental.finitedifferences.second_order_mixed_derivative_op import (
    SecondOrderMixedDerivativeOp,
)
from pquantlib.methods.finitedifferences.meshers.fdm_mesher_composite import (
    FdmMesherComposite,
)
from pquantlib.methods.finitedifferences.meshers.uniform_1d_mesher import (
    Uniform1dMesher,
)
from pquantlib.testing import tolerance


def test_nine_point_apply_zero_op() -> None:
    """Default NinePointLinearOp (all coefficients zero) -> apply returns zeros.

    The base class does NOT populate coefficients; only subclasses do.
    """
    m1 = Uniform1dMesher(0.0, 1.0, 5)
    m2 = Uniform1dMesher(0.0, 1.0, 5)
    mesher = FdmMesherComposite(m1, m2)
    op = NinePointLinearOp(0, 1, mesher)
    r = np.arange(25, dtype=np.float64)
    out = op.apply(r)
    assert np.array_equal(out, np.zeros(25))


def test_second_order_mixed_derivative_interior_const_zero() -> None:
    """On a constant function the mixed second derivative is zero everywhere.

    TIGHT: d2/(dx dy) of a constant is exactly 0.
    """
    m1 = Uniform1dMesher(0.0, 1.0, 5)
    m2 = Uniform1dMesher(0.0, 1.0, 5)
    mesher = FdmMesherComposite(m1, m2)
    op = SecondOrderMixedDerivativeOp(0, 1, mesher)
    ones = np.ones(25, dtype=np.float64)
    out = op.apply(ones)
    # Boundaries may not be exactly 0 due to the one-sided stencil; check
    # the central node (coord (2, 2)) is exactly 0.
    central_idx = mesher.layout().index((2, 2))
    tolerance.tight(float(out[central_idx]), 0.0)


def test_second_order_mixed_derivative_bilinear_xy() -> None:
    """For f(x, y) = x * y, d2f/dxdy = 1 at the interior.

    TIGHT: closed-form result.
    """
    m1 = Uniform1dMesher(0.0, 1.0, 7)
    m2 = Uniform1dMesher(0.0, 1.0, 7)
    mesher = FdmMesherComposite(m1, m2)
    op = SecondOrderMixedDerivativeOp(0, 1, mesher)
    # Build x*y at every node.
    layout = mesher.layout()
    r = np.empty(layout.size(), dtype=np.float64)
    for iter_ in layout.iter():
        x = mesher.location(iter_, 0)
        y = mesher.location(iter_, 1)
        r[iter_.index] = x * y
    out = op.apply(r)
    # Pick an interior coordinate (3, 3).
    central_idx = layout.index((3, 3))
    tolerance.tight(float(out[central_idx]), 1.0)


def test_nine_point_mult_scales_coefficients() -> None:
    """``mult`` scales every coefficient by ``u[i]`` -> apply scales linearly.

    TIGHT.
    """
    m1 = Uniform1dMesher(0.0, 1.0, 5)
    m2 = Uniform1dMesher(0.0, 1.0, 5)
    mesher = FdmMesherComposite(m1, m2)
    base = SecondOrderMixedDerivativeOp(0, 1, mesher)
    layout = mesher.layout()
    n = layout.size()
    r = np.empty(n, dtype=np.float64)
    for iter_ in layout.iter():
        x = mesher.location(iter_, 0)
        y = mesher.location(iter_, 1)
        r[iter_.index] = x * y

    scaled = base.mult(np.full(n, 3.0, dtype=np.float64))
    base_out = base.apply(r)
    scaled_out = scaled.apply(r)
    central_idx = layout.index((2, 2))
    tolerance.tight(float(scaled_out[central_idx]), 3.0 * float(base_out[central_idx]))
